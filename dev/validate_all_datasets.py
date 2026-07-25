import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import yaml
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, precision_recall_curve, precision_score, r2_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from textblob import TextBlob

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
PROCESSED_DIR = os.path.join("data", "processed")
REPORT_DIR = os.path.join("data", "validation_reports")
CONFIG_FILE = "dataset_configs.yaml"

if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        SCHEMA = yaml.safe_load(f) or {}
else:
    SCHEMA = {}

ENGINEERED_NUM_COLS = ["tx_count", "tx_sum", "tx_avg", "review_count", "avg_review_length", "avg_sentiment"]


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_md(file_path, text):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(text + "\n\n")


def safe_markdown(df_or_series, index=True):
    try:
        return df_or_series.to_markdown(index=index)
    except ImportError:
        return df_or_series.to_string(index=index)


# ==========================================
# DATA ENGINEERING & CLEANING
# ==========================================
def clean_corrupted_features(df, dataset_name):
    df = df.copy()
    config = SCHEMA.get(dataset_name) or {}
    for col in config.get("numeric_columns") or []:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace("ERR_NAN", "", case=False, regex=False)
            df[col] = df[col].str.replace(r"[^\d\.\-]", "", regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_analytical_base_table(profiles, transactions, reviews, dataset_name):
    abt = clean_corrupted_features(profiles, dataset_name)

    if not transactions.empty and "Target_ID" in transactions.columns and "Trans_Amount" in transactions.columns:
        tx_agg = transactions.groupby("Target_ID").agg(tx_count=("Target_ID", "count"), tx_sum=("Trans_Amount", "sum"), tx_avg=("Trans_Amount", "mean")).reset_index()
        abt = abt.merge(tx_agg, on="Target_ID", how="left")
        abt[["tx_count", "tx_sum", "tx_avg"]] = abt[["tx_count", "tx_sum", "tx_avg"]].fillna(0)

    if not reviews.empty and "Target_ID" in reviews.columns and "Review_Text" in reviews.columns:
        reviews["review_length"] = reviews["Review_Text"].astype(str).apply(len)
        reviews["sentiment"] = reviews["Review_Text"].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
        rev_agg = reviews.groupby("Target_ID").agg(review_count=("Target_ID", "count"), avg_review_length=("review_length", "mean"), avg_sentiment=("sentiment", "mean")).reset_index()
        abt = abt.merge(rev_agg, on="Target_ID", how="left")
        abt[["review_count", "avg_review_length", "avg_sentiment"]] = abt[["review_count", "avg_review_length", "avg_sentiment"]].fillna(0)
    return abt


# ==========================================
# PHASE 2.1 & 2.2: EDA, CLUSTERING & NLP
# ==========================================
def run_eda_pipeline(abt, reviews, dataset_name, out_dir, md_file):
    write_md(md_file, f"# Глава 2. Анализ данных: {dataset_name}")
    write_md(md_file, "## 2.1. Оценка качества и исследовательский анализ (Wide ABT)")

    write_md(md_file, f"**Итоговая размерность Wide ABT:** {abt.shape[0]} строк, {abt.shape[1]} колонок.")

    dtypes = abt.dtypes.value_counts().reset_index()
    dtypes.columns = ["Тип данных", "Количество"]
    write_md(md_file, "**Распределение типов данных:**\n" + safe_markdown(dtypes, index=False))

    missing = (abt.isnull().sum() / len(abt) * 100).round(2)
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        write_md(md_file, "**Анализ пропусков (% от общего числа):**\n" + safe_markdown(missing.to_frame(name="% Пропусков")))

    config = SCHEMA.get(dataset_name) or {}
    base_num = config.get("numeric_columns") or []
    target = config.get("target_column", None)
    num_cols = [c for c in base_num + ENGINEERED_NUM_COLS if c in abt.columns]

    # [01] Missing Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(abt.isnull(), cbar=False, cmap="viridis", yticklabels=False)
    plt.title("01. Матрица пропусков (Data Completeness)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "01_missing_heatmap.png"))
    plt.close("all")

    write_md(md_file, "## 2.2. Описательная статистика и визуализация")
    if num_cols:
        write_md(md_file, "### Описательная статистика непрерывных признаков")
        stats = abt[num_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]].round(2)
        write_md(md_file, safe_markdown(stats))

    # [02] Target Dist
    if target and target in abt.columns:
        write_md(md_file, f"### Анализ целевой переменной: `{target}`")
        target_counts = abt[target].value_counts(normalize=True).round(4) * 100
        write_md(md_file, "**Баланс классов (%):**\n" + safe_markdown(target_counts.to_frame(name="Доля (%)")))

        plt.figure(figsize=(6, 4))
        sns.countplot(data=abt, x=target, palette="Set2")
        plt.title(f"02. Распределение целевой: {target}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "02_target_distribution.png"))
        plt.close("all")

    # [03 & 04] Histograms and Boxplots
    if num_cols:
        plot_cols = num_cols[:4]
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for i, col in enumerate(plot_cols):
            sns.histplot(abt, x=col, kde=True, ax=axes.flatten()[i], color="blue")
            axes.flatten()[i].set_title(f"Гистограмма: {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "03_numeric_histograms.png"))
        plt.close("all")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for i, col in enumerate(plot_cols):
            sns.boxplot(x=abt[col], ax=axes.flatten()[i], color="orange")
            axes.flatten()[i].set_title(f"Boxplot (Выбросы): {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "04_numeric_boxplots.png"))
        plt.close("all")

    # [05] Correlation Wide
    if len(num_cols) > 1:
        write_md(md_file, "### Корреляционный анализ")
        corr_cols = num_cols + ([target] if target in abt.columns else [])
        corr = abt[corr_cols].corr()
        write_md(md_file, "Матрица корреляций (Pearson):\n" + safe_markdown(corr.round(2)))

        plt.figure(figsize=(14, 12))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", center=0, annot_kws={"size": 8})
        plt.title("05. Матрица корреляций признаков (Wide ABT)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "05_correlation_matrix_wide.png"))
        plt.close("all")

        # [06] PCA Clustering
        write_md(md_file, "### Сегментация клиентов (K-Means Clustering)")

        # Filter to only the core continuous business variables to avoid Zero-Inflation banding
        base_cluster_cols = [c for c in base_num if c in abt.columns]

        if len(base_cluster_cols) >= 2:
            # 1. Автоматический детектор мультиколлинеарности (корреляция > 0.85)
            corr_matrix = abt[base_cluster_cols].corr().abs()
            # Берем только верхний треугольник матрицы, чтобы не удалить обе переменные из пары
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

            # Находим колонки, которые слишком сильно зависят от других
            to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
            indep_cols = [c for c in base_cluster_cols if c not in to_drop]

            # Защита от "пустого" датасета: если удалилось всё, оставляем первые два признака принудительно
            if len(indep_cols) < 2:
                indep_cols = base_cluster_cols[:2]
                to_drop = [c for c in to_drop if c not in indep_cols]

            # 2. Масштабирование и Кластеризация (только по независимым признакам)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(abt[indep_cols].fillna(abt[indep_cols].median()))

            kmeans = KMeans(n_clusters=3, random_state=42)
            abt["Customer_Segment"] = kmeans.fit_predict(X_scaled).astype(str)

            # 3. Отчет для студентов в Markdown
            write_md(md_file, "### Сегментация клиентов (Авто-разрешение мультиколлинеарности)")
            if to_drop:
                write_md(md_file, f"*Алгоритм обнаружил мультиколлинеарность (корреляция > 0.85) и автоматически исключил зависимые признаки (`{', '.join(to_drop)}`). Кластеризация K-Means (K=3) выполнена по чистым независимым осям: `{', '.join(indep_cols)}`.*")
            else:
                write_md(md_file, f"*Кластеризация K-Means (K=3) выполнена по независимым признакам: `{', '.join(indep_cols)}`.*")

            cluster_centers = abt.groupby("Customer_Segment")[num_cols].mean().round(2).reset_index()
            write_md(md_file, "**Сводные метрики центров распределений сегментов:**\n" + safe_markdown(cluster_centers, index=False))

            # 4. Динамическая визуализация
            plt.figure(figsize=(9, 6))
            if len(indep_cols) == 2:
                # Если ровно 2 признака — строим красивый 2D график по реальным осям
                sns.scatterplot(data=abt, x=indep_cols[0], y=indep_cols[1], hue="Customer_Segment", palette="Dark2", alpha=0.7)
                plt.title(f"06. Сегментация (K-Means K=3: {indep_cols[0]} vs {indep_cols[1]})")
            else:
                # Если независимых признаков больше 2 — применяем PCA исключительно для визуализации на плоскости
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=abt["Customer_Segment"], palette="Dark2", alpha=0.7)
                plt.title(f"06. Сегментация (K-Means K=3 на {len(indep_cols)} признаках, проекция PCA)")

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "06_kmeans_cluster_scatter.png"))
            plt.close("all")

    # [07 & 08] TF-IDF Split
    if not reviews.empty and "Review_Text" in reviews.columns and target in abt.columns:
        write_md(md_file, "### Углубленный семантический анализ отзывов (TF-IDF)")
        rev_merged = reviews.merge(abt[["Target_ID", target]], on="Target_ID", how="inner")
        for idx, (class_val, class_name) in enumerate(zip([0, 1], ["Лояльные (0)", "Отток/Целевые (1)"])):
            text_data = rev_merged[rev_merged[target] == class_val]["Review_Text"].dropna().astype(str)
            if len(text_data) > 10:
                tfidf = TfidfVectorizer(stop_words="english", max_features=15)
                tfidf_matrix = tfidf.fit_transform(text_data)
                tfidf_scores = pd.DataFrame({"Term": tfidf.get_feature_names_out(), "Score": tfidf_matrix.mean(axis=0).A1}).sort_values(by="Score", ascending=False)

                write_md(md_file, f"**Топ-10 маркеров для класса '{class_name}':**\n" + safe_markdown(tfidf_scores.head(10).round(4), index=False))

                plt.figure(figsize=(8, 5))
                sns.barplot(data=tfidf_scores, x="Score", y="Term", palette="Blues_r" if class_val == 0 else "Reds_r")
                plt.title(f"0{7 + idx}. Топ-15 маркеров для {class_name}")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"0{7 + idx}_nlp_tfidf_class_{class_val}.png"))
                plt.close("all")

    # [09] Sentiment vs Target
    if "avg_sentiment" in abt.columns and target in abt.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=abt, x=target, y="avg_sentiment", palette="Set3")
        plt.title(f"09. Влияние тональности отзывов на {target}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "09_nlp_sentiment_vs_target.png"))
        plt.close("all")


# ==========================================
# PHASE 2.3: MODELING, XAI, AND TIME SERIES
# ==========================================
def run_modeling_pipeline(abt, transactions, dataset_name, out_dir, md_file):
    write_md(md_file, "## 2.3. Построение моделей (Регрессия, Сравнение классификаторов, Временные ряды)")

    config = SCHEMA.get(dataset_name) or {}
    target = config.get("target_column")
    num_cols = [c for c in (config.get("numeric_columns") or []) + ENGINEERED_NUM_COLS if c in abt.columns]
    cat_cols = [c for c in (config.get("categorical_columns") or []) if c in abt.columns]

    # NEW: Force the modeling pipeline to use the unsupervised cluster as a predictive feature
    if "Customer_Segment" in abt.columns and "Customer_Segment" not in cat_cols:
        cat_cols.append("Customer_Segment")

    if target not in abt.columns:
        return

    # IMPUTATION
    inference_mask = abt[target].isna()
    df_train = abt[~inference_mask][num_cols + cat_cols + [target]].copy()
    df_infer = abt[inference_mask][num_cols + cat_cols].copy()

    for col in num_cols:
        df_train[col] = df_train[col].fillna(df_train[col].median())
        df_infer[col] = df_infer[col].fillna(df_train[col].median())
    for col in cat_cols:
        mode_val = df_train[col].mode()[0] if not df_train[col].mode().empty else "Unknown"
        df_train[col] = df_train[col].fillna(mode_val)
        df_infer[col] = df_infer[col].fillna(mode_val)

    df_train.dropna(subset=[target], inplace=True)
    if df_train.empty:
        return

    X_train_full = pd.get_dummies(df_train.drop(columns=[target]), columns=cat_cols, drop_first=True)
    y_train_full = df_train[target]
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    # [10] REGRESSION
    reg_target = next((col for col in ["tx_sum", "TotalCharges"] if col in X_train_full.columns), None)
    if reg_target:
        write_md(md_file, f"### 2.3.1. Регрессионный бенчмарк: Прогнозирование `{reg_target}`")
        X_reg = X_train_full.drop(columns=[reg_target])
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, X_train_full[reg_target], test_size=0.2, random_state=42)
        rf_reg = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
        rf_reg.fit(Xr_train, yr_train)

        reg_metrics = pd.DataFrame({"Модель": ["Random Forest Regressor"], "R2 Score": [r2_score(yr_test, rf_reg.predict(Xr_test))], "RMSE": [np.sqrt(mean_squared_error(yr_test, rf_reg.predict(Xr_test)))], "MAE": [mean_absolute_error(yr_test, rf_reg.predict(Xr_test))]})
        write_md(md_file, "**Метрики качества регрессии:**\n" + safe_markdown(reg_metrics.round(4), index=False))

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=yr_test, y=rf_reg.predict(Xr_test), alpha=0.5, color="green")
        plt.plot([yr_test.min(), yr_test.max()], [yr_test.min(), yr_test.max()], "k--", lw=2)
        plt.title(f"10. Random Forest Регрессия: Факт vs Прогноз ({reg_target})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "10_reg_actual_vs_pred.png"))
        plt.close("all")

    # CLASSIFICATION COMPARISON
    write_md(md_file, f"### 2.3.2. Сравнение моделей классификации: Прогнозирование `{target}`")
    clf_base = DecisionTreeClassifier(max_depth=3, random_state=42)  # Seminar 5 Default
    clf_adv = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=42)  # Refined

    clf_base.fit(Xc_train, yc_train)
    clf_adv.fit(Xc_train, yc_train)

    base_prob = clf_base.predict_proba(Xc_test)[:, 1]
    adv_prob = clf_adv.predict_proba(Xc_test)[:, 1]
    base_pred = clf_base.predict(Xc_test)
    adv_pred = clf_adv.predict(Xc_test)

    metrics_df = pd.DataFrame(
        {
            "Модель": ["Seminar 5 (Decision Tree)", "Refined (Random Forest)"],
            "Accuracy": [accuracy_score(yc_test, base_pred), accuracy_score(yc_test, adv_pred)],
            "Precision": [precision_score(yc_test, base_pred, zero_division=0), precision_score(yc_test, adv_pred, zero_division=0)],
            "Recall": [recall_score(yc_test, base_pred, zero_division=0), recall_score(yc_test, adv_pred, zero_division=0)],
            "ROC-AUC": [roc_auc_score(yc_test, base_prob), roc_auc_score(yc_test, adv_prob)],
        }
    )
    write_md(md_file, "**Таблица сравнения качества (Baseline vs Продвинутая модель):**\n" + safe_markdown(metrics_df.round(4), index=False))

    report = classification_report(yc_test, adv_pred, digits=4)
    write_md(md_file, "**Детализированный отчет классификации (Random Forest - Продвинутая):**\n```text\n" + report + "\n```")

    # [11] Metrics Bar Chart
    metrics_melted = metrics_df.melt(id_vars="Модель", var_name="Метрика", value_name="Значение")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_melted, x="Метрика", y="Значение", hue="Модель", palette="Set1")
    plt.title("11. Сравнение метрик (Базовая vs Продвинутая модель)")
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "11_clf_metrics_comparison.png"))
    plt.close("all")

    # [12] Dual ROC Curve
    plt.figure(figsize=(8, 6))
    fpr_b, tpr_b, _ = roc_curve(yc_test, base_prob)
    fpr_a, tpr_a, _ = roc_curve(yc_test, adv_prob)
    plt.plot(fpr_b, tpr_b, color="gray", lw=2, linestyle="--", label=f"Baseline AUC = {roc_auc_score(yc_test, base_prob):.2f}")
    plt.plot(fpr_a, tpr_a, color="darkorange", lw=2, label=f"Refined AUC = {roc_auc_score(yc_test, adv_prob):.2f}")
    plt.plot([0, 1], [0, 1], color="black", lw=1, linestyle=":")
    plt.title("12. ROC Кривая (Сравнение моделей)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "12_clf_roc_curve.png"))
    plt.close("all")

    # [13] Precision-Recall Curve (Advanced)
    precision, recall, _ = precision_recall_curve(yc_test, adv_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="purple", lw=2)
    plt.title("13. Precision-Recall Кривая (Refined Model)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "13_clf_pr_curve.png"))
    plt.close("all")

    # [14] Confusion Matrix (Advanced)
    cm = confusion_matrix(yc_test, adv_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("14. Матрица ошибок (Refined Model)")
    plt.ylabel("Истинный класс")
    plt.xlabel("Предсказанный класс")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "14_clf_confusion_matrix.png"))
    plt.close("all")

    # [15] SHAP Summary
    write_md(md_file, "### 2.3.3. Объяснимый ИИ (Explainable AI - SHAP)")
    write_md(md_file, "На графике `15_clf_shap_summary.png` отображены ключевые признаки, повлиявшие на решения продвинутой модели (Random Forest). Цвет точек указывает на значение признака (красный — высокое, синий — низкое), а положение на оси X — на силу и направление влияния на предсказание.")

    explainer = shap.TreeExplainer(clf_adv)
    X_sample = Xc_test.sample(min(300, len(Xc_test)), random_state=42)
    shap_values = explainer.shap_values(X_sample)
    vals = shap_values[1] if isinstance(shap_values, list) else (shap_values[:, :, 1] if len(np.shape(shap_values)) == 3 else shap_values)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(np.array(vals), X_sample, show=False)
    plt.title("15. Векторы влияния SHAP (Refined Model)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "15_clf_shap_summary.png"))
    plt.close("all")

    # RESTORE INFERENCE
    if not df_infer.empty:
        write_md(md_file, "### 2.3.4. Инференс: Восстановление пропусков целевой переменной")
        X_infer_full = pd.get_dummies(df_infer, columns=cat_cols, drop_first=True).reindex(columns=X_train_full.columns, fill_value=0)
        predicted_targets = clf_adv.predict(X_infer_full)
        abt.loc[df_infer.index, target] = predicted_targets
        write_md(md_file, f"Успешно восстановлено **{len(predicted_targets)}** скрытых значений `{target}` с помощью обученной модели Random Forest. Восстановленный вектор передан в алгоритм анализа временных рядов (Prophet).")

    # [16] PROPHET FORECAST
    if not transactions.empty and "Trans_Date" in transactions.columns:
        write_md(md_file, "### 2.3.5. Анализ временных рядов (Prophet Forecast)")
        tx_restored = transactions.merge(abt[["Target_ID", target]], on="Target_ID", how="inner")
        target_tx = tx_restored[tx_restored[target] == 1]

        if not target_tx.empty:
            ts_data = target_tx.groupby(pd.to_datetime(target_tx["Trans_Date"]).dt.date).size().reset_index(name="y")
            ts_data.rename(columns={"Trans_Date": "ds"}, inplace=True)
            if len(ts_data) >= 14:
                ts_data["floor"] = 0
                ts_data["cap"] = ts_data["y"].max() * 1.5

                m = Prophet(growth="logistic", changepoint_prior_scale=0.5, changepoint_range=0.98)
                m.fit(ts_data)

                future = m.make_future_dataframe(periods=30)
                future["floor"], future["cap"] = 0, ts_data["y"].max() * 1.5
                forecast = m.predict(future)
                forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)

                write_md(md_file, "*Настройки модели Prophet:* `growth='logistic', changepoint_prior_scale=0.5, changepoint_range=0.98, floor=0`. Применено жесткое отсечение нижней границы доверительного интервала (`yhat_lower = 0`) во избежание отрицательных прогнозов транзакционной активности.")

                fig = m.plot(forecast, figsize=(10, 5))
                plt.ylim(bottom=0)
                plt.title(f"16. Прогноз активности для класса {target}=1 (С учетом обрыва)")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "16_ts_prophet_restored_forecast.png"))
                plt.close("all")


def main():
    if not os.path.exists(CONFIG_FILE):
        return
    ensure_dir(REPORT_DIR)
    datasets = [d for d in os.listdir(PROCESSED_DIR) if os.path.isdir(os.path.join(PROCESSED_DIR, d))]

    for dataset in sorted(datasets):
        if dataset == "validation_reports":
            continue
        print(f"🚀 Running End-to-End Validation: {dataset}...")
        ds_path = os.path.join(PROCESSED_DIR, dataset)
        out_dir = os.path.join(REPORT_DIR, dataset)
        ensure_dir(out_dir)

        md_file = os.path.join(out_dir, "chapter_2_full_report.md")
        if os.path.exists(md_file):
            os.remove(md_file)

        try:
            profiles = pd.read_csv(os.path.join(ds_path, "profiles.csv")) if os.path.exists(os.path.join(ds_path, "profiles.csv")) else pd.DataFrame()
            transactions = pd.read_csv(os.path.join(ds_path, "transactions.csv")) if os.path.exists(os.path.join(ds_path, "transactions.csv")) else pd.DataFrame()
            reviews = pd.read_csv(os.path.join(ds_path, "reviews.csv")) if os.path.exists(os.path.join(ds_path, "reviews.csv")) else pd.DataFrame()
            if profiles.empty:
                continue

            abt = build_analytical_base_table(profiles, transactions, reviews, dataset)
            run_eda_pipeline(abt, reviews, dataset, out_dir, md_file)
            run_modeling_pipeline(abt, transactions, dataset, out_dir, md_file)

            print(f"  ✅ Comprehensive deep-dive complete for {dataset}.")
        except Exception as e:
            print(f"  ❌ Error processing {dataset}: {str(e)}")


if __name__ == "__main__":
    main()
