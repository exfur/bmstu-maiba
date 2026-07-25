from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from transformers import pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def clean_and_normalise_dataframe(df: pd.DataFrame, datetime_cols: list = None, drop_dup: bool = True) -> pd.DataFrame:
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.strip().str.replace(" ", "_")

    # Remove hidden string placeholders
    placeholders = [r"^\s*$", r"^\?$", r"(?i)^null$", r"(?i)^n/a$", r"(?i)^none$"]
    target_text_cols = df_clean.select_dtypes(include=["object", "category"]).columns
    for pattern in placeholders:
        df_clean[target_text_cols] = df_clean[target_text_cols].replace(pattern, np.nan, regex=True)

    if datetime_cols:
        for col in datetime_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce")

    # Numeric formatting handling commas and currency symbols
    numeric_targets = [col for col in df_clean.columns if any(kw in col.lower() for kw in ["charge", "num", "amount", "total", "monthly", "tenure"])]
    for col in numeric_targets:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.replace(",", ".", regex=False)
            df_clean[col] = df_clean[col].str.replace(r"[^\d.]", "", regex=True)
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

            if "tenure" in df_clean.columns:
                df_clean[col] = np.where((df_clean["tenure"] == 0) & (df_clean[col].isnull()), 0.0, df_clean[col])

            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Binary flags normalization
    true_vars = {"YES", "Y", "1", "TRUE", "1.0"}
    false_vars = {"NO", "N", "0", "FALSE", "0.0"}

    for col in df_clean.select_dtypes(include=["object", "category"]).columns:
        unique_vals = set(df_clean[col].dropna().astype(str).str.strip().str.upper().unique())
        if unique_vals.issubset(true_vars.union(false_vars)) and len(unique_vals) > 0:
            str_col = df_clean[col].astype(str).str.strip().str.upper()
            df_clean.loc[str_col.isin(true_vars), col] = True
            df_clean.loc[str_col.isin(false_vars), col] = False
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            df_clean[col] = df_clean[col].astype(bool)

    if drop_dup:
        df_clean = df_clean.drop_duplicates()

    return df_clean


def compute_rfm_clusters(df_trans: pd.DataFrame, customer_id_col: str, date_col: str, amount_col: str, n_clusters: int = 3) -> pd.DataFrame:
    df = df_trans.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[amount_col] > 0]

    q99 = df[amount_col].quantile(0.99)
    df = df[df[amount_col] <= q99]
    if df.empty:
        return pd.DataFrame()

    snapshot_date = df[date_col].max() + pd.Timedelta(days=1)

    df_rfm = df.groupby(customer_id_col).agg(Recency=(date_col, lambda x: (snapshot_date - x.max()).days), Frequency=(amount_col, "count"), Monetary=(amount_col, "sum")).reset_index()

    actual_clusters = min(n_clusters, len(df_rfm))
    if actual_clusters < 2:
        df_rfm["Cluster_ID"] = 0
        return df_rfm

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(df_rfm[["Recency", "Frequency", "Monetary"]])

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
    df_rfm["Cluster_ID"] = kmeans.fit_predict(rfm_scaled)

    return df_rfm


def engineer_profile_features(df: pd.DataFrame, categorical_to_encode: list = None, numeric_to_scale: list = None) -> pd.DataFrame:
    df_eng = df.copy()

    if all(c in df_eng.columns for c in ["TotalCharges", "tenure"]):
        df_eng["Avg_Charge_Per_Month"] = np.where(df_eng["tenure"] > 0, df_eng["TotalCharges"] / df_eng["tenure"], 0.0)

    # Fixed binning upper bound bug: np.inf avoids NaN for tenure > 100
    if "tenure" in df_eng.columns:
        df_eng["Loyalty_Tier"] = pd.cut(df_eng["tenure"], bins=[-1, 12, 48, np.inf], labels=["Newbie", "Loyal", "Veteran"])
        cat_list = list(categorical_to_encode) if categorical_to_encode else []
        if "Loyalty_Tier" not in cat_list:
            cat_list.append("Loyalty_Tier")
        categorical_to_encode = cat_list

    if categorical_to_encode:
        actual_cat = [c for c in categorical_to_encode if c in df_eng.columns]
        if actual_cat:
            df_eng = pd.get_dummies(df_eng, columns=actual_cat, drop_first=True, dtype=float)

    if numeric_to_scale:
        actual_num = [c for c in numeric_to_scale if c in df_eng.columns]
        if actual_num:
            scaler = StandardScaler()
            df_eng[actual_num] = scaler.fit_transform(df_eng[actual_num])

    return df_eng


def extract_sentiment_features(df: pd.DataFrame, text_col: str, id_col: str, model_name: str = "tabularisai/multilingual-sentiment-analysis") -> pd.DataFrame:
    df_pipe = df.copy()
    clean_col = "Clean_Text_Tmp"
    df_pipe[clean_col] = df_pipe[text_col].astype(str).str.lower()
    df_pipe[clean_col] = df_pipe[clean_col].str.replace(r"[^\w\s]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()

    # Real Hugging Face Transformer inference with local weight caching
    if TRANSFORMERS_AVAILABLE:
        try:
            cache_dir = Path(__file__).resolve().parents[1] / "data" / "seminar_4_nlp_sentiment" / "models"
            cache_dir.mkdir(parents=True, exist_ok=True)
            nlp_pipeline = pipeline("text-classification", model=model_name, model_kwargs={"cache_dir": str(cache_dir)})

            def _score_fn(t):
                if not t or str(t).strip() == "":
                    return 0.0
                res = nlp_pipeline(str(t)[:512])[0]
                lbl = str(res["label"]).lower()
                if any(k in lbl for k in ["1", "2", "neg"]):
                    return -1.0
                if any(k in lbl for k in ["4", "5", "pos"]):
                    return 1.0
                return 0.0
        except Exception:
            _score_fn = lambda t: -1.0 if any(w in str(t).lower() for w in ["bad", "poor", "ужас"]) else (1.0 if any(w in str(t).lower() for w in ["good", "great", "отличн"]) else 0.0)
    else:
        _score_fn = lambda t: 0.0

    df_pipe["Sentiment_Score"] = df_pipe[clean_col].apply(_score_fn)
    df_result = df_pipe.groupby(id_col)["Sentiment_Score"].mean().reset_index()
    df_result.rename(columns={"Sentiment_Score": "Mean_Sentiment"}, inplace=True)
    return df_result


def assemble_abt(dfs_list: list, on_col: str) -> pd.DataFrame:
    if not dfs_list:
        return pd.DataFrame()
    df_final = dfs_list[0].copy()
    for df in dfs_list[1:]:
        if not df.empty:
            df_final = pd.merge(df_final, df, on=on_col, how="left")

    # CRITICAL FIX: Exclude target columns from fillna(0) to preserve batch inference NaNs
    target_cols = ["Churn", "Target_Flag", "Exited", "Default", "Converted"]
    feature_cols = [c for c in df_final.select_dtypes(include=[np.number]).columns if c not in target_cols]
    df_final[feature_cols] = df_final[feature_cols].fillna(0)

    if "Mean_Sentiment" in df_final.columns:
        df_final["Mean_Sentiment"] = df_final["Mean_Sentiment"].fillna(0.0)

    return df_final
