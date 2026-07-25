from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier

    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

PYCARET_AVAILABLE = False
USE_PYCARET_OOP = False

try:
    try:
        from pycaret.classification import ClassificationExperiment

        USE_PYCARET_OOP = True
        PYCARET_AVAILABLE = True
    except ImportError:
        from pycaret.classification import compare_models, finalize_model, pull, setup  # type: ignore

        USE_PYCARET_OOP = False
        PYCARET_AVAILABLE = True
except Exception:
    PYCARET_AVAILABLE = False


def get_project_models_dir() -> Path:
    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_default_output_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "powerbi"


def train_baseline_models(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list,
    task_type: str = "classification",
    output_dir: Path | str | None = None,
) -> dict:
    if output_dir is None:
        output_dir = get_default_output_dir()
    output_dir = Path(output_dir)

    df_historical = df[df[target_col].notna()]
    df_inference = df[df[target_col].isna()]

    if df_historical.empty:
        return {"error": "No labeled data found for training."}

    X = df_historical[feature_cols]
    y = df_historical[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    models_dir = get_project_models_dir()
    joblib.dump(model, models_dir / "model.pkl")

    if not df_inference.empty:
        df_inference[target_col + "_Predicted"] = model.predict(df_inference[feature_cols])
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "recovered_targets.csv"
        df_inference.to_csv(out_path, index=False)

    return {"model": model, "metrics": metrics, "inference_shape": df_inference.shape}


def _run_native_multimodel_benchmark(df: pd.DataFrame, target_col: str, output_dir: Path) -> tuple:
    df_clean = df.dropna(subset=[target_col])
    feature_cols = [c for c in df_clean.columns if c not in ["Target_ID", target_col, "Target_Flag_Predicted"] and pd.api.types.is_numeric_dtype(df_clean[c])]

    X = df_clean[feature_cols]
    y = df_clean[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    if LGBM_AVAILABLE:
        candidate_models["LightGBM"] = LGBMClassifier(random_state=42, verbose=-1)
    else:
        candidate_models["Gradient Boosting"] = GradientBoostingClassifier(random_state=42)

    leaderboard_rows = []
    best_model = None
    best_f1 = -1.0

    for name, clf in candidate_models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else 0.0
        auc = roc_auc_score(y_test, y_proba) if hasattr(clf, "predict_proba") else 0.0

        leaderboard_rows.append(
            {
                "Model": name,
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "AUC": round(auc, 4),
                "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                "Prec.": round(precision_score(y_test, y_pred, zero_division=0), 4),
                "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            }
        )

        f1 = f1_score(y_test, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_model = clf

    leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values(by="F1", ascending=False).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_df.to_csv(output_dir / "automl_leaderboard.csv", index=False)
    joblib.dump(best_model, get_project_models_dir() / "model.pkl")

    explainer = None
    if SHAP_AVAILABLE and best_model is not None:
        try:
            explainer = shap.Explainer(best_model, X_train)
        except Exception:
            explainer = None

    return best_model, explainer, leaderboard_df


def run_automl_and_explain(
    df: pd.DataFrame,
    target_col: str,
    task_type: str = "classification",
    output_dir: Path | str | None = None,
) -> tuple:
    if output_dir is None:
        output_dir = get_default_output_dir()
    output_dir = Path(output_dir)

    if not PYCARET_AVAILABLE:
        return _run_native_multimodel_benchmark(df, target_col, output_dir)

    try:
        if USE_PYCARET_OOP:
            exp = ClassificationExperiment()
            exp.setup(
                data=df,
                target=target_col,
                ignore_features=["Target_ID"] if "Target_ID" in df.columns else None,
                session_id=42,
                verbose=False,
            )
            best_model = exp.compare_models(include=["lr", "dt", "rf", "lightgbm"], verbose=False)
            leaderboard = exp.pull()
            final_model = exp.finalize_model(best_model)
        else:
            setup(
                data=df,
                target=target_col,
                ignore_features=["Target_ID"] if "Target_ID" in df.columns else None,
                session_id=42,
                verbose=False,
                html=False,
            )
            best_model = compare_models(include=["lr", "dt", "rf", "lightgbm"], verbose=False)
            leaderboard = pull()
            final_model = finalize_model(best_model)

        output_dir.mkdir(parents=True, exist_ok=True)
        leaderboard.to_csv(output_dir / "automl_leaderboard.csv", index=False)
        joblib.dump(final_model, get_project_models_dir() / "model.pkl")

        X_train = df.drop(columns=[target_col, "Target_ID"], errors="ignore")
        explainer = None
        if SHAP_AVAILABLE:
            try:
                explainer = shap.TreeExplainer(final_model)
            except Exception:
                explainer = None

        return final_model, explainer, leaderboard

    except Exception:
        return _run_native_multimodel_benchmark(df, target_col, output_dir)
