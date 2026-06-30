import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

try:
    import shap

    # type: ignore is used here because PyCaret dynamically loads modules at runtime,
    # which causes static analysis tools like Pylance to report missing attributes.
    from pycaret.classification import (  # type: ignore
        compare_models,  # type: ignore
        finalize_model,  # type: ignore
        pull,  # type: ignore
        save_model,  # type: ignore
        setup,  # type: ignore
    )
except ImportError:
    pass


def train_baseline_models(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list,
    task_type: str = "classification",
) -> dict:
    """
    Isolates inference pool, splits historical data, and trains baseline classical models.
    """
    df_historical = df[df[target_col].notna()]
    df_inference = df[df[target_col].isna()]

    if df_historical.empty:
        return {"error": "No labeled data found for training."}

    X = df_historical[feature_cols]
    y = df_historical[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    # Batch Inference on unlabeled dataset
    if not df_inference.empty:
        df_inference[target_col + "_Predicted"] = model.predict(
            df_inference[feature_cols]
        )
        out_path = os.path.join(
            os.getcwd(), "data", "processed", "recovered_targets.csv"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_inference.to_csv(out_path, index=False)

    return {"model": model, "metrics": metrics, "inference_shape": df_inference.shape}


def run_automl_and_explain(
    df: pd.DataFrame, target_col: str, task_type: str = "classification"
) -> tuple:
    """
    Industrial AutoML optimization utilizing PyCaret and initializing SHAP Explainer.
    Saves final artifact to disk.
    """
    print("Initializing PyCaret AutoML Environment...")
    # Removed the 'clf_setup =' assignment to resolve Ruff's F841 unused variable warning.
    # PyCaret's setup builds the context statefully behind the scenes.
    setup(
        data=df,
        target=target_col,
        ignore_features=["Target_ID"] if "Target_ID" in df.columns else None,
        session_id=42,
        verbose=False,
        html=False,
    )

    # Rapid prototype search over fast tree/linear algorithms
    best_model = compare_models(include=["lr", "dt", "rf", "lightgbm"], verbose=False)
    leaderboard = pull()

    final_model = finalize_model(best_model)

    X_train = df.drop(columns=[target_col, "Target_ID"], errors="ignore")

    try:
        explainer = shap.TreeExplainer(final_model)
        _ = explainer.shap_values(X_train.iloc[:5])
    except Exception:
        actual_model = (
            final_model.steps[-1][1] if hasattr(final_model, "steps") else final_model
        )
        try:
            explainer = shap.TreeExplainer(actual_model)
        except Exception:
            explainer = shap.Explainer(actual_model, X_train)

    os.makedirs("models", exist_ok=True)
    save_model(final_model, "models/model")

    return final_model, explainer, leaderboard
