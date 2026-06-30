import os
import pickle

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.data_pipeline import (
    assemble_abt,
    clean_and_normalise_dataframe,
    compute_rfm_clusters,
    engineer_profile_features,
    extract_sentiment_features,
)

app = FastAPI(title="MIIBA BI Integration Core API")

# Setup relative paths pointing to the generated serialized artifact
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None


class ClientData(BaseModel):
    features: dict
    transactions: list = []
    reviews: list = []


@app.post("/predict")
def predict_client_behavior(data: ClientData):
    if not model:
        return {
            "error": "Model artifact not found. Execute main.py orchestration first."
        }

    df_profile = pd.DataFrame([data.features])
    df_trans = pd.DataFrame(data.transactions) if data.transactions else pd.DataFrame()
    df_reviews = pd.DataFrame(data.reviews) if data.reviews else pd.DataFrame()

    df_profile_clean = clean_and_normalise_dataframe(
        df_profile, datetime_cols=["Join_Date"]
    )
    df_profile_eng = engineer_profile_features(
        df_profile_clean,
        categorical_to_encode=["gender", "Contract", "PaymentMethod", "TechSupport"],
        numeric_to_scale=["tenure", "MonthlyCharges", "TotalCharges"],
    )

    dfs_to_merge = [df_profile_eng]

    if not df_trans.empty:
        dfs_to_merge.append(
            compute_rfm_clusters(df_trans, "Target_ID", "Trans_Date", "Trans_Amount")
        )

    if not df_reviews.empty:
        dfs_to_merge.append(
            extract_sentiment_features(df_reviews, "Review_Text", "Target_ID")
        )

    df_abt = assemble_abt(dfs_to_merge, on_col="Target_ID")

    # Safe transform mapping to match model input vectors
    expected_features = (
        model.feature_names_in_
        if hasattr(model, "feature_names_in_")
        else df_abt.columns.drop("Target_ID")
    )
    for col in expected_features:
        if col not in df_abt.columns:
            df_abt[col] = 0.0

    X_processed = df_abt[expected_features]
    probability = model.predict_proba(X_processed)[0][1]

    return {
        "score": float(probability),
        "status": "High Risk" if probability > 0.65 else "Low Risk",
        "action": "Retention required" if probability > 0.65 else "Standard workflow",
    }
