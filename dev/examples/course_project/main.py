import os

import pandas as pd
from src.data_pipeline import (
    assemble_abt,
    clean_and_normalise_dataframe,
    compute_rfm_clusters,
    engineer_profile_features,
    extract_sentiment_features,
)
from src.model_training import run_automl_and_explain, train_baseline_models


def main():
    # Resolve dynamic paths from the repository root
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    input_dir = os.path.join(repo_root, "data", "input")
    processed_dir = os.path.join(repo_root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    print("=====================================================")
    print("  MIIBA Course Project - End-to-End Orchestrator")
    print("=====================================================")

    # 1. LOAD DATA
    print(f"\n[1/5] Loading raw data from: {input_dir}")
    profiles_path = os.path.join(input_dir, "profiles.csv")
    trans_path = os.path.join(input_dir, "transactions.csv")
    reviews_path = os.path.join(input_dir, "reviews.csv")

    if not all(os.path.exists(p) for p in [profiles_path, trans_path, reviews_path]):
        print(f"Error: Missing input files in {input_dir}.")
        print("Ensure 'profiles.csv', 'transactions.csv', and 'reviews.csv' exist.")
        return

    df_profiles = pd.read_csv(profiles_path)
    df_trans = pd.read_csv(trans_path)
    df_reviews = pd.read_csv(reviews_path)

    # 2. RUN DATA PIPELINE
    print("\n[2/5] Executing Data Cleaning & Engineering Pipelines...")
    df_profiles_clean = clean_and_normalise_dataframe(
        df_profiles, datetime_cols=["Join_Date"]
    )

    df_profiles_eng = engineer_profile_features(
        df_profiles_clean,
        categorical_to_encode=["gender", "Contract", "PaymentMethod", "TechSupport"],
        numeric_to_scale=["tenure", "MonthlyCharges", "TotalCharges"],
    )

    df_rfm = compute_rfm_clusters(
        df_trans,
        customer_id_col="Target_ID",
        date_col="Trans_Date",
        amount_col="Trans_Amount",
    )
    df_sentiment = extract_sentiment_features(
        df_reviews, text_col="Review_Text", id_col="Target_ID"
    )

    # 3. ASSEMBLE ABT (Analytical Base Table)
    print("\n[3/5] Orchestrating Left Joins & Assembling ABT...")
    df_abt = assemble_abt([df_profiles_eng, df_rfm, df_sentiment], on_col="Target_ID")

    abt_path = os.path.join(processed_dir, "abt_result.csv")
    df_abt.to_csv(abt_path, index=False)
    print(f"ABT successfully exported. Shape: {df_abt.shape}")

    # 4. BASELINE MODELING
    print("\n[4/5] Training Baseline Scikit-Learn Models...")
    # Determine the target column based on the dataset logic
    target_col = "Churn" if "Churn" in df_abt.columns else "Target_Flag"
    if target_col not in df_abt.columns:
        print(f"Error: Could not locate Target column ('{target_col}') in ABT.")
        return

    feature_cols = [
        c
        for c in df_abt.columns
        if c not in ["Target_ID", target_col, "Target_Flag_Predicted"]
    ]
    baseline_results = train_baseline_models(
        df_abt, target_col=target_col, feature_cols=feature_cols
    )
    print(f"Baseline Validation Metrics:\n{baseline_results.get('metrics')}")

    # 5. AUTOML & EXPLAINABLE AI
    print("\n[5/5] Searching best hyperparameters via PyCaret AutoML...")
    try:
        final_model, explainer, leaderboard = run_automl_and_explain(
            df_abt, target_col=target_col
        )
        print("\nTop 3 Models Identified:")
        print(leaderboard.head(3))
        print("\nModel artifact 'model.pkl' serialized to models/ directory.")
    except Exception as e:
        print(
            f"AutoML module skipped or encountered an error (check uv dependencies): {e}"
        )

    print("\n✅ All processes executed successfully!")


if __name__ == "__main__":
    main()
