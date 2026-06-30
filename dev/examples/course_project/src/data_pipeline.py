import os
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from transformers import pipeline

# Suppress Hugging Face/TensorFlow warnings for clean output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")


def clean_and_normalise_dataframe(
    df: pd.DataFrame, datetime_cols: list = None, drop_dup: bool = True
) -> pd.DataFrame:
    """
    Industrial ETL function for end-to-end cleaning and normalization of the profiles dataframe.
    Ensures numerical columns like 'tenure' are explicitly cast to prevent comparison type errors.
    """
    df_clean = df.copy()

    # 1. Clean headers
    df_clean.columns = df_clean.columns.str.strip().str.replace(" ", "_")

    # 2. Fix text placeholders
    placeholders = [r"^\s*$", r"^\?$", r"(?i)^null$", r"(?i)^n/a$", r"(?i)^none$"]
    for pattern in placeholders:
        df_clean = df_clean.replace(pattern, np.nan, regex=True)

    # 3. Process datetime fields
    if datetime_cols is not None:
        for col in datetime_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce")

    # 4. Clean tenure column first so it can be safely used for numerical conditional logic downstream
    if "tenure" in df_clean.columns:
        df_clean["tenure"] = pd.to_numeric(df_clean["tenure"], errors="coerce")
        if df_clean["tenure"].isnull().sum() > 0:
            df_clean["tenure"] = df_clean["tenure"].fillna(df_clean["tenure"].median())

    # 5. Clean other numerics & Fill NaN
    numeric_targets = ["TotalCharges", "MonthlyCharges", "Num_Feature_X"]
    for col in numeric_targets:
        if col in df_clean.columns:
            df_clean[col] = (
                df_clean[col].astype(str).str.replace(r"[^\d.]", "", regex=True)
            )
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
            if "tenure" in df_clean.columns:
                # Business logic: If customer is completely new (tenure == 0), assign zero charges
                df_clean[col] = np.where(
                    (df_clean["tenure"] == 0) & (df_clean[col].isnull()),
                    0.0,
                    df_clean[col],
                )
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # 6. Normalise Booleans and Categories
    true_variants = ["YES", "Y", "1", "TRUE", "1.0"]
    false_variants = ["NO", "N", "0", "FALSE", "0.0"]
    binary_cols = [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "Churn",
        "TechSupport",
        "Target_Flag",
    ]
    categorical_cols = ["gender", "Contract", "PaymentMethod", "InternetService"]

    for col in binary_cols:
        if col in df_clean.columns:
            str_col = df_clean[col].astype(str).str.strip().str.upper()
            df_clean.loc[str_col.isin(true_variants), col] = True
            df_clean.loc[str_col.isin(false_variants), col] = False
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            df_clean[col] = df_clean[col].astype(bool)

    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()

    # 7. Remove Duplicates
    if drop_dup:
        df_clean = df_clean.drop_duplicates()

    return df_clean


def engineer_profile_features(
    df: pd.DataFrame,
    categorical_to_encode: list | None = None,
    numeric_to_scale: list | None = None,
) -> pd.DataFrame:
    """
    Parametrized industrial function for Feature Engineering.
    Includes synthetic metrics calculation, OHE-encoding, and scaling.
    """
    df_engineered = df.copy()

    # 1. Synthetic Metrics
    if all(col in df_engineered.columns for col in ["TotalCharges", "tenure"]):
        df_engineered["Avg_Charge_Per_Month"] = np.where(
            df_engineered["tenure"] > 0,
            df_engineered["TotalCharges"] / df_engineered["tenure"],
            0.0,
        )

    # 2. Discretization (Loyalty Tiers based on tenure)
    if "tenure" in df_engineered.columns:
        df_engineered["Loyalty_Tier"] = pd.cut(
            df_engineered["tenure"],
            bins=[-1, 12, 48, 100],
            labels=["newbie", "loyal", "veteran"],
        )
        if categorical_to_encode and "Loyalty_Tier" not in categorical_to_encode:
            categorical_to_encode.append("Loyalty_Tier")

    # 3. One-Hot Encoding
    if categorical_to_encode:
        actual_cat = [c for c in categorical_to_encode if c in df_engineered.columns]
        if actual_cat:
            df_engineered = pd.get_dummies(
                df_engineered, columns=actual_cat, drop_first=True, dtype=float
            )

    # 4. Scaling
    if numeric_to_scale:
        actual_num = [c for c in numeric_to_scale if c in df_engineered.columns]
        if actual_num:
            scaler = StandardScaler()
            df_engineered[actual_num] = scaler.fit_transform(df_engineered[actual_num])

    return df_engineered


def compute_rfm_clusters(
    df_trans: pd.DataFrame,
    customer_id_col: str,
    date_col: str,
    amount_col: str,
    n_clusters: int = 3,
) -> pd.DataFrame:
    """
    Aggregates transaction logs, calculates behavioral RFM metrics, and segments users via K-Means.
    """
    df = df_trans.copy()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[amount_col] > 0]

    q99 = df[amount_col].quantile(0.99)
    df = df[df[amount_col] <= q99]

    if df.empty:
        return pd.DataFrame()

    snapshot_date = df[date_col].max() + pd.Timedelta(days=1)

    df_rfm = (
        df.groupby(customer_id_col)
        .agg(
            {
                date_col: lambda x: (snapshot_date - x.max()).days,
                customer_id_col: "count",
                amount_col: "sum",
            }
        )
        .rename(columns={customer_id_col: "Frequency"})
    )

    df_rfm = df_rfm.rename_axis(customer_id_col).reset_index()
    df_rfm.rename(columns={date_col: "Recency", amount_col: "Monetary"}, inplace=True)

    actual_clusters = min(n_clusters, len(df_rfm))

    if actual_clusters < 2:
        df_rfm["Cluster_ID"] = 0
        return df_rfm

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(df_rfm[["Recency", "Frequency", "Monetary"]])

    kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
    df_rfm["Cluster_ID"] = kmeans.fit_predict(rfm_scaled)

    return df_rfm


def extract_sentiment_features(
    df: pd.DataFrame,
    text_col: str,
    id_col: str,
    model_name: str = "data/seminar_4_nlp_sentiment/models",
) -> pd.DataFrame:
    """
    NLP sentiment extraction pipeline. Cleans text, runs batched Transformer inference.
    If local weights are missing or empty at model_name, automatically downloads them
    from the Hugging Face Hub, saves them locally to that path for future offline runs, and proceeds.
    """
    df_pipe = df.copy()

    # 1. Clean text
    clean_col = "Clean_Text_Tmp"
    df_pipe[clean_col] = df_pipe[text_col].astype(str).str.lower()
    df_pipe[clean_col] = df_pipe[clean_col].str.replace(r"[^\w\s]", " ", regex=True)
    df_pipe[clean_col] = (
        df_pipe[clean_col].str.replace(r"\s+", " ", regex=True).str.strip()
    )

    # 2. Robust Absolute Path Resolver
    resolved_path = os.path.abspath(model_name)

    # Fallback: Check climbing up from dev/ folder structure context
    if not os.path.exists(resolved_path) or not os.listdir(resolved_path):
        root_fallback = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", model_name)
        )
        if os.path.exists(root_fallback) and os.listdir(root_fallback):
            resolved_path = root_fallback

    # Determine if a valid local directory with cached model files is present
    is_local_present = (
        os.path.exists(resolved_path)
        and os.path.isdir(resolved_path)
        and len(os.listdir(resolved_path)) > 0
    )

    # 3. Conditional Pipeline Loading (Offline vs Self-Healing Installer)
    if is_local_present:
        print(
            f"⏳ Loading local Transformer weights from absolute path: {resolved_path}"
        )
        nlp_classifier = pipeline(
            "text-classification",
            model=resolved_path,
            tokenizer=resolved_path,
            truncation=True,
            max_length=512,
            local_files_only=True,  # Strictly lock framework to local files
        )
    else:
        # 🚀 "Install if not present" pipeline download & compilation engine
        remote_repo = "tabularisai/multilingual-sentiment-analysis"
        print(f"⚠️ Local weights not found or folder is empty at: '{resolved_path}'")
        print(f"📥 Downloading weights for '{remote_repo}' from Hugging Face Hub...")
        print(
            f"💾 Saving components locally to '{resolved_path}' for future offline runs..."
        )

        # Build the exact local directory tree structure
        os.makedirs(resolved_path, exist_ok=True)

        # Instantiate from the remote web registry
        nlp_classifier = pipeline(
            "text-classification",
            model=remote_repo,
            truncation=True,
            max_length=512,
            local_files_only=False,
        )

        # Serialize the structures down to your custom folder path
        nlp_classifier.model.save_pretrained(resolved_path)
        nlp_classifier.tokenizer.save_pretrained(resolved_path)
        print("✅ Model downloaded and successfully cached to your local data folder!")

    # 4. Batch process all reviews simultaneously for high performance
    raw_texts = df_pipe[clean_col].tolist()
    print(f"🚀 Running batched inference on {len(raw_texts)} reviews...")
    model_outputs = nlp_classifier(raw_texts, batch_size=16)

    # 5. Map textual star labels directly to categorical numeric intervals
    mapped_scores = []
    for out in model_outputs:
        label = str(out["label"]).lower()
        if "1" in label or "2" in label or "negative" in label:
            mapped_scores.append(-1.0)
        elif "4" in label or "5" in label or "positive" in label:
            mapped_scores.append(1.0)
        else:
            mapped_scores.append(0.0)

    df_pipe["Sentiment_Score"] = mapped_scores

    # 6. Collapse multiple user reviews into a single mean metric per unique ID
    df_result = df_pipe.groupby(id_col)["Sentiment_Score"].mean().reset_index()
    df_result.rename(columns={"Sentiment_Score": "Mean_Sentiment"}, inplace=True)

    return df_result


def assemble_abt(dfs_list: list, on_col: str) -> pd.DataFrame:
    """
    Orchestrator for assembling the Analytical Base Table via sequential Left Joins.
    """
    if not dfs_list:
        return pd.DataFrame()

    df_final = dfs_list[0].copy()
    for df in dfs_list[1:]:
        if not df.empty:
            df_final = pd.merge(df_final, df, on=on_col, how="left")

    # Fill NaN for numeric features spawned after the join
    numeric_cols = df_final.select_dtypes(include=[np.number]).columns
    df_final[numeric_cols] = df_final[numeric_cols].fillna(0)

    return df_final
