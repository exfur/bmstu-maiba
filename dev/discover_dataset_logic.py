import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_and_clean_data(data_path: str):
    print(f"--- 1. Loading and Cleaning Data from {data_path} ---")

    # Load raw data
    profiles_raw = pd.read_csv(os.path.join(data_path, "profiles.csv"), index_col=0)
    transactions = pd.read_csv(os.path.join(data_path, "transactions.csv"))
    reviews = pd.read_csv(os.path.join(data_path, "reviews.csv"))

    # 1. Clean Dirty Columns (Trailing/Leading spaces)
    profiles_raw.columns = profiles_raw.columns.str.strip()

    profiles = profiles_raw.copy()

    # 2. Clean 'TotalCharges' (ERR_NUMERIC_AS_OBJECT)
    if "TotalCharges" in profiles.columns:
        # Strip out any non-numeric characters (like '$', ' USD', or spaces)
        profiles["TotalCharges"] = (
            profiles["TotalCharges"].astype(str).str.replace(r"[^\d.]", "", regex=True)
        )
        # Convert back to float, coercing empty strings to NaN
        profiles["TotalCharges"] = pd.to_numeric(
            profiles["TotalCharges"], errors="coerce"
        )
        # Fill missing values with median
        profiles["TotalCharges"] = profiles["TotalCharges"].fillna(
            profiles["TotalCharges"].median()
        )

    # 3. Clean 'TechSupport' (ERR_MIXED_BOOLEAN)
    if "TechSupport" in profiles.columns:
        # Standardize variations of Yes/No
        yes_vals = ["Yes", "Y", "1", "True", "1.0"]
        no_vals = ["No", "N", "0", "False", "0.0"]

        profiles["TechSupport"] = (
            profiles["TechSupport"].astype(str).str.strip().str.title()
        )
        profiles.loc[profiles["TechSupport"].isin(yes_vals), "TechSupport"] = "Yes"
        profiles.loc[profiles["TechSupport"].isin(no_vals), "TechSupport"] = "No"
        # Fill placeholders/NaNs with the mode
        valid_mask = profiles["TechSupport"].isin(["Yes", "No"])
        mode_val = profiles.loc[valid_mask, "TechSupport"].mode()[0]
        profiles.loc[~valid_mask, "TechSupport"] = mode_val

    # 4. Clean 'Gender' (ERR_STRING_PLACEHOLDER / ERR_WHITESPACE_NAN)
    if "Gender" in profiles.columns:
        profiles["Gender"] = profiles["Gender"].astype(str).str.strip().str.title()
        valid_mask = profiles["Gender"].isin(["Male", "Female"])
        mode_val = (
            profiles.loc[valid_mask, "Gender"].mode()[0]
            if not profiles.loc[valid_mask, "Gender"].empty
            else "Unknown"
        )
        profiles.loc[~valid_mask, "Gender"] = mode_val

    if "Churn" in profiles.columns:
        # Force to numeric, turning any weird string noise into NaN
        profiles["Churn"] = pd.to_numeric(profiles["Churn"], errors="coerce")
        # Fill those NaNs with the most common class (0) and cast to integer
        profiles["Churn"] = profiles["Churn"].fillna(0).astype(int)

    print(f"Data cleaned. Profiles shape: {profiles.shape}")
    return profiles, transactions, reviews


def build_wide_table(profiles, transactions, reviews):
    print("--- 2. Building Wide Table ---")

    # Aggregate Transactions
    trans_agg = (
        transactions.groupby("Target_ID")
        .agg(
            trans_count=("Trans_Amount", "count"),
            trans_sum=("Trans_Amount", "sum"),
            trans_mean=("Trans_Amount", "mean"),
        )
        .fillna(0)
    )

    transactions["Trans_Date"] = pd.to_datetime(transactions["Trans_Date"])
    max_date = transactions["Trans_Date"].max()
    recency = transactions.groupby("Target_ID").agg(
        days_since_last_trans=("Trans_Date", lambda x: (max_date - x.max()).days)
    )

    # Merge everything
    wide_df = profiles.join(trans_agg, how="left").join(recency, how="left")

    wide_df["days_since_last_trans"] = wide_df["days_since_last_trans"].fillna(999)
    wide_df["trans_count"] = wide_df["trans_count"].fillna(0)

    # Convert Target_ID back to column if it's the index
    if wide_df.index.name == "Target_ID":
        wide_df = wide_df.reset_index()

    return wide_df


def generate_discovery_report(wide_df, transactions):
    print("--- 3. Generating Visual Evidence ---")

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "Data Generation Logic Discovery Report", fontsize=20, fontweight="bold", y=0.98
    )

    # ---------------------------------------------------------
    # PLOT 1: Time-Series Dynamics (The "Fade Out" and Hard Stop)
    # ---------------------------------------------------------
    ax1 = plt.subplot(2, 2, 1)

    # Prepare time-series for plotting by merging Churn status
    tx_merged = transactions.merge(
        wide_df[["Target_ID", "Churn"]], on="Target_ID", how="inner"
    )
    tx_merged["Trans_Date"] = pd.to_datetime(tx_merged["Trans_Date"])
    tx_merged["MonthYear"] = tx_merged["Trans_Date"].dt.to_period("M").astype(str)

    # Group by month and churn status
    ts_trend = (
        tx_merged.groupby(["MonthYear", "Churn"])["Trans_Amount"].count().reset_index()
    )

    sns.lineplot(
        data=ts_trend,
        x="MonthYear",
        y="Trans_Amount",
        hue="Churn",
        marker="o",
        ax=ax1,
        palette=["#2ecc71", "#e74c3c"],
    )
    ax1.set_title(
        "1. Transaction Volume over Time (Sparsity & Hard Stop)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xlabel("Time (Months)")
    ax1.set_ylabel("Total Number of Transactions")
    ax1.tick_params(axis="x", rotation=45)
    ax1.legend(
        title="Churn", labels=["0 (Active - Stable)", "1 (Churned - Drops to 0)"]
    )

    # ---------------------------------------------------------
    # PLOT 2: The Non-Linear SHAP Interaction
    # ---------------------------------------------------------
    ax2 = plt.subplot(2, 2, 2)

    # Create bins for TotalCharges to easily visualize the threshold logic
    bins = [0, 2000, 4000, 6000, 10000]
    labels = ["Low (0-2k)", "Medium (2k-4k)", "High (4k-6k)", "Very High (6k+)"]
    wide_df["Charge_Tier"] = pd.cut(wide_df["TotalCharges"], bins=bins, labels=labels)

    # Calculate churn rate by TechSupport and Charge Tier
    interaction_df = (
        wide_df.groupby(["TechSupport", "Charge_Tier"], observed=True)["Churn"]
        .mean()
        .unstack()
    )

    sns.heatmap(
        interaction_df,
        annot=True,
        fmt=".0%",
        cmap="Reds",
        vmin=0,
        vmax=1,
        ax=ax2,
        cbar_kws={"label": "Churn Rate %"},
    )
    ax2.set_title(
        "2. The Interaction Rule (High Charges + No TechSupport)",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xlabel("Total Charges Tier")
    ax2.set_ylabel("Has Tech Support?")

    # ---------------------------------------------------------
    # PLOT 3: Linear Feature Impact (Base Logits)
    # ---------------------------------------------------------
    ax3 = plt.subplot(2, 2, 3)

    sns.barplot(
        data=wide_df,
        x="TechSupport",
        y="Churn",
        errorbar=None,
        palette=["#3498db", "#95a5a6"],
        ax=ax3,
    )
    ax3.set_title(
        "3. Linear Impact: Tech Support lowers Churn", fontsize=14, fontweight="bold"
    )
    ax3.set_ylabel("Average Churn Rate")
    ax3.set_ylim(0, 1)

    for p in ax3.patches:
        ax3.annotate(
            f"{p.get_height():.0%}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            fontsize=12,
            color="black",
            xytext=(0, 10),
            textcoords="offset points",
        )

    # ---------------------------------------------------------
    # PLOT 4: Ethical Bias Injection
    # ---------------------------------------------------------
    ax4 = plt.subplot(2, 2, 4)

    if "Gender" in wide_df.columns:
        sns.barplot(
            data=wide_df,
            x="Gender",
            y="Churn",
            errorbar=None,
            palette=["#9b59b6", "#f1c40f"],
            ax=ax4,
        )
        ax4.set_title(
            "4. Protected Attribute Bias (Gender vs Churn)",
            fontsize=14,
            fontweight="bold",
        )
        ax4.set_ylabel("Average Churn Rate")
        ax4.set_ylim(0, 1)

        for p in ax4.patches:
            ax4.annotate(
                f"{p.get_height():.0%}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                fontsize=12,
                color="black",
                xytext=(0, 10),
                textcoords="offset points",
            )
    else:
        ax4.text(
            0.5, 0.5, "Gender Column Not Found", ha="center", va="center", fontsize=14
        )

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()


if __name__ == "__main__":
    # You can change this hardcoded path or pass it via command line
    DEFAULT_PATH = "data/processed/01_telco_customer_churn"

    parser = argparse.ArgumentParser(
        description="Discover logic in generated datasets."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_PATH,
        help="Path to the dataset directory",
    )
    args = parser.parse_args()

    if os.path.exists(args.data_dir):
        profiles_df, transactions_df, reviews_df = load_and_clean_data(args.data_dir)
        wide_table = build_wide_table(profiles_df, transactions_df, reviews_df)
        generate_discovery_report(wide_table, transactions_df)
    else:
        print(
            f"Error: Directory '{args.data_dir}' not found. Please generate the data first."
        )
