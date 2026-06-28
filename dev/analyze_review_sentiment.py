import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from transformers import pipeline


def run_sentiment_analysis(data_path: str):
    print(f"1. Loading datasets from {data_path}...")
    try:
        profiles = pd.read_csv(os.path.join(data_path, "profiles.csv"))
        reviews = pd.read_csv(os.path.join(data_path, "reviews.csv"))
    except FileNotFoundError:
        print(f"Error: Could not find data in {data_path}. Generate data first!")
        return

    # --- КРИТИЧЕСКИЙ ФИКС: Очищаем имена колонок от случайных пробелов (шума) ---
    profiles.columns = profiles.columns.str.strip()
    reviews.columns = reviews.columns.str.strip()

    # Переносим Target_ID в индекс, если он загрузился как обычная колонка
    if "Target_ID" in profiles.columns:
        profiles = profiles.set_index("Target_ID")

    # Clean Churn column
    if "Churn" in profiles.columns:
        profiles["Churn"] = (
            pd.to_numeric(profiles["Churn"], errors="coerce").fillna(0).astype(int)
        )

    # Clean Gender column if it exists
    if "Gender" in profiles.columns:
        profiles["Gender"] = profiles["Gender"].astype(str).str.strip().str.title()
        valid_mask = profiles["Gender"].isin(["Male", "Female"])
        profiles.loc[~valid_mask, "Gender"] = "Unknown"
    else:
        print("⚠️ Внимание: Колонка 'Gender' не найдена в profiles.csv.")
        print(
            "Пожалуйста, убедитесь, что вы перезапустили скрипт генерации данных с новым конфигом!"
        )

    # --- ДИНАМИЧЕСКИЙ СБОР КОЛОНОК ДЛЯ МЕРДЖА ---
    # Собираем только те колонки, которые реально существуют в датасете
    cols_to_merge = [col for col in ["Gender", "Churn"] if col in profiles.columns]

    # Мерджим данные без падения по KeyError
    df = reviews.merge(profiles[cols_to_merge], on="Target_ID", how="left")
    print(f"2. Merged {len(df)} reviews with profile demographics.")

    print(
        "3. Loading Transformer Pipeline (tabularisai/multilingual-sentiment-analysis)..."
    )
    # Using the exact model you specified
    classifier = pipeline(
        "text-classification",
        model="tabularisai/multilingual-sentiment-analysis",
        truncation=True,
        max_length=512,
    )

    print(
        "4. Running Sentiment Analysis (This might take a minute depending on your CPU/GPU)..."
    )

    # Process reviews (using a list comprehension for simplicity)
    texts = df["Review_Text"].astype(str).tolist()

    # Run the classifier in batches for better performance
    results = classifier(texts, batch_size=16)

    # Extract labels and scores
    df["Predicted_Sentiment"] = [res["label"] for res in results]
    df["Sentiment_Confidence"] = [res["score"] for res in results]

    # Standardize labels (just in case the model returns variations like 'positive', 'POSITIVE', etc.)
    df["Predicted_Sentiment"] = df["Predicted_Sentiment"].str.title()

    print("5. Analysis complete! Generating Bias Discovery Report...")
    generate_bias_report(df)


def generate_bias_report(df: pd.DataFrame):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Ethical Bias Discovery via NLP Sentiment Analysis",
        fontsize=16,
        fontweight="bold",
    )

    # --- PLOT 1: Overall Sentiment by Churn Status ---
    # We expect Churn=1 to be highly negative, and Churn=0 to be mostly positive
    sns.histplot(
        data=df,
        x="Predicted_Sentiment",
        hue="Churn",
        multiple="dodge",
        stat="percent",
        shrink=0.8,
        palette=["#2ecc71", "#e74c3c"],
        ax=axes[0],
    )
    axes[0].set_title("Expected Logic: Sentiment matches Churn Status")
    axes[0].set_ylabel("Percentage of Reviews")
    axes[0].set_xlabel("Transformer Predicted Sentiment")

    # --- PLOT 2: The Ethical Bias (Focusing only on Churned Users) ---
    # Here we prove the bias: Even among users who canceled (Churn=1),
    # the system generated softer/more positive reviews for women.
    churned_df = df[df["Churn"] == 1]

    if not churned_df.empty and "Gender" in churned_df.columns:
        # Calculate percentage of negative vs positive reviews by gender for churned users
        bias_calc = (
            churned_df.groupby("Gender")["Predicted_Sentiment"]
            .value_counts(normalize=True)
            .unstack()
            .fillna(0)
            * 100
        )

        bias_calc.plot(
            kind="bar",
            stacked=True,
            ax=axes[1],
            color=["#e74c3c", "#95a5a6", "#3498db"],
        )

        axes[1].set_title("The Bias Trap: Sentiment of CHURNED Users by Gender")
        axes[1].set_ylabel("Percentage (%)")
        axes[1].set_xlabel("Gender (Churn = 1 Only)")
        axes[1].tick_params(axis="x", rotation=0)
        axes[1].legend(title="Predicted Sentiment")
    else:
        axes[1].text(
            0.5, 0.5, "Not enough data for Churned Users", ha="center", va="center"
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Point this to your generated dataset folder
    target_directory = "data/processed/01_telco_customer_churn"
    run_sentiment_analysis(target_directory)
