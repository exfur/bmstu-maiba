import argparse
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from googleapiclient.discovery import build
from tqdm import tqdm
from utils import get_creds

# ==========================================
# CONFIGURATION & NETWORK SETTINGS
# ==========================================
GSHEET_ID = os.getenv("SOURCE_GSHEET")
SHEET_NAME = "legends"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL_NAME = "gemma3:4b"
MAX_CONCURRENT_REQUESTS = (
    8  # Slightly higher than parallel settings to keep the queue saturated
)


def clean_json_string(raw_str: str) -> str:
    """
    Cleans Google Sheets string artifacts by removing outer quotes
    and resolving doubled double-quotes ("") back into valid JSON format.
    """
    if not isinstance(raw_str, str):
        return raw_str

    cleaned = raw_str.strip()
    if cleaned.startswith('"""') and cleaned.endswith('"""'):
        cleaned = cleaned[3:-3]
    elif cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1]

    cleaned = cleaned.replace('""', '"')
    return cleaned.strip()


def fetch_config_from_gsheet(target_variant: int = 1) -> dict:
    """Loads and normalizes the variant configuration directly from Google Sheets."""
    if not GSHEET_ID:
        print(
            "Warning: SOURCE_GSHEET variable not set. Falling back to development mock."
        )
        return get_default_config()

    try:
        creds = get_creds()
        service = build("sheets", "v4", credentials=creds)
        result = (
            service.spreadsheets()
            .values()
            .get(spreadsheetId=GSHEET_ID, range=SHEET_NAME)
            .execute()
        )
        values = result.get("values", [])

        config = {}
        if not values:
            return get_default_config()

        for row in values:
            if len(row) > target_variant and row[0].strip():
                key = row[0].strip()
                if key.lower() == "variant":
                    continue

                raw_value = clean_json_string(row[target_variant])

                if raw_value.startswith("{") or raw_value.startswith("["):
                    try:
                        config[key] = json.loads(raw_value)
                    except json.JSONDecodeError:
                        config[key] = raw_value
                else:
                    config[key] = raw_value
        return config
    except Exception as e:
        print(
            f"Sheets API Connection failure: {e}. Transitioning to fallback structure."
        )
        return get_default_config()


def get_default_config() -> dict:
    """Stable development mock configuration."""
    return {
        "dataset_id": "01_telco_customer_churn",
        "target_column": "Churn",
        "SCHEMA_NUM_FEATURE_X": ["tenure", "MonthlyCharges", "TotalCharges"],
        "SCHEMA_CAT_FEATURE_Y": ["gender", "Contract", "PaymentMethod", "TechSupport"],
    }


def ask_ollama(prompt: str) -> str:
    """Queries local LLM context engines for user review generation text."""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {"model": OLLAMA_MODEL_NAME, "prompt": prompt, "stream": False}
    try:
        # Increased timeout to 60s to safely allow queueing under load
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception:
        return "Error generating review text due to system timeout parameters."


# ==========================================
# DATA GENERATOR ENGINE (THE GHOST PIPELINE)
# ==========================================
class MultimodalDataGenerator:
    def __init__(
        self, config: dict, num_inference_rows: int = 300, no_ai: bool = False
    ):
        self.config = config
        self.num_inference_rows = num_inference_rows
        self.no_ai = no_ai

        self.profiles = pd.DataFrame()
        self.transactions = pd.DataFrame()
        self.reviews = pd.DataFrame()
        self.ghost_abt = pd.DataFrame()

        self.target_col = self.config.get("target_column", "Target_Flag")

        self.base_gen = self._ensure_dict("BASE_GENERATION")
        self.dgp_equations = self._ensure_dict("DGP_EQUATIONS")
        self.behavioral_logic = self._ensure_dict("BEHAVIORAL_LOGIC")
        self.sentiment_config = self._ensure_dict("SENTIMENT_CONFIG")
        self.noise_injection = self._ensure_dict("NOISE_INJECTION")

        self.num_users = self.base_gen.get("num_rows", 5000)

    def _ensure_dict(self, key: str) -> dict:
        val = self.config.get(key, {})
        if isinstance(val, str):
            cleaned = clean_json_string(val)
            if not cleaned.startswith("{") and "}" in cleaned:
                cleaned = "{" + cleaned
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return {}
        return val if isinstance(val, dict) else {}

    def _ensure_list(self, key: str) -> list:
        val = self.config.get(key, [])
        if isinstance(val, str):
            cleaned = clean_json_string(val)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return []
        return val if isinstance(val, list) else []

    def run_pipeline(self):
        self._generate_raw_profiles()
        self._generate_base_transactions()
        self._generate_base_nlp_sentiments()
        self._internal_feature_engineering()
        self._apply_math_model()
        self._apply_behavioral_logic()
        self._generate_llm_reviews()
        self._apply_profile_noise()
        self._mask_inference_targets()
        return self._purge_and_export()

    def _generate_raw_profiles(self):
        columns_cfg = self.base_gen.get("columns", {})
        df_data = {"Target_ID": [f"USR_{i:05d}" for i in range(self.num_users)]}

        for col_name, rules in columns_cfg.items():
            if col_name == self.target_col:
                continue

            col_type = rules.get("type")
            if col_type == "categorical":
                df_data[col_name] = np.random.choice(
                    rules["values"], size=self.num_users, p=rules.get("weights")
                )
            elif col_type == "numeric":
                vals = np.random.uniform(
                    rules.get("min", 0.0), rules.get("max", 100.0), size=self.num_users
                )
                if "round" in rules:
                    vals = np.round(vals, rules["round"])
                df_data[col_name] = vals
            elif col_type == "date":
                start = pd.to_datetime(rules.get("start", "2020-01-01"))
                end = pd.to_datetime(
                    rules.get("end", datetime.now().strftime("%Y-%m-%d"))
                )
                days_diff = (end - start).days
                random_days = np.random.randint(0, days_diff, size=self.num_users)
                df_data[col_name] = (
                    start + pd.to_timedelta(random_days, unit="D")
                ).strftime("%Y-%m-%d")

        self.profiles = pd.DataFrame(df_data)

    def _generate_base_transactions(self):
        transactions = []
        end_date = datetime.now()

        for _, row in self.profiles.iterrows():
            target_id = row["Target_ID"]
            num_trans = int(np.random.exponential(scale=15)) + 1
            for _ in range(num_trans):
                days_ago = random.randint(0, 365)
                amt = round(random.uniform(10.0, 150.0), 2)
                t_date = end_date - timedelta(days=days_ago)
                transactions.append(
                    {
                        "Transaction_ID": f"TXN-{random.randint(100000, 999999)}",
                        "Target_ID": target_id,
                        "Trans_Date": t_date.strftime("%Y-%m-%d %H:%M:%S"),
                        "Trans_Amount": amt,
                    }
                )
        self.transactions = pd.DataFrame(transactions)

    def _generate_base_nlp_sentiments(self):
        sentiments = []
        biases = self.sentiment_config.get("biases", [])

        for _, row in self.profiles.iterrows():
            base_pos, base_neu, base_neg = 0.33, 0.33, 0.34

            for bias in biases:
                col, op, val = (
                    bias.get("column"),
                    bias.get("operator"),
                    bias.get("value"),
                )
                if col not in row:
                    continue
                try:
                    if op == "==" and str(row[col]) == str(val):
                        shift = bias.get("shift", {})
                        base_pos += shift.get("positive", 0.0)
                        base_neg += shift.get("negative", 0.0)
                except ValueError:
                    pass

            base_pos, base_neg = max(0, base_pos), max(0, base_neg)
            total = (
                base_pos + base_neu + base_neg
                if (base_pos + base_neu + base_neg) > 0
                else 1.0
            )
            mean_sentiment = (base_pos / total) * 1.0 + (base_neg / total) * -1.0
            sentiments.append(
                {"Target_ID": row["Target_ID"], "Mean_Sentiment": mean_sentiment}
            )

        self.ghost_abt_sentiments = pd.DataFrame(sentiments)

    def _internal_feature_engineering(self):
        end_date = pd.to_datetime(datetime.now())
        df_tx = self.transactions.copy()
        df_tx["Trans_Date"] = pd.to_datetime(df_tx["Trans_Date"])

        rfm = (
            df_tx.groupby("Target_ID")
            .agg(
                Recency=("Trans_Date", lambda x: (end_date - x.max()).days),
                Frequency=("Transaction_ID", "count"),
                Monetary=("Trans_Amount", "sum"),
            )
            .reset_index()
        )

        self.ghost_abt = self.profiles.copy()
        self.ghost_abt = self.ghost_abt.merge(rfm, on="Target_ID", how="left").fillna(
            {"Recency": 999, "Frequency": 0, "Monetary": 0}
        )
        self.ghost_abt = self.ghost_abt.merge(
            self.ghost_abt_sentiments, on="Target_ID", how="left"
        )

        cat_cols = self._ensure_list("SCHEMA_CAT_FEATURE_Y")
        existing_cat = [c for c in cat_cols if c in self.ghost_abt.columns]
        self.ghost_abt = pd.get_dummies(self.ghost_abt, columns=existing_cat)
        self.ghost_abt.columns = [
            re.sub(r"[^a-zA-Z0-9_]", "_", c) for c in self.ghost_abt.columns
        ]

    def _apply_math_model(self):
        z_scores = np.zeros(len(self.ghost_abt))
        base_context = {col: self.ghost_abt[col] for col in self.ghost_abt.columns}
        base_context.update(
            {
                "exp": np.exp,
                "maximum": np.maximum,
                "log1p": np.log1p,
                "random_normal": lambda mu, sigma: np.random.normal(
                    mu, sigma, len(self.ghost_abt)
                ),
            }
        )

        for component_name, rules in self.dgp_equations.items():
            template = rules["Equation_Template"]
            coeffs = rules["Coefficients_JSON"]
            component_context = {**base_context, **coeffs}

            for var in rules["Dependent_Variables"]:
                var_safe = re.sub(r"[^a-zA-Z0-9_]", "_", var)
                if var_safe not in component_context:
                    component_context[var_safe] = 0.0

            z_scores += eval(template, {"__builtins__": None}, component_context)

        probs = 1 / (1 + np.exp(-z_scores))
        self.profiles[self.target_col] = np.random.binomial(1, probs)

    def _apply_behavioral_logic(self):
        cutoff = self.behavioral_logic.get("STORY_LIFESPAN_CUTOFF", {})
        trends = self.behavioral_logic.get("STORY_TRANSACTION_TRENDS", {})

        if cutoff.get("action") == "truncate_timeline":
            churn_targets = self.profiles[self.profiles[self.target_col] == 1][
                "Target_ID"
            ]
            drop_days = (
                trends.get("mapping", {}).get("1", {}).get("drop_off_last_n_days", 45)
            )
            cutoff_date = datetime.now() - timedelta(days=drop_days)

            mask = ~(
                (self.transactions["Target_ID"].isin(churn_targets))
                & (pd.to_datetime(self.transactions["Trans_Date"]) > cutoff_date)
            )
            self.transactions = self.transactions[mask]

    def _generate_llm_reviews(self):
        """
        Compiles and dispatches multiple review requests simultaneously to the
        Ollama server using ThreadPoolExecutor for highly parallelized execution.
        """
        reviews = []
        generation_tasks = []
        cfg = self.sentiment_config
        mapping = cfg.get("mapping", {})
        review_probs = cfg.get("review_count_probs", [0.60, 0.25, 0.10, 0.05])
        review_counts = [0, 1, 2, 3]

        print("Analyzing schemas and formatting evaluation prompt packages...")
        for _, row in self.profiles.iterrows():
            num_reviews = np.random.choice(review_counts, p=review_probs)
            if num_reviews == 0:
                continue

            target_val = str(row[self.target_col])
            behavior = mapping.get(target_val, mapping.get("0", {})).copy()
            length_instruction = (
                "Output exactly 2 sentences of the review text and nothing else."
            )

            for bias in cfg.get("biases", []):
                col, op, val = (
                    bias.get("column"),
                    bias.get("operator"),
                    bias.get("value"),
                )
                if col in row and str(row[col]) == str(val):
                    if bias.get("effect") == "sentiment_shift":
                        shift = bias.get("shift", {})
                        behavior["positive"] += shift.get("positive", 0.0)
                        behavior["negative"] += shift.get("negative", 0.0)
                    elif bias.get("effect") == "length_override":
                        length_instruction = bias.get(
                            "length_instruction", length_instruction
                        )

            behavior["positive"] = max(0.0, behavior["positive"])
            behavior["negative"] = max(0.0, behavior["negative"])
            total = (
                behavior["positive"]
                + behavior.get("neutral", 0.15)
                + behavior["negative"]
            )
            p_dist = [
                behavior["positive"] / total,
                behavior.get("neutral", 0.15) / total,
                behavior["negative"] / total,
            ]

            for _ in range(num_reviews):
                chosen_sentiment = np.random.choice(
                    ["positive", "neutral", "negative"], p=p_dist
                )
                aspects_to_mention = []

                if target_val == "0":
                    if chosen_sentiment == "positive" and "advantages" in behavior:
                        aspects_to_mention = random.sample(
                            behavior["advantages"],
                            k=min(2, len(behavior["advantages"])),
                        )
                    elif (
                        chosen_sentiment == "negative"
                        and "minor_annoyances" in behavior
                    ):
                        aspects_to_mention = random.sample(
                            behavior["minor_annoyances"], k=1
                        )
                    else:
                        aspects_to_mention = [
                            "general baseline satisfaction with services"
                        ]
                elif target_val == "1":
                    if (
                        chosen_sentiment == "negative"
                        and "critical_problems" in behavior
                    ):
                        aspects_to_mention = random.sample(
                            behavior["critical_problems"],
                            k=min(2, len(behavior["critical_problems"])),
                        )
                    elif (
                        chosen_sentiment == "positive"
                        and "faded_advantages" in behavior
                    ):
                        aspects_to_mention = random.sample(
                            behavior["faded_advantages"], k=1
                        )
                    else:
                        aspects_to_mention = [
                            "overall platform failure and service degradation"
                        ]

                context_str = ", ".join(
                    [a.replace("_", " ") for a in aspects_to_mention]
                )

                # Context payloads saved for execution mapping
                task_metadata = {
                    "Target_ID": row["Target_ID"],
                    "Review_Date": (
                        datetime.now() - timedelta(days=random.randint(1, 100))
                    ).strftime("%Y-%m-%d"),
                    "chosen_sentiment": chosen_sentiment,
                    "aspects_to_mention": aspects_to_mention,
                    "context_str": context_str,
                    "length_instruction": length_instruction,
                }
                generation_tasks.append(task_metadata)

        # --- PROCESS PIPELINE ACCORDING TO RUNTIME FLAGS ---
        if self.no_ai:
            for task in tqdm(generation_tasks, desc="Generating Mock Reviews (No-AI)"):
                text = f"""{{"sentiment": "{task["chosen_sentiment"]}", "explicit_aspects": [{", ".join([f"'{a}'" for a in task["aspects_to_mention"]])}]}}"""
                reviews.append(
                    {
                        "Review_ID": f"REV-{random.randint(100000, 999999)}",
                        "Target_ID": task["Target_ID"],
                        "Review_Date": task["Review_Date"],
                        "Review_Text": text,
                    }
                )
        else:
            # Multi-threaded dynamic execution loop mapping directly into the concurrent Ollama worker framework
            print(
                f"Spamming system queue with {len(generation_tasks)} contextual requests across {MAX_CONCURRENT_REQUESTS} concurrent execution pipelines..."
            )

            with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
                future_to_task = {}

                for task in generation_tasks:
                    system_instruction = (
                        "You are an automated corporate review generator mimicking realistic customer feedback. "
                        "CRITICAL: Output ONLY the raw text response. Never include explanations, pleasantries, intro, or markdown ticks. "
                        f"{task['length_instruction']}"
                    )
                    user_prompt = (
                        f"Generate a realistic customer review with strict {task['chosen_sentiment'].upper()} emotional tone.\n"
                        f"The customer must explicitly focus on the following details: {task['context_str']}.\n"
                        "Raw Review Text:"
                    )
                    full_prompt = f"<start_of_turn>user\n{system_instruction}\n\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"

                    # Submit to thread execution bank immediately
                    future = executor.submit(ask_ollama, full_prompt)
                    future_to_task[future] = task

                # Collect and assemble results natively as compute windows clear out
                for future in tqdm(
                    as_completed(future_to_task),
                    total=len(future_to_task),
                    desc="Processing Ollama Compute Stream",
                ):
                    task = future_to_task[future]
                    try:
                        text = future.result()
                        text = text.strip().strip('"').strip("'")
                    except Exception:
                        text = "System processing failure exception occurred."

                    reviews.append(
                        {
                            "Review_ID": f"REV-{random.randint(100000, 999999)}",
                            "Target_ID": task["Target_ID"],
                            "Review_Date": task["Review_Date"],
                            "Review_Text": text,
                        }
                    )

        self.reviews = pd.DataFrame(reviews)

    def _get_noise_target_cols(self, cfg_target, df: pd.DataFrame) -> list:
        if cfg_target == "all":
            return [c for c in df.columns if c not in ["Target_ID", self.target_col]]
        if isinstance(cfg_target, str) and cfg_target in self.config:
            target_list = self._ensure_list(cfg_target)
            return [c for c in target_list if c in df.columns]
        if isinstance(cfg_target, list):
            return [c for c in cfg_target if c in df.columns]
        return []

    def _apply_profile_noise(self):
        cfg_block = self.noise_injection
        if not cfg_block:
            return

        # 1. ERR_NAN
        if "ERR_NAN" in cfg_block:
            cfg = cfg_block["ERR_NAN"]
            for col in self._get_noise_target_cols(
                cfg.get("target_columns"), self.profiles
            ):
                mask = np.random.rand(len(self.profiles)) < cfg["ratio"]
                self.profiles.loc[mask, col] = np.nan

        # 2. ERR_WHITESPACE_NAN
        if "ERR_WHITESPACE_NAN" in cfg_block:
            cfg = cfg_block["ERR_WHITESPACE_NAN"]
            for col in self._get_noise_target_cols(
                cfg.get("target_columns"), self.profiles
            ):
                self.profiles[col] = self.profiles[col].astype("object")
                mask = np.random.rand(len(self.profiles)) < cfg["ratio"]
                self.profiles.loc[mask, col] = [
                    random.choice(cfg.get("values", [" "])) for _ in range(mask.sum())
                ]

        # 3. ERR_STRING_PLACEHOLDER
        if "ERR_STRING_PLACEHOLDER" in cfg_block:
            cfg = cfg_block["ERR_STRING_PLACEHOLDER"]
            for col in self._get_noise_target_cols(
                cfg.get("target_columns"), self.profiles
            ):
                self.profiles[col] = self.profiles[col].astype("object")
                mask = np.random.rand(len(self.profiles)) < cfg["ratio"]
                self.profiles.loc[mask, col] = [
                    random.choice(cfg["values"]) for _ in range(mask.sum())
                ]

        # 4. ERR_NUMERIC_AS_OBJECT
        if "ERR_NUMERIC_AS_OBJECT" in cfg_block:
            cfg = cfg_block["ERR_NUMERIC_AS_OBJECT"]
            for col in self._get_noise_target_cols(
                cfg.get("target_columns"), self.profiles
            ):
                self.profiles[col] = self.profiles[col].astype("object")
                mask = np.random.rand(len(self.profiles)) < cfg["ratio"]
                self.profiles.loc[mask, col] = self.profiles.loc[mask, col].astype(
                    str
                ) + [random.choice(cfg["values"]) for _ in range(mask.sum())]

        # 5. ERR_MIXED_BOOLEAN
        if "ERR_MIXED_BOOLEAN" in cfg_block:
            cfg = cfg_block["ERR_MIXED_BOOLEAN"]
            for col in self._get_noise_target_cols(
                cfg.get("target_columns"), self.profiles
            ):
                self.profiles[col] = self.profiles[col].astype("object")
                mask = np.random.rand(len(self.profiles)) < cfg["ratio"]
                self.profiles.loc[mask, col] = [
                    random.choice(cfg["values"]) for _ in range(mask.sum())
                ]

        # 6. ERR_CASE_INCONSISTENCY
        if "ERR_CASE_INCONSISTENCY" in cfg_block:
            cfg = cfg_block["ERR_CASE_INCONSISTENCY"]
            for col in self._get_noise_target_cols(
                cfg.get("target_columns"), self.profiles
            ):
                mask = np.random.rand(len(self.profiles)) < cfg["ratio"]
                for idx in self.profiles[mask].index:
                    val = str(self.profiles.loc[idx, col])
                    self.profiles.loc[idx, col] = (
                        val.lower()
                        if random.choice(cfg.get("values", ["lowercase"]))
                        == "lowercase"
                        else val.upper()
                    )

        # 7. ERR_ROW_DUPLICATE
        if "ERR_ROW_DUPLICATE" in cfg_block:
            cfg = cfg_block["ERR_ROW_DUPLICATE"]
            dup_mask = np.random.rand(len(self.profiles)) < cfg["ratio"]
            self.profiles = pd.concat(
                [self.profiles, self.profiles[dup_mask].copy()], ignore_index=False
            )

    def _mask_inference_targets(self):
        mask_indices = self.profiles.index[-self.num_inference_rows :]
        self.profiles.loc[mask_indices, self.target_col] = np.nan

    def _purge_and_export(self):
        num_features = self._ensure_list("SCHEMA_NUM_FEATURE_X")
        cat_features = self._ensure_list("SCHEMA_CAT_FEATURE_Y")

        allowed_p_cols = ["Target_ID", self.target_col] + num_features + cat_features
        export_p_cols = [c for c in allowed_p_cols if c in self.profiles.columns]

        final_profiles = self.profiles[export_p_cols].copy()
        final_transactions = self.transactions[
            ["Transaction_ID", "Target_ID", "Trans_Date", "Trans_Amount"]
        ].copy()
        final_reviews = self.reviews[
            ["Review_ID", "Target_ID", "Review_Date", "Review_Text"]
        ].copy()

        del self.ghost_abt
        del self.ghost_abt_sentiments

        return final_profiles, final_transactions, final_reviews


def main():
    parser = argparse.ArgumentParser(description="DGP Multimodal Generator.")
    parser.add_argument("-v", "--variant", type=int, default=1)
    parser.add_argument("--no-ai", action="store_true")
    args = parser.parse_args()

    print(f"Executing Configuration Fetch for Variant Group #{args.variant}...")
    config = fetch_config_from_gsheet(args.variant)

    dataset_name = config.get("dataset_id", f"variant_{args.variant}").strip()
    out_dir = os.path.join("data", "processed", dataset_name)
    print(f"Target Output Directory Verified: {out_dir}")

    generator = MultimodalDataGenerator(
        config, num_inference_rows=300, no_ai=args.no_ai
    )
    p_df, t_df, r_df = generator.run_pipeline()

    os.makedirs(out_dir, exist_ok=True)
    p_df.to_csv(os.path.join(out_dir, "profiles.csv"), index=False)
    t_df.to_csv(os.path.join(out_dir, "transactions.csv"), index=False)
    r_df.to_csv(os.path.join(out_dir, "reviews.csv"), index=False)
    print("Process Finished. Production files exported successfully.")


if __name__ == "__main__":
    main()
