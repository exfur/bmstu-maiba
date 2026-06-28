import argparse
import json
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Загружаем переменные окружения
load_dotenv()

# ==========================================
# КОНФИГУРАЦИЯ ПУТЕЙ И GOOGLE API
# ==========================================
GOOGLE_API_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar",
]

GOOGLE_API_SECRET_PATH = os.path.expanduser("~/google_api_client_secret.json")
GOOGLE_API_TOKEN_PATH = os.path.expanduser("~/google_api_token.json")

GSHEET_ID = os.getenv("SOURCE_GSHEET")
SHEET_NAME = "legends"

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL_NAME = "gemma3:4b"


# ==========================================
# АВТОРИЗАЦИЯ И ПОЛУЧЕНИЕ ДАННЫХ
# ==========================================
def get_creds(scopes=None):
    """
    Returns User Credentials (3-Legged OAuth).
    Opens a browser window for the first login, then uses a saved JSON token.
    """
    if scopes is None:
        scopes = GOOGLE_API_SCOPES

    creds = None
    token_path = GOOGLE_API_TOKEN_PATH

    # 1. Load existing token if available
    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_file(token_path, scopes)
        except Exception:
            # If the file is corrupted or incompatible, ignore it
            creds = None

    # 2. If no valid credentials, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                # Refresh the token silently
                creds.refresh(Request())
            except Exception:
                # If refresh fails (e.g. revoked), force re-login
                creds = None

        if not creds:
            if not os.path.exists(GOOGLE_API_CLIENT_SECRET_PATH):
                raise FileNotFoundError(
                    f"OAuth Client Secret not found at: {GOOGLE_API_CLIENT_SECRET_PATH}. "
                    "Please download it from Google Cloud Console (Credentials -> Create -> OAuth Client ID -> Desktop App) "
                    "and rename it to client_secret.json"
                )

            # Open browser for login
            flow = InstalledAppFlow.from_client_secrets_file(
                GOOGLE_API_CLIENT_SECRET_PATH, scopes
            )

            # Фикс: принудительно запрашиваем offline доступ для генерации долговечного refresh_token
            creds = flow.run_local_server(
                port=0, prompt="consent", access_type="offline"
            )

        # 3. Save the credentials as JSON
        with open(token_path, "w", encoding="utf-8") as token:
            token.write(creds.to_json())

    return creds


def fetch_config_from_gsheet(target_variant: int = 1) -> dict:
    """
    Загружает конфигурацию из Google Таблицы для конкретного варианта (от 1 до 15).
    target_variant=1 означает чтение из столбца B.
    """
    if not GSHEET_ID:
        print("Внимание: SOURCE_GSHEET не задан. Разворачиваю дефолтный конфиг.")
        return get_default_config()

    try:
        creds = get_creds()
        service = build("sheets", "v4", credentials=creds)
        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=GSHEET_ID, range=SHEET_NAME).execute()
        values = result.get("values", [])

        config = {}
        if not values:
            print("Таблица пуста. Перехожу на резервный конфиг.")
            return get_default_config()

        for row in values:
            # Проверяем, что в строке есть ключ (Столбец A) и значение для выбранного варианта
            if len(row) > target_variant and row[0].strip():
                key = row[0].strip()

                # Пропускаем строку с номерами вариантов ("variant, 1, 2, 3...")
                if key.lower() == "variant":
                    continue

                raw_value = row[target_variant].strip()

                # Очищаем от мусорных кавычек при экспорте из Sheets (например, """Churn""")
                if raw_value.startswith('"""') and raw_value.endswith('"""'):
                    raw_value = raw_value[3:-3]
                elif raw_value.startswith('"') and raw_value.endswith('"'):
                    raw_value = raw_value[1:-1]

                # Если это похоже на JSON (список или словарь), парсим его
                if raw_value.startswith("{") or raw_value.startswith("["):
                    try:
                        config[key] = json.loads(raw_value)
                    except json.JSONDecodeError as e:
                        print(
                            f"Внимание: Ошибка парсинга JSON для {key} - {e}. Оставляю как строку."
                        )
                        config[key] = raw_value
                else:
                    # Иначе сохраняем как обычную строку (например, "01_telco_customer_churn" или "Churn")
                    config[key] = raw_value

        return config

    except Exception as e:
        print(
            f"Ошибка при работе со Sheets API: {e}. Перехожу на резервный локальный конфиг."
        )
        return get_default_config()


def get_default_config() -> dict:
    """Возвращает стабильный локальный конфиг для разработки."""
    return {
        "dataset_id": "01_telco_customer_churn",
        "target_column": "Churn",
        "SCHEMA_NUM_FEATURE_X": ["tenure", "MonthlyCharges", "TotalCharges"],
        "SCHEMA_CAT_FEATURE_Y": ["gender", "Contract", "PaymentMethod", "TechSupport"],
        "ERR_NAN": {"ratio": 0.05, "target_columns": "all", "values": [None]},
        "ERR_NUMERIC_AS_OBJECT": {
            "ratio": 0.08,
            "target_columns": ["TotalCharges", "MonthlyCharges"],
            "noise_type": "currency",
            "values": ["$", " USD", " "],
        },
        "ERR_MIXED_BOOLEAN": {
            "ratio": 0.10,
            "target_columns": ["TechSupport"],
            "values": ["Y", "N", "1", "0", "True", "False"],
        },
        "STORY_TRANSACTION_TRENDS": {
            "condition_column": "Churn",
            "mapping": {
                "0": {
                    "trend": "stable_usage",
                    "frequency_multiplier": 1.0,
                    "days_since_last_max": 15,
                },
                "1": {
                    "trend": "fading_last_3_months",
                    "frequency_multiplier": 0.4,
                    "days_since_last_max": 75,
                },
            },
        },
        "SENTIMENT_CONFIG": {
            "condition_column": "Churn",
            "mapping": {
                "0": {
                    "positive": 0.70,
                    "neutral": 0.20,
                    "negative": 0.10,
                    "context": "happy_fast_internet_good_coverage",
                },
                "1": {
                    "positive": 0.05,
                    "neutral": 0.15,
                    "negative": 0.80,
                    "context": "angry_network_drops_hidden_fees",
                },
            },
        },
    }


def ask_ollama(prompt: str) -> str:
    """Обращается к локальной LLM для генерации отзыва."""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {"model": OLLAMA_MODEL_NAME, "prompt": prompt, "stream": False}
    try:
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception:
        return "Error generating review due to connection timeout."


class BaseDataGenerator:
    def __init__(self, config: dict):
        self.full_config = config

        import json

        raw_gen_config = config.get("BASE_GENERATION")
        if isinstance(raw_gen_config, str):
            try:
                self.gen_config = json.loads(raw_gen_config)
            except json.JSONDecodeError:
                raw_gen_config = None
        else:
            self.gen_config = raw_gen_config

        if not self.gen_config:
            # Fallback (as defined in previous steps)
            self.gen_config = {}  # Replace with your full fallback dict

    def generate(self) -> pd.DataFrame:
        num_rows = self.gen_config.get("num_rows", 250)
        columns_cfg = self.gen_config.get("columns", {})
        target_col = self.full_config.get("target_column", "Churn")

        df_data = {}

        # --- STEP 1: Generate Independent Features (Skip Target) ---
        for col_name, rules in columns_cfg.items():
            if col_name == target_col:
                continue

            col_type = rules.get("type")
            if col_type == "categorical":
                df_data[col_name] = np.random.choice(
                    rules["values"], size=num_rows, p=rules.get("weights")
                )
            elif col_type == "numeric":
                vals = np.random.uniform(
                    rules.get("min", 0.0), rules.get("max", 100.0), size=num_rows
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
                random_days = np.random.randint(0, days_diff, size=num_rows)
                df_data[col_name] = (
                    start + pd.to_timedelta(random_days, unit="D")
                ).strftime("%Y-%m-%d")

        df = pd.DataFrame(df_data)
        df.index.name = "Target_ID"

        # --- STEP 2: Inject SHAP Dynamics for Target Column ---
        shap_logic = self.gen_config.get("SHAP_LOGIC", {})

        if shap_logic and target_col in columns_cfg:
            num_rows = len(df)
            logits = np.full(num_rows, shap_logic.get("intercept", 0.0))

            # A. Add linear feature effects
            for col_name, logic in shap_logic.get("features", {}).items():
                if col_name not in df.columns:
                    continue

                if columns_cfg[col_name]["type"] == "numeric":
                    logits += df[col_name] * logic.get("weight", 0.0)
                elif columns_cfg[col_name]["type"] == "categorical":
                    weights = df[col_name].map(logic.get("mapping", {})).fillna(0.0)
                    logits += np.array(weights)

            # B. Add Non-Linear Interactions (For Tree-based models)
            for interaction in shap_logic.get("interactions", []):
                weight = interaction.get("weight", 0.0)
                mask = np.ones(num_rows, dtype=bool)

                for cond in interaction.get("conditions", []):
                    col = cond.get("column")
                    op = cond.get("operator")
                    val = cond.get("value")

                    if col not in df.columns:
                        continue

                    if op == "==":
                        mask &= df[col] == val
                    elif op == ">":
                        mask &= df[col] > val
                    elif op == "<":
                        mask &= df[col] < val

                logits += np.where(mask, weight, 0.0)

            # C. Convert to probabilities and sample
            probs = 1 / (1 + np.exp(-logits))
            df[target_col] = np.random.binomial(1, probs)
        else:
            # Fallback random generation
            rules = columns_cfg.get(
                target_col, {"values": [0, 1], "weights": [0.5, 0.5]}
            )
            df[target_col] = np.random.choice(
                rules["values"], size=len(df), p=rules.get("weights")
            )

        return df


# ==========================================
# БЛОКИ СИМУЛЯЦИИ ДАННЫХ И ШУМА
# ==========================================
class ProfileCorruptor:
    def __init__(self, config: dict):
        self.config = config

    def _get_target_cols(self, cfg_target, df: pd.DataFrame) -> list:
        """
        Helper to resolve which columns to apply noise to.
        Handles 'all', list of columns, or config key references (e.g. 'SCHEMA_CAT_FEATURE_Y').
        """
        if cfg_target == "all":
            return list(df.columns)

        # If the target is a string referencing another key in the config
        if isinstance(cfg_target, str) and cfg_target in self.config:
            target_list = self.config[cfg_target]
            return [c for c in target_list if c in df.columns]

        # If the target is explicitly a list of columns
        if isinstance(cfg_target, list):
            return [c for c in cfg_target if c in df.columns]

        return []

    def apply_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        df_noisy = df.copy()

        # 1. ERR_NAN (True nulls)
        if "ERR_NAN" in self.config:
            cfg = self.config["ERR_NAN"]
            cols = self._get_target_cols(cfg.get("target_columns"), df_noisy)
            for col in cols:
                mask = np.random.rand(len(df_noisy)) < cfg["ratio"]
                df_noisy.loc[mask, col] = np.nan

        # 2. ERR_WHITESPACE_NAN (Spaces pretending to be nulls)
        if "ERR_WHITESPACE_NAN" in self.config:
            cfg = self.config["ERR_WHITESPACE_NAN"]
            cols = self._get_target_cols(cfg.get("target_columns"), df_noisy)
            for col in cols:
                mask = np.random.rand(len(df_noisy)) < cfg["ratio"]
                df_noisy.loc[mask, col] = [
                    random.choice(cfg.get("values", [" "])) for _ in range(mask.sum())
                ]

        # 3. ERR_STRING_PLACEHOLDER ("N/A", "?", "null")
        if "ERR_STRING_PLACEHOLDER" in self.config:
            cfg = self.config["ERR_STRING_PLACEHOLDER"]
            cols = self._get_target_cols(cfg.get("target_columns"), df_noisy)
            for col in cols:
                mask = np.random.rand(len(df_noisy)) < cfg["ratio"]
                df_noisy.loc[mask, col] = [
                    random.choice(cfg["values"]) for _ in range(mask.sum())
                ]

        # 4. ERR_NUMERIC_AS_OBJECT (e.g., adding "$", " USD")
        if "ERR_NUMERIC_AS_OBJECT" in self.config:
            cfg = self.config["ERR_NUMERIC_AS_OBJECT"]
            cols = self._get_target_cols(cfg.get("target_columns"), df_noisy)
            for col in cols:
                df_noisy[col] = df_noisy[col].astype("object")
                mask = np.random.rand(len(df_noisy)) < cfg["ratio"]
                df_noisy.loc[mask, col] = df_noisy.loc[mask, col].astype(str) + [
                    random.choice(cfg["values"]) for _ in range(mask.sum())
                ]

        # 5. ERR_MIXED_BOOLEAN ("True", "Y", "1")
        if "ERR_MIXED_BOOLEAN" in self.config:
            cfg = self.config["ERR_MIXED_BOOLEAN"]
            cols = self._get_target_cols(cfg.get("target_columns"), df_noisy)
            for col in cols:
                mask = np.random.rand(len(df_noisy)) < cfg["ratio"]
                df_noisy.loc[mask, col] = [
                    random.choice(cfg["values"]) for _ in range(mask.sum())
                ]

        # 6. ERR_CASE_INCONSISTENCY (Random uppercase/lowercase)
        if "ERR_CASE_INCONSISTENCY" in self.config:
            cfg = self.config["ERR_CASE_INCONSISTENCY"]
            cols = self._get_target_cols(cfg.get("target_columns"), df_noisy)
            for col in cols:
                df_noisy[col] = df_noisy[col].astype("object")
                mask = np.random.rand(len(df_noisy)) < cfg["ratio"]

                # Apply transformation
                for idx in df_noisy[mask].index:
                    val = str(df_noisy.loc[idx, col])
                    case_choice = random.choice(
                        cfg.get("values", ["lowercase", "uppercase"])
                    )
                    if case_choice == "lowercase":
                        df_noisy.loc[idx, col] = val.lower()
                    elif case_choice == "uppercase":
                        df_noisy.loc[idx, col] = val.upper()

        # 7. ERR_DATE_AS_OBJECT (Format breaking)
        if "ERR_DATE_AS_OBJECT" in self.config:
            cfg = self.config["ERR_DATE_AS_OBJECT"]
            cols = self._get_target_cols(cfg.get("target_columns"), df_noisy)
            fmt = cfg.get("format", "%Y-%m-%d %H:%M:%S")
            for col in cols:
                try:
                    # ФИКС: Добавлен errors='coerce', чтобы игнорировать ранее вставленные пробелы и мусор
                    df_noisy[col] = pd.to_datetime(df_noisy[col], errors="coerce")

                    mask = np.random.rand(len(df_noisy)) < cfg["ratio"]
                    df_noisy.loc[mask, col] = df_noisy.loc[mask, col].dt.strftime(fmt)
                except Exception as e:
                    print(f"Warning: Could not format date for column {col} - {e}")

        # 8. ERR_ROW_DUPLICATE
        if "ERR_ROW_DUPLICATE" in self.config:
            cfg = self.config["ERR_ROW_DUPLICATE"]
            dup_mask = np.random.rand(len(df_noisy)) < cfg["ratio"]
            duplicates = df_noisy[dup_mask].copy()
            # Append duplicates while keeping the original Target_ID index
            df_noisy = pd.concat([df_noisy, duplicates], ignore_index=False)

        # 9. ERR_DIRTY_COLUMNS (Trailing/leading spaces in headers)
        # Applied last so it doesn't break column referencing for earlier steps
        if "ERR_DIRTY_COLUMNS" in self.config:
            cfg = self.config["ERR_DIRTY_COLUMNS"]
            new_columns = []
            for col in df_noisy.columns:
                if np.random.rand() < cfg["ratio"]:
                    pos = random.choice(cfg.get("position", ["trailing"]))
                    if pos == "trailing":
                        new_columns.append(col + " ")
                    elif pos == "leading":
                        new_columns.append(" " + col)
                    else:
                        new_columns.append(col)
                else:
                    new_columns.append(col)
            df_noisy.columns = new_columns

        return df_noisy


class TransactionSimulator:
    def __init__(self, config: dict, target_col: str):
        self.config = config
        self.target_col = target_col

    def generate(self, profiles_df: pd.DataFrame) -> pd.DataFrame:
        cfg_trends = self.config.get("STORY_TRANSACTION_TRENDS", {})
        cfg_cutoff = self.config.get("STORY_LIFESPAN_CUTOFF", {})

        mapping = cfg_trends.get("mapping", {})
        periods = cfg_trends.get("periods_days", 730)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=periods - 1)
        date_range = pd.date_range(start=start_date, periods=periods, freq="D")

        t = np.arange(periods)
        transactions = []

        for target_id, row in profiles_df.iterrows():
            target_val = str(row.get(self.target_col, "0"))
            behavior = mapping.get(target_val, mapping.get("0", {}))

            # --- 1. Determine Lifespan Cutoff (Hard Stop) ---
            cutoff_date = None
            is_churned = target_val == "1"

            if is_churned and cfg_cutoff.get("action") == "truncate_timeline":
                min_days, max_days = cfg_cutoff.get("churn_days_ago_range", [1, 180])
                # Randomize the exact day they canceled their account
                days_ago = random.randint(min_days, max_days)
                cutoff_date = end_date - timedelta(days=days_ago)

            # --- 2. Extract Mathematical Parameters ---
            base_amt = behavior.get("base_amount", 50.0)
            trend_slope = behavior.get("trend_slope", 0.0)
            weekly_amp = behavior.get("weekly_seasonality_amp", 0.0)
            yearly_amp = behavior.get("yearly_seasonality_amp", 0.0)
            noise_scale = behavior.get("noise_scale", 5.0)
            daily_prob = behavior.get("daily_prob", 0.10)
            drop_off_days = behavior.get("drop_off_last_n_days")

            # --- 3. Build the Time Series Components ---
            trend = t * trend_slope
            weekly_seasonality = weekly_amp * np.sin(2 * np.pi * t / 7)
            yearly_seasonality = yearly_amp * np.sin(2 * np.pi * t / 365.25)
            noise = np.random.normal(0, noise_scale, periods)
            amounts = base_amt + trend + weekly_seasonality + yearly_seasonality + noise

            prob_mask = np.full(periods, daily_prob)

            # --- 4. Apply Behavioral Fade-Out (Before the Hard Stop) ---
            if drop_off_days:
                drop_start_idx = periods - drop_off_days
                if drop_start_idx < 0:
                    drop_start_idx = 0
                decay = np.linspace(1.0, 0.05, periods - drop_start_idx)
                prob_mask[drop_start_idx:] *= decay
                amounts[drop_start_idx:] *= decay

            amounts = np.clip(amounts, 1.0, None)
            transaction_days_mask = np.random.rand(periods) < prob_mask

            active_dates = date_range[transaction_days_mask]
            active_amounts = amounts[transaction_days_mask]

            # --- 5. Filter & Append Transactions ---
            for d, amt in zip(active_dates, active_amounts):
                # ENFORCE THE HARD STOP: If date is after churn, skip it entirely
                if cutoff_date and d > cutoff_date:
                    continue

                random_time = timedelta(
                    hours=random.randint(0, 23), minutes=random.randint(0, 59)
                )
                exact_dt = d + random_time

                transactions.append(
                    {
                        "Target_ID": target_id,
                        "Trans_Date": exact_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "Trans_Amount": round(amt, 2),
                    }
                )

        return pd.DataFrame(transactions)


class ReviewSynthesizer:
    def __init__(self, config: dict, target_col: str):
        self.config = config
        self.target_col = target_col

    def generate(self, profiles_df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config.get("SENTIMENT_CONFIG", {})
        mapping = cfg.get("mapping", {})
        biases = cfg.get("biases", [])

        review_counts = [0, 1, 2, 3]
        review_probs = cfg.get("review_count_probs", [0.50, 0.30, 0.15, 0.05])

        reviews = []

        for target_id, row in profiles_df.iterrows():
            num_reviews = np.random.choice(review_counts, p=review_probs)
            if num_reviews == 0:
                continue

            target_val = str(row.get(self.target_col, "0"))

            # 1. Load Base Behavior
            behavior = (
                mapping.get(target_val)
                or mapping.get("0")
                or {
                    "positive": 0.33,
                    "neutral": 0.33,
                    "negative": 0.34,
                    "context": "user",
                }
            ).copy()

            # Default Prompt Length
            length_instruction = (
                "Output exactly 2 sentences of the review text and nothing else."
            )

            # 2. Evaluate Dynamic Biases
            for bias in biases:
                col = bias.get("column")
                op = bias.get("operator")
                val = bias.get("value")

                if col not in row:
                    continue

                # Safe type casting for comparison
                row_val = row[col]
                condition_met = False
                try:
                    if op == "==" and str(row_val) == str(val):
                        condition_met = True
                    elif op == ">" and float(row_val) > float(val):
                        condition_met = True
                    elif op == "<" and float(row_val) < float(val):
                        condition_met = True
                    elif op == ">=" and float(row_val) >= float(val):
                        condition_met = True
                    elif op == "<=" and float(row_val) <= float(val):
                        condition_met = True
                except ValueError:
                    pass  # Ignore if we can't cast to float

                # Apply the effect if condition is met
                if condition_met:
                    effect = bias.get("effect")
                    if effect == "sentiment_shift":
                        shift = bias.get("shift", {})
                        behavior["positive"] += shift.get("positive", 0.0)
                        behavior["neutral"] += shift.get("neutral", 0.0)
                        behavior["negative"] += shift.get("negative", 0.0)
                    elif effect == "length_override":
                        length_instruction = bias.get(
                            "length_instruction", length_instruction
                        )

            # 3. Clean and Normalize Probabilities
            # Ensure no probability drops below 0 due to negative shifts
            behavior["positive"] = max(0.0, behavior["positive"])
            behavior["neutral"] = max(0.0, behavior["neutral"])
            behavior["negative"] = max(0.0, behavior["negative"])

            # Re-normalize so they perfectly sum to 1.0
            total = behavior["positive"] + behavior["neutral"] + behavior["negative"]
            if total == 0:
                behavior["neutral"] = 1.0
                total = 1.0

            behavior["positive"] /= total
            behavior["neutral"] /= total
            behavior["negative"] /= total

            # 4. Generate the Reviews via LLM
            for _ in range(num_reviews):
                chosen_sentiment = np.random.choice(
                    ["positive", "neutral", "negative"],
                    p=[behavior["positive"], behavior["neutral"], behavior["negative"]],
                )

                system_instruction = (
                    "You are an automated backend database generator. You output ONLY raw data. "
                    "CRITICAL: Never talk to the user. Never write intro, notes, explanations, or wrapping quotes. "
                    f"{length_instruction}"
                )

                user_prompt = (
                    f"Generate a realistic customer review with strict {chosen_sentiment.upper()} sentiment.\n"
                    f"Context topic: {behavior['context']}\n"
                    "Raw Review Text:"
                )

                full_prompt = f"<start_of_turn>user\n{system_instruction}\n\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"

                review_text = ask_ollama(full_prompt)
                review_text = review_text.strip().strip('"').strip("'")

                reviews.append(
                    {
                        "Review_ID": f"REV-{random.randint(100000, 999999)}",
                        "Target_ID": target_id,
                        "Review_Date": (
                            datetime.now() - timedelta(days=random.randint(1, 100))
                        ).strftime("%Y-%m-%d"),
                        "Review_Text": review_text,
                    }
                )

        return pd.DataFrame(reviews)


# ==========================================
# ГЛАВНЫЙ ЗАПУСК
# ==========================================
def main():
    # 1. Настройка парсера аргументов командной строки
    parser = argparse.ArgumentParser(
        description="Скрипт для динамической генерации мультимодальных учебных датасетов."
    )
    parser.add_argument(
        "-v",
        "--variant",
        type=int,
        default=1,
        choices=range(1, 16),
        help="Номер варианта для генерации (целое число от 1 до 15). По умолчанию: 1.",
    )

    # Читаем аргументы
    args = parser.parse_args()
    target_variant = args.variant

    print(f"1. Получение конфигурации для варианта {target_variant}...")
    config = fetch_config_from_gsheet(target_variant)

    print("2. Динамическая генерация базового датасета...")
    base_generator = BaseDataGenerator(config)
    raw_df = base_generator.generate()
    print(f"Сгенерировано {len(raw_df)} строк.")

    target_column = config.get("target_column", "Churn")

    print("3. Применение шума к profiles.csv...")
    corruptor = ProfileCorruptor(config)
    profiles_df = corruptor.apply_noise(raw_df)

    print("4. Симуляция поведенческих логов transactions.csv...")
    simulator = TransactionSimulator(config, target_col=target_column)
    transactions_df = simulator.generate(raw_df)

    print("5. Генерация LLM текстов для reviews.csv...")
    synthesizer = ReviewSynthesizer(config, target_col=target_column)
    reviews_df = synthesizer.generate(raw_df)

    # ---------------------------------------------------------
    # НОВАЯ ЛОГИКА СОХРАНЕНИЯ: Группировка по папкам вариантов
    # ---------------------------------------------------------
    # Берем dataset_id из Google Sheet (или используем fallback имя)
    dataset_name = config.get("dataset_id", f"variant_{target_variant}")
    output_dir = os.path.join("data", "processed", dataset_name)

    print(f"6. Сохранение датасетов в папку: {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    profiles_df.to_csv(os.path.join(output_dir, "profiles.csv"))
    transactions_df.to_csv(os.path.join(output_dir, "transactions.csv"), index=False)
    reviews_df.to_csv(os.path.join(output_dir, "reviews.csv"), index=False)

    print("Генерация успешно завершена!")


if __name__ == "__main__":
    main()
