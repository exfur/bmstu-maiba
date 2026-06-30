import logging
import warnings

import pandas as pd

# Suppress Prophet's aggressive logging for cleaner production outputs
logging.getLogger("prophet").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
except ImportError:
    pass


def forecast_trends(
    df: pd.DataFrame, date_col: str, value_col: str, periods: int, freq: str = "D"
) -> pd.DataFrame:
    """
    Industrial module for automated time-series trend analysis and forecasting.
    Extracts underlying components and handles optimistic/pessimistic horizons.
    """
    # 1. Isolate and prepare standard Prophet format columns
    df_clean = df[[date_col, value_col]].copy()
    df_clean = df_clean.rename(columns={date_col: "ds", value_col: "y"})
    df_clean["ds"] = pd.to_datetime(df_clean["ds"])

    # 2. Initialize and fit the Meta Prophet model
    model = Prophet(
        yearly_seasonality=False,  # type: ignore
        weekly_seasonality=True,  # type: ignore
        daily_seasonality=False,  # type: ignore
    )
    model.fit(df_clean)

    # 3. Generate future date frames and inference
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    # 4. Reconstruct original business terms for downstream BI tool mapping
    result_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    result_df = result_df.rename(columns={"ds": date_col})

    return result_df
