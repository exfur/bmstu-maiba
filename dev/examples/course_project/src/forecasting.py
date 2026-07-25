import logging
import warnings
from pathlib import Path

import pandas as pd

logging.getLogger("prophet").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
except ImportError:
    pass


def forecast_trends(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    periods: int,
    freq: str = "D",
    output_dir: Path | str | None = None,
) -> pd.DataFrame:
    """
    Time-series trend analysis exporting predictions directly to Power BI folder.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / "data" / "powerbi"
    output_dir = Path(output_dir)

    df_clean = df[[date_col, value_col]].copy()
    df_clean = df_clean.rename(columns={date_col: "ds", value_col: "y"})
    df_clean["ds"] = pd.to_datetime(df_clean["ds"])

    model = Prophet(
        yearly_seasonality=False,  # type: ignore
        weekly_seasonality=True,  # type: ignore
        daily_seasonality=False,  # type: ignore
    )
    model.fit(df_clean)

    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    result_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    result_df = result_df.rename(columns={"ds": date_col})

    output_dir.mkdir(parents=True, exist_ok=True)
    export_path = output_dir / "forecast_trends.csv"
    result_df.to_csv(export_path, index=False)
    print(f"💾 Saved forecast trends to: {export_path.resolve()}")

    return result_df
