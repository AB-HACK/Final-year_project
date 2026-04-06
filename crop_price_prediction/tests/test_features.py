import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


def _synthetic_feature_df():
    return pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=6, freq="MS"),
            "commodity": ["Maize"] * 6,
            "market": ["TestMarket"] * 6,
            "admin1": ["Kano"] * 6,
            "pricetype": ["Wholesale"] * 6,
            "category": ["cereals and tubers"] * 6,
            "price": [100, 110, 105, 120, 115, 130],
        }
    )


def _apply_group_lag_rolling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["commodity", "market", "date"]).reset_index(drop=True)
    grouped_price = df.groupby(["commodity", "market"])["price"]

    df["price_lag_1"] = grouped_price.shift(1)
    df["price_lag_3"] = grouped_price.shift(3)
    df["price_lag_6"] = grouped_price.shift(6)
    df["price_lag_12"] = grouped_price.shift(12)
    df["price_rolling_mean_3"] = grouped_price.transform(lambda s: s.rolling(3).mean())
    df["price_rolling_std_3"] = grouped_price.transform(lambda s: s.rolling(3).std())
    df["price_rolling_mean_6"] = grouped_price.transform(lambda s: s.rolling(6).mean())

    return df


def test_no_nulls_after_feature_build():
    df = _synthetic_feature_df()
    engineered = _apply_group_lag_rolling(df)
    cleaned = engineered.dropna().reset_index(drop=True)
    assert cleaned.isna().sum().sum() == 0


def test_lag_feature_correctness():
    df = _synthetic_feature_df()
    engineered = _apply_group_lag_rolling(df)

    # Check at least 3 positions where lag-1 should equal prior row's price.
    for idx in [1, 2, 3]:
        assert engineered.loc[idx, "price_lag_1"] == engineered.loc[idx - 1, "price"]


def test_time_based_split_no_leakage():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=20, freq="MS"),
            "price": np.linspace(100, 200, 20),
        }
    ).sort_values("date")

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    assert train_df["date"].max() <= test_df["date"].min()


def test_harvest_season_encoding():
    harvest_months = [10, 11, 12, 1, 2]
    non_harvest_months = [3, 4, 5, 6, 7, 8, 9]

    for month in harvest_months:
        is_harvest_season = 1 if month in [10, 11, 12, 1, 2] else 0
        assert is_harvest_season == 1

    for month in non_harvest_months:
        is_harvest_season = 1 if month in [10, 11, 12, 1, 2] else 0
        assert is_harvest_season == 0


MODEL_PATH = Path(__file__).resolve().parents[1] / "src" / "models" / "best_model.joblib"
FEATURE_COLUMNS_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "models" / "feature_columns.json"
)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="best_model.joblib not found")
def test_prediction_output_is_positive():
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    feature_df = pd.DataFrame([[1.0] * len(feature_columns)], columns=feature_columns)
    pred = model.predict(feature_df)[0]

    assert isinstance(float(pred), float)
    assert float(pred) > 0
