import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


FINAL_FEATURE_COLUMNS = [
    "commodity_encoded",
    "market_encoded",
    "admin1_encoded",
    "pricetype_encoded",
    "category_encoded",
    "year",
    "month",
    "quarter",
    "is_harvest_season",
    "price_lag_1",
    "price_lag_3",
    "price_lag_6",
    "price_lag_12",
    "price_rolling_mean_3",
    "price_rolling_std_3",
    "price_rolling_mean_6",
]
TARGET_COLUMN = "price"


def build_features() -> None:
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "data" / "processed" / "cleaned_prices.csv"
    processed_dir = project_root / "data" / "processed"
    encoders_dir = project_root / "src" / "models" / "encoders"
    feature_columns_path = project_root / "src" / "models" / "feature_columns.json"
    output_path = processed_dir / "features_dataset.csv"

    # Ensure output directories exist
    processed_dir.mkdir(parents=True, exist_ok=True)
    encoders_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and Prepare
    df = pd.read_csv(input_path, parse_dates=["date"])
    df = df.sort_values(["commodity", "market", "date"]).reset_index(drop=True)

    print("Shape after load and sort:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())

    # 2. Encode Categorical Columns
    categorical_columns = ["commodity", "market", "admin1", "pricetype", "category"]
    encoders = {}

    for col in categorical_columns:
        encoder = LabelEncoder()
        df[f"{col}_encoded"] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder
        joblib.dump(encoder, encoders_dir / f"{col}_encoder.pkl")

    commodity_mapping = dict(
        zip(
            encoders["commodity"].classes_,
            encoders["commodity"].transform(encoders["commodity"].classes_),
        )
    )
    print("\nCommodity encoding mapping:")
    print(commodity_mapping)

    # 3. Create Lag Features
    grouped_price = df.groupby(["commodity", "market"])["price"]
    df["price_lag_1"] = grouped_price.shift(1)
    df["price_lag_3"] = grouped_price.shift(3)
    df["price_lag_6"] = grouped_price.shift(6)
    df["price_lag_12"] = grouped_price.shift(12)

    # 4. Create Rolling Features
    # Rolling windows are computed within each commodity+market sequence.
    df["price_rolling_mean_3"] = grouped_price.transform(
        lambda s: s.rolling(window=3).mean()
    )
    df["price_rolling_std_3"] = grouped_price.transform(
        lambda s: s.rolling(window=3).std()
    )
    df["price_rolling_mean_6"] = grouped_price.transform(
        lambda s: s.rolling(window=6).mean()
    )

    # 5. Add Calendar Features
    df["quarter"] = ((df["month"] - 1) // 3 + 1).astype(int)
    harvest_months = {10, 11, 12, 1, 2}
    df["is_harvest_season"] = df["month"].isin(harvest_months).astype(int)

    # 6. Drop NaN rows
    shape_before_dropna = df.shape
    df = df.dropna().reset_index(drop=True)
    shape_after_dropna = df.shape
    print("\nShape before dropping NaN:", shape_before_dropna)
    print("Shape after dropping NaN:", shape_after_dropna)

    # 7. Select Final Feature Columns
    with open(feature_columns_path, "w", encoding="utf-8") as f:
        json.dump(FINAL_FEATURE_COLUMNS, f, indent=2)

    # 8. Save Output
    df.to_csv(output_path, index=False)
    print("\nSaved feature dataset to:", output_path)
    print("Final dataframe shape:", df.shape)
    print("\nTarget column:", TARGET_COLUMN)

    print("\nBasic stats of final feature columns:")
    print(df[FINAL_FEATURE_COLUMNS].describe().T)


if __name__ == "__main__":
    build_features()
