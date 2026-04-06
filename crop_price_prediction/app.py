import datetime
import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request


app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "src" / "models"
ENCODERS_DIR = MODELS_DIR / "encoders"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return joblib.load(path)


# Startup loads
model = _load_pickle(MODELS_DIR / "best_model.joblib")

with open(MODELS_DIR / "feature_columns.json", "r", encoding="utf-8") as f:
    feature_columns = json.load(f)

with open(MODELS_DIR / "model_metadata.json", "r", encoding="utf-8") as f:
    model_metadata = json.load(f)

encoders = {
    "commodity": _load_pickle(ENCODERS_DIR / "commodity_encoder.pkl"),
    "market": _load_pickle(ENCODERS_DIR / "market_encoder.pkl"),
    "admin1": _load_pickle(ENCODERS_DIR / "admin1_encoder.pkl"),
    "category": _load_pickle(ENCODERS_DIR / "category_encoder.pkl"),
}

# Handle potential file naming inconsistency requested by user
pricetype_encoder_path = ENCODERS_DIR / "pricetype_encoded.pkl"
if not pricetype_encoder_path.exists():
    pricetype_encoder_path = ENCODERS_DIR / "pricetype_encoder.pkl"
encoders["pricetype"] = _load_pickle(pricetype_encoder_path)

# Build commodity -> category mapping at startup
cleaned_prices_path = DATA_PROCESSED_DIR / "cleaned_prices.csv"
mapping_df = pd.read_csv(cleaned_prices_path, usecols=["commodity", "category"]).dropna()
commodity_category_map = (
    mapping_df.drop_duplicates(subset=["commodity"])
    .set_index("commodity")["category"]
    .to_dict()
)

# Build commodity -> state -> markets mapping
df_map = pd.read_csv(
    cleaned_prices_path, usecols=["commodity", "admin1", "market"]
).dropna()
commodity_state_market = {}
for commodity, grp in df_map.groupby("commodity"):
    commodity_state_market[commodity] = {}
    for state, sgrp in grp.groupby("admin1"):
        markets = sorted(sgrp["market"].unique().tolist())
        commodity_state_market[commodity][state] = markets

print("Startup load complete:")
print(f"- Model loaded: {MODELS_DIR / 'best_model.joblib'}")
print(f"- Feature columns loaded: {MODELS_DIR / 'feature_columns.json'}")
print(f"- Model metadata loaded: {MODELS_DIR / 'model_metadata.json'}")
print(f"- Encoders loaded: {list(encoders.keys())}")
print(f"- Commodity-category mappings loaded: {len(commodity_category_map)}")
print(f"- Commodity-state-market mappings loaded: {len(commodity_state_market)}")


def _encode_value(encoder_key: str, value: str) -> int:
    encoder = encoders[encoder_key]
    value_str = str(value)
    if value_str not in set(encoder.classes_):
        raise ValueError(f"Unknown {encoder_key}: {value_str}")
    return int(encoder.transform([value_str])[0])


@app.route("/", methods=["GET"])
def index():
    commodities = sorted(list(encoders["commodity"].classes_))
    current_year = datetime.datetime.now().year
    model_info = {
        "model_name": model_metadata.get("model_name", "Unknown Model"),
        "r2": model_metadata.get("r2", 0.0),
    }
    return render_template(
        "index.html",
        commodities=commodities,
        commodity_state_market=json.dumps(commodity_state_market),
        current_year=current_year,
        model_info=model_info,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        commodity = request.form.get("commodity", "").strip()
        market = request.form.get("market", "").strip()
        admin1 = request.form.get("admin1", "").strip()
        pricetype = request.form.get("pricetype", "").strip()

        if not commodity or not admin1 or not market:
            return render_template(
                "error.html",
                error_message=(
                    "Please complete all dropdown selections before predicting."
                ),
            )

        previous_price = float(request.form.get("previous_price", "0"))
        year = int(request.form.get("year", "0"))
        month = int(request.form.get("month", "0"))

        if previous_price <= 0:
            raise ValueError("Previous price must be greater than zero.")
        if month < 1 or month > 12:
            raise ValueError("Month must be between 1 and 12.")

        commodity_encoded = _encode_value("commodity", commodity)
        market_encoded = _encode_value("market", market)
        admin1_encoded = _encode_value("admin1", admin1)
        pricetype_encoded = _encode_value("pricetype", pricetype)

        category = commodity_category_map.get(commodity)
        if not category:
            raise ValueError(
                f"Could not determine category for commodity '{commodity}'."
            )
        category_encoded = _encode_value("category", category)

        quarter = ((month - 1) // 3) + 1
        is_harvest_season = 1 if month in [10, 11, 12, 1, 2] else 0

        feature_values = {
            "commodity_encoded": commodity_encoded,
            "market_encoded": market_encoded,
            "admin1_encoded": admin1_encoded,
            "pricetype_encoded": pricetype_encoded,
            "category_encoded": category_encoded,
            "year": year,
            "month": month,
            "quarter": quarter,
            "is_harvest_season": is_harvest_season,
            "price_lag_1": previous_price,
            "price_lag_3": previous_price,
            "price_lag_6": previous_price,
            "price_lag_12": previous_price,
            "price_rolling_mean_3": previous_price,
            "price_rolling_std_3": 25.0,
            "price_rolling_mean_6": previous_price,
        }

        feature_df = pd.DataFrame([feature_values], columns=feature_columns)
        predicted_price = float(model.predict(feature_df)[0])

        if predicted_price > previous_price * 1.05:
            trend = "Increasing"
        elif predicted_price < previous_price * 0.95:
            trend = "Decreasing"
        else:
            trend = "Stable"

        confidence = f"{float(model_metadata.get('r2', 0.0)) * 100:.1f}%"

        return render_template(
            "result.html",
            predicted_price=predicted_price,
            trend=trend,
            confidence=confidence,
            commodity=commodity,
            market=market,
            year=year,
            month=month,
            previous_price=previous_price,
            model_name=model_metadata.get("model_name", "Unknown Model"),
        )
    except Exception as exc:
        return render_template(
            "error.html",
            error_message=(
                "We could not process your prediction request. "
                f"Please check your inputs and try again. Details: {exc}"
            ),
        )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model": model_metadata.get("model_name"),
            "r2": model_metadata.get("r2"),
        }
    )


@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
