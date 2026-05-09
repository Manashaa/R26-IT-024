import os
import joblib
import pandas as pd
import numpy as np

DATA_PATH = "data/processed/FINAL_FEATURED_DATASET.csv"
MODEL_PATH = "models/mangrove_trend_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
OUTPUT_PATH = "outputs/future_predictions.csv"

FEATURES = [
    "year",
    "ndvi",
    "ndvi_previous_year",
    "ndwi_mean",
    "ndwi_previous_year",
    "rainfall_mm",
    "temperature_celsius",
    "elevation_mean",
    "coastal_distance_km"
]

FORECAST_FEATURES = [
    "ndvi",
    "ndwi_mean",
    "rainfall_mm",
    "temperature_celsius"
]

STATIC_FEATURES = [
    "longitude",
    "latitude",
    "elevation_mean",
    "coastal_distance_km"
]

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

future_years = [2026, 2027, 2028, 2029, 2030]
all_future_rows = []

def forecast_value(group, feature, future_year):
    group = group.sort_values("year")

    years = group["year"].values
    values = group[feature].values

    valid_mask = ~pd.isna(values)
    years = years[valid_mask]
    values = values[valid_mask]

    if len(values) < 2:
        return values[-1] if len(values) > 0 else np.nan

    slope, intercept = np.polyfit(years, values, 1)
    predicted_value = slope * future_year + intercept

    return predicted_value

for polygon_id, group in df.groupby("polygon_id"):

    group = group.sort_values("year")
    latest_row = group.iloc[-1]

    for future_year in future_years:

        future_row = {}

        future_row["record_id"] = f"{polygon_id}_{future_year}"
        future_row["polygon_id"] = polygon_id
        future_row["year"] = future_year

        for feature in STATIC_FEATURES:
            future_row[feature] = latest_row[feature]

        # Forecast current-year environmental values
        future_row["ndvi"] = forecast_value(group, "ndvi", future_year)
        future_row["ndwi_mean"] = forecast_value(group, "ndwi_mean", future_year)
        future_row["rainfall_mm"] = forecast_value(group, "rainfall_mm", future_year)
        future_row["temperature_celsius"] = forecast_value(group, "temperature_celsius", future_year)

        # Previous-year values for future prediction
        previous_year = future_year - 1

        if previous_year <= 2025:
            previous_row = group[group["year"] == previous_year]

            if len(previous_row) > 0:
                future_row["ndvi_previous_year"] = previous_row.iloc[0]["ndvi"]
                future_row["ndwi_previous_year"] = previous_row.iloc[0]["ndwi_mean"]
            else:
                future_row["ndvi_previous_year"] = latest_row["ndvi"]
                future_row["ndwi_previous_year"] = latest_row["ndwi_mean"]
        else:
            # For 2027–2030, use forecasted previous year values
            future_row["ndvi_previous_year"] = forecast_value(group, "ndvi", previous_year)
            future_row["ndwi_previous_year"] = forecast_value(group, "ndwi_mean", previous_year)

        future_row["actual_trend"] = "Not Available"
        future_row["factor_source"] = "Forecasted environmental factors using temporal trend analysis"

        all_future_rows.append(future_row)

future_df = pd.DataFrame(all_future_rows)

future_df = future_df.dropna(subset=FEATURES).copy()

X_future = future_df[FEATURES]

predicted_encoded = model.predict(X_future)
predicted_labels = label_encoder.inverse_transform(predicted_encoded)

prediction_probabilities = model.predict_proba(X_future)
confidence_scores = prediction_probabilities.max(axis=1)

future_df["predicted_trend"] = predicted_labels
future_df["prediction_confidence"] = confidence_scores

final_columns = [
    "record_id",
    "polygon_id",
    "year",
    "longitude",
    "latitude",
    "actual_trend",
    "ndvi",
    "ndvi_previous_year",
    "ndwi_mean",
    "ndwi_previous_year",
    "rainfall_mm",
    "temperature_celsius",
    "elevation_mean",
    "coastal_distance_km",
    "predicted_trend",
    "prediction_confidence",
    "factor_source"
]

future_df = future_df[final_columns]

future_df.to_csv(OUTPUT_PATH, index=False)

print("Future predictions generated successfully.")
print("Saved:", OUTPUT_PATH)
print("Future years:", future_years)
print("Total records:", len(future_df))
print(future_df.head())