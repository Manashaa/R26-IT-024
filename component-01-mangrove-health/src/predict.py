import os
import joblib
import pandas as pd

DATA_PATH = "data/processed/FINAL_FEATURED_DATASET.csv"
MODEL_PATH = "models/mangrove_trend_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
OUTPUT_PATH = "outputs/predictions.csv"

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

KEEP_COLUMNS = [
    "record_id",
    "polygon_id",
    "year",
    "longitude",
    "latitude",
    "trend_label",
    "ndvi",
    "ndvi_previous_year",
    "ndwi_mean",
    "ndwi_previous_year",
    "rainfall_mm",
    "temperature_celsius",
    "elevation_mean",
    "coastal_distance_km"
]

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

df_clean = df.dropna(subset=FEATURES).copy()

X = df_clean[FEATURES]

predicted_encoded = model.predict(X)
predicted_labels = label_encoder.inverse_transform(predicted_encoded)

prediction_probabilities = model.predict_proba(X)
confidence_scores = prediction_probabilities.max(axis=1)

result_df = df_clean[KEEP_COLUMNS].copy()
result_df = result_df.rename(columns={"trend_label": "actual_trend"})

result_df["predicted_trend"] = predicted_labels
result_df["prediction_confidence"] = confidence_scores
result_df["factor_source"] = "Observed environmental data"

result_df.to_csv(OUTPUT_PATH, index=False)

print("Prediction completed successfully.")
print("Saved:", OUTPUT_PATH)
print("Years:", sorted(result_df["year"].unique()))
print("Total records:", len(result_df))
print(result_df.head())