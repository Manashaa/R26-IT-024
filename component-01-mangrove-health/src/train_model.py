import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

DATA_PATH = "data/processed/FINAL_FEATURED_DATASET.csv"

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

TARGET = "trend_label"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print("Original shape:", df.shape)

df = df.dropna(subset=FEATURES + [TARGET])

print("After removing missing values:", df.shape)

train_df = df[df["year"] <= 2023]
test_df = df[df["year"] >= 2024]

print("\nTraining years: 2001–2023")
print("Testing years: 2024–2025")
print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train_encoded)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test_encoded, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(
    y_test_encoded,
    y_pred,
    target_names=label_encoder.classes_
))

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

joblib.dump(model, "models/mangrove_trend_model.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

cm = confusion_matrix(y_test_encoded, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Temporal Validation")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.close()

importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

importance_df.to_csv("outputs/feature_importance.csv", index=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=importance_df, x="Importance", y="Feature")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()

print("\nModel saved successfully")
print("Confusion Matrix Saved")
print("Feature Importance Saved")
print("\nTraining completed successfully.")