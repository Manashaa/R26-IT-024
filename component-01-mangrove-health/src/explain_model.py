import pandas as pd

FEATURE_IMPORTANCE_FILE = "outputs/feature_importance.csv"

feature_df = pd.read_csv(FEATURE_IMPORTANCE_FILE)

print("Explainable AI - Feature Importance")
print("------------------------------------")

for index, row in feature_df.iterrows():
    print(f"{index + 1}. {row['Feature']} : {row['Importance']:.4f}")

print("\nInterpretation:")
print("The model explains predictions using Random Forest feature importance.")
print("Higher importance means that feature had stronger influence on prediction decisions.")
print("SHAP-based local explainability can be added in the next development stage.")