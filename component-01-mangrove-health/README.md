# Component 01 — AI-Based Mangrove Ecosystem Health Monitoring, Change Detection, and Future Trend Prediction

## 📌 Overview

This component focuses on monitoring mangrove ecosystem health using Artificial Intelligence (AI), Remote Sensing, and GIS technologies.

The system analyzes satellite imagery and environmental datasets to:

- Monitor mangrove vegetation health
- Detect spatial and temporal environmental changes
- Predict future mangrove ecosystem trends
- Visualize prediction results using GIS-based interactive maps and dashboards

The selected study area for this research is the Puttalam District in Sri Lanka.

---

# 🎯 Objectives

- Analyze mangrove vegetation health using NDVI and environmental indicators
- Detect ecosystem changes across multiple years
- Predict future mangrove ecosystem conditions using Machine Learning
- Generate GIS-based visualizations and prediction maps
- Support conservation and environmental monitoring decision-making

---

# 🛰️ Technologies Used

## GIS & Remote Sensing
- Google Earth Engine
- QGIS
- Sentinel-2 Satellite Imagery
- Global Mangrove Watch Dataset

## Programming & Machine Learning
- Python
- Pandas
- Scikit-learn
- Random Forest Classifier

## Visualization
- Streamlit
- Folium
- Matplotlib

---

# 📂 Folder Structure

```text
component-01-mangrove-health/
│
├── api/
│   └── main.py
│
├── dashboard/
│   └── app.py
│
├── data/
│   └── processed/
│       └── FINAL_FEATURED_DATASET.csv
│
├── models/
│   ├── label_encoder.pkl
│   └── mangrove_trend_model.pkl
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── feature_importance.csv
│   ├── feature_importance.png
│   ├── future_predictions.csv
│   ├── prediction_map.html
│   └── predictions.csv
│
├── src/
│   ├── train_model.py
│   ├── predict.py
│   ├── explain_model.py
│   ├── generate_future_predictions.py
│   └── create_map.py
│
├── requirements.txt
└── README.md
```

---

# 📊 Dataset Information

The component uses processed environmental and remote sensing datasets containing:

- NDVI values
- Rainfall data
- Temperature data
- Polygon-based GIS information
- Historical environmental trends

The dataset was prepared using satellite image processing and GIS-based feature extraction methods.

---

# 🧠 Machine Learning Workflow

## 1. Data Collection
Satellite imagery and environmental datasets are collected from multiple open-source platforms.

## 2. Data Preprocessing
The datasets are cleaned, merged, and transformed into machine learning-ready formats.

## 3. Feature Engineering
Environmental indicators and temporal features are generated for model training.

## 4. Model Training
A Random Forest machine learning model is trained to identify mangrove ecosystem trends.

## 5. Prediction Generation
The trained model predicts future ecosystem conditions and trend categories.

## 6. GIS Visualization
Prediction outputs are visualized using interactive GIS-based maps and dashboards.

---

# 📈 Current Features

✅ Dataset preprocessing pipeline  
✅ Machine learning model training  
✅ Trend prediction generation  
✅ Feature importance analysis  
✅ Confusion matrix evaluation  
✅ GIS prediction map generation  
✅ Dashboard integration  

---

# 📌 Generated Outputs

The system currently generates:

- Mangrove trend predictions
- Feature importance analysis
- Confusion matrix evaluation
- Future prediction datasets
- Interactive GIS prediction maps

---

# 🌍 Expected Impact

This component aims to support:

- Environmental monitoring
- Mangrove conservation planning
- Coastal ecosystem analysis
- Future environmental risk prediction
- Data-driven conservation decision-making

---

# ⚠️ Research Prototype Notice

This component is currently developed as part of an undergraduate research project. Features and models may continue to improve during future development phases.

---

# 👨‍🎓 Academic Information

Undergraduate Research Project  
Faculty of Computing  
Sri Lanka Institute of Information Technology (SLIIT)

Academic Year: 2025 / 2026