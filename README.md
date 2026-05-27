# 🌊 FloodHunger Ghana

## Flood-Driven Food Insecurity Early Warning System for Ghana

FloodHunger Ghana is a district-level machine learning pipeline that predicts food insecurity risks caused by flooding, rainfall shocks, rising food prices, and conflict events across Ghana. The project combines climate, market, and conflict datasets to generate actionable early warning insights for humanitarian agencies, policymakers, and local communities.

---

## 🚀 Project Overview

This project integrates:

- **CHIRPS v3 Rainfall Data**
- **WFP VAM Food Price Data**
- **ACLED Conflict Event Data**

to build an AI-powered early warning system capable of:

- Predicting IPC food insecurity phases
- Forecasting food price trends
- Detecting abnormal climate and conflict events
- Supporting district-level humanitarian response planning

---

## 📊 Scope

| Feature | Details |
|---|---|
| Coverage | 53 districts across Ghana |
| Time Range | 2003 – 2024 |
| Data Volume | ~14,000 records |
| Regions | 16 Ghanaian regions |
| Pipeline Type | End-to-end ML workflow |
| Outputs | Alerts, forecasts, SHAP explainability, anomaly detection |

---

## 🧠 Machine Learning Models

### 1. XGBoost IPC Classifier
Predicts food insecurity phases using rainfall, food prices, and conflict indicators.

### 2. Random Forest Regressor
Forecasts commodity price movements and market stress.

### 3. Isolation Forest
Detects unusual climate or conflict anomalies that may indicate emerging crises.

---

## ⚙️ Pipeline Workflow

1. Load and inspect raw datasets
2. Clean and standardize regional data
3. Aggregate district-level rainfall indicators
4. Merge rainfall, price, and conflict datasets
5. Engineer predictive features
6. Train and evaluate ML models
7. Generate visualizations and insights
8. Export alerts and explainability outputs

---

## 📁 Project Structure

```bash
FloodHunger_Ghana/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   └── FloodHunger_Ghana_Pipeline.ipynb
│
├── outputs/
│   ├── figures/
│   ├── reports/
│   └── alerts/
│
├── models/
│
├── requirements.txt
└── README.md
```

---

## 📈 Key Features

- District-level food insecurity monitoring
- Climate-driven risk prediction
- Flood event analysis
- Conflict impact integration
- Market price forecasting
- Explainable AI using SHAP
- Humanitarian early warning support

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- SHAP
- Jupyter Notebook

---

## 🌍 Potential Impact

FloodHunger Ghana aims to help:

- Humanitarian organizations
- Government agencies
- Disaster response teams
- Researchers and policymakers
- Smallholder farming communities

make faster and data-driven decisions during climate and food crises.

---

## 📌 Future Improvements

- Real-time weather integration
- SMS alert system for farmers
- Satellite imagery support
- Interactive dashboard deployment
- API integration for humanitarian agencies

---

## 👨‍💻 Author

Created by Abdul Wahab Osman and collaborators as part of an AI-driven humanitarian and climate resilience initiative focused on Ghana.

---

## 📜 License

This project is open-source and available for educational, research, and humanitarian purposes.
