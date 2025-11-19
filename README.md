# 📉 Telecom Customer Churn Prediction (Python / Scikit-learn)

Dette prosjektet er et komplett, praktisk churn-prediksjonssystem for en fiktiv teleoperatør.  
Målet er å forutsi hvilke kunder som mest sannsynlig avslutter abonnementet sitt basert på kundedata, kontrakter, tjenester og betalingsinformasjon.

Prosjektet demonstrerer både dataforståelse, dataforberedelse og maskinlæring – fra rådata til ferdig trent modell.

---

## 🎯 Hva prosjektet demonstrerer

- Databehandling og feature engineering i Python  
- Tren/test-split, modelltrening og evaluering  
- Sammenligning av flere klassifikasjonsmodeller (LogReg, RandomForest, XGBoost)  
- ROC AUC, presisjon, recall, F1-score  
- Lagre modeller med `joblib`  
- Produksjonsklar prosjektstruktur

---

## 🗂️ Prosjektstruktur

```text
telecom-churn-prediction/
├─ data/
│  └─ telco_customer_churn.csv
│
├─ notebooks/
│  └─ 01_telco_eda_and_modeling.ipynb
│
├─ models/
│  ├─ churn_model.pkl
│  └─ scaler.pkl
│
├─ reports/
│  └─ evaluation_metrics.json
│
├─ src/
│  ├─ data_loader.py
│  ├─ preprocess.py
│  ├─ train_models.py
│  └─ utils.py
│
├─ requirements.txt
└─ README.md
```

## ▶️ Kom i gang
1️⃣ Opprett og aktiver virtuelt miljø

```bash
python -m venv .venv

# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

2️⃣ Tren modellen
```bash
python -m src.train_models
```

3️⃣ Utdata
- Trenet modell ligger i /models
- Evalueringsresultater i /reports
- Notebook med EDA i /notebooks

---

## 📊 Modellresultater
- ROC AUC
- Accuracy
- Precision
- Recall
- F1-score

(verdier varierer etter kjøring)

---

## 🚀 Videre arbeid
- Legge til SHAP feature importance
- Lage en REST-API (FastAPI eller Flask) for prediksjoner
- Legge til hyperparameter-tuning (GridSearch eller Optuna)

---

## 👤 Forfatter
Runar Olsen
Data Analyst – Python | Power BI | Machine Learning