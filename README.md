# Telecom Customer Churn Prediction (Python / Scikit-learn)

Dette prosjektet er et end-to-end churn-prediksjonsprosjekt for en fiktiv teleoperatør.  
Målet er å forutsi om en kunde kommer til å avslutte abonnementet (churn) basert på
kundedata, kontraktstype, betalingsmetode og hvilke tjenester kunden bruker.

Prosjektet er laget for å vise forståelse for:
- Databehandling og feature engineering i Python
- Bruk av `scikit-learn`-pipelines (preprocessing + modell)
- Tren/test-splitt, evaluering med ROC AUC, presisjon, recall og F1
- Sammenligning av flere klassifikasjonsmodeller
- Lagre trenet modell til disk for senere bruk

## Datasett

Datasettet er IBM sitt **Telco Customer Churn**-datasett (ofte brukt i churn-eksempler):

- Hver rad = én kunde
- Target-kolonne: `Churn` (Yes/No)
- Eksempler på featurer:
  - Demografi: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - Konto: `tenure` (antall måneder), `Contract`, `PaymentMethod`,
    `PaperlessBilling`
  - Tjenester: `PhoneService`, `InternetService`, `OnlineSecurity`,
    `StreamingTV`, `StreamingMovies`
  - Økonomi: `MonthlyCharges`, `TotalCharges`

Datasettet kan lastes ned fra Kaggle (søk etter **"Telco Customer Churn"**).  
Lagre filen `WA_Fn-UseC_-Telco-Customer-Churn.csv` inn i `data/raw/`.

> Merk: Selve datafilen er **ikke** inkludert i repoet (ligger i `.gitignore`).

## Prosjektstruktur

```text
telecom-churn-prediction/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ data/
│  ├─ raw/
│  └─ processed/
├─ notebooks/
│  └─ 01_telco_churn_eda_and_modeling.ipynb
├─ src/
│  ├─ train_models.py
│  └─ utils.py
├─ models/
└─ reports/
   ├─ figures/
   └─ metrics.txt

## Utforskende dataanalyse (EDA)

Notebooken [01_telco_churn_eda_and_modeling.ipynb](notebooks/01_telco_churn_eda_and_modeling.ipynb)
inneholder en trinnvis utforskning av churn-datasettet.

Her viser jeg:
- hvordan data lastes inn og ryddes
- churn-rate og fordeling på sentrale variabler
- sammenhenger mellom churn og både numeriske og kategoriske features
- en enkel Random Forest-modell for å identifisere de viktigste churn-driverne
- visualisering av ROC-kurve og feature importance

Notebooken er laget for å demonstrere hvordan man går fra rå data til
innsikt og første modell — en realistisk dataanalytikerprosess i Python.
