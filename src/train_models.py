# src/train_models.py

from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
    # NB: Hvis du får import-feil på noe her senere, sier du ifra.
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from .utils import (
    configure_logging,
    get_models_path,
    get_raw_data_path,
    get_reports_path,
)

logger = logging.getLogger(__name__)


def load_data(filename: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv") -> pd.DataFrame:
    """Laster inn churn-datasettet fra data/raw/."""
    data_path = get_raw_data_path() / filename
    if not data_path.exists():
        raise FileNotFoundError(
            f"Fant ikke datafilen: {data_path}. "
            "Last ned filen fra Kaggle og legg den i data/raw/"
        )

    df = pd.read_csv(data_path)
    logger.info("Data shape før rydding: %s", df.shape)

    # Konverter TotalCharges til numerisk, og fjern rader med manglende verdi
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    # Fjern customerID – unik nøkkel, ikke informativ som feature
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    logger.info("Data shape etter rydding: %s", df.shape)
    return df


def preprocess_features(df: pd.DataFrame):
    """Splitter i X/y og setter opp kolonnelister for numeriske og kategoriske features."""
    # Target
    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn"])

    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [col for col in X.columns if col not in numeric_features]

    return X, y, numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    """ColumnTransformer som skalerer numeriske og one-hot-koder kategoriske variabler."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def get_models():
    """Returnerer et dictionary med flere modeller som skal trenes og sammenlignes."""
    models = {
        "log_reg": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
    }

    if HAS_XGBOOST:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

    return models


def evaluate_model(name: str, y_test, y_pred, y_proba) -> dict:
    """Beregner evalueringsmetrikker for en modell."""
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)

    logger.info("=== Resultater for %s ===", name)
    logger.info("Accuracy: %.4f", acc)
    logger.info("ROC AUC : %.4f", auc)
    logger.info("Confusion matrix:\n%s", cm)
    logger.info("Classification report:\n%s", report)

    return {
        "model": name,
        "accuracy": acc,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def save_best_model(best_pipeline: Pipeline, model_name: str):
    """Lagrer beste modell til models/-mappen."""
    models_dir = get_models_path()
    out_path = models_dir / f"best_model_{model_name}.joblib"
    joblib.dump(best_pipeline, out_path)
    logger.info("Lagret beste modell til: %s", out_path)


def save_metrics(all_results: list[dict]):
    """Lagrer evalueringsresultater til reports/metrics.txt."""
    reports_dir = get_reports_path()
    out_path = reports_dir / "metrics.txt"

    lines = []
    for res in all_results:
        lines.append(f"Model: {res['model']}")
        lines.append(f"Accuracy: {res['accuracy']:.4f}")
        lines.append(f"ROC_AUC: {res['roc_auc']:.4f}")
        lines.append("Classification report:")
        lines.append(res["classification_report"])
        lines.append("-" * 60)

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Lagret metrikk-rapport til: %s", out_path)


def main():
    configure_logging()
    logger.info("Starter churn-trening...")

    # 1. Last data
    df = load_data()

    # 2. Preprocess
    X, y, numeric_features, categorical_features = preprocess_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("Train size: %s, Test size: %s", X_train.shape, X_test.shape)

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    models = get_models()

    all_results = []
    best_auc = -np.inf
    best_model_name = None
    best_pipeline = None

    # 3. Tren og evaluer hver modell
    for name, model in models.items():
        logger.info("Trener modell: %s", name)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        res = evaluate_model(name, y_test, y_pred, y_proba)
        all_results.append(res)

        if res["roc_auc"] > best_auc:
            best_auc = res["roc_auc"]
            best_model_name = name
            best_pipeline = pipeline

    logger.info("Beste modell: %s med ROC AUC = %.4f", best_model_name, best_auc)

    # 4. Lagre beste modell + metrikkrapporter
    if best_pipeline is not None and best_model_name is not None:
        save_best_model(best_pipeline, best_model_name)
        save_metrics(all_results)

    logger.info("Ferdig!")


if __name__ == "__main__":
    main()
