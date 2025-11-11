# src/utils.py
from pathlib import Path
import logging

# Prosjektrot = mappen som inneholder src-mappen (to nivåer opp fra denne filen)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_data_path() -> Path:
    """Returnerer path til data-mappen."""
    return PROJECT_ROOT / "data"


def get_raw_data_path() -> Path:
    """Returnerer path til raw data."""
    return get_data_path() / "raw"


def get_reports_path() -> Path:
    """Returnerer path til reports-mappen."""
    path = PROJECT_ROOT / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_models_path() -> Path:
    """Returnerer path til models-mappen."""
    path = PROJECT_ROOT / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_logging():
    """Setter opp enkel logging til console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
