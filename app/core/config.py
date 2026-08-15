from pathlib import Path


class Settings:
    """Application configuration."""

    APP_NAME = "PyTorch Fraud Detection API"
    VERSION = "1.0.0"

    BASE_DIR = Path(__file__).resolve().parents[2]

    DATA_PATH = (
        BASE_DIR / "data" / "raw" / "creditcard.csv"
    )

    MODEL_DIR = BASE_DIR / "models"

    MODEL_PATH = MODEL_DIR / "fraud_detector.pt"
    SCALER_PATH = MODEL_DIR / "scaler.pkl"
    FEATURE_PATH = MODEL_DIR / "feature_names.json"

    INPUT_DIM = 30

    THRESHOLD = 0.50


settings = Settings()