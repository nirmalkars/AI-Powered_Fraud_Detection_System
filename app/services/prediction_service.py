import json
from pathlib import Path

import joblib
import numpy as np
import torch

from app.core.config import settings
from app.models.fraud_detector import FraudDetector


class PredictionService:
    """Handle fraud model inference."""

    def __init__(self):

        self.model = None
        self.scaler = None
        self.feature_names = None

        self._load_artifacts()

    def _load_artifacts(self):

        if not Path(
            settings.MODEL_PATH
        ).exists():

            raise FileNotFoundError(
                "Trained model not found. "
                "Run training first."
            )

        if not Path(
            settings.SCALER_PATH
        ).exists():

            raise FileNotFoundError(
                "Scaler not found. "
                "Run training first."
            )

        if not Path(
            settings.FEATURE_PATH
        ).exists():

            raise FileNotFoundError(
                "Feature metadata not found. "
                "Run training first."
            )

        self.scaler = joblib.load(
            settings.SCALER_PATH
        )

        with open(
            settings.FEATURE_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            self.feature_names = json.load(
                file
            )

        self.model = FraudDetector(
            input_dim=len(
                self.feature_names
            )
        )

        state_dict = torch.load(
            settings.MODEL_PATH,
            map_location="cpu",
            weights_only=True,
        )

        self.model.load_state_dict(
            state_dict
        )

        self.model.eval()

    def predict(
        self,
        features,
    ):

        if len(features) != len(
            self.feature_names
        ):

            raise ValueError(
                f"Expected "
                f"{len(self.feature_names)} "
                f"features, received "
                f"{len(features)}."
            )

        features_array = np.array(
            features,
            dtype=np.float32,
        ).reshape(1, -1)

        scaled_features = (
            self.scaler.transform(
                features_array
            )
        )

        tensor = torch.tensor(
            scaled_features,
            dtype=torch.float32,
        )

        with torch.no_grad():

            logits = self.model(
                tensor
            )

            probability = torch.sigmoid(
                logits
            ).item()

        prediction = (
            "FRAUD"
            if probability >= settings.THRESHOLD
            else "NORMAL"
        )

        risk_level = (
            self._get_risk_level(
                probability
            )
        )

        return {
            "fraud_probability": round(
                probability,
                4,
            ),
            "prediction": prediction,
            "risk_level": risk_level,
        }

    @staticmethod
    def _get_risk_level(
        probability: float,
    ) -> str:

        if probability >= 0.70:
            return "HIGH"

        if probability >= 0.30:
            return "MEDIUM"

        return "LOW"