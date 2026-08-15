import json

import joblib
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from app.core.config import settings
from app.models.fraud_detector import FraudDetector
from app.utils.preprocessing import (
    load_data,
    prepare_features,
)


def evaluate():

    print("Loading dataset...")

    dataframe = load_data(
        settings.DATA_PATH
    )

    features, labels = prepare_features(
        dataframe
    )

    scaler = joblib.load(
        settings.SCALER_PATH
    )

    with open(
        settings.FEATURE_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        feature_names = json.load(file)

    features = features[
        feature_names
    ]

    scaled_features = scaler.transform(
        features
    )

    X = torch.tensor(
        scaled_features,
        dtype=torch.float32,
    )

    model = FraudDetector(
        input_dim=len(feature_names)
    )

    state_dict = torch.load(
        settings.MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    with torch.no_grad():

        logits = model(X)

        probabilities = torch.sigmoid(
            logits
        ).squeeze(1).numpy()

    predictions = (
        probabilities >= settings.THRESHOLD
    ).astype(int)

    y_true = labels.values

    print("\nEvaluation Results")
    print("=" * 40)

    print(
        f"Accuracy : "
        f"{accuracy_score(y_true, predictions):.4f}"
    )

    print(
        f"Precision: "
        f"{precision_score(y_true, predictions, zero_division=0):.4f}"
    )

    print(
        f"Recall   : "
        f"{recall_score(y_true, predictions, zero_division=0):.4f}"
    )

    print(
        f"F1 Score : "
        f"{f1_score(y_true, predictions, zero_division=0):.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{roc_auc_score(y_true, probabilities):.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_true,
            predictions,
        )
    )


if __name__ == "__main__":
    evaluate()