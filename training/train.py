import json
from pathlib import Path

import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
)

from torch import nn
from torch.utils.data import DataLoader

from app.core.config import settings
from app.models.fraud_detector import FraudDetector
from app.utils.preprocessing import (
    load_data,
    prepare_features,
    fit_scaler,
    save_scaler,
)
from training.dataset import FraudDataset


BATCH_SIZE = 512
EPOCHS = 20
LEARNING_RATE = 0.001
RANDOM_STATE = 42


def train_model():

    print("Loading dataset...")

    dataframe = load_data(
        settings.DATA_PATH
    )

    print(
        f"Dataset shape: {dataframe.shape}"
    )

    features, labels = prepare_features(
        dataframe
    )

    feature_names = list(
        features.columns
    )

    # Train/test split
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    # Fit scaler ONLY on training data
    scaler = fit_scaler(X_train)

    X_train_scaled = scaler.transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # Save preprocessing artifacts
    save_scaler(
        scaler,
        settings.SCALER_PATH,
    )

    with open(
        settings.FEATURE_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            feature_names,
            file,
            indent=4,
        )

    # Create datasets
    train_dataset = FraudDataset(
        X_train_scaled,
        y_train.values,
    )

    test_dataset = FraudDataset(
        X_test_scaled,
        y_test.values,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    input_dim = X_train_scaled.shape[1]

    model = FraudDetector(
        input_dim=input_dim
    )

    # Calculate class weight
    negative_count = np.sum(
        y_train.values == 0
    )

    positive_count = np.sum(
        y_train.values == 1
    )

    pos_weight = (
        negative_count / positive_count
    )

    print(
        f"Normal transactions: {negative_count}"
    )

    print(
        f"Fraud transactions: {positive_count}"
    )

    print(
        f"Positive class weight: "
        f"{pos_weight:.2f}"
    )

    pos_weight_tensor = torch.tensor(
        [pos_weight],
        dtype=torch.float32,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight_tensor
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # Training
    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for batch_x, batch_y in train_loader:

            optimizer.zero_grad()

            logits = model(batch_x)

            loss = criterion(
                logits,
                batch_y,
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss / len(train_loader)
        )

        print(
            f"Epoch "
            f"{epoch + 1}/{EPOCHS} "
            f"| Loss: "
            f"{average_loss:.4f}"
        )

    # Evaluation
    model.eval()

    probabilities = []
    actual_labels = []

    with torch.no_grad():

        for batch_x, batch_y in test_loader:

            logits = model(batch_x)

            probs = torch.sigmoid(
                logits
            )

            probabilities.extend(
                probs.squeeze(1).numpy()
            )

            actual_labels.extend(
                batch_y.squeeze(1).numpy()
            )

    probabilities = np.array(
        probabilities
    )

    actual_labels = np.array(
        actual_labels
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    print("\nClassification Report:")

    print(
        classification_report(
            actual_labels,
            predictions,
            digits=4,
        )
    )

    auc = roc_auc_score(
        actual_labels,
        probabilities,
    )

    print(
        f"ROC-AUC: {auc:.4f}"
    )

    # Save model
    Path(
        settings.MODEL_DIR
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        settings.MODEL_PATH,
    )

    print(
        f"\nModel saved to: "
        f"{settings.MODEL_PATH}"
    )


if __name__ == "__main__":
    train_model()