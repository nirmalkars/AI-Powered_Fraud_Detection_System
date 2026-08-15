from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "Class"


def load_data(
    data_path: str,
) -> pd.DataFrame:
    """Load fraud dataset."""

    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    return pd.read_csv(path)


def prepare_features(
    dataframe: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "not found."
        )

    data = dataframe.copy()

    # Time is not directly useful in its raw form.
    # Amount is retained as a numerical feature.
    features = data.drop(
        columns=[TARGET_COLUMN]
    )

    labels = data[TARGET_COLUMN]

    return features, labels


def fit_scaler(
    features: pd.DataFrame,
) -> StandardScaler:

    scaler = StandardScaler()

    scaler.fit(features)

    return scaler


def save_scaler(
    scaler: StandardScaler,
    path: str,
) -> None:

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        scaler,
        path,
    )


def load_scaler(
    path: str,
) -> StandardScaler:

    return joblib.load(path)