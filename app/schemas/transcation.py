from typing import List

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """
    Transaction input for fraud prediction.

    Features must follow the same order as the
    training dataset.
    """

    features: List[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description=(
            "30 transaction features in "
            "training order."
        ),
    )


class FraudPredictionResponse(BaseModel):

    fraud_probability: float

    prediction: str

    risk_level: str