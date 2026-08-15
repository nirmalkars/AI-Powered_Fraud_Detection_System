from fastapi import (
    APIRouter,
    HTTPException,
)

from app.schemas.transaction import (
    TransactionRequest,
    FraudPredictionResponse,
)
from app.services.prediction_service import (
    PredictionService,
)


router = APIRouter(
    prefix="/api/v1",
    tags=["Fraud Detection"],
)


prediction_service = PredictionService()


@router.post(
    "/predict",
    response_model=FraudPredictionResponse,
)
def predict_fraud(
    request: TransactionRequest,
):

    try:

        result = prediction_service.predict(
            request.features
        )

        return result

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Prediction failed: {exc}"
            ),
        )