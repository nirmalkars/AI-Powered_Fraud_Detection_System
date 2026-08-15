from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description=(
        "PyTorch-based Credit Card "
        "Fraud Detection API"
    ),
)


app.include_router(router)


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": settings.APP_NAME,
    }