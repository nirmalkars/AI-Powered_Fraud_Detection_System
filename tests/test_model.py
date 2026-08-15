import torch

from app.models.fraud_detector import FraudDetector


def test_model_output_shape():

    model = FraudDetector(
        input_dim=30
    )

    sample = torch.randn(
        4,
        30,
    )

    output = model(sample)

    assert output.shape == (
        4,
        1,
    )