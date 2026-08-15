import torch
import torch.nn as nn


class FraudDetector(nn.Module):
    """
    Feed-forward neural network for binary fraud classification.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.30),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.20),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.network(x)