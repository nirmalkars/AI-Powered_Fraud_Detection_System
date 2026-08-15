import torch

from torch.utils.data import Dataset


class FraudDataset(Dataset):
    """
    PyTorch Dataset for fraud classification.
    """

    def __init__(
        self,
        features,
        labels,
    ) -> None:

        self.features = torch.tensor(
            features,
            dtype=torch.float32,
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.float32,
        ).view(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):

        return (
            self.features[index],
            self.labels[index],
        )