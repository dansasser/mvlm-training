import torch
from torch.utils.data import DataLoader

class PrioritaryTrainer:
    """Simple training helper for Prioritary MVLM."""

    def __init__(self, model, dataset, batch_size: int = 8):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size

    def train_epoch(self):
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        for batch in loader:
            inputs = batch.to(self.model.device)
            self.model(inputs, labels=inputs)
