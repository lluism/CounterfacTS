from abc import ABC
import os
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn

from .trainer import Trainer
from ..utils.evaluation import mean_mase, mean_mse, mean_smape


class BaseModel(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    def _extract_input_from_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def calculate_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def fit(self, trainer: Trainer, datadir: str) -> None:
        if self.path is None:
            raise ValueError("Path cannot be None when fitting model")

        datadir = os.path.join(datadir, "training_data")
        os.makedirs(datadir, exist_ok=True)
        trainer.train(self, 0.001, datadir, early_stopping=True)

        torch.save(self.state_dict(), os.path.join(self.path, "model.pth"))

    def predict(self, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        raise NotImplementedError

    def validate(self, batch: Dict[str, torch.Tensor], sp: int = 1) -> Tuple[float, float, float]:
        x = self._extract_input_from_batch(batch)
        y = batch["future_target"].numpy()
        scaled_x = self.scaler.fit_transform(x)

        with torch.no_grad():
            output = self(scaled_x)
            output = self.scaler.inverse_transform(output).squeeze(dim=-1)

            x = x.cpu().numpy()
            output = output.cpu().numpy()
            return mean_mase(y, output, x, sp=sp), mean_smape(y, output), mean_mse(output, y)
