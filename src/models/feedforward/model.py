from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn

from ..base import BaseModel
from ..scalers import get_scaler


class FeedForward(BaseModel):
    """Adapted from https://github.com/awslabs/gluon-ts/blob/master/examples/pytorch_predictor_example.ipynb
    """

    def __init__(self,
                 prediction_length: int,
                 context_length: int,
                 hidden_dimensions: List[int],
                 device: torch.device,
                 path: str,
                 scaler: str = "mean_abs") -> None:
        super().__init__()

        assert prediction_length > 0
        assert context_length > 0
        assert len(hidden_dimensions) > 0

        self.prediction_length = prediction_length
        self.context_length = context_length
        self.hidden_dimensions = hidden_dimensions
        self.device = device
        self.path = path
        self.scaler = get_scaler(scaler)(self.device)
        self.criterion = nn.MSELoss()

        dimensions = [context_length] + hidden_dimensions[:-1]
        modules = []
        for in_size, out_size in zip(dimensions[:-1], dimensions[1:]):
            modules += [self.__make_lin(in_size, out_size), nn.ReLU()]

        modules.append(self.__make_lin(dimensions[-1], hidden_dimensions[-1]))
        self.nn = nn.Sequential(*modules)
        self.output = nn.Sequential(self.__make_lin(hidden_dimensions[-1], prediction_length))

    @staticmethod
    def __make_lin(dim_in: int, dim_out: int) -> nn.Linear:
        lin = nn.Linear(dim_in, dim_out)
        torch.nn.init.uniform_(lin.weight, -0.07, 0.07)
        torch.nn.init.zeros_(lin.bias)
        return lin

    def _extract_input_from_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["past_target"].type(torch.float32).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.size()) > 2:
            x = x.squeeze(dim=-1)

        nn_out = self.nn(x)
        return self.output(nn_out)

    def calculate_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # x: [batch_size, context_length]
        # y: [batch_size, prediction_length]
        x = self._extract_input_from_batch(batch)
        y = batch["future_target"].type(torch.float32).to(self.device)

        x = self.scaler.fit_transform(x)
        y = self.scaler.transform(y)

        outputs = self(x)
        loss = self.criterion(outputs, y)
        return loss

    def predict(self, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        self.eval()

        x = self._extract_input_from_batch(batch)
        if len(x.shape) > 2:
            x = x.squeeze(dim=-1)

        x = self.scaler.fit_transform(x)
        with torch.no_grad():
            forecast = self(x)
            forecast = self.scaler.inverse_transform(forecast)
            forecast = forecast.cpu().numpy()

        mean = np.expand_dims(forecast, axis=-1)
        lower = np.full(mean.shape, np.nan)
        upper = np.full(mean.shape, np.nan)

        return np.concatenate([mean, lower, upper], axis=-1)
