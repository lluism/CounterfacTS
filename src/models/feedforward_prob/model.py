from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn

from ..base import BaseModel
from ..scalers import get_scaler
from ...utils.evaluation import mean_mase, mean_smape, mean_mse


class FeedForwardProb(BaseModel):
    """Adapted from https://github.com/awslabs/gluon-ts/blob/master/examples/pytorch_predictor_example.ipynb
    """

    distr_type = torch.distributions.StudentT

    def __init__(self,
                 prediction_length: int,
                 context_length: int,
                 hidden_dimensions: List[int],
                 device: torch.device,
                 path: str,
                 num_samples: int = 100,
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
        self.num_samples = num_samples
        self.scaler = get_scaler(scaler)(self.device)

        dimensions = [context_length] + hidden_dimensions[:-1]

        modules = []
        for in_size, out_size in zip(dimensions[:-1], dimensions[1:]):
            modules += [self.__make_lin(in_size, out_size), nn.ReLU()]

        modules.append(self.__make_lin(dimensions[-1], prediction_length * hidden_dimensions[-1]))
        self.nn = nn.Sequential(*modules)

        self.df_proj = nn.Sequential(self.__make_lin(hidden_dimensions[-1], 1), nn.Softplus())
        self.loc_proj = self.__make_lin(hidden_dimensions[-1], 1)
        self.scale_proj = nn.Sequential(self.__make_lin(hidden_dimensions[-1], 1), nn.Softplus())

    @staticmethod
    def __make_lin(dim_in: int, dim_out: int) -> nn.Linear:
        lin = nn.Linear(dim_in, dim_out)
        torch.nn.init.uniform_(lin.weight, -0.07, 0.07)
        torch.nn.init.zeros_(lin.bias)
        return lin

    def _extract_input_from_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["past_target"].type(torch.float32).to(self.device).unsqueeze(dim=-1)

        return x

    def forward(self, x: torch.Tensor) -> torch.distributions:
        nn_out = self.nn(x)
        nn_out_reshaped = nn_out.reshape(-1, self.prediction_length, self.hidden_dimensions[-1])

        distr_args = (
            2.0 + self.df_proj(nn_out_reshaped).squeeze(dim=-1),
            self.loc_proj(nn_out_reshaped).squeeze(dim=-1),
            self.scale_proj(nn_out_reshaped).squeeze(dim=-1),
        )
        distr = self.distr_type(*distr_args)

        return distr

    def calculate_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self._extract_input_from_batch(batch)
        y = batch["future_target"].type(torch.float32).to(self.device).unsqueeze(dim=-1)

        scaled_x = self.scaler.fit_transform(x)
        distr = self(scaled_x)
        scaled_y = self.scaler.transform(y)

        loss = -distr.log_prob(scaled_y)
        return loss.mean()

    def predict(self, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        # x: [batch_size, context_length, input_size]
        # output: [batch_size, prediction_length, 3]
        self.eval()
        x = self._extract_input_from_batch(batch)
        x = self.scaler.fit_transform(x).squeeze(dim=-1)

        with torch.no_grad():
            distribution = self(x)
            output = distribution.sample([self.num_samples]).permute(1, 0, 2)
            output = self.scaler.inverse_transform(output).cpu().numpy()

        # shape [batch_size, prediction_length]
        mean = np.mean(output, axis=1)
        lower = np.quantile(output, 0.05, axis=1)
        upper = np.quantile(output, 0.95, axis=1)

        return np.concatenate([mean[..., np.newaxis], lower[..., np.newaxis], upper[..., np.newaxis]], axis=-1)

    def validate(self, batch: Dict[str, torch.Tensor], sp: int = 1) -> Tuple[float, float, float]:
        x = self._extract_input_from_batch(batch)
        y = batch["future_target"].numpy()
        scaled_x = self.scaler.fit_transform(x).squeeze(dim=-1)

        with torch.no_grad():
            distribution = self(scaled_x)
            output = distribution.sample([self.num_samples]).permute(1, 0, 2)
            output = self.scaler.inverse_transform(output)
            output = output.mean(dim=1)

            x = x.cpu().numpy()
            output = output.cpu().numpy()
            return mean_mase(y, output, x, sp=sp), mean_smape(y, output), mean_mse(output, y)
