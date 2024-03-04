from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn

from .losses import mase_loss
from ..base import BaseModel
from ...utils.evaluation import mean_mase, mean_smape, mean_mse


class NBeats(BaseModel):
    """
    N-Beats Model.

    Source: https://github.com/ElementAI/N-BEATS
    """
    def __init__(self, blocks: nn.ModuleList, sp: int, path: str, device: torch.device) -> None:
        super().__init__()
        self.blocks = blocks
        self.sp = sp
        self.path = path
        self.device = device
        self.loss_fn = mase_loss
        self.to(device)

    def _extract_input_from_batch(self,
                                  batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = batch["past_target"].type(torch.float32).to(self.device)
        mask = batch["past_observed_values"].type(torch.float32).to(self.device)

        return x, mask

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        residuals = x.flip(dims=(1,))
        input_mask = input_mask.flip(dims=(1,))
        forecast = x[:, -1:]
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * input_mask
            forecast = forecast + block_forecast

        return forecast

    def calculate_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x, mask = self._extract_input_from_batch(batch)
        y = batch["future_target"].type(torch.float32).to(self.device)
        outputs = self(x, mask)
        return self.loss_fn(insample=x, freq=self.sp, forecast=outputs, target=y)

    def validate(self, batch: Dict[str, torch.Tensor], sp: int = 1) -> Tuple[float, float, float]:
        x, mask = self._extract_input_from_batch(batch)
        y = batch["future_target"].numpy()

        with torch.no_grad():
            output = self(x, mask).cpu().numpy()
            x = x.cpu().numpy()
            return mean_mase(y, output, x, sp=sp), mean_smape(y, output), mean_mse(output, y)

    def predict(self, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        self.eval()

        x, mask = self._extract_input_from_batch(batch)
        if len(x.shape) > 2:
            x = x.squeeze(dim=-1)

        with torch.no_grad():
            forecast = self(x, mask).cpu().numpy()

        mean = np.expand_dims(forecast, axis=-1)
        lower = np.full(mean.shape, np.nan)
        upper = np.full(mean.shape, np.nan)

        return np.concatenate([mean, lower, upper], axis=-1)


class NBeatsBlock(nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """
    def __init__(self,
                 context_length: int,
                 theta_size: int,
                 basis_function: nn.Module,
                 layers: int,
                 layer_size: int) -> None:
        """
        N-BEATS block.
        :param context_length: Insample size.
        :param theta_size:  Number of parameters for the basis function.
        :param basis_function: Basis function which takes the parameters and produces backcast and forecast.
        :param layers: Number of layers.
        :param layer_size: Layer size.
        """
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_features=context_length, out_features=layer_size)] +
                                    [nn.Linear(in_features=layer_size, out_features=layer_size)
                                     for _ in range(layers - 1)])
        self.basis_parameters = nn.Linear(in_features=layer_size, out_features=theta_size)
        self.basis_function = basis_function

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        block_input = x
        for layer in self.layers:
            block_input = torch.relu(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)
        return self.basis_function(basis_parameters)


class GenericBasis(nn.Module):
    """
    Generic basis function.
    """
    def __init__(self, context_length: int, prediction_length: int) -> None:
        super().__init__()
        self.backcast_size = context_length
        self.forecast_size = prediction_length

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]


class TrendBasis(nn.Module):
    """
    Polynomial function to model trend.
    """
    def __init__(self, degree_of_polynomial: int, context_length: int, prediction_length: int) -> None:
        super().__init__()
        self.polynomial_size = degree_of_polynomial + 1  # degree of polynomial with constant term
        self.backcast_time = nn.Parameter(
            torch.tensor(np.concatenate([np.power(np.arange(context_length, dtype=np.float) / context_length, i)[None, :]
                                         for i in range(self.polynomial_size)]), dtype=torch.float32),
            requires_grad=False)
        self.forecast_time = nn.Parameter(
            torch.tensor(
                np.concatenate([np.power(np.arange(prediction_length, dtype=np.float) / prediction_length, i)[None, :]
                                for i in range(self.polynomial_size)]), dtype=torch.float32), requires_grad=False)

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        backcast = torch.einsum('bp,pt->bt', theta[:, self.polynomial_size:], self.backcast_time)
        forecast = torch.einsum('bp,pt->bt', theta[:, :self.polynomial_size], self.forecast_time)
        return backcast, forecast


class SeasonalityBasis(nn.Module):
    """
    Harmonic functions to model seasonality.
    """
    def __init__(self, harmonics: int, context_length: int, prediction_length: int) -> None:
        super().__init__()
        self.frequency = np.append(np.zeros(1, dtype=np.float32),
                                   np.arange(harmonics, harmonics / 2 * prediction_length,
                                             dtype=np.float32) / harmonics)[None, :]
        backcast_grid = -2 * np.pi * (
                np.arange(context_length, dtype=np.float32)[:, None] / prediction_length) * self.frequency
        forecast_grid = 2 * np.pi * (
                np.arange(prediction_length, dtype=np.float32)[:, None] / prediction_length) * self.frequency
        self.backcast_cos_template = nn.Parameter(torch.tensor(np.transpose(np.cos(backcast_grid)),
                                                               dtype=torch.float32), requires_grad=False)
        self.backcast_sin_template = nn.Parameter(torch.tensor(np.transpose(np.sin(backcast_grid)),
                                                               dtype=torch.float32), requires_grad=False)
        self.forecast_cos_template = nn.Parameter(torch.tensor(np.transpose(np.cos(forecast_grid)),
                                                               dtype=torch.float32), requires_grad=False)
        self.forecast_sin_template = nn.Parameter(torch.tensor(np.transpose(np.sin(forecast_grid)),
                                                               dtype=torch.float32), requires_grad=False)

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        params_per_harmonic = theta.shape[1] // 4

        backcast_harmonics_cos = torch.einsum('bp,pt->bt', theta[:, 2 * params_per_harmonic:3 * params_per_harmonic],
                                              self.backcast_cos_template)
        backcast_harmonics_sin = torch.einsum('bp,pt->bt', theta[:, 3 * params_per_harmonic:],
                                              self.backcast_sin_template)
        backcast = backcast_harmonics_sin + backcast_harmonics_cos

        forecast_harmonics_cos = torch.einsum('bp,pt->bt', theta[:, :params_per_harmonic], self.forecast_cos_template)
        forecast_harmonics_sin = torch.einsum('bp,pt->bt', theta[:, params_per_harmonic:2 * params_per_harmonic],
                                              self.forecast_sin_template)
        forecast = forecast_harmonics_sin + forecast_harmonics_cos

        return backcast, forecast


def create_interpretable_nbeats(context_length: int,
                                prediction_length: int,
                                trend_blocks: int,
                                trend_layers: int,
                                trend_layer_size: int,
                                degree_of_polynomial: int,
                                seasonality_blocks: int,
                                seasonality_layers: int,
                                seasonality_layer_size: int,
                                num_of_harmonics: int,
                                sp: int,
                                path: str,
                                device: torch.device) -> NBeats:
    """
    Create N-BEATS interpretable model.
    """
    trend_block = NBeatsBlock(context_length=context_length,
                              theta_size=2 * (degree_of_polynomial + 1),
                              basis_function=TrendBasis(degree_of_polynomial=degree_of_polynomial,
                                                        context_length=context_length,
                                                        prediction_length=prediction_length),
                              layers=trend_layers,
                              layer_size=trend_layer_size)

    seasonality_block = NBeatsBlock(context_length=context_length,
                                    theta_size=4 * int(
                                        np.ceil(num_of_harmonics / 2 * prediction_length) - (num_of_harmonics - 1)),
                                    basis_function=SeasonalityBasis(harmonics=num_of_harmonics,
                                                                    context_length=context_length,
                                                                    prediction_length=prediction_length),
                                    layers=seasonality_layers,
                                    layer_size=seasonality_layer_size)

    return NBeats(nn.ModuleList([trend_block for _ in range(trend_blocks)] +
                                [seasonality_block for _ in range(seasonality_blocks)]),
                  sp=sp, path=path, device=device)


def create_generic_nbeats(context_length: int, prediction_length: int, stacks: int, layers: int,
                          layer_size: int, sp: int, path: str, device: torch.device) -> NBeats:
    """
    Create N-BEATS generic model.
    """
    return NBeats(nn.ModuleList([NBeatsBlock(context_length=context_length,
                                             theta_size=context_length + prediction_length,
                                             basis_function=GenericBasis(context_length=context_length,
                                                                         prediction_length=prediction_length),
                                             layers=layers,
                                             layer_size=layer_size)
                                 for _ in range(stacks)]),
                  sp=sp, path=path, device=device)
