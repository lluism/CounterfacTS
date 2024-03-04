from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from ..scalers import get_scaler
from ...utils.evaluation import mean_mase, mean_mse, mean_smape


class TCN(BaseModel):
    """
    A TCN model based on Conditional Time Series Forecasting with Convolutional Neural Networks.

    Adapted from https://github.com/albertogaspar/dts/blob/9fcad2c672cdcf5d2c6bd005dae05afc65f97e58/dts/models/TCN.py
    """
    def __init__(self, context_length: int, prediction_length: int, input_size: int, conditional_size: int,
                 num_channels: int, k: int, cardinality: int, path: str, device: torch.device,
                 scaler: str = "mean_abs") -> None:
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.input_size = input_size
        self.conditional_size = conditional_size
        self.num_channels = num_channels
        self.k = k
        self.cardinality = cardinality
        self.path = path
        self.device = device
        self.scaler = get_scaler(scaler)(self.device)
        self.criterion = nn.MSELoss()

        self.cat_embedding = nn.Embedding(cardinality, 5)
        self.blocks = nn.ModuleList()
        i = 0
        dilation_size = 1
        while dilation_size < context_length:
            if i == 0:
                in_channels = input_size
                self.blocks.append(
                    ConditionalBlock(dilation_size, input_in_channels=in_channels,
                                     conditional_in_channels=conditional_size, out_channels=num_channels, k=k)
                )
            else:
                in_channels = num_channels
                self.blocks.append(
                    ResidualBlock(dilation_size, in_channels=in_channels, out_channels=num_channels, k=k)
                )

            i += 1
            dilation_size = 2 ** i

        self.output_conv = nn.Conv1d(in_channels=num_channels, out_channels=1, kernel_size=1)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        def forward_pass(x, z):
            block_output = self.blocks[0](x, z)

            for block in self.blocks[1:]:
                block_output = block(block_output)

            return self.output_conv(block_output)

        if self.training:
            return forward_pass(x, z)

        for i in range(self.prediction_length):
            conditional = z[:, :, i:-self.prediction_length + i]
            prediction = forward_pass(x, conditional)[:, :, -1:]
            x = torch.cat([x, prediction], dim=-1)[:, :, 1:]

        return x[:, :, -self.prediction_length:].permute(0, 2, 1)

    def _extract_input_from_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = batch["past_target"].unsqueeze(dim=-1).to(self.device)

        past_time_feat = batch["past_time_feat"].to(self.device)
        future_time_feat = batch["future_time_feat"].to(self.device)
        time_feat = torch.cat([past_time_feat, future_time_feat], dim=1)

        past_observed = batch["past_observed_values"].to(self.device)
        future_observed = batch["future_observed_values"].to(self.device)
        observed = torch.cat([past_observed, future_observed], dim=-1).unsqueeze(dim=-1)

        static_cat = batch["feat_static_cat"].to(self.device)
        embedded_cat = self.cat_embedding(static_cat.type(torch.long))
        repeated_cat = embedded_cat.repeat_interleave(self.context_length + self.prediction_length, dim=1).float()

        features = torch.cat([repeated_cat, time_feat, observed], dim=-1)
        return x, features

    def calculate_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x, features = self._extract_input_from_batch(batch)
        y = batch["future_target"].unsqueeze(dim=-1).to(self.device)

        x = self.scaler.fit_transform(x, dim=1)
        y = self.scaler.transform(y)

        # reshape to [batch size, num channels, context length]
        x = torch.cat([x[:, :, :], y[:, :-1, :]], dim=1)
        x = x.permute(0, 2, 1)
        features = features[:, :-1, :].permute(0, 2, 1)

        output = self(x, features).permute(0, 2, 1)[:, -self.prediction_length:, :]
        loss = self.criterion(output, y)
        return loss

    def validate(self, batch: Dict[str, torch.Tensor], sp: int = 1) -> Tuple[float, float, float]:
        x, features = self._extract_input_from_batch(batch)
        y = batch["future_target"].numpy()
        scaled_x = self.scaler.fit_transform(x)

        with torch.no_grad():
            output = self(scaled_x.permute(0, 2, 1), features.permute(0, 2, 1))
            output = self.scaler.inverse_transform(output).squeeze(dim=-1)

            x = x.cpu().numpy()
            output = output.cpu().numpy()
            return mean_mase(y, output, x, sp=sp), mean_smape(y, output), mean_mse(output, y)

    def predict(self, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        self.eval()
        x, features = self._extract_input_from_batch(batch)
        scaled_x = self.scaler.fit_transform(x)

        with torch.no_grad():
            forecast = self(scaled_x.permute(0, 2, 1), features.permute(0, 2, 1))
            forecast = self.scaler.inverse_transform(forecast)
            forecast = forecast.cpu().numpy()

        mean = forecast
        lower = np.full(mean.shape, np.nan)
        upper = np.full(mean.shape, np.nan)

        return np.concatenate([mean, lower, upper], axis=-1)


class ConditionalBlock(nn.Module):
    """
    A conditional block.

    The input and condition is passed through a residual block before summing them together. This block is used as the
    first layer in the TCN.
    """
    def __init__(self, d: int, input_in_channels: int, conditional_in_channels: int, out_channels: int, k: int) -> None:
        super().__init__()
        self.d = d
        self.input_in_channels = input_in_channels
        self.conditional_in_channels = conditional_in_channels
        self.out_channels = out_channels
        self.k = k

        self.input_block = ResidualBlock(d, input_in_channels, out_channels, k)
        self.conditional_block = ResidualBlock(d, conditional_in_channels, out_channels, k)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        input_out = self.input_block(x)
        conditional_out = self.conditional_block(z)
        return input_out + conditional_out


class ResidualBlock(nn.Module):
    """
    A residual block.

    The input is convolved using a 1d causal convolution. The result is summed with a parameterized residual connection.
    """
    def __init__(self, d: int, in_channels: int, out_channels: int, k: int) -> None:
        super().__init__()
        self.d = d
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k

        self.conv_block = nn.Sequential(
            CausalConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=k, dilation=d),
            nn.ReLU()
        )
        self.skip = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x) + self.skip(x)


class CausalConv1d(torch.nn.Conv1d):
    """
    A causal convolution layer.

    This is simply a wrapper that pads the left side of the input before passing it on to nn.Conv1d

    Source: https://github.com/pytorch/pytorch/issues/1333#issuecomment-453702879
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.__padding, 0))
        return super().forward(x)
