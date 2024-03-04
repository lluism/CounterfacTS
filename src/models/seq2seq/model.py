from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn

from ..base import BaseModel
from ..scalers import get_scaler
from ...utils.evaluation import mean_mase, mean_mse, mean_smape


class EncoderRNN(torch.nn.Module):
    def __init__(self,
                 device: torch.device,
                 input_size: int = 1,
                 hidden_size: int = 128,
                 nlayers: int = 1,
                 bidirectional: bool = False) -> None:
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.rnn_directions = 2 if bidirectional else 1
        self.layers_directions = self.nlayers * self.rnn_directions

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=nlayers,
                           bidirectional=bidirectional, batch_first=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: [batch_size, context_length, input_size]
        init_hidden = (torch.zeros(self.nlayers * self.rnn_directions, x.size(0), self.hidden_size, device=self.device),
                       torch.zeros(self.nlayers * self.rnn_directions, x.size(0), self.hidden_size, device=self.device))
        output, hidden = self.rnn(x, init_hidden)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self,
                 prediction_length: int,
                 device: torch.device,
                 input_size: int = 1,
                 hidden_size: int = 128,
                 nlayers: int = 1) -> None:
        super().__init__()
        self.prediction_length = prediction_length
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nlayers = nlayers

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=nlayers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, z: torch.Tensor, h: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # x: [batch_size, context_length, input_size]
        # h: two tuples with dims [nlayers * bidirectional, batch_size, rnn_units]
        # output: [batch_size, prediction_length, input_size]
        output = x[:, -1, :].unsqueeze(1)
        outputs = torch.zeros([x.size(0), self.prediction_length, 1], device=self.device)
        for i in range(self.prediction_length):
            decoder_input = torch.cat([output, z[:, -self.prediction_length + i, :].unsqueeze(dim=1)], dim=-1)
            decoder_output, h = self.rnn(decoder_input, h)
            output = self.output_layer(decoder_output)
            outputs[:, i, :] = output.squeeze(dim=1)

        return outputs


class Seq2Seq(BaseModel):
    def __init__(self,
                 prediction_length: int,
                 device: torch.device,
                 cardinality: int,
                 input_size: int = 12,
                 hidden_size: int = 128,
                 nlayers: int = 1,
                 bidirectional: bool = False,
                 scaler: str = "mean_abs",
                 path: str = None) -> None:
        super().__init__()
        self.prediction_length = prediction_length
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cardinality = cardinality

        self.scaler = get_scaler(scaler)(self.device)
        self.criterion = nn.MSELoss()
        self.path = path

        self.cat_embedding = nn.Embedding(cardinality, 5)
        self.encoder = EncoderRNN(device, input_size, hidden_size, nlayers, bidirectional)
        self.decoder = DecoderRNN(prediction_length, device, input_size, hidden_size, nlayers)

    def _extract_input_from_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = batch["past_target"].type(torch.float32).to(self.device).unsqueeze(dim=-1)

        past_time_feat = batch["past_time_feat"].to(self.device)
        future_time_feat = batch["future_time_feat"].to(self.device)
        time_feat = torch.cat([past_time_feat, future_time_feat], dim=1)

        past_observed = batch["past_observed_values"].to(self.device)
        future_observed = batch["future_observed_values"].to(self.device)
        observed = torch.cat([past_observed, future_observed], dim=-1).unsqueeze(dim=-1)

        static_cat = batch["feat_static_cat"].to(self.device)
        embedded_cat = self.cat_embedding(static_cat.type(torch.long))
        repeated_cat = embedded_cat.repeat_interleave(x.size(1) + self.prediction_length, dim=1).float()

        features = torch.cat([repeated_cat, time_feat, observed], dim=-1)

        return x, features

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, context_length + prediction_length, input_size]
        # output: [batch_size, prediction_length, 1]
        encoder_in = torch.cat([x, z[:, :-self.prediction_length, :]], dim=-1)
        _, encoder_hidden = self.encoder(encoder_in)
        output = self.decoder(x, z, encoder_hidden)
        return output

    def calculate_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # x: [batch_size, context_length, input_size]
        # y: [batch_size, prediction_length, input_size]
        x, features = self._extract_input_from_batch(batch)
        y = batch["future_target"].type(torch.float32).to(self.device).unsqueeze(dim=-1)

        x = self.scaler.fit_transform(x)
        y = self.scaler.transform(y)

        outputs = self(x, features)
        loss = self.criterion(outputs, y)
        return loss

    def validate(self, batch: Dict[str, torch.Tensor], sp: int = 1) -> Tuple[float, float, float]:
        x, features = self._extract_input_from_batch(batch)
        y = batch["future_target"].to(self.device)
        scaled_x = self.scaler.fit_transform(x)

        with torch.no_grad():
            output = self(scaled_x, features)
            output = self.scaler.inverse_transform(output).squeeze(dim=-1)

            x = x.cpu().numpy()
            y = y.cpu().numpy()
            output = output.cpu().numpy()
            return mean_mase(y, output, x, sp=sp), mean_smape(y, output), mean_mse(output, y)

    def predict(self, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        self.eval()
        x, features = self._extract_input_from_batch(batch)
        x = self.scaler.fit_transform(x)

        with torch.no_grad():
            forecast = self(x, features)
            forecast = self.scaler.inverse_transform(forecast)
            forecast = forecast.cpu().numpy()

        mean = forecast
        lower = np.full(mean.shape, np.nan)
        upper = np.full(mean.shape, np.nan)

        return np.concatenate([mean, lower, upper], axis=-1)
