from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .utils import PositionalEncoder
from ..base import BaseModel
from ..scalers import get_scaler
from ...utils.evaluation import mean_mase, mean_mse, mean_smape


class Transformer(BaseModel):

    def __init__(self, context_length: int, prediction_length: int, cardinality: int, d_model: int, nheads: int,
                 num_encoder_layers: int, num_decoder_layers: int, dim_feedforward: int, dropout: float, path: str,
                 device: torch.device, scaler: str = "mean_abs") -> None:
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.cardinality = cardinality
        self.d_model = d_model
        self.nheads = nheads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.path = path
        self.device = device
        self.scaler = get_scaler(scaler)(self.device)
        self.criterion = nn.MSELoss()

        self.cat_embedding = nn.Embedding(cardinality, 5)
        self.positional_enc = PositionalEncoder(d_model, context_length, dropout)

        self.encoder_embedding = nn.Linear(12, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nheads, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

        self.decoder_embedding = nn.Linear(10, d_model)
        decoder_layers = nn.TransformerDecoderLayer(d_model, nheads, dim_feedforward, dropout)
        self.decoder = nn.TransformerDecoder(decoder_layers, num_decoder_layers)
        self.projection = nn.Linear(d_model, 1)

    def _extract_input_from_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = batch["past_target"].unsqueeze(dim=-1).to(self.device)
        past_observed = batch["past_observed_values"].type(torch.float32).to(self.device)

        # scale input and target
        x = self.scaler.fit_transform(x, mask=past_observed.unsqueeze(dim=-1))

        # repeat static cat features to get [batch_size, context_len + prediction_len, 1]
        static_cat = batch["feat_static_cat"].to(self.device)
        repeated_static_cat = static_cat.repeat_interleave(self.context_length, dim=-1).type(torch.long)
        embedded_static_cat = self.cat_embedding(repeated_static_cat)

        past_time_feat = batch["past_time_feat"].to(self.device)
        future_time_feat = batch["future_time_feat"].to(self.device)

        # concat all features and permute to [context_len, batch_size, num_features]
        encoder_x = torch.cat([x, embedded_static_cat, past_time_feat, past_observed.unsqueeze(dim=-1)], dim=-1)
        encoder_x = encoder_x.permute(1, 0, 2).type(torch.float32).to(self.device)

        decoder_x = torch.cat([embedded_static_cat[:, :self.prediction_length, :], future_time_feat], dim=-1)
        decoder_x = decoder_x.permute(1, 0, 2).type(torch.float32).to(self.device)

        return encoder_x, decoder_x

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        encoder_input = self.encoder_embedding(src)
        encoder_input = self.positional_enc(encoder_input)
        encoder_out = self.encoder(encoder_input)

        decoder_input = self.decoder_embedding(tgt)
        decoder_input = self.positional_enc(decoder_input)
        decoder_out = self.decoder(decoder_input, encoder_out)

        # project decoder output to [batch_size, prediction_length, 1]
        output = self.projection(decoder_out).permute(1, 0, 2)
        return output

    def calculate_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        encoder_x, decoder_x = self._extract_input_from_batch(batch)
        output = self(encoder_x, decoder_x)

        y = batch["future_target"].unsqueeze(dim=-1).to(self.device)
        y = self.scaler.transform(y)
        loss = self.criterion(output, y)
        return loss

    def predict(self, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        self.eval()
        encoder_x, decoder_x = self._extract_input_from_batch(batch)

        with torch.no_grad():
            forecast = self(encoder_x, decoder_x)
            forecast = self.scaler.inverse_transform(forecast)
            forecast = forecast.cpu().numpy()

        mean = forecast
        lower = np.full(mean.shape, np.nan)
        upper = np.full(mean.shape, np.nan)

        return np.concatenate([mean, lower, upper], axis=-1)

    def validate(self, batch: Dict[str, torch.Tensor], sp: Optional[int] = None) -> Tuple[float, float, float]:
        encoder_x, decoder_x = self._extract_input_from_batch(batch)
        x = batch["past_target"].numpy()
        y = batch["future_target"].numpy()

        with torch.no_grad():
            output = self(encoder_x, decoder_x)
            output = self.scaler.inverse_transform(output).squeeze(dim=-1)

            output = output.cpu().numpy()
            return mean_mase(y, output, x, sp=sp), mean_smape(y, output), mean_mse(output, y)
