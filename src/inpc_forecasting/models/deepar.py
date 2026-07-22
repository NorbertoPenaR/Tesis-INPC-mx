from __future__ import annotations

import torch
from torch import nn

from .base import BaseForecastModel, TrainingConfig


class DeepARForecast(BaseForecastModel):
    """DeepAR univariado con verosimilitud gaussiana y covariables futuras."""

    def __init__(
        self,
        input_size: int,
        exog_size: int,
        horizon: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        training: TrainingConfig = TrainingConfig(),
    ) -> None:
        super().__init__(horizon, training)
        self.exog_size = exog_size
        self.encoder = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.LSTMCell(1 + exog_size, hidden_size)
        self.mean_head = nn.Linear(hidden_size, 1)
        self.scale_head = nn.Linear(hidden_size, 1)

    def forward(self, history, future_exog, teacher=None):
        _, (hidden, cell) = self.encoder(history)
        h, c = hidden[-1], cell[-1]
        previous = history[:, -1, 0:1]
        means, scales = [], []
        for step in range(self.horizon):
            decoder_input = torch.cat([previous, future_exog[:, step, :]], dim=-1)
            h, c = self.decoder(decoder_input, (h, c))
            mean = self.mean_head(h)
            scale = torch.nn.functional.softplus(self.scale_head(h)) + 1e-4
            means.append(mean)
            scales.append(scale)
            previous = teacher[:, step : step + 1] if teacher is not None else mean
        return torch.cat(means, dim=1), torch.cat(scales, dim=1)
