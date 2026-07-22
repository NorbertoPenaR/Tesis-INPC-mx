from __future__ import annotations

import torch
from torch import nn

from .base import BaseForecastModel, TrainingConfig


class _RecurrentForecast(BaseForecastModel):
    cell_type = "rnn"

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
        recurrent_cls = nn.RNN if self.cell_type == "rnn" else nn.LSTM
        kwargs = {"nonlinearity": "tanh"} if recurrent_cls is nn.RNN else {}
        self.encoder = recurrent_cls(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            **kwargs,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + exog_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, history, future_exog, teacher=None):
        encoded, _ = self.encoder(history)
        context = encoded[:, -1, :].unsqueeze(1).expand(-1, self.horizon, -1)
        joined = torch.cat([context, future_exog], dim=-1)
        return self.head(joined).squeeze(-1)


class RNNForecast(_RecurrentForecast):
    cell_type = "rnn"


class LSTMForecast(_RecurrentForecast):
    cell_type = "lstm"
