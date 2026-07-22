from __future__ import annotations

import math

import torch
from torch import nn

from .base import BaseForecastModel, TrainingConfig


class SinusoidalPosition(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * div)
        encoding[:, 1::2] = torch.cos(position * div)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.size(1)]


class TransformerForecast(BaseForecastModel):
    def __init__(
        self,
        input_size: int,
        exog_size: int,
        horizon: int,
        d_model: int = 32,
        num_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        training: TrainingConfig = TrainingConfig(),
    ) -> None:
        super().__init__(horizon, training)
        if d_model % nhead:
            raise ValueError("d_model debe ser divisible entre nhead.")
        self.projection = nn.Linear(input_size, d_model)
        self.position = SinusoidalPosition(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model + exog_size, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def causal_mask(length: int, device: torch.device | None = None) -> torch.Tensor:
        return torch.triu(torch.full((length, length), float("-inf"), device=device), diagonal=1)

    def forward(self, history, future_exog, teacher=None):
        embedded = self.position(self.projection(history))
        mask = self.causal_mask(embedded.size(1), embedded.device)
        encoded = self.encoder(embedded, mask=mask)
        context = encoded[:, -1, :].unsqueeze(1).expand(-1, self.horizon, -1)
        return self.head(torch.cat([context, future_exog], dim=-1)).squeeze(-1)
