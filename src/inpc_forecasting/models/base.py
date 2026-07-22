from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
import os
import random

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 80
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 12
    gradient_clip: float = 1.0
    validation_fraction: float = 0.2
    seed: int = 119
    device: str = "auto"


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=True)


class BaseForecastModel(nn.Module, ABC):
    def __init__(self, horizon: int, training: TrainingConfig) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.train_config = training
        self._input_size: int | None = None

    @property
    def device(self) -> torch.device:
        if self.train_config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.train_config.device)

    @abstractmethod
    def forward(
        self,
        history: torch.Tensor,
        future_exog: torch.Tensor,
        teacher: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _loss(self, output: torch.Tensor | tuple[torch.Tensor, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        if isinstance(output, tuple):
            mean, scale = output
            return (torch.log(scale) + 0.5 * ((target - mean) / scale) ** 2).mean()
        return torch.nn.functional.l1_loss(output, target)

    def fit(self, history: np.ndarray, future_exog: np.ndarray, targets: np.ndarray) -> "BaseForecastModel":
        set_deterministic_seed(self.train_config.seed)
        x = torch.as_tensor(history, dtype=torch.float32)
        f = torch.as_tensor(future_exog, dtype=torch.float32)
        y = torch.as_tensor(targets, dtype=torch.float32)
        if not (len(x) == len(f) == len(y)) or len(x) < 2:
            raise ValueError("Se requieren al menos dos ventanas alineadas para entrenar.")
        split = max(1, min(len(x) - 1, int(len(x) * (1.0 - self.train_config.validation_fraction))))
        train_ds = TensorDataset(x[:split], f[:split], y[:split])
        val_ds = TensorDataset(x[split:], f[split:], y[split:])
        generator = torch.Generator().manual_seed(self.train_config.seed)
        loader = DataLoader(train_ds, batch_size=self.train_config.batch_size, shuffle=True, generator=generator)
        self.to(self.device)
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.train_config.learning_rate, weight_decay=self.train_config.weight_decay
        )
        best_state: dict[str, torch.Tensor] | None = None
        best_loss = float("inf")
        stale = 0
        for _ in range(self.train_config.epochs):
            self.train()
            for bx, bf, by in loader:
                bx, bf, by = bx.to(self.device), bf.to(self.device), by.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = self._loss(self(bx, bf, teacher=by), by)
                if not torch.isfinite(loss):
                    raise FloatingPointError("La perdida de entrenamiento no es finita.")
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.train_config.gradient_clip)
                optimizer.step()
            self.eval()
            with torch.no_grad():
                vx, vf, vy = (tensor.to(self.device) for tensor in val_ds.tensors)
                val_loss = float(self._loss(self(vx, vf), vy).cpu())
            if val_loss < best_loss - 1e-7:
                best_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in self.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.train_config.patience:
                    break
        if best_state is not None:
            self.load_state_dict(best_state)
        self.to(self.device)
        return self

    def predict(self, history: np.ndarray, future_exog: np.ndarray) -> np.ndarray:
        self.eval()
        x = torch.as_tensor(history, dtype=torch.float32, device=self.device)
        f = torch.as_tensor(future_exog, dtype=torch.float32, device=self.device)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if f.ndim == 2:
            f = f.unsqueeze(0)
        with torch.no_grad():
            output = self(x, f)
            mean = output[0] if isinstance(output, tuple) else output
        return mean.squeeze(0).detach().cpu().numpy().astype(np.float64)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "training": asdict(self.train_config)}, path)

    def load(self, path: str | Path) -> "BaseForecastModel":
        payload = torch.load(Path(path), map_location=self.device, weights_only=True)
        self.load_state_dict(payload["state_dict"])
        self.to(self.device)
        return self
