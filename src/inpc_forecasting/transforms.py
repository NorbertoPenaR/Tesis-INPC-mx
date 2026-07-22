from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


SUPPORTED_TRANSFORMS = ("diff", "diff_logp1", "pct", "logp1", "none", "diff2")


@dataclass
class TrendTransform:
    name: str = "diff"
    anchors: list[float] = field(default_factory=list, init=False)

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        y = np.asarray(values, dtype=np.float64).reshape(-1)
        if self.name not in SUPPORTED_TRANSFORMS:
            raise ValueError(f"Transformacion no soportada: {self.name}")
        if self.name == "none":
            self.anchors = [float(y[-1])]
            return y.copy()
        if self.name == "logp1":
            self._require_log_domain(y)
            self.anchors = [float(y[-1])]
            return np.log1p(y)
        if self.name == "diff":
            self.anchors = [float(y[-1])]
            return np.diff(y)
        if self.name == "diff_logp1":
            self._require_log_domain(y)
            self.anchors = [float(y[-1])]
            return np.diff(np.log1p(y))
        if self.name == "pct":
            if np.any(np.isclose(y[:-1], 0.0)):
                raise ValueError("pct no puede aplicarse cuando el denominador es cero.")
            self.anchors = [float(y[-1])]
            return np.diff(y) / y[:-1] * 100.0
        self.anchors = [float(y[-2]), float(y[-1])]
        return np.diff(y, n=2)

    def inverse_forecast(self, transformed: np.ndarray) -> np.ndarray:
        z = np.asarray(transformed, dtype=np.float64).reshape(-1)
        if not self.anchors:
            raise RuntimeError("La transformacion debe ajustarse antes de invertir.")
        if self.name == "none":
            return z
        if self.name == "logp1":
            return np.expm1(z)
        last = self.anchors[-1]
        if self.name == "diff":
            return last + np.cumsum(z)
        if self.name == "diff_logp1":
            return np.expm1(np.log1p(last) + np.cumsum(z))
        if self.name == "pct":
            out = []
            current = last
            for change in z:
                current = current * (1.0 + change / 100.0)
                out.append(current)
            return np.asarray(out)
        previous, current = self.anchors
        out = []
        for second_difference in z:
            nxt = second_difference + 2.0 * current - previous
            out.append(nxt)
            previous, current = current, nxt
        return np.asarray(out)

    @staticmethod
    def _require_log_domain(values: np.ndarray) -> None:
        if np.any(values <= -1.0):
            raise ValueError("log1p requiere valores mayores que -1.")


@dataclass
class Standardizer:
    mean_: float = field(default=0.0, init=False)
    scale_: float = field(default=1.0, init=False)

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        y = np.asarray(values, dtype=np.float64)
        self.mean_ = float(y.mean())
        std = float(y.std())
        self.scale_ = std if std > 1e-12 else 1.0
        return (y - self.mean_) / self.scale_

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean_) / self.scale_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.scale_ + self.mean_
