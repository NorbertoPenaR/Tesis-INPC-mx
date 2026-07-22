from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter


_OBSERVATIONS_PER_YEAR = {
    "A": 1.0,
    "Y": 1.0,
    "Q": 4.0,
    "M": 12.0,
    "ME": 12.0,
    "MS": 12.0,
    "W": 52.0,
    "W-MON": 52.0,
    "D": 365.25,
    "B": 260.0,
}


def hp_lambda_for_frequency(frequency: str) -> float:
    """Ravn-Uhlig scaling relative to lambda=1600 for quarterly data."""
    normalized = frequency.upper()
    if normalized.startswith("W-"):
        normalized = "W-MON"
    if normalized not in _OBSERVATIONS_PER_YEAR:
        raise ValueError(f"Frecuencia HP no soportada: {frequency!r}")
    ratio = _OBSERVATIONS_PER_YEAR[normalized] / 4.0
    return float(1600.0 * ratio**4)


@dataclass(frozen=True)
class HPDecomposer:
    frequency: str = "W-MON"
    lamb: float | None = None

    @property
    def smoothing(self) -> float:
        return hp_lambda_for_frequency(self.frequency) if self.lamb is None else float(self.lamb)

    def fit_transform(self, train: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(train, dtype=np.float64).reshape(-1)
        if len(values) < 4:
            raise ValueError("El filtro HP requiere al menos cuatro observaciones.")
        if not np.isfinite(values).all():
            raise ValueError("La serie de entrenamiento contiene valores no finitos.")
        cycle, trend = hpfilter(values, lamb=self.smoothing)
        return np.asarray(trend, dtype=np.float64), np.asarray(cycle, dtype=np.float64)
