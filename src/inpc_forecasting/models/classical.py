from __future__ import annotations

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def _recover_series(history: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Recover the unwindowed target used to build overlapping samples."""
    first = np.asarray(history[0, :, 0], dtype=float)
    leading = np.asarray(targets[:, 0], dtype=float)
    tail = np.asarray(targets[-1, 1:], dtype=float)
    return np.concatenate([first, leading, tail])


class HoltWintersForecast:
    """ETS baseline with the same fit/predict surface as the neural models."""

    def __init__(self, horizon: int, trend: str | None = None, damped_trend: bool = False,
                 seasonal: str | None = None, seasonal_periods: int | None = None,
                 initialization_method: str = "estimated", **_: object) -> None:
        self.horizon = int(horizon)
        self.options = {
            "trend": trend,
            "damped_trend": damped_trend if trend else False,
            "seasonal": seasonal,
            "seasonal_periods": seasonal_periods if seasonal else None,
            "initialization_method": initialization_method,
        }
        self.result = None

    def fit(self, history: np.ndarray, future_exog: np.ndarray, targets: np.ndarray):
        del future_exog
        self.result = ExponentialSmoothing(_recover_series(history, targets), **self.options).fit(
            optimized=True, remove_bias=False
        )
        return self

    def predict(self, history: np.ndarray, future_exog: np.ndarray) -> np.ndarray:
        del history, future_exog
        if self.result is None:
            raise RuntimeError("Holt-Winters debe ajustarse antes de pronosticar.")
        return np.asarray(self.result.forecast(self.horizon), dtype=float)


class XGBoostForecast:
    """Direct multi-horizon XGBoost baseline over the common window tensors."""

    def __init__(self, horizon: int, n_estimators: int = 300, max_depth: int = 4,
                 learning_rate: float = 0.03, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, reg_alpha: float = 0.0,
                 reg_lambda: float = 1.0, random_state: int = 119, lag_count: int = 52,
                 n_jobs: int = 1, **_: object) -> None:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Instala el extra 'classical' para usar XGBoost.") from exc
        self.horizon = int(horizon)
        self.lag_count = int(lag_count)
        self.history_: np.ndarray | None = None
        self.model = XGBRegressor(
            objective="reg:absoluteerror", tree_method="hist",
            n_estimators=int(n_estimators), max_depth=int(max_depth),
            learning_rate=float(learning_rate), subsample=float(subsample),
            colsample_bytree=float(colsample_bytree), reg_alpha=float(reg_alpha),
            reg_lambda=float(reg_lambda), random_state=int(random_state), n_jobs=int(n_jobs),
        )

    def fit(self, history: np.ndarray, future_exog: np.ndarray, targets: np.ndarray):
        series = _recover_series(history, targets)
        historical_exog = np.concatenate(
            [history[0, :, 1:], future_exog[:, 0, :], future_exog[-1, 1:, :]], axis=0
        )
        if self.lag_count >= len(series):
            raise ValueError("lag_count debe ser menor que la historia disponible.")
        x = np.asarray([
            np.r_[series[end - self.lag_count:end], historical_exog[end]]
            for end in range(self.lag_count, len(series))
        ])
        self.model.fit(x, series[self.lag_count:])
        self.history_ = series.copy()
        return self

    def predict(self, history: np.ndarray, future_exog: np.ndarray) -> np.ndarray:
        if self.history_ is None:
            raise RuntimeError("XGBoost debe ajustarse antes de pronosticar.")
        values = list(self.history_)
        for step in range(self.horizon):
            x = np.r_[values[-self.lag_count:], future_exog[step]][None, :]
            values.append(float(self.model.predict(x)[0]))
        return np.asarray(values[-self.horizon:], dtype=float)
