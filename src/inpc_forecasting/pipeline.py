from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from hashlib import sha256
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .decomposition import HPDecomposer
from .factory import create_model
from .features import component_features
from .transforms import Standardizer, TrendTransform


RESULT_COLUMNS = [
    "ds", "unique_id", "cutoff", "horizon", "step", "model", "transformation", "yhat_trend",
    "yhat_cycle", "yhat", "y_true", "mae", "rmse", "mape", "execution_time", "config_hash",
]


def _future_dates(last: pd.Timestamp, periods: int, frequency: str) -> pd.DatetimeIndex:
    return pd.date_range(start=last, periods=periods + 1, freq=frequency)[1:]


def _windows(target: np.ndarray, exog: np.ndarray, context: int, horizon: int):
    histories, futures, targets = [], [], []
    merged = np.column_stack([target, exog])
    for end in range(context, len(target) - horizon + 1):
        histories.append(merged[end - context : end])
        futures.append(exog[end : end + horizon])
        targets.append(target[end : end + horizon])
    if not histories:
        raise ValueError("No hay observaciones suficientes para contexto y horizonte.")
    return np.asarray(histories), np.asarray(futures), np.asarray(targets)


@dataclass
class _PreparedComponent:
    transformed: np.ndarray
    historical_exog: np.ndarray
    future_exog: np.ndarray
    scaler: Standardizer
    trend_transform: TrendTransform | None


class ComponentForecastPipeline:
    def __init__(self, config: dict) -> None:
        self.config = config

    def _prepare(
        self,
        values: np.ndarray,
        dates: pd.DatetimeIndex,
        future_dates: pd.DatetimeIndex,
        is_trend: bool,
    ) -> _PreparedComponent:
        component_cfg = self.config["components"]["trend" if is_trend else "cycle"]
        transform = TrendTransform(component_cfg.get("transform", "diff")) if is_trend else None
        transformed = transform.fit_transform(values) if transform else np.asarray(values, dtype=float)
        lost = len(values) - len(transformed)
        transformed_dates = dates[lost:]
        historical_exog, future_exog = component_features(
            transformed,
            transformed_dates,
            future_dates,
            bool(component_cfg.get("calendar", True)),
            int(component_cfg.get("fft_top_k", 0)),
        )
        scaler = Standardizer()
        scaled = scaler.fit_transform(transformed)
        return _PreparedComponent(scaled, historical_exog, future_exog, scaler, transform)

    def _forecast_component(
        self, name: str, prepared: _PreparedComponent, horizon: int, model_cfg: dict, component: str
    ):
        model_cfg = deepcopy(model_cfg)
        override = model_cfg.pop("component_overrides", {}).get(component, {})
        for section in ("architecture", "training"):
            if section in override:
                model_cfg.setdefault(section, {}).update(override[section])
        if "context_length" in override:
            model_cfg["context_length"] = override["context_length"]
        context = int(model_cfg["context_length"])
        histories, futures, targets = _windows(
            prepared.transformed, prepared.historical_exog, context=context, horizon=horizon
        )
        model = create_model(name, histories.shape[-1], futures.shape[-1], horizon, model_cfg)
        model.fit(histories, futures, targets)
        latest_history = np.column_stack([prepared.transformed, prepared.historical_exog])[-context:]
        prediction = model.predict(latest_history, prepared.future_exog)
        prediction = prepared.scaler.inverse_transform(prediction)
        if prepared.trend_transform:
            prediction = prepared.trend_transform.inverse_forecast(prediction)
        return prediction

    def fit_predict(self, data: pd.DataFrame, cutoff: str | pd.Timestamp | None = None) -> pd.DataFrame:
        started = time.perf_counter()
        required = {"ds", "unique_id", "y"}
        if not required.issubset(data.columns):
            raise ValueError(f"El DataFrame debe contener {sorted(required)}")
        frame = data.copy()
        frame["ds"] = pd.to_datetime(frame["ds"])
        unique_id = self.config["data"].get("unique_id", "Inflacion")
        frame = frame[frame["unique_id"] == unique_id].sort_values("ds")
        cutoff_ts = pd.Timestamp(cutoff or self.config["data"]["cutoff"])
        train = frame[frame["ds"] <= cutoff_ts]
        if train.empty:
            raise ValueError("No existen observaciones anteriores al corte.")
        horizon = int(self.config["experiment"]["horizon"])
        frequency = self.config["data"]["frequency"]
        dates = pd.DatetimeIndex(train["ds"])
        forecast_dates = _future_dates(dates[-1], horizon, frequency)
        decomposer = HPDecomposer(frequency, self.config["hp"].get("lambda"))
        trend, cycle = decomposer.fit_transform(train["y"].to_numpy())
        trend_data = self._prepare(trend, dates, forecast_dates, True)
        cycle_data = self._prepare(cycle, dates, forecast_dates, False)
        model_name = self.config["experiment"]["model"]
        model_cfg = self.config["models"][model_name.lower()]
        trend_hat = self._forecast_component(model_name, trend_data, horizon, model_cfg, "trend")
        cycle_hat = self._forecast_component(model_name, cycle_data, horizon, model_cfg, "cycle")
        yhat = trend_hat + cycle_hat
        actual = frame.set_index("ds")["y"].reindex(forecast_dates).to_numpy(dtype=float)
        valid = np.isfinite(actual)
        mae = float(mean_absolute_error(actual[valid], yhat[valid])) if valid.any() else np.nan
        rmse = float(mean_squared_error(actual[valid], yhat[valid]) ** 0.5) if valid.any() else np.nan
        nonzero = valid & ~np.isclose(actual, 0.0)
        mape = float(np.mean(np.abs((actual[nonzero] - yhat[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else np.nan
        config_json = json.dumps(self.config, sort_keys=True, ensure_ascii=True)
        result = pd.DataFrame(
            {
                "ds": forecast_dates,
                "unique_id": unique_id,
                "cutoff": cutoff_ts,
                "horizon": horizon,
                "step": np.arange(1, horizon + 1),
                "model": model_name,
                "transformation": self.config["components"]["trend"]["transform"],
                "yhat_trend": trend_hat,
                "yhat_cycle": cycle_hat,
                "yhat": yhat,
                "y_true": actual,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "execution_time": time.perf_counter() - started,
                "config_hash": sha256(config_json.encode()).hexdigest()[:12],
            }
        )
        return result[RESULT_COLUMNS]
