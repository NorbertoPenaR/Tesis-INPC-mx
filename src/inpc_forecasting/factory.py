from __future__ import annotations

from .models import (
    DeepARForecast, HoltWintersForecast, LSTMForecast, RNNForecast,
    TrainingConfig, TransformerForecast, XGBoostForecast,
)
from .models.base import set_deterministic_seed


def create_model(name: str, input_size: int, exog_size: int, horizon: int, config: dict):
    normalized = name.lower()
    architecture = dict(config.get("architecture", {}))
    if normalized == "holt_winters":
        return HoltWintersForecast(horizon=horizon, **architecture)
    if normalized == "xgboost":
        return XGBoostForecast(horizon=horizon, **architecture)
    training_keys = TrainingConfig.__dataclass_fields__.keys()
    training = TrainingConfig(**{key: value for key, value in config.get("training", {}).items() if key in training_keys})
    # La semilla debe fijarse antes de construir las capas; hacerlo solo dentro
    # de fit deja la inicializacion de pesos dependiente del orden de ejecucion.
    set_deterministic_seed(training.seed)
    common = dict(input_size=input_size, exog_size=exog_size, horizon=horizon, training=training)
    if normalized == "rnn":
        return RNNForecast(**common, **architecture)
    if normalized == "lstm":
        return LSTMForecast(**common, **architecture)
    if normalized == "deepar":
        return DeepARForecast(**common, **architecture)
    if normalized == "transformer":
        return TransformerForecast(**common, **architecture)
    raise ValueError(f"Modelo no soportado: {name}")
