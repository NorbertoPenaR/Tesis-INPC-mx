from __future__ import annotations

from .models import DeepARForecast, LSTMForecast, RNNForecast, TrainingConfig, TransformerForecast


def create_model(name: str, input_size: int, exog_size: int, horizon: int, config: dict):
    training_keys = TrainingConfig.__dataclass_fields__.keys()
    training = TrainingConfig(**{key: value for key, value in config.get("training", {}).items() if key in training_keys})
    architecture = dict(config.get("architecture", {}))
    common = dict(input_size=input_size, exog_size=exog_size, horizon=horizon, training=training)
    normalized = name.lower()
    if normalized == "rnn":
        return RNNForecast(**common, **architecture)
    if normalized == "lstm":
        return LSTMForecast(**common, **architecture)
    if normalized == "deepar":
        return DeepARForecast(**common, **architecture)
    if normalized == "transformer":
        return TransformerForecast(**common, **architecture)
    raise ValueError(f"Modelo no soportado: {name}")
