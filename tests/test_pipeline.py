from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from inpc_forecasting.pipeline import ComponentForecastPipeline, RESULT_COLUMNS


def _config(model):
    architecture = {"hidden_size": 8, "num_layers": 1, "dropout": 0.0}
    if model == "transformer":
        architecture = {"d_model": 8, "num_layers": 1, "nhead": 2, "dim_feedforward": 16, "dropout": 0.0}
    return {
        "data": {"unique_id": "Inflacion", "cutoff": "2022-04-04", "frequency": "W-MON"},
        "hp": {"lambda": None},
        "components": {
            "trend": {"transform": "diff", "calendar": True, "fft_top_k": 1},
            "cycle": {"transform": "none", "calendar": True, "fft_top_k": 1},
        },
        "experiment": {"model": model, "horizon": 3},
        "models": {
            model: {
                "context_length": 12,
                "architecture": architecture,
                "training": {"epochs": 1, "batch_size": 8, "patience": 1, "device": "cpu", "seed": 119},
            }
        },
    }


@pytest.mark.parametrize("model", ["rnn", "lstm", "deepar", "transformer"])
def test_end_to_end_component_pipeline(model):
    dates = pd.date_range("2019-01-07", periods=180, freq="W-MON")
    t = np.arange(len(dates))
    frame = pd.DataFrame(
        {"ds": dates, "unique_id": "Inflacion", "y": 4 + 0.005 * t + 0.2 * np.sin(2 * np.pi * t / 13)}
    )
    result = ComponentForecastPipeline(_config(model)).fit_predict(frame)
    assert list(result.columns) == RESULT_COLUMNS
    assert len(result) == 3
    numeric = [
        "yhat_trend", "yhat_cycle", "yhat", "y_true_trend", "y_true_cycle",
        "y_true_trend_realtime", "y_true_cycle_realtime",
        "mae_trend", "rmse_trend", "mae_cycle", "rmse_cycle",
        "mae_trend_realtime", "mae_cycle_realtime",
    ]
    assert np.isfinite(result[numeric].to_numpy()).all()
    np.testing.assert_allclose(result["yhat"], result["yhat_trend"] + result["yhat_cycle"])
    np.testing.assert_allclose(result["y_true"], result["y_true_trend"] + result["y_true_cycle"])
    np.testing.assert_allclose(
        result["y_true"], result["y_true_trend_realtime"] + result["y_true_cycle_realtime"]
    )


def test_cycle_training_window_is_applied_before_scaling():
    config = _config("rnn")
    config["components"]["cycle"]["training_window"] = 20
    pipe = ComponentForecastPipeline(config)
    dates = pd.date_range("2020-01-06", periods=50, freq="W-MON")
    future = pd.date_range("2020-12-21", periods=3, freq="W-MON")
    prepared = pipe._prepare(np.arange(50, dtype=float), dates, future, False)
    assert len(prepared.transformed) == 20
    assert prepared.scaler.mean_ == pytest.approx(np.arange(30, 50).mean())


def test_future_values_only_change_evaluation_labels_not_predictions():
    dates = pd.date_range("2019-01-07", periods=180, freq="W-MON")
    t = np.arange(len(dates))
    frame = pd.DataFrame(
        {"ds": dates, "unique_id": "Inflacion", "y": 4 + 0.005 * t + 0.2 * np.sin(2 * np.pi * t / 13)}
    )
    config = _config("rnn")
    first = ComponentForecastPipeline(config).fit_predict(frame)
    changed = frame.copy()
    changed.loc[changed["ds"] > pd.Timestamp(config["data"]["cutoff"]), "y"] += 10.0
    second = ComponentForecastPipeline(config).fit_predict(changed)

    np.testing.assert_allclose(first["yhat_trend"], second["yhat_trend"])
    np.testing.assert_allclose(first["yhat_cycle"], second["yhat_cycle"])
    assert not np.allclose(first["y_true_trend"], second["y_true_trend"])
