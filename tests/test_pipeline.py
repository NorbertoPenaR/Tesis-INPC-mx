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
    assert np.isfinite(result[["yhat_trend", "yhat_cycle", "yhat"]].to_numpy()).all()
    np.testing.assert_allclose(result["yhat"], result["yhat_trend"] + result["yhat_cycle"])
