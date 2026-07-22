import numpy as np
import pytest

from inpc_forecasting.models.classical import HoltWintersForecast, XGBoostForecast
from inpc_forecasting.pipeline import _windows


def _samples(horizon=4):
    t = np.arange(90, dtype=float)
    target = np.sin(2 * np.pi * t / 13)
    exog = np.column_stack([np.sin(2 * np.pi * t / 52), np.cos(2 * np.pi * t / 52)])
    return _windows(target, exog, context=20, horizon=horizon), target, exog


def test_holt_winters_forecast_is_finite():
    (history, future, targets), target, exog = _samples()
    model = HoltWintersForecast(horizon=4).fit(history, future, targets)
    prediction = model.predict(np.column_stack([target, exog])[-20:], exog[-4:])
    assert prediction.shape == (4,)
    assert np.isfinite(prediction).all()


def test_xgboost_recursive_forecast_is_finite():
    pytest.importorskip("xgboost")
    (history, future, targets), target, exog = _samples()
    model = XGBoostForecast(horizon=4, n_estimators=5, max_depth=2, lag_count=13)
    model.fit(history, future, targets)
    prediction = model.predict(np.column_stack([target, exog])[-20:], exog[-4:])
    assert prediction.shape == (4,)
    assert np.isfinite(prediction).all()
