import numpy as np
import pytest

from inpc_forecasting.transforms import Standardizer, TrendTransform


@pytest.mark.parametrize("name", ["diff", "diff_logp1", "pct", "logp1", "none", "diff2"])
def test_transform_forecast_inverse(name):
    history = np.linspace(2.0, 5.0, 40) ** 1.2
    future = np.linspace(5.1, 6.0, 6) ** 1.2
    full = np.r_[history, future]
    transform = TrendTransform(name)
    transform.fit_transform(history)
    if name == "none":
        future_transformed = future
    elif name == "logp1":
        future_transformed = np.log1p(future)
    elif name == "diff":
        future_transformed = np.diff(np.r_[history[-1], future])
    elif name == "diff_logp1":
        future_transformed = np.diff(np.log1p(np.r_[history[-1], future]))
    elif name == "pct":
        base = np.r_[history[-1], future]
        future_transformed = np.diff(base) / base[:-1] * 100
    else:
        future_transformed = np.diff(np.r_[history[-2:], future], n=2)
    np.testing.assert_allclose(transform.inverse_forecast(future_transformed), future, rtol=1e-10)


def test_standardizer_roundtrip():
    values = np.array([-2.0, 0.0, 4.0, 8.0])
    scaler = Standardizer()
    scaled = scaler.fit_transform(values)
    np.testing.assert_allclose(scaler.inverse_transform(scaled), values)
