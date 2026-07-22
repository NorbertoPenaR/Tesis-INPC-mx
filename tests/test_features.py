import numpy as np
import pandas as pd

from inpc_forecasting.features import component_features, extrapolated_fft_features


def test_fft_is_fit_only_from_values_passed_by_training():
    values = np.sin(2 * np.pi * np.arange(80) / 10)
    first = extrapolated_fft_features(values, total_steps=100, top_k=2)
    second = extrapolated_fft_features(values.copy(), total_steps=100, top_k=2)
    np.testing.assert_array_equal(first, second)


def test_component_features_align_history_and_future():
    dates = pd.date_range("2020-01-06", periods=80, freq="W-MON")
    future = pd.date_range(dates[-1], periods=5, freq="W-MON")[1:]
    historical, forecast = component_features(np.arange(80), dates, future, True, 2)
    assert historical.shape == (80, 10)
    assert forecast.shape == (4, 10)
