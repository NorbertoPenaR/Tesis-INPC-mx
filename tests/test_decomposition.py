import numpy as np
import pandas as pd

from inpc_forecasting.decomposition import HPDecomposer, hp_lambda_for_frequency


def test_weekly_lambda_uses_ravn_uhlig_scaling():
    assert hp_lambda_for_frequency("W-MON") == 45_697_600.0
    assert hp_lambda_for_frequency("M") == 129_600.0
    assert hp_lambda_for_frequency("Q") == 1_600.0


def test_hp_components_recompose_original_series():
    t = np.arange(160)
    series = 3 + 0.01 * t + 0.3 * np.sin(2 * np.pi * t / 13)
    trend, cycle = HPDecomposer("W-MON").fit_transform(pd.Series(series))
    np.testing.assert_allclose(trend + cycle, series, atol=1e-9)


def test_future_changes_do_not_change_training_decomposition():
    rng = np.random.default_rng(119)
    history = np.cumsum(rng.normal(size=120))
    future_a = np.zeros(20)
    future_b = np.full(20, 1_000_000.0)
    decomposer = HPDecomposer("W-MON")
    trend_a, cycle_a = decomposer.fit_transform(np.r_[history, future_a][:120])
    trend_b, cycle_b = decomposer.fit_transform(np.r_[history, future_b][:120])
    np.testing.assert_array_equal(trend_a, trend_b)
    np.testing.assert_array_equal(cycle_a, cycle_b)
