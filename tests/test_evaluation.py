import pandas as pd
import pytest

from inpc_forecasting.evaluation import rolling_cutoffs


def test_rolling_cutoffs_are_ordered_and_use_observed_dates():
    dates = pd.date_range("2020-01-06", periods=30, freq="W-MON")
    frame = pd.DataFrame({"ds": dates, "unique_id": "Inflacion", "y": range(30)})
    cutoffs = rolling_cutoffs(frame, "Inflacion", dates[-1], windows=3, step=5)
    assert cutoffs == [dates[-11], dates[-6], dates[-1]]


def test_rolling_cutoffs_reject_incomplete_evaluation_horizon():
    dates = pd.date_range("2020-01-06", periods=30, freq="W-MON")
    frame = pd.DataFrame({"ds": dates, "unique_id": "Inflacion", "y": range(30)})

    with pytest.raises(ValueError, match="solo existen 2"):
        rolling_cutoffs(
            frame,
            "Inflacion",
            dates[-3],
            windows=1,
            step=5,
            required_horizon=3,
        )


def test_rolling_cutoffs_accept_complete_evaluation_horizon():
    dates = pd.date_range("2020-01-06", periods=30, freq="W-MON")
    frame = pd.DataFrame({"ds": dates, "unique_id": "Inflacion", "y": range(30)})

    cutoffs = rolling_cutoffs(
        frame,
        "Inflacion",
        dates[-4],
        windows=1,
        step=5,
        required_horizon=3,
    )

    assert cutoffs == [dates[-4]]
