from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import ComponentForecastPipeline


def rolling_cutoffs(
    data: pd.DataFrame,
    unique_id: str,
    final_cutoff: str | pd.Timestamp,
    windows: int,
    step: int,
    required_horizon: int = 0,
) -> list[pd.Timestamp]:
    dates = pd.DatetimeIndex(
        data.loc[data["unique_id"] == unique_id, "ds"].pipe(pd.to_datetime).sort_values().unique()
    )
    last_position = int(dates.searchsorted(pd.Timestamp(final_cutoff), side="right") - 1)
    positions = [last_position - step * index for index in reversed(range(windows))]
    if not positions or min(positions) < 0:
        raise ValueError("No existen observaciones suficientes para las ventanas rolling solicitadas.")
    if required_horizon < 0:
        raise ValueError("El horizonte requerido no puede ser negativo.")
    if max(positions) + required_horizon >= len(dates):
        available = len(dates) - max(positions) - 1
        raise ValueError(
            "La ultima ventana rolling no tiene valores reales suficientes: "
            f"requiere {required_horizon} y solo existen {available} despues del corte."
        )
    return [pd.Timestamp(dates[position]) for position in positions]


def run_rolling_benchmark(config: dict, data: pd.DataFrame, output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment = config["experiment"]
    rolling = experiment["rolling"]
    cutoffs = rolling_cutoffs(
        data,
        config["data"]["unique_id"],
        config["data"]["cutoff"],
        int(rolling["windows"]),
        int(rolling["step"]),
        required_horizon=max(int(value) for value in experiment["rolling_horizons"]),
    )
    predictions = []
    for horizon in experiment["rolling_horizons"]:
        for model in experiment["models"]:
            for cutoff in cutoffs:
                run_config = deepcopy(config)
                run_config["experiment"]["model"] = model
                run_config["experiment"]["horizon"] = int(horizon)
                predictions.append(ComponentForecastPipeline(run_config).fit_predict(data, cutoff=cutoff))
    detailed = pd.concat(predictions, ignore_index=True)
    scores = detailed[
        [
            "model", "trend_model", "cycle_model", "transformation", "horizon", "cutoff",
            "mae_trend", "rmse_trend", "mae_cycle", "rmse_cycle",
            "mae_trend_realtime", "mae_cycle_realtime", "mae", "rmse", "mape",
        ]
    ].drop_duplicates()
    summary = (
        scores.groupby(["model", "trend_model", "cycle_model", "transformation", "horizon"], as_index=False)
        .agg(
            mae_trend_mean=("mae_trend", "mean"),
            mae_trend_sd=("mae_trend", "std"),
            mae_cycle_mean=("mae_cycle", "mean"),
            mae_cycle_sd=("mae_cycle", "std"),
            mae_trend_realtime_mean=("mae_trend_realtime", "mean"),
            mae_cycle_realtime_mean=("mae_cycle_realtime", "mean"),
            mae_mean=("mae", "mean"),
            mae_sd=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            mape_mean=("mape", "mean"),
            windows=("cutoff", "nunique"),
        )
    )
    detailed.to_csv(output_dir / "rolling_predictions.csv", index=False)
    summary.to_csv(output_dir / "rolling_summary.csv", index=False)
    return detailed, summary
