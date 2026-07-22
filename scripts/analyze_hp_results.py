from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from inpc_forecasting.data import load_inpc_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audita y resume un benchmark HP rolling-origin.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--frequency", default="W-MON")
    parser.add_argument("--unique-id", default="Inflacion")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def analyze(
    predictions_path: Path,
    data_path: Path,
    frequency: str,
    unique_id: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(predictions_path, parse_dates=["ds", "cutoff"])
    numeric = [
        "yhat_trend", "yhat_cycle", "yhat", "y_true", "mae", "rmse", "mape", "execution_time",
    ]
    if predictions[numeric].isna().any().any() or not np.isfinite(predictions[numeric].to_numpy()).all():
        raise ValueError("El benchmark contiene valores faltantes o no finitos.")
    keys = ["ds", "cutoff", "horizon", "model"]
    if predictions.duplicated(keys).any():
        raise ValueError("El benchmark contiene pronosticos duplicados.")

    data = load_inpc_csv(data_path, frequency)
    series = data[data["unique_id"] == unique_id].set_index("ds")["y"]
    scores = predictions[
        ["model", "horizon", "cutoff", "mae", "rmse", "mape", "execution_time"]
    ].drop_duplicates()
    summary = (
        scores.groupby(["horizon", "model"])
        .agg(
            mae_mean=("mae", "mean"),
            mae_sd=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            mape_mean=("mape", "mean"),
            time_mean=("execution_time", "mean"),
        )
        .reset_index()
        .sort_values(["horizon", "mae_mean"])
    )

    baseline_rows = []
    for (cutoff, horizon), group in predictions.groupby(["cutoff", "horizon"]):
        actual = group[["ds", "y_true"]].drop_duplicates().sort_values("ds")
        last_value = float(series.loc[cutoff])
        error = actual["y_true"].to_numpy() - last_value
        baseline_rows.append(
            {
                "cutoff": cutoff,
                "horizon": horizon,
                "persistence_mae": np.abs(error).mean(),
                "persistence_rmse": np.sqrt(np.mean(error**2)),
                "persistence_mape": np.mean(np.abs(error / actual["y_true"].to_numpy())) * 100.0,
            }
        )
    baselines = pd.DataFrame(baseline_rows)
    comparison = scores.merge(baselines, on=["cutoff", "horizon"])
    comparison["delta_vs_persistence"] = comparison["mae"] - comparison["persistence_mae"]

    point_errors = predictions.assign(
        abs_full=lambda frame: np.abs(frame["y_true"] - frame["yhat"]),
        abs_trend=lambda frame: np.abs(frame["y_true"] - frame["yhat_trend"]),
    )
    components = (
        point_errors.groupby(["horizon", "model", "cutoff"])
        .agg(
            full_mae=("abs_full", "mean"),
            trend_only_mae=("abs_trend", "mean"),
            cycle_abs_mean=("yhat_cycle", lambda values: np.abs(values).mean()),
            cycle_mean=("yhat_cycle", "mean"),
        )
        .reset_index()
    )
    components["cycle_delta"] = components["full_mae"] - components["trend_only_mae"]

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "audited_summary.csv", index=False)
    comparison.to_csv(output_dir / "comparison_with_persistence.csv", index=False)
    components.to_csv(output_dir / "component_diagnostics.csv", index=False)
    return summary, comparison, components


def main() -> None:
    args = parse_args()
    summary, comparison, components = analyze(
        args.predictions,
        args.data,
        args.frequency,
        args.unique_id,
        args.output_dir,
    )
    versus = (
        comparison.groupby(["horizon", "model"])
        .agg(
            model_mae=("mae", "mean"),
            persistence_mae=("persistence_mae", "mean"),
            delta=("delta_vs_persistence", "mean"),
            wins=("delta_vs_persistence", lambda values: int((values < 0).sum())),
        )
        .reset_index()
        .sort_values(["horizon", "model_mae"])
    )
    component_summary = (
        components.groupby(["horizon", "model"])
        .agg(
            full_mae=("full_mae", "mean"),
            trend_only_mae=("trend_only_mae", "mean"),
            cycle_delta=("cycle_delta", "mean"),
            cycle_abs_mean=("cycle_abs_mean", "mean"),
        )
        .reset_index()
        .sort_values(["horizon", "full_mae"])
    )
    print("\nResumen por modelo\n", summary.round(4).to_string(index=False))
    print("\nComparacion contra persistencia\n", versus.round(4).to_string(index=False))
    print("\nEfecto del pronostico ciclico\n", component_summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
