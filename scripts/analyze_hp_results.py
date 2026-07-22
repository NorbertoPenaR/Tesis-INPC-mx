from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from inpc_forecasting.data import load_inpc_csv  # noqa: E402
from inpc_forecasting.decomposition import HPDecomposer  # noqa: E402


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
        "yhat_trend", "yhat_cycle", "yhat", "y_true_trend", "y_true_cycle", "y_true",
        "mae_trend", "rmse_trend", "mae_cycle", "rmse_cycle", "mae", "rmse", "mape",
        "execution_time",
    ]
    if predictions[numeric].isna().any().any() or not np.isfinite(predictions[numeric].to_numpy()).all():
        raise ValueError("El benchmark contiene valores faltantes o no finitos.")
    keys = ["ds", "cutoff", "horizon", "model"]
    if predictions.duplicated(keys).any():
        raise ValueError("El benchmark contiene pronosticos duplicados.")

    data = load_inpc_csv(data_path, frequency)
    series = data[data["unique_id"] == unique_id].set_index("ds")["y"]
    scores = predictions[
        [
            "model", "trend_model", "cycle_model", "transformation", "horizon", "cutoff",
            "mae_trend", "rmse_trend", "mae_cycle", "rmse_cycle", "mae", "rmse", "mape",
            "execution_time",
        ]
    ].drop_duplicates()
    summary = (
        scores.groupby(["horizon", "transformation", "model"])
        .agg(
            mae_trend_mean=("mae_trend", "mean"),
            mae_trend_sd=("mae_trend", "std"),
            mae_cycle_mean=("mae_cycle", "mean"),
            mae_cycle_sd=("mae_cycle", "std"),
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
        actual = group[["ds", "y_true", "y_true_trend", "y_true_cycle"]].drop_duplicates().sort_values("ds")
        last_value = float(series.loc[cutoff])
        error = actual["y_true"].to_numpy() - last_value
        training_values = series.loc[:cutoff].to_numpy(dtype=float)
        training_trend, _ = HPDecomposer(frequency).fit_transform(training_values)
        trend_persistence_error = actual["y_true_trend"].to_numpy() - training_trend[-1]
        baseline_rows.append(
            {
                "cutoff": cutoff,
                "horizon": horizon,
                "persistence_mae": np.abs(error).mean(),
                "persistence_rmse": np.sqrt(np.mean(error**2)),
                "persistence_mape": np.mean(np.abs(error / actual["y_true"].to_numpy())) * 100.0,
                "trend_persistence_mae": np.abs(trend_persistence_error).mean(),
                "zero_cycle_mae": np.abs(actual["y_true_cycle"].to_numpy()).mean(),
            }
        )
    baselines = pd.DataFrame(baseline_rows)
    comparison = scores.merge(baselines, on=["cutoff", "horizon"])
    comparison["delta_vs_persistence"] = comparison["mae"] - comparison["persistence_mae"]
    comparison["delta_trend_vs_persistence"] = comparison["mae_trend"] - comparison["trend_persistence_mae"]
    comparison["delta_cycle_vs_zero"] = comparison["mae_cycle"] - comparison["zero_cycle_mae"]

    components = scores[
        [
            "horizon", "transformation", "model", "trend_model", "cycle_model", "cutoff",
            "mae_trend", "rmse_trend", "mae_cycle", "rmse_cycle", "mae", "rmse",
        ]
    ].copy()

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
        comparison.groupby(["horizon", "transformation", "model"])
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
        components.groupby(["horizon", "transformation", "model"])
        .agg(
            mae_trend=("mae_trend", "mean"),
            rmse_trend=("rmse_trend", "mean"),
            mae_cycle=("mae_cycle", "mean"),
            rmse_cycle=("rmse_cycle", "mean"),
            mae_total=("mae", "mean"),
        )
        .reset_index()
        .sort_values(["horizon", "mae_total"])
    )
    component_baselines = (
        comparison.groupby(["horizon", "transformation", "model"])
        .agg(
            mae_trend=("mae_trend", "mean"),
            trend_persistence=("trend_persistence_mae", "mean"),
            trend_delta=("delta_trend_vs_persistence", "mean"),
            mae_cycle=("mae_cycle", "mean"),
            zero_cycle=("zero_cycle_mae", "mean"),
            cycle_delta=("delta_cycle_vs_zero", "mean"),
        )
        .reset_index()
        .sort_values(["horizon", "mae_trend"])
    )
    print("\nResumen por modelo\n", summary.round(4).to_string(index=False))
    print("\nComparacion contra persistencia\n", versus.round(4).to_string(index=False))
    print("\nMetricas retrospectivas por componente\n", component_summary.round(4).to_string(index=False))
    print("\nComponentes contra lineas base\n", component_baselines.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
