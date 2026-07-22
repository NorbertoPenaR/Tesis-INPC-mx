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
    parser = argparse.ArgumentParser(description="Evalua modelos analiticos para la tendencia HP.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--cycle-model", default="lstm")
    parser.add_argument("--frequency", default="W-MON")
    parser.add_argument("--unique-id", default="Inflacion")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def extrapolate(trend: np.ndarray, horizon: int, window: int | None, damping: float | None) -> np.ndarray:
    if window is None:
        return np.repeat(trend[-1], horizon)
    local = trend[-window:]
    slope, intercept = np.polyfit(np.arange(len(local), dtype=float), local, 1)
    if damping is None:
        future_x = np.arange(len(local), len(local) + horizon, dtype=float)
        return intercept + slope * future_x
    increments = slope * np.cumsum(damping ** np.arange(horizon, dtype=float))
    return trend[-1] + increments


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions, parse_dates=["ds", "cutoff"])
    cycle_rows = predictions[predictions["cycle_model"] == args.cycle_model]
    data = load_inpc_csv(args.data, args.frequency)
    series = data[data["unique_id"] == args.unique_id].set_index("ds")["y"]
    candidates: list[tuple[str, int | None, float | None]] = [("hp_persistence", None, None)]
    for window in (26, 52, 104, 156):
        candidates.append((f"hp_linear_{window}", window, None))
        for damping in (0.90, 0.95, 0.98):
            candidates.append((f"hp_damped_{window}_{damping:.2f}", window, damping))

    score_rows = []
    prediction_rows = []
    for (horizon, cutoff), group in cycle_rows.groupby(["horizon", "cutoff"]):
        group = group.sort_values("ds")
        training = series.loc[:cutoff].to_numpy(dtype=float)
        train_trend, _ = HPDecomposer(args.frequency).fit_transform(training)
        for name, window, damping in candidates:
            trend_hat = extrapolate(train_trend, int(horizon), window, damping)
            cycle_variants = [
                (args.cycle_model, group["yhat_cycle"].to_numpy(dtype=float)),
                ("zero", np.zeros(int(horizon), dtype=float)),
            ]
            for cycle_name, cycle_hat in cycle_variants:
                yhat = trend_hat + cycle_hat
                trend_error = group["y_true_trend"].to_numpy(dtype=float) - trend_hat
                total_error = group["y_true"].to_numpy(dtype=float) - yhat
                actual = group["y_true"].to_numpy(dtype=float)
                score_rows.append(
                    {
                        "trend_model": name,
                        "cycle_model": cycle_name,
                        "horizon": horizon,
                        "cutoff": cutoff,
                        "mae_trend": np.abs(trend_error).mean(),
                        "rmse_trend": np.sqrt(np.mean(trend_error**2)),
                        "mae": np.abs(total_error).mean(),
                        "rmse": np.sqrt(np.mean(total_error**2)),
                        "mape": np.mean(np.abs(total_error / actual)) * 100.0,
                    }
                )
                candidate_predictions = group[
                    [
                        "ds", "unique_id", "cutoff", "horizon", "step",
                        "y_true_trend", "y_true_cycle", "y_true",
                    ]
                ].copy()
                candidate_predictions["trend_model"] = name
                candidate_predictions["cycle_model"] = cycle_name
                candidate_predictions["yhat_trend"] = trend_hat
                candidate_predictions["yhat_cycle"] = cycle_hat
                candidate_predictions["yhat"] = yhat
                prediction_rows.append(candidate_predictions)

    scores = pd.DataFrame(score_rows)
    summary = (
        scores.groupby(["horizon", "trend_model", "cycle_model"])
        .agg(
            mae_trend_mean=("mae_trend", "mean"),
            mae_trend_sd=("mae_trend", "std"),
            mae_mean=("mae", "mean"),
            mae_sd=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            mape_mean=("mape", "mean"),
        )
        .reset_index()
        .sort_values(["horizon", "mae_trend_mean"])
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_dir / "trend_baseline_scores.csv", index=False)
    summary.to_csv(output_dir / "trend_baseline_summary.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        output_dir / "trend_baseline_predictions.csv", index=False
    )
    print(
        summary.sort_values(["horizon", "mae_mean"]).groupby("horizon").head(12).round(4).to_string(index=False)
    )


if __name__ == "__main__":
    main()
