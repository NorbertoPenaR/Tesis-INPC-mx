from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["ds", "unique_id", "cutoff", "horizon", "step", "transformation"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combina pronosticos independientes de tendencia y ciclo.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--trend-model", required=True)
    parser.add_argument("--cycle-model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def combine(
    predictions_path: Path,
    trend_model: str,
    cycle_model: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(predictions_path, parse_dates=["ds", "cutoff"])
    trend = predictions[predictions["trend_model"] == trend_model].copy()
    cycle = predictions[predictions["cycle_model"] == cycle_model].copy()
    if trend.empty or cycle.empty:
        raise ValueError("No se encontraron ambos modelos en el archivo de pronosticos.")
    trend = trend[
        KEYS + ["yhat_trend", "y_true_trend", "y_true_cycle", "y_true", "mae_trend", "rmse_trend"]
    ]
    cycle = cycle[KEYS + ["yhat_cycle", "mae_cycle", "rmse_cycle"]]
    hybrid = trend.merge(cycle, on=KEYS, validate="one_to_one")
    hybrid["trend_model"] = trend_model
    hybrid["cycle_model"] = cycle_model
    hybrid["model"] = f"{trend_model}+{cycle_model}"
    hybrid["yhat"] = hybrid["yhat_trend"] + hybrid["yhat_cycle"]

    score_rows = []
    for (horizon, cutoff), group in hybrid.groupby(["horizon", "cutoff"]):
        error = group["y_true"].to_numpy() - group["yhat"].to_numpy()
        actual = group["y_true"].to_numpy()
        score_rows.append(
            {
                "horizon": horizon,
                "cutoff": cutoff,
                "mae": np.abs(error).mean(),
                "rmse": np.sqrt(np.mean(error**2)),
                "mape": np.mean(np.abs(error / actual)) * 100.0,
            }
        )
    scores = pd.DataFrame(score_rows)
    hybrid = hybrid.merge(scores, on=["horizon", "cutoff"], validate="many_to_one")
    summary = (
        scores.groupby("horizon")
        .agg(
            mae_mean=("mae", "mean"),
            mae_sd=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            mape_mean=("mape", "mean"),
            windows=("cutoff", "nunique"),
        )
        .reset_index()
    )
    summary.insert(0, "model", f"{trend_model}+{cycle_model}")
    summary.insert(1, "trend_model", trend_model)
    summary.insert(2, "cycle_model", cycle_model)
    summary.insert(3, "transformation", hybrid["transformation"].iat[0])

    output_dir.mkdir(parents=True, exist_ok=True)
    hybrid.to_csv(output_dir / "hybrid_predictions.csv", index=False)
    summary.to_csv(output_dir / "hybrid_summary.csv", index=False)
    return hybrid, summary


def main() -> None:
    args = parse_args()
    _, summary = combine(args.predictions, args.trend_model, args.cycle_model, args.output_dir)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
