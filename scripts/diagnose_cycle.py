"""Diagnostico reproducible del ciclo HP y baselines clasicos."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from inpc_forecasting.data import load_inpc_csv
from inpc_forecasting.decomposition import HPDecomposer
from inpc_forecasting.evaluation import rolling_cutoffs
from inpc_forecasting.models.classical import HoltWintersForecast, XGBoostForecast
from inpc_forecasting.pipeline import ComponentForecastPipeline, _future_dates, _windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnostica escala y pronostico del ciclo HP")
    parser.add_argument("--config", default="configs/hp_pytorch.yaml")
    parser.add_argument("--output-dir", default="outputs/hp_cycle_diagnostics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = load_inpc_csv(config["data"]["csv_path"], config["data"]["frequency"])
    series = data[data["unique_id"] == config["data"]["unique_id"]].sort_values("ds")
    decomposer = HPDecomposer(config["data"]["frequency"], config["hp"].get("lambda"))
    latest_cutoff = pd.Timestamp(config["data"]["cutoff"])
    latest = series[series["ds"] <= latest_cutoff]
    _, latest_cycle = decomposer.fit_transform(latest["y"].to_numpy())
    history = pd.DataFrame({"ds": latest["ds"].to_numpy(), "cycle": latest_cycle})
    history.to_csv(output / "cycle_history.csv", index=False)

    scales = []
    for years in (3, 5, 8, 10, 15, 20):
        sample = history[history["ds"] >= latest_cutoff - pd.DateOffset(years=years)]["cycle"]
        scales.append({
            "years": years, "observations": len(sample), "mean": sample.mean(),
            "std": sample.std(ddof=0), "minimum": sample.min(), "maximum": sample.max(),
        })
    scales.append({
        "years": "all", "observations": len(history), "mean": history["cycle"].mean(),
        "std": history["cycle"].std(ddof=0), "minimum": history["cycle"].min(),
        "maximum": history["cycle"].max(),
    })
    pd.DataFrame(scales).to_csv(output / "cycle_scale_by_window.csv", index=False)

    cutoffs = rolling_cutoffs(
        data, config["data"]["unique_id"], latest_cutoff,
        int(config["experiment"]["rolling"]["windows"]),
        int(config["experiment"]["rolling"]["step"]),
        required_horizon=max(map(int, config["experiment"]["rolling_horizons"])),
    )
    rows = []
    for horizon in config["experiment"]["rolling_horizons"]:
        for cutoff in cutoffs:
            train = series[series["ds"] <= cutoff]
            dates = pd.DatetimeIndex(train["ds"])
            future_dates = _future_dates(dates[-1], int(horizon), config["data"]["frequency"])
            _, cycle = decomposer.fit_transform(train["y"].to_numpy())
            pipeline = ComponentForecastPipeline(config)
            prepared = pipeline._prepare(cycle, dates, future_dates, False)
            context = int(config["models"]["xgboost"]["context_length"])
            histories, futures, targets = _windows(
                prepared.transformed, prepared.historical_exog,
                context, int(horizon),
            )
            latest_history = np.column_stack([prepared.transformed, prepared.historical_exog])[-context:]
            actual = series.set_index("ds")["y"].reindex(future_dates).to_numpy(dtype=float)
            _, actual_cycle = decomposer.fit_transform(np.r_[train["y"].to_numpy(), actual])
            actual_cycle = actual_cycle[-int(horizon):]
            models = {
                "xgboost": XGBoostForecast(horizon=int(horizon), **config["models"]["xgboost"]["architecture"]),
                "holt_winters": HoltWintersForecast(
                    horizon=int(horizon), **config["models"]["holt_winters"]["architecture"]
                ),
            }
            for name, model in models.items():
                model.fit(histories, futures, targets)
                prediction = prepared.scaler.inverse_transform(model.predict(latest_history, prepared.future_exog))
                for step, (date, observed, predicted) in enumerate(
                    zip(future_dates, actual_cycle, prediction), start=1
                ):
                    rows.append({
                        "cutoff": cutoff, "horizon": horizon, "step": step, "ds": date,
                        "model": name, "y_true_cycle": observed, "yhat_cycle": predicted,
                        "absolute_error": abs(observed - predicted),
                    })
    predictions = pd.DataFrame(rows)
    predictions.to_csv(output / "cycle_baseline_predictions.csv", index=False)
    predictions.groupby(["horizon", "model"], as_index=False)["absolute_error"].mean().rename(
        columns={"absolute_error": "mae_cycle"}
    ).to_csv(output / "cycle_baseline_summary.csv", index=False)


if __name__ == "__main__":
    main()
