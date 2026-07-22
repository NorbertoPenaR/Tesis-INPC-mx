from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from .data import load_inpc_csv
from .evaluation import run_rolling_benchmark
from .pipeline import ComponentForecastPipeline
from .transforms import SUPPORTED_TRANSFORMS


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def smoke_config(config: dict) -> dict:
    result = deepcopy(config)
    result["experiment"]["horizon"] = min(4, int(result["experiment"]["horizon"]))
    for model in result["models"].values():
        model["context_length"] = min(16, int(model["context_length"]))
        model.setdefault("training", {})["epochs"] = 2
        model["training"]["patience"] = 2
        model["training"]["batch_size"] = 16
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pronostico HP por tendencia y ciclo")
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rolling", action="store_true", help="Ejecuta todas las ventanas y horizontes configurados")
    parser.add_argument("--model", choices=["rnn", "lstm", "deepar", "transformer"])
    parser.add_argument("--trend-model", choices=["rnn", "lstm", "deepar", "transformer"])
    parser.add_argument("--cycle-model", choices=["rnn", "lstm", "deepar", "transformer"])
    parser.add_argument("--trend-transform", choices=SUPPORTED_TRANSFORMS)
    parser.add_argument("--output-dir", default="outputs/hp_pytorch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.trend_transform:
        config["components"]["trend"]["transform"] = args.trend_transform
    if args.trend_model:
        config["experiment"]["trend_model"] = args.trend_model
    if args.cycle_model:
        config["experiment"]["cycle_model"] = args.cycle_model
    if args.smoke:
        config = smoke_config(config)
    models = [args.model] if args.model else config["experiment"].get("models", [config["experiment"]["model"]])
    data = load_inpc_csv(config["data"]["csv_path"], config["data"]["frequency"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.rolling and not args.smoke:
        if args.model:
            config["experiment"]["models"] = [args.model]
        run_rolling_benchmark(config, data, output_dir)
        return
    outputs = []
    for model in models:
        model_config = deepcopy(config)
        model_config["experiment"]["model"] = model
        result = ComponentForecastPipeline(model_config).fit_predict(data)
        result.to_csv(output_dir / f"{model}_forecast.csv", index=False)
        outputs.append(result)
    pd.concat(outputs, ignore_index=True).to_csv(output_dir / "all_models_forecast.csv", index=False)


if __name__ == "__main__":
    main()
