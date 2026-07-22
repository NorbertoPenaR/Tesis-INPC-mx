from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INPC_COLUMNS = [
    "Inflacion",
    "Subyacente",
    "Mercancias",
    "Alimentos_bebidas_tabaco",
    "Mercancias_no_alimenticias",
    "Servicios",
    "Vivienda",
    "Educacion_colegiaturas",
    "Otros_servicios",
    "No_subyacente",
    "Agropecuarios",
    "Frutas_verduras",
    "Pecuarios",
    "Energeticos_tarifas_autorizadas_por_el_gobierno",
    "Energeticos",
    "Tarifas_autorizadas_por_el_gobierno",
]

MONTHS = {
    "Ene": 1, "Feb": 2, "Mar": 3, "Abr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Ago": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dic": 12,
}


def parse_fortnight(value: str) -> pd.Timestamp:
    fortnight, month_name, year = str(value).strip().split()
    day = 1 if fortnight == "1Q" else 16 if fortnight == "2Q" else None
    if day is None or month_name not in MONTHS:
        raise ValueError(f"Fecha quincenal invalida: {value!r}")
    return pd.Timestamp(int(year), MONTHS[month_name], day)


def load_inpc_csv(path: str | Path, frequency: str = "W-MON") -> pd.DataFrame:
    """Load either a tidy series or the original INEGI CA56 export."""
    path = Path(path)
    for encoding in ("utf-8", "latin1"):
        try:
            header = pd.read_csv(path, encoding=encoding, nrows=1)
            if {"ds", "unique_id", "y"}.issubset(header.columns):
                tidy = pd.read_csv(path, encoding=encoding)
                tidy["ds"] = pd.to_datetime(tidy["ds"])
                return tidy[["ds", "unique_id", "y"]].sort_values(["unique_id", "ds"])
            break
        except UnicodeDecodeError:
            continue

    raw = pd.read_csv(path, skiprows=9, encoding="latin1", header=None)
    first = raw.iloc[:, 0].astype(str).str.strip()
    valid = first.str.match(r"^[12]Q\s+[A-Za-z]{3}\s+\d{4}$")
    raw = raw.loc[valid, : len(INPC_COLUMNS)].copy()
    raw.columns = ["ds", *INPC_COLUMNS]
    raw["ds"] = raw["ds"].map(parse_fortnight)
    for column in INPC_COLUMNS:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
        raw[column] = raw[column].pct_change(24) * 100.0
    raw = raw.dropna(subset=INPC_COLUMNS)
    weekly = raw.set_index("ds")[INPC_COLUMNS].resample(frequency).mean().interpolate("linear")
    return (
        weekly.reset_index()
        .melt(id_vars="ds", var_name="unique_id", value_name="y")
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["y"])
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )
