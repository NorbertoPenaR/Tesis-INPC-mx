from __future__ import annotations

import numpy as np
import pandas as pd


def calendar_features(dates: pd.DatetimeIndex) -> np.ndarray:
    dates = pd.DatetimeIndex(dates)
    week = dates.isocalendar().week.to_numpy(dtype=float)
    month = dates.month.to_numpy(dtype=float)
    quarter = dates.quarter.to_numpy(dtype=float)
    return np.column_stack(
        [
            np.sin(2 * np.pi * week / 52.0),
            np.cos(2 * np.pi * week / 52.0),
            np.sin(2 * np.pi * month / 12.0),
            np.cos(2 * np.pi * month / 12.0),
            np.sin(2 * np.pi * quarter / 4.0),
            np.cos(2 * np.pi * quarter / 4.0),
        ]
    ).astype(np.float64)


def extrapolated_fft_features(values: np.ndarray, total_steps: int, top_k: int) -> np.ndarray:
    """Fit Fourier terms only on training values and extrapolate them."""
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    if top_k <= 0:
        return np.empty((total_steps, 0), dtype=np.float64)
    centered = y - y.mean()
    spectrum = np.fft.rfft(centered)
    frequencies = np.fft.rfftfreq(len(y))
    candidates = np.arange(1, len(spectrum))
    order = candidates[np.argsort(np.abs(spectrum[candidates]))[::-1]][:top_k]
    t = np.arange(total_steps, dtype=np.float64)
    columns: list[np.ndarray] = []
    for index in order:
        amplitude = 2.0 * np.abs(spectrum[index]) / len(y)
        phase = np.angle(spectrum[index])
        angle = 2.0 * np.pi * frequencies[index] * t + phase
        columns.extend([amplitude * np.cos(angle), amplitude * np.sin(angle)])
    return np.column_stack(columns) if columns else np.empty((total_steps, 0))


def component_features(
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    future_dates: pd.DatetimeIndex,
    use_calendar: bool,
    fft_top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_dates = dates.append(future_dates)
    total = len(all_dates)
    blocks = []
    if use_calendar:
        blocks.append(calendar_features(all_dates))
    if fft_top_k:
        blocks.append(extrapolated_fft_features(values, total, fft_top_k))
    if not blocks:
        empty = np.empty((total, 0), dtype=np.float64)
        return empty[: len(dates)], empty[len(dates) :]
    full = np.column_stack(blocks)
    return full[: len(dates)], full[len(dates) :]
