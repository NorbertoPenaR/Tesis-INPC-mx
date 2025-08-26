
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

@dataclass
class SeriesScaler:
    kind: str = "none"
    minmax_range: Tuple[float, float] = (0.0, 1.0)

    @staticmethod
    def transform(y: np.ndarray, params: Dict) -> np.ndarray:
        kind = params.get("kind", "none")
        if kind == "none":
            return y
        if kind == "zscore":
            mu = params["mean"]; sc = params["scale"]
            return (y - mu) / sc if sc != 0 else y - mu
        if kind == "minmax":
            data_min = params["data_min"]; data_max = params["data_max"]
            a, b = params["range"]
            return (y - data_min) * (b - a) / (data_max - data_min) + a if data_max != data_min else y*0 + a
        raise ValueError(f"Unknown scaler kind {kind}")


    def fit_transform(self, y: np.ndarray) -> Tuple[np.ndarray, Dict]:
        if self.kind == "none":
            return y, {"kind": "none"}
        if self.kind == "zscore":
            ss = StandardScaler(with_mean=True, with_std=True)
            y2 = ss.fit_transform(y.reshape(-1, 1)).reshape(-1)
            return y2, {"kind": "zscore", "mean": float(ss.mean_[0]), "scale": float(ss.scale_[0])}
        if self.kind == "minmax":
            a, b = self.minmax_range
            mm = MinMaxScaler(feature_range=(a, b))
            y2 = mm.fit_transform(y.reshape(-1, 1)).reshape(-1)
            return y2, {"kind": "minmax", "data_min": float(mm.data_min_[0]), "data_max": float(mm.data_max_[0]), "range": (a, b)}
        raise ValueError(f"Unknown scaler kind {self.kind}")

    @staticmethod
    def inverse_transform(y: np.ndarray, params: Dict) -> np.ndarray:
        kind = params.get("kind", "none")
        if kind == "none":
            return y
        if kind == "zscore":
            return y * params["scale"] + params["mean"]
        if kind == "minmax":
            data_min = params["data_min"]
            data_max = params["data_max"]
            a, b = params["range"]
            return (y - a) * (data_max - data_min) / (b - a) + data_min
        raise ValueError(f"Unknown scaler kind {kind}")
    
    @staticmethod
    def transform(y: np.ndarray, params: Dict) -> np.ndarray:
        kind = params.get("kind", "none")
        if kind == "none":
            return y
        if kind == "zscore":
            mu = params["mean"]; sc = params["scale"]; sc = sc if sc!=0 else 1.0
            return (y - mu) / sc
        if kind == "minmax":
            data_min = params["data_min"]; data_max = params["data_max"]
            a, b = params["range"]; rng = (data_max - data_min) if data_max!=data_min else 1.0
            return (y - data_min) * (b - a) / rng + a
        raise ValueError(f"Unknown scaler kind {kind}")


def _sincos(ix: np.ndarray, period: int) -> np.ndarray:
    # ix = [0..T-1]; devuelve [sin, cos]
    ang = 2*np.pi*ix/period
    return np.stack([np.sin(ang), np.cos(ang)], axis=1)  # [T,2]

def _time_feature_matrix(ds: pd.Series, names: list[str]) -> np.ndarray:
    """Devuelve matriz [T, d_t] con pares sin/cos para cada feature temporal."""
    T = len(ds)
    out = []
    idx = np.arange(T)
    s = pd.to_datetime(ds)
    if "dow" in names:
        out.append(_sincos(s.dt.dayofweek.to_numpy(), 7))
    if "month" in names:
        out.append(_sincos(s.dt.month.to_numpy()-1, 12))
    if "weekofyear" in names:
        # pandas week=ISO: 1..53 -> restamos 1
        out.append(_sincos(s.dt.isocalendar().week.to_numpy().astype(int)-1, 53))
    if "dayofyear" in names:
        # 1..365/366 -> restamos 1
        out.append(_sincos(s.dt.dayofyear.to_numpy()-1, 366))
    if not out:
        return np.zeros((T,0), dtype=np.float32)
    return np.concatenate(out, axis=1).astype(np.float32)

def _infer_step_days(ds: pd.Series) -> float:
    diffs = ds.sort_values().diff().dropna().dt.total_seconds()/86400.0
    if len(diffs)==0: return 1.0
    try:
        return float(diffs.mode().iloc[0])
    except Exception:
        return float(diffs.median())

def _fourier_matrix(T: int, step_days: float, cfg_list: list[dict]) -> np.ndarray:
    """cfg_list: [{'period': int, 'order': K} o {'period_days': float, 'order': K}]"""
    if not cfg_list: return np.zeros((T,0), dtype=np.float32)
    t = np.arange(T, dtype=np.float32)
    cols = []
    for c in cfg_list:
        if "period" in c:
            P = float(c["period"])              # en pasos
        elif "period_days" in c:
            P = float(c["period_days"])/max(step_days, 1e-6)  # a pasos
        else:
            continue
        K = int(c.get("order",1))
        for k in range(1, K+1):
            ang = 2*np.pi*k*t/P
            cols.append(np.sin(ang)); cols.append(np.cos(ang))
    return np.stack(cols, axis=1).T if cols else np.zeros((T,0), dtype=np.float32)

def _apply_window(x: np.ndarray, win: str | None):
    if win is None: return x
    if win.lower() == "hann":
        w = np.hanning(len(x))
        return x * w
    return x

def fft_sin_features_from_train(y: np.ndarray, top_k=4, scale_range=(0.0,1.0),
                                window: str | None = None,
                                extra_steps: int = 0, fit_len: int | None = None,
                                include_signal: bool = True):
    """
    Genera ondas senoidales (exógenas) a partir de FFT del tramo de entrenamiento.
    - y: serie (ya normalizada o no) de largo T
    - fit_len: usa y[:fit_len] para estimar frecuencias/amplitudes/fases (train only)
    - extra_steps: cuántos pasos futuros quieres (para 'predict', pon H)
    Devuelve:
      X_hist: [T, K]  (o [T, K+1] si include_signal)
      X_fut:  [H, K]  idem
    """
    T = len(y)
    N_fit = int(fit_len) if fit_len is not None else T
    N_fit = max(8, min(N_fit, T))  # sanidad

    y_fit = y[:N_fit].astype(np.float64)
    y_fit = _apply_window(y_fit, window)

    # FFT real positiva
    X = np.fft.rfft(y_fit)
    amps = np.abs(X)
    # ignorar componente DC (k=0) y tomar top_k
    idxs = np.argsort(amps)[::-1]
    idxs = [i for i in idxs if i != 0][:top_k]

    t_total = np.arange(T + extra_steps, dtype=np.float64)
    waves = []
    for k in idxs:
        # frecuencia normalizada a [0, 0.5] ciclos por muestra
        # rfft usa frecuencias k/N_fit
        freq = k / N_fit
        amp  = np.abs(X[k]) / N_fit
        phase= np.angle(X[k])
        wave = amp * np.cos(2*np.pi*freq*t_total + phase)
        waves.append(wave)

    if len(waves) == 0:
        X_total = np.zeros((T + extra_steps, 0), dtype=np.float32)
    else:
        X_total = np.vstack(waves).T.astype(np.float64)  # [T+H, K]
        # escalar cada columna con stats del tramo de train (evita leakage)
        K = X_total.shape[1]
        for j in range(K):
            scaler = MinMaxScaler(feature_range=scale_range)
            scaler.fit(X_total[:N_fit, j:j+1])  # fit solo en train
            X_total[:, j] = scaler.transform(X_total[:, j:j+1]).ravel()
        X_total = X_total.astype(np.float32)

    cols = [f"f_seno_{i+1}" for i in range(X_total.shape[1])]

    if include_signal:
        sig = X_total.sum(axis=1, keepdims=True) if X_total.size else np.zeros((T+extra_steps,1),np.float32)
        # escalar señal con train only
        scaler = MinMaxScaler(feature_range=scale_range)
        scaler.fit(sig[:N_fit])
        sig = scaler.transform(sig).astype(np.float32)
        X_total = np.concatenate([X_total, sig], axis=1)
        cols.append("fft_signal")

    X_hist = X_total[:T]
    X_fut  = X_total[T:]
    return X_hist, X_fut, cols

'''class WindowedTS(Dataset):
    """
    Si split_mode='series' (default): igual que antes.
    Si split_mode='time': hace holdout temporal por serie usando cutoff_idx_map y NO re-ajusta el scaler en validación.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 context_len: int,
                 horizon: int,
                 stride: int,
                 normalize: str = "zscore",
                 minmax_range=(0.0,1.0),
                 split_mode: str = "series",
                 part: str = "train",                 # 'train' o 'val' cuando split_mode='time'
                 cutoff_idx_map: Dict[str,int] | None = None):
        df = df.copy()
        assert {"ds","unique_id","y"}.issubset(df.columns), "CSV must have ds, unique_id, y"
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values(["unique_id","ds"]).reset_index(drop=True)
        self.context_len = context_len
        self.horizon = horizon
        self.stride = stride
        self.samples: List[Tuple[str, int, int]] = []
        self.series_data: Dict[str, Dict] = {}
        self.split_mode = split_mode
        self.part = part
        self.cutoff_idx_map = cutoff_idx_map or {}

        for uid, g in df.groupby("unique_id"):
            y = g["y"].to_numpy(dtype=np.float32)

            if self.split_mode == "time":
                # cutoff por índice (último índice de la parte de entrenamiento)
                cut = self.cutoff_idx_map.get(uid, len(y))
                cut = int(max(self.context_len + self.horizon, min(cut, len(y))))
                # fit scaler SOLO en la porción de entrenamiento
                scaler = SeriesScaler(kind=normalize, minmax_range=minmax_range)
                y_train_norm, params = scaler.fit_transform(y[:cut])
                # transforma TODO con los params del train
                y_norm = SeriesScaler.transform(y, params).astype(np.float32)
                scaler_params = params
            else:
                # como antes (fit por serie completa del dataset)
                scaler = SeriesScaler(kind=normalize, minmax_range=minmax_range)
                y_norm, scaler_params = scaler.fit_transform(y)
                y_norm = y_norm.astype(np.float32)

            self.series_data[uid] = {
                "y": y_norm,
                "scaler_params": scaler_params,
            }

            T = len(y_norm)
            start = self.context_len
            while start + self.horizon <= T:
                if self.split_mode == "time":
                    # Regla sin leakage:
                    #  - TRAIN: ventanas cuyo futuro termina ANTES o EN el corte
                    #  - VAL:   ventanas cuyo primer paso de futuro empieza DESPUÉS del corte
                    if self.part == "train":
                        if start + self.horizon <= cut:
                            self.samples.append((uid, start - self.context_len, start + self.horizon))
                    else:  # val
                        if start >= cut and start + self.horizon <= T:
                            self.samples.append((uid, start - self.context_len, start + self.horizon))
                else:
                    # split por series (comportamiento original)
                    self.samples.append((uid, start - self.context_len, start + self.horizon))
                start += self.stride

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        uid, a, b = self.samples[i]
        arr = self.series_data[uid]["y"]
        ctx = arr[a:a+self.context_len]
        fut = arr[a+self.context_len:b]
        return {
            "unique_id": uid,
            "context": torch.tensor(ctx).unsqueeze(-1),
            "future": torch.tensor(fut).unsqueeze(-1),
        }

    def inverse_future(self, uid: str, yhat: np.ndarray) -> np.ndarray:
        return SeriesScaler.inverse_transform(yhat, self.series_data[uid]["scaler_params"])
'''

class WindowedTS(Dataset):
    """
    Soporta exógenas:
      - time_features (sin/cos) y fourier (sin/cos)
    Split:
      - 'series': como antes (por unique_id)
      - 'time'  : holdout temporal por serie sin leakage (usa cutoff_idx_map)
    """
    def __init__(self,
                 df: pd.DataFrame,
                 context_len: int,
                 horizon: int,
                 stride: int,
                 normalize: str = "zscore",
                 minmax_range=(0.0,1.0),
                 split_mode: str = "series",
                 part: str = "train",                 # 'train' | 'val' si split_mode='time'
                 cutoff_idx_map: Dict[str,int] | None = None,
                 use_time_features: bool = False,
                 time_features: list[str] | None = None,
                 use_fourier: bool = False,
                 fourier_cfg: list[dict] | None = None,
                 cfg: list[dict] | None = None):
        df = df.copy()
        assert {"ds","unique_id","y"}.issubset(df.columns), "CSV must have ds, unique_id, y"
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values(["unique_id","ds"]).reset_index(drop=True)
        self.context_len = context_len
        self.horizon = horizon
        self.stride = stride
        self.samples: List[Tuple[str, int, int]] = []
        self.series_data: Dict[str, Dict] = {}
        self.split_mode = split_mode
        self.part = part
        self.cutoff_idx_map = cutoff_idx_map or {}
        self.d_exog = 0

        for uid, g in df.groupby("unique_id"):
            y = g["y"].to_numpy(dtype=np.float32)
            ds = g["ds"].reset_index(drop=True)
            T  = len(y)

            # --- scaler en train (si split temporal) ---
            if self.split_mode == "time":
                cut = int(self.cutoff_idx_map.get(uid, T))
                cut = int(max(self.context_len + self.horizon, min(cut, T)))
                scaler = SeriesScaler(kind=normalize, minmax_range=minmax_range)
                y_train_norm, params = scaler.fit_transform(y[:cut])
                y_norm = SeriesScaler.transform(y, params).astype(np.float32)
                scaler_params = params
            else:
                scaler = SeriesScaler(kind=normalize, minmax_range=minmax_range)
                y_norm, scaler_params = scaler.fit_transform(y)
                y_norm = y_norm.astype(np.float32)

            # --- exógenas determinísticas ---
            X_time = _time_feature_matrix(ds, time_features or []) if use_time_features else np.zeros((T,0),np.float32)
            step_days = _infer_step_days(ds)
            X_four = _fourier_matrix(T, step_days, fourier_cfg or []) if use_fourier else np.zeros((T,0),np.float32)
            # --- FFT features (data-driven) ---
            X_hist = np.zeros((T,0), dtype=np.float32)
            # _cfg_data
            #if cfg := getattr(self, "_cfg_data", None):  # lo seteará build_dataloaders
            #    print('Hola?')
            if cfg.get("use_fft_features", False):
                print('We are here')
                X_hist, _, _ = fft_sin_features_from_train(
                    y=y_norm,                                  # puedes usar y_norm o y (raw)
                    top_k=int(cfg.get("fft_top_k", 4)),
                    scale_range=tuple(cfg.get("fft_scale_range",[0.0,1.0])),
                    window=cfg.get("fft_window", None),
                    extra_steps=0,
                    fit_len=(cut if self.split_mode=="time" else T),
                    include_signal=bool(cfg.get("fft_include_signal", True)),
                )
                    
            print(X_hist.shape)
            print(X_time.shape)

            X = np.concatenate([X_time, X_hist], axis=1).astype(np.float32) if (X_time.size or X_hist.size) else np.zeros((T,0),np.float32)
            
            self.d_exog = max(self.d_exog, X.shape[1])

            self.series_data[uid] = {
                "y": y_norm,
                "X": X,                         # exógenas alignadas por tiempo
                "scaler_params": scaler_params,
                "ds": ds,
            }

            start = self.context_len
            while start + self.horizon <= T:
                if self.split_mode == "time":
                    if self.part == "train":
                        if start + self.horizon <= cut:
                            self.samples.append((uid, start - self.context_len, start + self.horizon))
                    else:
                        if start >= cut and start + self.horizon <= T:
                            self.samples.append((uid, start - self.context_len, start + self.horizon))
                else:
                    self.samples.append((uid, start - self.context_len, start + self.horizon))
                start += self.stride

    def __len__(self): return len(self.samples)

    def __getitem__(self, i: int):
        uid, a, b = self.samples[i]
        y = self.series_data[uid]["y"]
        X = self.series_data[uid]["X"]
        ctx_y = y[a:a+self.context_len]
        fut_y = y[a+self.context_len:b]
        ctx_x = X[a:a+self.context_len] if X.shape[1]>0 else np.zeros((self.context_len,0),dtype=np.float32)
        fut_x = X[a+self.context_len:b] if X.shape[1]>0 else np.zeros((self.horizon,0),dtype=np.float32)
        return {
            "unique_id": uid,
            "context": torch.tensor(ctx_y).unsqueeze(-1),
            "future":  torch.tensor(fut_y).unsqueeze(-1),
            "context_x": torch.tensor(ctx_x),
            "future_x":  torch.tensor(fut_x),
        }

    def inverse_future(self, uid: str, yhat: np.ndarray) -> np.ndarray:
        return SeriesScaler.inverse_transform(yhat, self.series_data[uid]["scaler_params"])


def build_dataloaders(csv_path: str,
                      context_len: int,
                      horizon: int,
                      stride: int,
                      cut_off_date: str,
                      from_date: str,
                      batch_size: int,
                      num_workers: int,
                      val_split: float,
                      normalize: str,
                      id_series: str | None = None,
                      minmax_range=(0.0,1.0),
                      split_mode: str = "series",
                      use_time_features: bool = False,
                      use_fft_features: bool = False, 
                      time_features: list[str] | None = None,
                      use_fourier: bool = False,
                      fourier_cfg: list[dict] | None = None,
                      cfg_data: list[dict] | None = None):
    
    df = pd.read_csv(csv_path)
    df = df[(df['ds']<= cut_off_date) & (df['ds']>= from_date)]
    if id_series == 'all' or id_series is None:
        print('Procesando todas las series temporales')
    else:
        df = df[df['unique_id']==id_series]

    if split_mode == "series":
        uids = sorted(df["unique_id"].unique().tolist())
        n_val = max(1, int(len(uids) * val_split))
        val_ids = set(uids[-n_val:])
        train_df = df[~df["unique_id"].isin(val_ids)].copy()
        val_df   = df[df["unique_id"].isin(val_ids)].copy()

        train_ds = WindowedTS(train_df, context_len, horizon, stride, normalize, minmax_range,
                              split_mode="series",
                              use_time_features=use_time_features, time_features=time_features,
                              use_fourier=use_fourier, fourier_cfg=fourier_cfg, cfg=cfg_data)
        
        val_ds   = WindowedTS(val_df,   context_len, horizon, stride, normalize, minmax_range,
                              split_mode="series",
                              use_time_features=use_time_features, time_features=time_features,
                              use_fourier=use_fourier, fourier_cfg=fourier_cfg, cfg=cfg_data)
        
    else:
        df["ds"] = pd.to_datetime(df["ds"])
        cutoff_idx_map = {}
        for uid, g in df.groupby("unique_id"):
            T = len(g)
            cut = int(np.floor((1.0 - float(val_split)) * T))
            cut = max(context_len + horizon, min(cut, T - 1))
            cutoff_idx_map[uid] = cut

        train_ds = WindowedTS(df, context_len, horizon, stride, normalize, minmax_range,
                              split_mode="time", part="train", cutoff_idx_map=cutoff_idx_map,
                              use_time_features=use_time_features, time_features=time_features,
                              use_fourier=use_fourier, fourier_cfg=fourier_cfg, cfg=cfg_data)
        
        val_ds   = WindowedTS(df, context_len, horizon, stride, normalize, minmax_range,
                              split_mode="time", part="val", cutoff_idx_map=cutoff_idx_map,
                              use_time_features=use_time_features, time_features=time_features,
                              use_fourier=use_fourier, fourier_cfg=fourier_cfg, cfg=cfg_data)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return train_ds, val_ds, train_dl, val_dl


'''def build_dataloaders(csv_path: str,
                      context_len: int,
                      cut_off_date: str,
                      from_date: str,
                      horizon: int,
                      stride: int,
                      batch_size: int,
                      num_workers: int,
                      val_split: float,
                      normalize: str,
                      minmax_range=(0.0,1.0),
                      split_mode: str = "series"):
    
    df = pd.read_csv(csv_path)
    df = df[(df['ds']<= cut_off_date) & (df['ds']>= from_date)]
    

    if split_mode == "series":
        # igual que antes
        uids = sorted(df["unique_id"].unique().tolist())
        n_val = max(1, int(len(uids) * val_split))
        val_ids = set(uids[-n_val:])
        train_df = df[~df["unique_id"].isin(val_ids)].copy()
        val_df   = df[df["unique_id"].isin(val_ids)].copy()

        train_ds = WindowedTS(train_df, context_len, horizon, stride, normalize, minmax_range, split_mode="series")
        val_ds   = WindowedTS(val_df,   context_len, horizon, stride, normalize, minmax_range, split_mode="series")

    else:  # split temporal por serie
        df["ds"] = pd.to_datetime(df["ds"])
        cutoff_idx_map = {}
        for uid, g in df.groupby("unique_id"):
            T = len(g)
            # corte en el (1 - val_split) de la longitud temporal
            cut = int(np.floor((1.0 - float(val_split)) * T))
            # Asegura que exista al menos 1 ventana de train y 1 de val si es posible
            cut = max(context_len + horizon, min(cut, T - 1))
            cutoff_idx_map[uid] = cut

        # Usamos el MISMO df; la clase filtra ventanas según 'part' y 'cutoff'
        train_ds = WindowedTS(df, context_len, horizon, stride, normalize, minmax_range,
                              split_mode="time", part="train", cutoff_idx_map=cutoff_idx_map)
        val_ds   = WindowedTS(df, context_len, horizon, stride, normalize, minmax_range,
                              split_mode="time", part="val",   cutoff_idx_map=cutoff_idx_map)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return train_ds, val_ds, train_dl, val_dl
'''