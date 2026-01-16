
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import yaml
import torch
from .data import SeriesScaler
from .data import SeriesScaler, fft_sin_features_from_train  # importa helper
from .data import _time_feature_matrix, _infer_step_days, _fourier_matrix
from .model import CVAE, D3VAE   # <-- antes solo CVAE

'''def load_model(ckpt: str, device: torch.device):
    import inspect
    obj = torch.load(ckpt, map_location=device)
    cfg = obj["config"]; state = obj["state_dict"]

    # 1) Inferir el input_size entrenado desde los pesos del encoder
    w_enc = state.get("ctx_enc.rnn.weight_ih_l0", None)
    if w_enc is None:
        # fallback por si cambia el nombre
        for k, v in state.items():
            if k.endswith("ctx_enc.rnn.weight_ih_l0"):
                w_enc = v; break
    if w_enc is None:
        # 2) Fallback desde el decoder (dec_in = 1 + latent + enc_hidden + exog)
        w_dec = state.get("decoder.rnn.weight_ih_l0", None)
        if w_dec is None:
            for k, v in state.items():
                if k.endswith("decoder.rnn.weight_ih_l0"):
                    w_dec = v; break
        if w_dec is None:
            raise RuntimeError("No pude inferir el input_size desde el checkpoint.")
        dec_in_dim = int(w_dec.shape[1])
        latent = int(cfg["model"]["latent_dim"])
        enc_h  = int(cfg["model"]["enc_hidden"])
        exog_size = dec_in_dim - (1 + latent + enc_h)
        trained_input_size = 1 + exog_size
    else:
        trained_input_size = int(w_enc.shape[1])   # = 1 + exog_size
        exog_size = trained_input_size - 1

    # 3) Construir el CVAE compatible
    ctor = {
        "context_len": cfg["data"]["context_len"],
        "horizon":     cfg["data"]["horizon"],
        "latent_dim":  cfg["model"]["latent_dim"],
        "enc_hidden":  cfg["model"]["enc_hidden"],
        "enc_layers":  cfg["model"]["enc_layers"],
        "dec_hidden":  cfg["model"]["dec_hidden"],
        "dec_layers":  cfg["model"]["dec_layers"],
        "dropout":     cfg["model"].get("dropout", 0.1),
        "beta_kl":     cfg["model"].get("beta_kl", 1.0),
        "teacher_forcing": 0.0,
        "predict_sigma":   cfg["model"].get("predict_sigma", False),
    }

    # Soportar ambas variantes del CVAE:
    if "exog_size" in inspect.signature(CVAE.__init__).parameters:
        # versión nueva (y + exógenas separadas)
        ctor["input_size"] = 1
        ctor["exog_size"]  = max(0, exog_size)
    else:
        # versión antigua (todo via input_size)
        ctor["input_size"] = trained_input_size

    m = CVAE(**ctor).to(device)
    m.load_state_dict(state)
    m.eval()
    # Mensaje útil para el log
    # Guardar en el modelo cuántas exógenas espera (para usarlo en prepare_contexts)
    m._expected_exog = exog_size
    try:
        print(f"[load_model] Detecté input_size entrenado={trained_input_size} (exog_size={exog_size}).")
    except Exception:
        pass
    return m, cfg'''



def load_model(ckpt: str, device: torch.device):
    import inspect
    obj = torch.load(ckpt, map_location=device)
    cfg = obj["config"]; state = obj["state_dict"]
    model_type = obj.get("model_type", None)

    # ¿Es un checkpoint de D3VAE? (por metadata o por prefijo 'cvae.')
    is_d3 = (model_type == "d3vae") or any(k.startswith("cvae.") for k in state.keys())

    # ---- Inferir exog_size mirando el encoder/decoder (tanto plano como cvae.*) ----
    def _get(shape_key_options):
        for k in shape_key_options:
            if k in state: return int(state[k].shape[1])
        # fallback por sufijo
        for k,v in state.items():
            for suf in shape_key_options:
                if k.endswith(suf): return int(v.shape[1])
        return None

    # pesos posibles para encoder (primera capa)
    enc_keys = ["ctx_enc.rnn.weight_ih_l0", "cvae.ctx_enc.rnn.weight_ih_l0"]
    dec_keys = ["decoder.rnn.weight_ih_l0", "cvae.decoder.rnn.weight_ih_l0"]

    enc_in = _get(enc_keys)
    if enc_in is None:
        dec_in = _get(dec_keys)
        if dec_in is None:
            raise RuntimeError("No pude inferir el input_size desde el checkpoint.")
        latent = int(cfg["model"]["latent_dim"])
        enc_h  = int(cfg["model"]["enc_hidden"])
        exog_size = dec_in - (1 + latent + enc_h)
        trained_input_size = 1 + exog_size
    else:
        trained_input_size = enc_in  # = 1 + exog_size
        exog_size = trained_input_size - 1

    # ---- Construir CVAE base ----
    ctor = {
        "context_len": cfg["data"]["context_len"],
        "horizon":     cfg["data"]["horizon"],
        "latent_dim":  cfg["model"]["latent_dim"],
        "enc_hidden":  cfg["model"]["enc_hidden"],
        "enc_layers":  cfg["model"]["enc_layers"],
        "dec_hidden":  cfg["model"]["dec_hidden"],
        "dec_layers":  cfg["model"]["dec_layers"],
        "dropout":     cfg["model"].get("dropout", 0.1),
        "beta_kl":     cfg["model"].get("beta_kl", 1.0),
        "teacher_forcing": 0.0,
        "predict_sigma":   cfg["model"].get("predict_sigma", False),
    }

    if "exog_size" in inspect.signature(CVAE.__init__).parameters:
        ctor["input_size"] = 1
        ctor["exog_size"]  = max(0, exog_size)
    else:
        ctor["input_size"] = trained_input_size

    cvae = CVAE(**ctor).to(device)

    # ---- Envolver como D3VAE si aplica ----
    if is_d3:
        d3 = cfg["model"].get("d3vae", {})
        model = D3VAE(
            cvae=cvae,
            T=d3.get("T", 50),
            schedule=d3.get("schedule", "linear"),
            beta_x=tuple(d3.get("beta_x", (1e-4, 2e-2))),
            beta_y=tuple(d3.get("beta_y", (1e-4, 2e-2))),
            time_emb_dim=d3.get("time_emb_dim", 32),
            dsm_weight=d3.get("dsm_weight", 0.1),
            tc_weight=d3.get("tc_weight", 0.0),
            jump_gamma=d3.get("jump_gamma", 0.0),
            jump_t=d3.get("jump_t", 0),
        ).to(device)
    else:
        model = cvae

    model.load_state_dict(state)
    model.eval()

    # guardar exog esperadas para preparar contextos
    try:
        setattr(model, "_expected_exog", exog_size)
        if hasattr(model, "cvae"):
            setattr(model.cvae, "_expected_exog", exog_size)
        #print(f"[load_model] Detecté input_size entrenado={1+exog_size} (exog_size={exog_size}). Tipo={'D3VAE' if is_d3 else 'CVAE'}.")
    except Exception:
        pass

    return model, cfg



def get_device(flag: str):
    if flag == "cpu": return torch.device("cpu")
    if flag == "cuda": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_exog_from_ds(ds_series: pd.Series, H: int, cfg_data: dict):
    ds_series = pd.to_datetime(ds_series).reset_index(drop=True)
    # 1) tiempo presentes
    X_time = _time_feature_matrix(ds_series, cfg_data.get("time_features", [])) if cfg_data.get("use_time_features", False) else np.zeros((len(ds_series),0),np.float32)
    # 2) fourier presentes
    step_days = _infer_step_days(ds_series)
    X_four = _fourier_matrix(len(ds_series), step_days, cfg_data.get("fourier", [])) if cfg_data.get("use_fourier", False) else np.zeros((len(ds_series),0),np.float32)
    X_hist = np.concatenate([X_time, X_four], axis=1) if (X_time.size or X_four.size) else np.zeros((len(ds_series),0),np.float32)

    # Fechas futuras
    user_freq = cfg_data.get("freq", None)
    if user_freq:
        fut_ds = pd.date_range(start=ds_series.iloc[-1], periods=H+1, freq=user_freq)[1:]
    else:
        try:
            f = pd.infer_freq(ds_series)
            fut_ds = pd.date_range(start=ds_series.iloc[-1], periods=H+1, freq=f)[1:]
        except Exception:
            delta = (ds_series.diff().dropna()).mode()
            delta = delta.iloc[0] if not delta.empty else pd.Timedelta(days=1)
            fut_ds = [ds_series.iloc[-1] + (h+1)*delta for h in range(H)]
            fut_ds = pd.to_datetime(fut_ds)

    # Exógenas futuras
    X_time_f = _time_feature_matrix(pd.Series(fut_ds), cfg_data.get("time_features", [])) if cfg_data.get("use_time_features", False) else np.zeros((H,0),np.float32)
    X_four_f = _fourier_matrix(H, step_days, cfg_data.get("fourier", [])) if cfg_data.get("use_fourier", False) else np.zeros((H,0),np.float32)
    X_fut = np.concatenate([X_time_f, X_four_f], axis=1) if (X_time_f.size or X_four_f.size) else np.zeros((H,0),np.float32)
    return X_hist.astype(np.float32), X_fut.astype(np.float32), pd.Series(fut_ds)


def prepare_contexts(df: pd.DataFrame, context_len: int, cfg_data: dict, expected_exog: int):
    """
    Devuelve:
      Xy: [B, Tc, 1]
      Xx: [B, Tc, D_exog] (padd/trunc a expected_exog)
      Fx: [B, H,  D_exog] (padd/trunc a expected_exog)
      meta: [(uid, scaler_params, last_ds, ds_series)]
    """
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(["unique_id","ds"])
    ctx_y, ctx_x, fut_x, meta = [], [], [], []
    H = cfg_data["horizon"]

    use_time = bool(cfg_data.get("use_time_features", False))
    use_four = bool(cfg_data.get("use_fourier", False))
    use_fft  = bool(cfg_data.get("use_fft_features", False))

    for uid, g in df.groupby("unique_id"):
        y = g["y"].to_numpy(dtype=np.float32)
        scaler = SeriesScaler(kind=cfg_data.get("normalize","zscore"),
                              minmax_range=tuple(cfg_data.get("minmax_range",[0.0,1.0])))
        y_norm, params = scaler.fit_transform(y)
        if len(y_norm) < context_len: 
            continue

        # --- TIME FEATURES (hist + fut) ---
        X_time_hist = np.zeros((len(g),0), np.float32)
        X_time_fut  = np.zeros((H,0), np.float32)
        if use_time:
            X_time_hist = _time_feature_matrix(g["ds"], cfg_data.get("time_features", []))
            # futuras fechas (usamos build_exog_from_ds para fechas/frecuencia)
            _, _, fut_ds = build_exog_from_ds(g["ds"], H=H, cfg_data=cfg_data)
            X_time_fut = _time_feature_matrix(pd.Series(fut_ds), cfg_data.get("time_features", []))

        # --- FOURIER MANUAL (hist + fut) ---
        X_four_hist = np.zeros((len(g),0), np.float32)
        X_four_fut  = np.zeros((H,0), np.float32)
        if use_four:
            
            step_days = _infer_step_days(g["ds"])
            X_four_hist = _fourier_matrix(len(g), step_days, cfg_data.get("fourier", []))
            X_four_fut  = _fourier_matrix(H,      step_days, cfg_data.get("fourier", []))

        # --- FFT DATA-DRIVEN (hist + fut) ---
        X_fft_hist = np.zeros((len(g),0), np.float32)
        X_fft_fut  = np.zeros((H,0), np.float32)
        if use_fft:
            X_fft_hist, X_fft_fut, _ = fft_sin_features_from_train(
                y=y_norm,  # usar normalizada está ok
                top_k=int(cfg_data.get("fft_top_k", 4)),
                scale_range=tuple(cfg_data.get("fft_scale_range",[0.0,1.0])),
                window=cfg_data.get("fft_window", None),
                extra_steps=H,
                fit_len=len(y_norm),  # historia disponible
                include_signal=bool(cfg_data.get("fft_include_signal", True)),
            )

        #print(X_time_hist.shape)
        #print(X_fft_hist.shape)
        #print(X_four_hist.shape)

        # --- CONCAT EXÓGENAS ---
        X_hist = np.concatenate([Z for Z in [X_time_hist, #X_four_hist, 
                                            X_fft_hist] if Z.shape[1]>0], axis=1) \
                 if (use_time or use_four or use_fft) else np.zeros((len(g),0), np.float32)
                
        X_fut  = np.concatenate([Z for Z in [X_time_fut,  #X_four_fut,  
                                            X_fft_fut ] if Z.shape[1]>0], axis=1) \
                 if (use_time or use_four or use_fft) else np.zeros((H,0), np.float32)

        # Asegurar dimensiones esperadas por el checkpoint
        #X_hist = _pad_or_trunc(X_hist, expected_exog)
        #X_fut  = _pad_or_trunc(X_fut,  expected_exog)

        #print(X_hist.shape)
        #print(X_fut.shape)

        # --- Cortar contexto y preparar tensores ---
        cx_y = y_norm[-context_len:]
        cx_x = X_hist[-context_len:] if expected_exog>0 else np.zeros((context_len,0), np.float32)

        ctx_y.append(torch.tensor(cx_y).unsqueeze(-1))
        ctx_x.append(torch.tensor(cx_x) if expected_exog>0 else torch.zeros((context_len,0)))
        fut_x.append(torch.tensor(X_fut) if expected_exog>0 else torch.zeros((H,0)))
        meta.append((uid, params, g["ds"].iloc[-1], g["ds"].reset_index(drop=True)))

    Xy = torch.stack(ctx_y, dim=0)
    Xx = torch.stack(ctx_x, dim=0) if expected_exog>0 else torch.zeros((len(ctx_y), context_len, 0))
    Fx = torch.stack(fut_x, dim=0) if expected_exog>0 else torch.zeros((len(ctx_y), H, 0))
    return Xy, Xx, Fx, meta

def main(config_path: str, ckpt_path: str, out_path: str, samples: int, device_flag: str):
    import pandas as pd
    device = get_device(device_flag); model, cfg = load_model(ckpt_path, device)
    expected_exog = int(getattr(model, "_expected_exog", 0))
    df = pd.read_csv(cfg["data"]["csv_path"])
    df = df[ (df['ds']<= cfg["data"]["cutoff"]) & ( df['ds']>= cfg["data"]["from_date"] )]
    if cfg["data"]['id'] == 'all' or cfg["data"]['id'] is None:
        print('Procesando todas las series temporales')
    else:
        df = df[df['unique_id']==cfg["data"]['id']]

    Xy, Xx, Fx, meta = prepare_contexts(
        df,
        context_len=cfg["data"]["context_len"],
        cfg_data=cfg["data"],
        expected_exog=expected_exog,
    )

    Xy = Xy.to(device)
    Xx = Xx.to(device) if Xx.numel()>0 else None
    Fx = Fx.to(device) if Fx.numel()>0 else None

    mean, median, p10, p90 = model.predict(Xy, context_x=Xx, future_x=Fx, samples=samples, deterministic=False)

    rows = []
    H = cfg["data"]["horizon"]
    user_freq = cfg["data"].get("freq", None)  # opcional en YAML

    for i, (uid, scaler_params, last_ds, ds_series) in enumerate(meta):
        m  = SeriesScaler.inverse_transform(mean[i].cpu().numpy().reshape(H),   scaler_params)
        md = SeriesScaler.inverse_transform(median[i].cpu().numpy().reshape(H), scaler_params)
        lo = SeriesScaler.inverse_transform(p10[i].cpu().numpy().reshape(H),    scaler_params)
        hi = SeriesScaler.inverse_transform(p90[i].cpu().numpy().reshape(H),    scaler_params)
        '''for h in range(H):
            rows.append({
                "unique_id": uid,
                "ds": pd.Timestamp(fut_ds.iloc[h]),
                "mean": float(m[h]), "median": float(md[h]),
                "p10": float(lo[h]), "p90": float(hi[h]),
            })
    out_df = pd.DataFrame(rows); out_df.to_csv(out_path, index=False)
    print(f"Saved forecasts to {out_path}")'''
        
        # 1) frecuencia desde config, 2) inferida, 3) delta modal como fallback
        freq = user_freq
        if freq is None:
            try:
                import pandas as pd
                freq = pd.infer_freq(ds_series)
            except Exception:
                freq = None

        if freq:
            # construye rango a partir del siguiente punto
            fut_ds = pd.date_range(start=last_ds, periods=H+1, freq=freq)[1:]
        else:
            # fallback robusto: usar delta modal (o mediana) de las diferencias
            diffs = (ds_series.diff().dropna())
            if not diffs.empty:
                # modo si existe, si no, mediana
                try:
                    delta = diffs.mode().iloc[0]
                except Exception:
                    delta = diffs.median()
            else:
                delta = pd.Timedelta(days=1)
            fut_ds = [last_ds + (h+1)*delta for h in range(H)]

        for h in range(H):
            rows.append({
                "unique_id": uid,
                "ds": pd.Timestamp(fut_ds[h]),
                "mean": float(m[h]),
                "median": float(md[h]),
                "p10": float(lo[h]),
                "p90": float(hi[h]),
            })

    out_df = pd.DataFrame(rows); out_df.to_csv(out_path, index=False); 
    #print(f"Saved forecasts to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    main(args.config, args.ckpt, args.out, args.samples, args.device)


