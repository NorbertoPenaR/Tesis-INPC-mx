from __future__ import annotations
import argparse, os, random
import yaml
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from .data import build_dataloaders
from .model import CVAE, D3VAE  # <-- antes solo CVAE

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_device(flag: str):
    if flag == "cpu": return torch.device("cpu")
    if flag == "cuda": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _as_list(x):
    if isinstance(x, (list, tuple)): return list(x)
    if isinstance(x, str): return [x]
    return ["cvae"]

def _build_cvae(model_cfg, data_cfg, exog_size, device):
    return CVAE(
        input_size=1,
        context_len=data_cfg["context_len"],
        horizon=data_cfg["horizon"],
        latent_dim=model_cfg["latent_dim"],
        enc_hidden=model_cfg["enc_hidden"],
        enc_layers=model_cfg["enc_layers"],
        dec_hidden=model_cfg["dec_hidden"],
        dec_layers=model_cfg["dec_layers"],
        dropout=model_cfg.get("dropout",0.1),
        beta_kl=model_cfg.get("beta_kl",1.0),
        teacher_forcing=model_cfg.get("teacher_forcing",0.5),
        predict_sigma=model_cfg.get("predict_sigma", False),
        exog_size=exog_size,
    ).to(device)

def _build_d3vae(model_cfg, data_cfg, exog_size, device):
    cvae = _build_cvae(model_cfg, data_cfg, exog_size, device)
    d3 = model_cfg.get("d3vae", {})
    return D3VAE(
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

def _forward_and_loss_train(model, batch, device):
    ctx = batch["context"].to(device); fut = batch["future"].to(device)
    ctx_x = batch.get("context_x"); fut_x = batch.get("future_x")
    if ctx_x is not None: ctx_x = ctx_x.to(device)
    if fut_x is not None: fut_x = fut_x.to(device)

    if isinstance(model, D3VAE):
        recon_t, logs = model(ctx, fut, context_x=ctx_x, future_x=fut_x)
        loss, parts = model.loss(recon_t, logs, future_clean=fut)
    else:
        recon, aux = model(ctx, fut, context_x=ctx_x, future_x=fut_x)
        loss, parts = model.loss(recon, fut, aux)
    return loss, parts

@torch.no_grad()
def _forward_and_loss_val(model, batch, device):
    ctx = batch["context"].to(device); fut = batch["future"].to(device)
    ctx_x = batch.get("context_x"); fut_x = batch.get("future_x")
    if ctx_x is not None: ctx_x = ctx_x.to(device)
    if fut_x is not None: fut_x = fut_x.to(device)

    if isinstance(model, D3VAE):
        recon_t, logs = model(ctx, fut, context_x=ctx_x, future_x=fut_x)
        loss, parts = model.loss(recon_t, logs, future_clean=fut)
    else:
        recon, aux = model(ctx, fut, context_x=ctx_x, future_x=fut_x)
        loss, parts = model.loss(recon, fut, aux)
    return loss, parts

def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    data_cfg, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["train"]

    set_seed(train_cfg.get("seed", 1337))
    device = get_device(train_cfg.get("device","auto"))
    root_save = train_cfg.get("save_dir", "runs/dvae"); os.makedirs(root_save, exist_ok=True)

    # dataloaders (idéntico a tu script actual)
    train_ds, val_ds, train_dl, val_dl = build_dataloaders(
        csv_path=data_cfg["csv_path"],
        context_len=data_cfg["context_len"],
        horizon=data_cfg["horizon"],
        cut_off_date=data_cfg['cutoff'],
        from_date = data_cfg['from_date'],
        stride=data_cfg["stride"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        val_split=data_cfg["val_split"],
        normalize=data_cfg.get("normalize","zscore"),
        minmax_range=tuple(data_cfg.get("minmax_range",[0.0,1.0])),
        split_mode=data_cfg.get("split_mode","series"),
        use_time_features=data_cfg.get("use_time_features", False),
        time_features=data_cfg.get("time_features", []),
        use_fourier=data_cfg.get("use_fourier", False),
        fourier_cfg=data_cfg.get("fourier", []),
        cfg_data=data_cfg,
        id_series=data_cfg["id"],
    )

    # exógenas
    try:
        sample = next(iter(train_dl))
        exog_size = int(sample.get("context_x", torch.zeros(1,1,0)).shape[-1])
    except StopIteration:
        exog_size = 0

    # qué modelos entrenar
    models_to_train = _as_list(data_cfg.get("models_to_train", "cvae"))

    for name in models_to_train:
        name = name.lower()
        if name == "cvae":
            model = _build_cvae(model_cfg, data_cfg, exog_size, device)
        elif name == "d3vae":
            model = _build_d3vae(model_cfg, data_cfg, exog_size, device)
        else:
            raise ValueError(f"Modelo no soportado: {name}")

        # annealings sobre el núcleo CVAE
        core = model.cvae if isinstance(model, D3VAE) else model
        kl_target = train_cfg.get("beta_kl", core.beta_kl)
        warmup_epochs = train_cfg.get("kl_warmup_epochs", 10)
        tf_start = train_cfg.get("teacher_forcing_start", core.teacher_forcing)
        tf_end   = train_cfg.get("teacher_forcing_end",   0.2)

        opt = optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
        best_val = float("inf"); patience = 0; max_patience = train_cfg.get("early_stop_patience", 10)
        epochs = train_cfg["epochs"]

        save_dir = os.path.join(root_save, name); os.makedirs(save_dir, exist_ok=True)
        print(f"\n=== Entrenando {name.upper()} ===")

        for epoch in range(1, epochs+1):
            if warmup_epochs > 0:
                core.beta_kl = kl_target * min(1.0, epoch / warmup_epochs)

            if tf_start is not None and tf_end is not None:
                frac = min(1.0, epoch / max(1, epochs))
                core.teacher_forcing = tf_start + (tf_end - tf_start) * frac

            model.train(); losses=[]
            for batch in tqdm(train_dl, desc=f"{name} Epoch {epoch}/{epochs} [train]"):
                opt.zero_grad()
                loss, parts = _forward_and_loss_train(model, batch, device)
                loss.backward()
                if train_cfg.get("grad_clip",0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg["grad_clip"])
                opt.step()
                losses.append(loss.item())

            model.eval(); vlosses=[]
            with torch.no_grad():
                for batch in tqdm(val_dl, desc=f"{name} Epoch {epoch}/{epochs} [valid]"):
                    vloss, _ = _forward_and_loss_val(model, batch, device)
                    vlosses.append(vloss.item())
            vmean = float(np.mean(vlosses))

            if vmean < best_val - 1e-6:
                best_val = vmean; patience = 0
                ckpt_path = os.path.join(save_dir, "best.ckpt")
                torch.save({"state_dict": model.state_dict(), "config": cfg, "model_type": name}, ckpt_path)
                print(f"[{name}] Saved checkpoint to {ckpt_path}  (val={vmean:.6f})")
            else:
                patience += 1
                print(f"[{name}] val={vmean:.6f} | best={best_val:.6f} | patience={patience}/{max_patience}")
                if patience >= max_patience:
                    print(f"[{name}] Early stopping. Best val={best_val:.6f}")
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
