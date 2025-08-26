
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def kl_divergence_normal(mu_q, logvar_q, mu_p, logvar_p):
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    kl = 0.5 * ( (var_q / var_p).sum(dim=-1)
                + ((mu_p - mu_q).pow(2) / var_p).sum(dim=-1)
                - mu_q.size(-1)
                + (logvar_p - logvar_q).sum(dim=-1) )
    return kl

class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=(dropout if num_layers>1 else 0.0))
    def forward(self, x):
        out, h = self.rnn(x)
        return h[-1]

class GRUDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size=1, predict_sigma=False):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=(dropout if num_layers>1 else 0.0))
        self.out = nn.Linear(hidden_size, output_size * (2 if predict_sigma else 1))
        self.predict_sigma = predict_sigma
    def forward(self, x, h0=None):
        out, h = self.rnn(x, h0)
        out = self.out(out)
        return out, h

class CVAE(nn.Module):
    def __init__(self, input_size, context_len, horizon, latent_dim, enc_hidden, enc_layers,
                 dec_hidden, dec_layers, dropout=0.1, beta_kl=1.0, teacher_forcing=0.5,
                 predict_sigma=False, exog_size: int = 0):
        super().__init__()
        self.context_len = context_len
        self.horizon = horizon
        self.latent_dim = latent_dim
        self.beta_kl = beta_kl
        self.teacher_forcing = teacher_forcing
        self.predict_sigma = predict_sigma
        self.exog_size = exog_size  # <--- NUEVO

        # encoders ven y (+ exógenas) -> input_size == 1 + exog_size
        enc_in = 1 + self.exog_size
        self.ctx_enc  = GRUEncoder(enc_in, enc_hidden, enc_layers, dropout)
        self.post_enc = GRUEncoder(enc_in, enc_hidden, enc_layers, dropout)

        self.to_prior = nn.Linear(enc_hidden, 2*latent_dim)
        self.to_post  = nn.Linear(enc_hidden, 2*latent_dim)

        # decoder recibe y_prev, z, ctx_embed y exógena(t)
        self.dec_in_dim = 1 + latent_dim + enc_hidden + self.exog_size
        self.decoder = GRUDecoder(self.dec_in_dim, dec_hidden, dec_layers, dropout, output_size=1, predict_sigma=predict_sigma)

    def prior(self, ctx_embed):
        p = self.to_prior(ctx_embed); mu_p, logvar_p = torch.chunk(p, 2, dim=-1); return mu_p, logvar_p
    def posterior(self, ctxf_embed):
        q = self.to_post(ctxf_embed); mu_q, logvar_q = torch.chunk(q, 2, dim=-1); return mu_q, logvar_q
    def reparameterize(self, mu, logvar):
        std = (0.5*logvar).exp(); eps = torch.randn_like(std); return mu + eps * std

    def decode_autoregressive(self, z, ctx_embed, future, teacher_forcing, y0, future_x=None):
        B = z.size(0); device = z.device
        y_prev = y0 if y0 is not None else torch.zeros(B, 1, 1, device=device)
        outs, h = [], None
        for t in range(self.horizon):
            x_t = future_x[:, t:t+1, :] if (future_x is not None and self.exog_size>0) else torch.zeros(B,1,self.exog_size, device=device)
            dec_in = torch.cat([y_prev, z.unsqueeze(1), ctx_embed.unsqueeze(1), x_t], dim=-1)
            o, h = self.decoder(dec_in, h)
            if self.predict_sigma:
                mu_t, logvar_t = torch.chunk(o, 2, dim=-1)
                outs.append(torch.cat([mu_t, logvar_t], dim=-1)); y_hat = mu_t
            else:
                outs.append(o); y_hat = o
            use_tf = (torch.rand((), device=device) < teacher_forcing).item()
            y_prev = future[:, t:t+1, :] if self.training and use_tf else y_hat.detach()
        return torch.cat(outs, dim=1)

    def forward(self, context, future, context_x=None, future_x=None):
        # concat y + x para los encoders
        if self.exog_size>0 and context_x is not None:
            ctx_in  = torch.cat([context, context_x], dim=-1)
            fut_in  = torch.cat([future,  future_x],  dim=-1) if future_x is not None else future
        else:
            ctx_in, fut_in = context, future

        ctx_embed = self.ctx_enc(ctx_in)
        ctxf_embed= self.post_enc(torch.cat([ctx_in, fut_in], dim=1))

        mu_p,  logvar_p  = self.prior(ctx_embed)
        mu_q,  logvar_q  = self.posterior(ctxf_embed)

        z = self.reparameterize(mu_q, logvar_q)
        y0 = context[:, -1:, :]
        recon = self.decode_autoregressive(z, ctx_embed, future, self.teacher_forcing, y0, future_x=future_x)
        out = {"mu_p":mu_p, "logvar_p":logvar_p, "mu_q":mu_q, "logvar_q":logvar_q}
        return recon, out

    def loss(self, recon, future, aux):
        if self.predict_sigma:
            mu, logvar = torch.chunk(recon, 2, dim=-1)
            nll = 0.5*(math.log(2*math.pi) + logvar + (future - mu).pow(2) / logvar.exp())
            rec_loss = nll.mean()
        else:
            rec_loss = F.mse_loss(recon, future)
        kl = kl_divergence_normal(aux["mu_q"], aux["logvar_q"], aux["mu_p"], aux["logvar_p"]).mean()
        return rec_loss + self.beta_kl * kl, {"rec": rec_loss.detach(), "kl": kl.detach()}

    @torch.no_grad()
    def predict(self, context, context_x=None, future_x=None, samples: int = 100, deterministic: bool=False):
        self.eval()
        B = context.size(0); device = context.device
        if self.exog_size>0 and context_x is not None:
            ctx_in = torch.cat([context, context_x], dim=-1)
        else:
            ctx_in = context
        ctx_embed = self.ctx_enc(ctx_in)
        mu_p, logvar_p = self.prior(ctx_embed)

        draws = []
        for _ in range(samples):
            z = mu_p if deterministic else (mu_p + torch.randn_like(logvar_p).mul((0.5*logvar_p).exp()))
            H = self.horizon
            y_prev = context[:, -1:, :]
            outs, h = [], None
            for t in range(H):
                x_t = future_x[:, t:t+1, :] if (future_x is not None and self.exog_size>0) else torch.zeros(B,1,self.exog_size, device=device)
                dec_in = torch.cat([y_prev, z.unsqueeze(1), ctx_embed.unsqueeze(1), x_t], dim=-1)
                o, h = self.decoder(dec_in, h)
                if self.predict_sigma:
                    mu_t, _ = torch.chunk(o, 2, dim=-1); outs.append(mu_t); y_prev = mu_t
                else:
                    outs.append(o); y_prev = o
            draws.append(torch.cat(outs, dim=1).unsqueeze(0))
        S = torch.cat(draws, dim=0)
        return S.mean(dim=0), S.median(dim=0).values, S.quantile(0.10, dim=0), S.quantile(0.90, dim=0)


# ------------------------------
# D3VAE helpers
# ------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- beta schedules / diffusion buffers ---
def make_beta_schedule(T: int, schedule: str = "linear",
                       beta_start: float = 1e-4, beta_end: float = 2e-2, device=None):
    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, T, device=device)
    elif schedule == "cosine":
        # Nichol & Dhariwal cosine schedule (simplificada)
        s = 0.008
        steps = torch.arange(T+1, device=device, dtype=torch.float32)
        alphas_bar = torch.cos(((steps / T) + s) / (1+s) * math.pi/2) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        betas = betas.clamp(1e-5, 0.999)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    return betas

def _extract_at_t(v: torch.Tensor, t: torch.Tensor, shape):
    """Gather per-sample scalar from v[t] and reshape to broadcast to `shape`."""
    # v: [T], t: [B]
    out = v.gather(0, t).view(-1, *([1]*(len(shape)-1)))
    return out

def q_sample(x0: torch.Tensor, t: torch.Tensor, alphas_bar: torch.Tensor):
    """x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps"""
    B = x0.size(0)
    a_bar_t = _extract_at_t(alphas_bar, t, x0.shape)
    eps = torch.randn_like(x0)
    x_t = a_bar_t.sqrt() * x0 + (1. - a_bar_t).sqrt() * eps
    return x_t, eps, a_bar_t

# --- time embedding (sinusoidal) ---
class SinTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # frequencies
        half = dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half).float() / half))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, t: torch.Tensor):
        # t: [B] integer timesteps
        t = t.float().unsqueeze(1)  # [B,1]
        freqs = t * self.inv_freq.unsqueeze(0)  # [B,half]
        emb = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)  # [B,dim]
        if emb.size(-1) < self.dim:  # odd
            emb = F.pad(emb, (0, self.dim - emb.size(-1)))
        return emb  # [B,dim]

# --- simple per-timestep score head (MLP sobre y_t + emb_t) ---
class ScoreHead(nn.Module):
    def __init__(self, time_emb_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.lin1 = nn.Linear(1 + time_emb_dim, hidden)
        self.lin2 = nn.Linear(hidden, 1)
        self.act = nn.SiLU()
    def forward(self, y_t: torch.Tensor, t_emb: torch.Tensor):
        # y_t: [B,H,1], t_emb: [B, time_emb_dim] -> expand along H
        B, H, _ = y_t.shape
        t_expand = t_emb.unsqueeze(1).expand(B, H, t_emb.size(-1))
        x = torch.cat([y_t, t_expand], dim=-1)  # [B,H,1+temb]
        h = self.act(self.lin1(x))
        s = self.lin2(h)  # [B,H,1]
        return s

# --- log N(z | mu, diag(var)) ---
def log_normal_diag(z, mu, logvar):
    return -0.5 * (math.log(2*math.pi) + logvar + (z - mu).pow(2) / logvar.exp())

# --- minibatch TC estimator (β-TCVAE style, Chen et al. 2018) ---
def estimate_total_correlation(z, mu_q, logvar_q):
    """
    z:      [B,D] samples reparam from q(z|x)
    mu_q:   [B,D]
    logvar_q:[B,D]
    Returns scalar TC ~ E[ log q(z) - sum_i log q(z_i) ]
    """
    B, D = z.shape
    # log q(z_j | x_i) for all i,j => [B,B]
    # compute per-sample joint:
    # log q(z_j) ≈ log( 1/B * sum_i exp(sum_d log N(z_jd | mu_id, var_id)) )
    # build matrix of shape [B,B] with logits for each pair (i -> j)
    var_q = logvar_q.exp()
    # (z - mu)^2 / var  and logvar terms
    # expand to [B,B,D]
    z_j = z.unsqueeze(0)          # [1,B,D]
    mu_i = mu_q.unsqueeze(1)      # [B,1,D]
    logv_i = logvar_q.unsqueeze(1)# [B,1,D]

    log_probs = -0.5 * (math.log(2*math.pi) + logv_i + (z_j - mu_i).pow(2) / logv_i.exp())  # [B,B,D]
    log_qz_matrix = log_probs.sum(dim=-1)  # [B,B]
    # log q(z_j)
    log_qz = torch.logsumexp(log_qz_matrix, dim=0) - math.log(B)  # [B]

    # For product of marginals: sum over dimensions of log q(z_jd)
    # compute per-dimension log q(z_jd) ≈ log(1/B sum_i exp(logN))
    log_qz_prod = 0.0
    for d in range(D):
        log_probs_d = -0.5 * (math.log(2*math.pi) + logvar_q[:, d].unsqueeze(1)
                              + (z[:, d].unsqueeze(0) - mu_q[:, d].unsqueeze(1)).pow(2)
                                / logvar_q[:, d].unsqueeze(1).exp())  # [B,B]
        log_qzd = torch.logsumexp(log_probs_d, dim=0) - math.log(B)     # [B]
        log_qz_prod = log_qz_prod + log_qzd

    tc = (log_qz - log_qz_prod).mean()
    return tc

# ------------------------------
# D3VAE (wraps CVAE + diffusion + DSM + TC)
# ------------------------------
class D3VAE(nn.Module):
    """
    D3VAE: BVAE backbone (tu CVAE) + coupled diffusion (X/Y) + DSM + TC regularization.
    - Durante entrenamiento, difunde contexto (x) y futuro (y) en un nivel t ~ U{0..T-1}
    - Reconstruye y_t (objetivo noised) con el CVAE condicionado en x_t
    - Minimiza: rec + beta_kl*KL(q||p) + lambda_dsm*DSM + lambda_tc*TC
    - En predicción, usa el prior condicional y puede aplicar un 'denoising jump' 1-step
    """
    def __init__(self,
                 cvae: CVAE,
                 T: int = 50,
                 schedule: str = "linear",
                 beta_x=(1e-4, 2e-2),
                 beta_y=(1e-4, 2e-2),
                 time_emb_dim: int = 32,
                 dsm_weight: float = 0.1,
                 tc_weight: float = 0.0,
                 jump_gamma: float = 0.0,   # 0 => sin denoising jump
                 jump_t: int = 0):
        super().__init__()
        self.cvae = cvae
        self.T = T
        device = next(cvae.parameters()).device if list(cvae.parameters()) else None

        # schedules acopladas para X (contexto) y Y (futuro)
        self.register_buffer("betas_x", make_beta_schedule(T, schedule, beta_x[0], beta_x[1], device), persistent=False)
        self.register_buffer("betas_y", make_beta_schedule(T, schedule, beta_y[0], beta_y[1], device), persistent=False)
        self.register_buffer("alphas_x", 1.0 - self.betas_x, persistent=False)
        self.register_buffer("alphas_y", 1.0 - self.betas_y, persistent=False)
        self.register_buffer("alphas_bar_x", torch.cumprod(1.0 - self.betas_x, dim=0), persistent=False)
        self.register_buffer("alphas_bar_y", torch.cumprod(1.0 - self.betas_y, dim=0), persistent=False)

        # score net (DSM) + time embedding
        self.time_emb = SinTimeEmbedding(time_emb_dim)
        self.score_head = ScoreHead(time_emb_dim=time_emb_dim, hidden=64)

        # pesos
        self.dsm_weight = dsm_weight
        self.tc_weight = tc_weight

        # denoising jump en inferencia
        self.jump_gamma = jump_gamma
        self.jump_t = jump_t

    def sample_t(self, B, device):
        return torch.randint(0, self.T, (B,), device=device, dtype=torch.long)

    def dsm_loss(self, y_clean, y_t, t_idx):
        """
        DSM multiescala: E|| s_theta(y_t,t) - grad_{y_t} log q(y_t|y) ||^2
        Con difusión gaussiana: target score = -(y_t - sqrt(a_bar)*y)/(1 - a_bar)
        """
        a_bar = _extract_at_t(self.alphas_bar_y, t_idx, y_t.shape)   # [B,1,1] broadcast
        sigma2_t = (1. - a_bar)
        target = -(y_t - a_bar.sqrt() * y_clean) / (sigma2_t + 1e-8)   # [B,H,1]
        t_emb = self.time_emb(t_idx)                                   # [B,temb]
        s_hat = self.score_head(y_t, t_emb)                            # [B,H,1]
        return F.mse_loss(s_hat, target)

    def forward(self, context, future, context_x=None, future_x=None):
        """
        Entrenamiento: aplica difusión acoplada, llama al CVAE y calcula aux para pérdidas.
        Devuelve:
          recon_t: reconstrucción de y_t (noised) [B,H,1] o [B,H,2] si sigma
          logs: dict con prior/post, z, t y tensores auxiliares
        """
        B = context.size(0); device = context.device
        # 1) muestrea nivel de ruido
        t = self.sample_t(B, device)  # [B]

        # 2) difunde X (contexto) y Y (futuro)
        x_t, eps_x, a_bar_x = q_sample(context, t, self.alphas_bar_x)
        y_t, eps_y, a_bar_y = q_sample(future,  t, self.alphas_bar_y)

        # 3) llama al CVAE con entradas difusas
        recon_t, aux = self.cvae.forward(x_t, y_t, context_x=context_x, future_x=future_x)

        # 4) recolecta info adicional para TC (z muestrea del posterior en cvae.reparameterize)
        # No tenemos z explícito aquí; replicamos la muestrea para logs
        with torch.no_grad():
            z_post = self.cvae.reparameterize(aux["mu_q"], aux["logvar_q"])  # [B,D]

        logs = {
            "t": t,
            "x_t": x_t, "y_t": y_t,
            "a_bar_x": a_bar_x, "a_bar_y": a_bar_y,
            "mu_p": aux["mu_p"], "logvar_p": aux["logvar_p"],
            "mu_q": aux["mu_q"], "logvar_q": aux["logvar_q"],
            "z_post": z_post
        }
        return recon_t, logs

    def loss(self, recon_t, logs, future_clean):
        """
        Loss total = rec(y_t) + beta_kl*KL(q||p) + λ_dsm*DSM(y_t) + λ_tc*TC(z)
        - future_clean: y sin ruido, para el target del DSM
        """
        # --- reconstrucción: igual que CVAE pero contra y_t ---
        if self.cvae.predict_sigma:
            mu, logvar = torch.chunk(recon_t, 2, dim=-1)
            nll = 0.5 * (math.log(2*math.pi) + logvar + (logs["y_t"] - mu).pow(2) / logvar.exp())
            rec_loss = nll.mean()
        else:
            rec_loss = F.mse_loss(recon_t, logs["y_t"])

        # --- KL (como CVAE) ---
        kl = kl_divergence_normal(logs["mu_q"], logs["logvar_q"],
                                  logs["mu_p"], logs["logvar_p"]).mean()
        kl_term = self.cvae.beta_kl * kl

        # --- DSM ---
        dsm = self.dsm_loss(future_clean, logs["y_t"], logs["t"])

        # --- TC ---
        tc = estimate_total_correlation(logs["z_post"], logs["mu_q"], logs["logvar_q"]) if self.tc_weight > 0 else torch.tensor(0.0, device=recon_t.device)

        total = rec_loss + kl_term + self.dsm_weight * dsm + self.tc_weight * tc
        return total, {"rec": rec_loss.detach(),
                       "kl": kl.detach(),
                       "dsm": dsm.detach(),
                       "tc": tc.detach()}

    @torch.no_grad()
    def predict(self, context, context_x=None, future_x=None, samples: int = 100, deterministic: bool=False):
        """
        Predicción: delega a CVAE.predict; si jump_gamma>0 aplica un denoising jump 1-step
        usando el score en un nivel pequeño de ruido (jump_t).
        """
        mean, median, p10, p90 = self.cvae.predict(context, context_x=context_x, future_x=future_x,
                                                   samples=samples, deterministic=deterministic)

        if self.jump_gamma > 0:
            B, H, C = mean.shape
            device = mean.device
            # aplica score en y_hat tratado como y_t con t=jump_t
            t_idx = torch.full((B,), int(self.jump_t), device=device, dtype=torch.long)
            a_bar = _extract_at_t(self.alphas_bar_y, t_idx, mean.shape)   # [B,1,1]
            # sin y_clean en inferencia, usamos 'score-only' denoise: y <- y + gamma * sigma^2 * s_theta(y,t)
            t_emb = self.time_emb(t_idx)                                   # [B,temb]
            s_hat = self.score_head(mean, t_emb)                           # [B,H,1]
            sigma2 = (1. - a_bar)
            mean = mean + self.jump_gamma * sigma2 * s_hat
            median = median + self.jump_gamma * sigma2 * self.score_head(median, t_emb)
            p10 = p10 + self.jump_gamma * sigma2 * self.score_head(p10, t_emb)
            p90 = p90 + self.jump_gamma * sigma2 * self.score_head(p90, t_emb)

        return mean, median, p10, p90
