# container3_dynamics/c3_conditioned_attention.py
# Global conditioning token + cross-asset attention + GRU dynamics
# JAX + Haiku, XLA-jittable, static-shape (no retracing as long as N,D,G fixed)

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import haiku as hk


jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class C3Config:
    n_assets: int          # N
    feature_dim: int       # D_in  (features_z_plus dim)
    global_dim: int        # G     (ext_global_z dim)
    d_model: int           # transformer width
    num_heads: int         # attention heads
    num_layers: int        # transformer blocks
    mlp_mult: int          # MLP expansion (e.g. 4)
    rnn_hidden: int        # GRU hidden size
    out_alpha_dim: int = 1 # alpha per asset (or whatever head you want)
    dtype: jnp.dtype = jnp.float64
    eps: float = 1e-12


# -----------------------------
# Transformer block (token-wise)
# -----------------------------
class TransformerBlock(hk.Module):
    def __init__(self, cfg: C3Config, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [B, L, d_model]
        cfg = self.cfg
        d = cfg.d_model

        ln1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="ln1")
        ln2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="ln2")

        mha = hk.MultiHeadAttention(
            num_heads=cfg.num_heads,
            key_size=d // cfg.num_heads,
            model_size=d,
            name="mha",
        )

        h = ln1(x)
        a = mha(h, h, h)          # self-attn
        x = x + a                 # residual

        h = ln2(x)
        mlp = hk.Sequential([
            hk.Linear(cfg.mlp_mult * d, name="fc1"),
            jax.nn.gelu,
            hk.Linear(d, name="fc2"),
        ])
        m = mlp(h)
        x = x + m                 # residual
        return x


# -------------------------------------------------------
# C3 step: global token conditions attention (no retracing)
# -------------------------------------------------------
def c3_forward_step(
    x_t: jnp.ndarray,      # [N, D_in]     (C2 features_z_plus at time t)
    g_t: jnp.ndarray,      # [G]           (C2 ext_global_z at time t)
    h_prev: jnp.ndarray,   # [N, H]        (previous hidden state)
    cfg: C3Config,
) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:
    """
    Returns:
      outputs: dict of per-asset heads + optional diagnostics
      h_next : [N, H]
    """
    dt = cfg.dtype
    eps = jnp.asarray(cfg.eps, dtype=dt)

    x_t = jnp.asarray(x_t, dtype=dt)
    g_t = jnp.asarray(g_t, dtype=dt)
    h_prev = jnp.asarray(h_prev, dtype=dt)

    # ---- 1) Embed per-asset features to model width
    x_emb = hk.Linear(cfg.d_model, name="x_embed")(x_t)  # [N, d_model]

    # ---- 2) Build GLOBAL CONDITIONING TOKEN (single token)
    # shape: [d_model]
    g_tok = hk.Sequential([
        hk.Linear(cfg.d_model, name="g_fc1"),
        jax.nn.tanh,
        hk.Linear(cfg.d_model, name="g_fc2"),
    ])(g_t)

    # ---- 3) Concatenate token + assets into attention sequence (static length L=N+1)
    # tokens: [L, d_model] with token at index 0
    tokens = jnp.concatenate([g_tok[None, :], x_emb], axis=0)  # [N+1, d_model]

    # Add batch dim for Haiku MHA: [B, L, d_model]
    tok = tokens[None, :, :]

    # ---- 4) Token-conditioned cross-asset attention stack
    for i in range(cfg.num_layers):
        tok = TransformerBlock(cfg, name=f"block_{i}")(tok)

    tok = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="final_ln")(tok)
    tok = tok[0]  # [L, d_model]

    g_ctx = tok[0]      # [d_model]  (contextualized global token)
    x_ctx = tok[1:]     # [N, d_model] (contextualized per-asset embeddings)

    # Optional: FiLM-style modulation using global token (cheap + strong)
    gamma = hk.Linear(cfg.d_model, name="film_gamma")(g_ctx)  # [d_model]
    beta  = hk.Linear(cfg.d_model, name="film_beta")(g_ctx)   # [d_model]
    x_ctx = x_ctx * (F64(1.0) + gamma[None, :]) + beta[None, :]

    # ---- 5) Dynamics core (GRU) over assets as batch
    # GRU input: concatenate asset ctx + broadcast global ctx
    g_b = jnp.broadcast_to(g_ctx[None, :], (cfg.n_assets, cfg.d_model))  # [N, d_model]
    rnn_in = jnp.concatenate([x_ctx, g_b], axis=1)  # [N, 2*d_model]

    rnn_in = hk.Linear(cfg.rnn_hidden, name="rnn_in_proj")(rnn_in)
    rnn_in = jax.nn.gelu(rnn_in)

    gru = hk.GRU(cfg.rnn_hidden, name="gru")
    h_next, _ = gru(rnn_in, h_prev)  # output==state for GRUCore semantics

    # ---- 6) Heads (example: alpha + (optional) regime logits)
    alpha = hk.Linear(cfg.out_alpha_dim, name="head_alpha")(h_next)  # [N,1]
    alpha = alpha[:, 0] if cfg.out_alpha_dim == 1 else alpha

    # A small regime/logit head (optional) conditioned on global token
    # If you don’t want it, delete.
    regime_logits = hk.Linear(8, name="head_regime")(g_ctx)  # [R] with fixed R=8 here

    outputs = {
        "alpha": alpha,                 # [N] or [N,k]
        "h_next": h_next,               # [N,H]
        "g_ctx": g_ctx,                 # [d_model]
        "regime_logits": regime_logits, # [R]
    }
    return outputs, h_next


# -----------------------
# Haiku transform wrappers
# -----------------------
def build_c3():
    def _f(x_t, g_t, h_prev, cfg: C3Config):
        return c3_forward_step(x_t, g_t, h_prev, cfg)
    return hk.transform(_f)


# -----------------------
# JIT step (no retracing)
# -----------------------
@partial(jax.jit, static_argnames=("cfg",))
def c3_step_jit(
    params: hk.Params,
    x_t: jnp.ndarray,     # [N,D_in]
    g_t: jnp.ndarray,     # [G]
    h_prev: jnp.ndarray,  # [N,H]
    cfg: C3Config,
) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:
    f = build_c3()
    return f.apply(params, None, x_t, g_t, h_prev, cfg)


def init_c3_params(
    key: jax.Array,
    cfg: C3Config,
) -> hk.Params:
    f = build_c3()

    x0 = jnp.zeros((cfg.n_assets, cfg.feature_dim), dtype=cfg.dtype)
    g0 = jnp.zeros((cfg.global_dim,), dtype=cfg.dtype)
    h0 = jnp.zeros((cfg.n_assets, cfg.rnn_hidden), dtype=cfg.dtype)

    params = f.init(key, x0, g0, h0, cfg)
    return params


def init_c3_state(cfg: C3Config) -> jnp.ndarray:
    return jnp.zeros((cfg.n_assets, cfg.rnn_hidden), dtype=cfg.dtype)


# -----------------------
# Optional AOT compilation
# -----------------------
def aot_compile_c3_step(cfg: C3Config):
    f = build_c3()

    # abstract values
    x_aval = jax.ShapeDtypeStruct((cfg.n_assets, cfg.feature_dim), cfg.dtype)
    g_aval = jax.ShapeDtypeStruct((cfg.global_dim,), cfg.dtype)
    h_aval = jax.ShapeDtypeStruct((cfg.n_assets, cfg.rnn_hidden), cfg.dtype)

    # init params once (outside compile), then lower apply with avals
    params = f.init(jax.random.PRNGKey(0), jnp.zeros(x_aval.shape, cfg.dtype),
                    jnp.zeros(g_aval.shape, cfg.dtype),
                    jnp.zeros(h_aval.shape, cfg.dtype),
                    cfg)

    lowered = jax.jit(lambda x, g, h: f.apply(params, None, x, g, h, cfg)).lower(
        x_aval, g_aval, h_aval
    )
    return lowered.compile(), params


# -----------------------
# Minimal smoke test
# -----------------------
if __name__ == "__main__":
    cfg = C3Config(
        n_assets=200,
        feature_dim=128,   # must equal your features_z_plus width (F+E)
        global_dim=14,     # must equal ext_global_z length
        d_model=256,
        num_heads=8,
        num_layers=2,
        mlp_mult=4,
        rnn_hidden=128,
        dtype=jnp.float64,
    )

    key = jax.random.PRNGKey(42)
    params = init_c3_params(key, cfg)
    h = init_c3_state(cfg)

    x = jax.random.normal(key, (cfg.n_assets, cfg.feature_dim), dtype=cfg.dtype)
    g = jax.random.normal(key, (cfg.global_dim,), dtype=cfg.dtype)

    out, h2 = c3_step_jit(params, x, g, h, cfg)
    _ = out["alpha"].block_until_ready()

    print("alpha:", out["alpha"].shape)
    print("h_next:", h2.shape)
    print("g_ctx:", out["g_ctx"].shape)
    print("regime_logits:", out["regime_logits"].shape)

