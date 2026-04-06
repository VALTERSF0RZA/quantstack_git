# calibration_common.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

EPS = 1e-14


@dataclass(frozen=True)
class RobustLossConfig:
    huber_k: float = 0.0015
    l2_reg: float = 1e-6
    min_iv: float = 1e-6
    max_iv: float = 5.0
    eps: float = EPS


def safe_log(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.log(jnp.maximum(x, EPS))


def huber(x: jnp.ndarray, k: float) -> jnp.ndarray:
    ax = jnp.abs(x)
    return jnp.where(ax <= k, 0.5 * x * x, k * (ax - 0.5 * k))


def build_valid_iv_mask(
    K: jnp.ndarray,
    T: jnp.ndarray,
    iv: jnp.ndarray,
    cfg: RobustLossConfig,
) -> jnp.ndarray:
    return (
        jnp.isfinite(K)
        & jnp.isfinite(T)
        & jnp.isfinite(iv)
        & (K > 0.0)
        & (T > 0.0)
        & (iv >= cfg.min_iv)
        & (iv <= cfg.max_iv)
    )


def default_slice_weights(
    F: float,
    K: jnp.ndarray,
    T: jnp.ndarray,
    moneyness_scale: float = 8.0,
    term_scale: float = 0.2,
) -> jnp.ndarray:
    m = safe_log(K / float(F))
    return 1.0 / (1.0 + moneyness_scale * (m * m) + term_scale * T)


def robust_weighted_objective(
    err: jnp.ndarray,
    w: jnp.ndarray,
    cfg: RobustLossConfig,
    u: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    data = jnp.sum(w * huber(err, cfg.huber_k)) / jnp.maximum(jnp.sum(w), cfg.eps)
    reg = cfg.l2_reg * jnp.sum(u * u) if u is not None else 0.0
    return data + reg


def rmse_mae(
    y_hat: jnp.ndarray,
    y_true: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if mask is None:
        mask = jnp.isfinite(y_hat) & jnp.isfinite(y_true)
    y_hat_m = y_hat[mask]
    y_true_m = y_true[mask]
    if y_hat_m.shape[0] == 0:
        nan = jnp.array(jnp.nan, dtype=jnp.float64)
        return nan, nan
    err = y_hat_m - y_true_m
    rmse = jnp.sqrt(jnp.mean(err * err))
    mae = jnp.mean(jnp.abs(err))
    return rmse, mae

