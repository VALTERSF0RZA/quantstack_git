# sabr_calibration.py
# =============================================================================
# FP64 SABR CALIBRATION (JAX)
# - Per-expiry smile calibration (Hagan lognormal approximation)
# - Fixed-beta (default) or optional beta-fit
# - Robust weighted loss (Huber)
# - Deterministic multi-start + JIT Adam optimizer
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

import jax
import jax.numpy as jnp

# -------------------------
# FP64 (must be set early)
# -------------------------
jax.config.update("jax_enable_x64", True)

EPS = 1e-14


# =============================================================================
# Config / Result
# =============================================================================
@dataclass(frozen=True)
class SABRCalibConfig:
    # Model mode
    beta: float = 0.7
    fit_beta: bool = False
    beta_bounds: Tuple[float, float] = (0.0, 1.0)

    # Parameter guards
    rho_clip: float = 0.999
    alpha_floor: float = 1e-8
    nu_floor: float = 1e-8

    # Loss
    huber_k: float = 0.0015       # IV-space robust threshold
    l2_reg: float = 1e-6          # small Tikhonov regularization

    # Optimizer
    steps: int = 1200
    lr: float = 0.03

    # Input hygiene
    min_iv: float = 1e-6
    max_iv: float = 5.0


@dataclass(frozen=True)
class SABRParams:
    alpha: float
    beta: float
    rho: float
    nu: float


# =============================================================================
# Utilities
# =============================================================================
def _safe_log(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.log(jnp.maximum(x, EPS))


def _softplus_inv(y: jnp.ndarray) -> jnp.ndarray:
    # inverse softplus with stability
    y = jnp.maximum(y, 1e-12)
    return jnp.log(jnp.expm1(y) + 1e-12)


def _z_over_xz(z: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
    """
    Stable z/x(z), with Taylor fallback near z=0.
    """
    one_minus_rho = jnp.maximum(1.0 - rho, 1e-12)
    sqrt_term = jnp.sqrt(jnp.maximum(1.0 - 2.0 * rho * z + z * z, 1e-24))
    xz = jnp.log((sqrt_term + z - rho) / one_minus_rho)

    # z/x(z) expansion around z=0
    taylor = 1.0 - 0.5 * rho * z + ((2.0 - 3.0 * rho * rho) / 12.0) * z * z
    return jnp.where(jnp.abs(z) < 1e-7, taylor, z / jnp.maximum(xz, 1e-14))


def _huber(x: jnp.ndarray, k: float) -> jnp.ndarray:
    ax = jnp.abs(x)
    return jnp.where(ax <= k, 0.5 * x * x, k * (ax - 0.5 * k))


# =============================================================================
# SABR Hagan Lognormal IV (vectorized, FP64)
# =============================================================================
@jax.jit
def sabr_lognormal_iv(
    F: jnp.ndarray,
    K: jnp.ndarray,
    T: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    rho: jnp.ndarray,
    nu: jnp.ndarray,
) -> jnp.ndarray:
    """
    Hagan 2002 lognormal SABR implied volatility approximation.
    Inputs may be arrays or scalars (broadcastable).
    """
    F = jnp.maximum(F, EPS)
    K = jnp.maximum(K, EPS)
    T = jnp.maximum(T, EPS)

    omb = 1.0 - beta
    logFK = _safe_log(F / K)

    FK = F * K
    FK_omb2 = jnp.power(FK, 0.5 * omb)    # (FK)^((1-beta)/2)
    FK_omb  = jnp.power(FK, omb)          # (FK)^(1-beta)

    z = (nu / jnp.maximum(alpha, EPS)) * FK_omb2 * logFK
    zox = _z_over_xz(z, rho)

    denom = FK_omb2 * (
        1.0
        + (omb * omb / 24.0) * (logFK * logFK)
        + (omb**4 / 1920.0) * (logFK**4)
    )
    A = alpha / jnp.maximum(denom, EPS)

    C = 1.0 + T * (
        (omb * omb / 24.0) * (alpha * alpha / jnp.maximum(FK_omb, EPS))
        + 0.25 * rho * beta * nu * alpha / jnp.maximum(FK_omb2, EPS)
        + ((2.0 - 3.0 * rho * rho) / 24.0) * (nu * nu)
    )

    sigma = A * zox * C

    # ATM branch (logFK -> 0)
    sigma_atm = (alpha / jnp.maximum(jnp.power(F, omb), EPS)) * (
        1.0
        + T * (
            (omb * omb / 24.0) * (
                alpha * alpha / jnp.maximum(jnp.power(F, 2.0 * omb), EPS)
            )
            + 0.25 * rho * beta * nu * alpha / jnp.maximum(jnp.power(F, omb), EPS)
            + ((2.0 - 3.0 * rho * rho) / 24.0) * nu * nu
        )
    )

    sigma = jnp.where(jnp.abs(logFK) < 1e-10, sigma_atm, sigma)
    return jnp.maximum(sigma, 1e-12)


# =============================================================================
# Parameter transforms (unconstrained <-> constrained)
# =============================================================================
def _pack_unconstrained(
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    cfg: SABRCalibConfig,
) -> jnp.ndarray:
    a_u = _softplus_inv(jnp.array(max(alpha, cfg.alpha_floor), dtype=jnp.float64))
    r_u = jnp.arctanh(jnp.clip(jnp.array(rho, dtype=jnp.float64), -cfg.rho_clip, cfg.rho_clip))
    n_u = _softplus_inv(jnp.array(max(nu, cfg.nu_floor), dtype=jnp.float64))

    if cfg.fit_beta:
        lo, hi = cfg.beta_bounds
        b = jnp.clip(jnp.array(beta, dtype=jnp.float64), lo + 1e-8, hi - 1e-8)
        b01 = (b - lo) / (hi - lo)
        b_u = _safe_log(b01 / (1.0 - b01))
        return jnp.array([a_u, b_u, r_u, n_u], dtype=jnp.float64)

    return jnp.array([a_u, r_u, n_u], dtype=jnp.float64)


def _unpack_constrained(u: jnp.ndarray, cfg: SABRCalibConfig):
    alpha = jax.nn.softplus(u[0]) + cfg.alpha_floor

    if cfg.fit_beta:
        lo, hi = cfg.beta_bounds
        beta = lo + (hi - lo) * jax.nn.sigmoid(u[1])
        rho = cfg.rho_clip * jnp.tanh(u[2])
        nu = jax.nn.softplus(u[3]) + cfg.nu_floor
    else:
        beta = jnp.array(cfg.beta, dtype=jnp.float64)
        rho = cfg.rho_clip * jnp.tanh(u[1])
        nu = jax.nn.softplus(u[2]) + cfg.nu_floor

    return alpha, beta, rho, nu


# =============================================================================
# Loss + optimizer
# =============================================================================
def _default_weights(F: float, K: jnp.ndarray, T: jnp.ndarray, iv: jnp.ndarray) -> jnp.ndarray:
    """
    Default desk weighting:
      - downweight extreme wings
      - mild downweight for longer T
    """
    m = _safe_log(K / F)
    return 1.0 / (1.0 + 8.0 * (m * m) + 0.2 * T)


def _loss_from_u(
    u: jnp.ndarray,
    F: float,
    K: jnp.ndarray,
    T: jnp.ndarray,
    iv_mkt: jnp.ndarray,
    w: jnp.ndarray,
    cfg: SABRCalibConfig,
) -> jnp.ndarray:
    alpha, beta, rho, nu = _unpack_constrained(u, cfg)
    iv_model = sabr_lognormal_iv(F, K, T, alpha, beta, rho, nu)
    err = iv_model - iv_mkt

    robust = _huber(err, cfg.huber_k)
    data_term = jnp.sum(w * robust) / jnp.maximum(jnp.sum(w), 1e-12)
    reg = cfg.l2_reg * jnp.sum(u * u)
    return data_term + reg


def _adam_optimize(
    u0: jnp.ndarray,
    F: float,
    K: jnp.ndarray,
    T: jnp.ndarray,
    iv_mkt: jnp.ndarray,
    w: jnp.ndarray,
    cfg: SABRCalibConfig,
):
    b1, b2, eps = 0.9, 0.999, 1e-8
    vg = jax.value_and_grad(_loss_from_u)

    def body(carry, t):
        u, m, v = carry
        loss, g = vg(u, F, K, T, iv_mkt, w, cfg)

        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * (g * g)

        t1 = t + 1.0
        m_hat = m / (1.0 - b1**t1)
        v_hat = v / (1.0 - b2**t1)

        u = u - cfg.lr * m_hat / (jnp.sqrt(v_hat) + eps)
        return (u, m, v), loss

    ts = jnp.arange(cfg.steps, dtype=jnp.float64)
    init = (u0, jnp.zeros_like(u0), jnp.zeros_like(u0))
    (u_star, _, _), losses = jax.lax.scan(body, init, ts)
    return u_star, losses[-1]


_adam_optimize = jax.jit(_adam_optimize, static_argnames=("cfg",))


# =============================================================================
# Public calibration API
# =============================================================================
def calibrate_sabr_slice(
    F: float,
    K: jnp.ndarray,
    T: jnp.ndarray,
    iv_mkt: jnp.ndarray,
    cfg: SABRCalibConfig = SABRCalibConfig(),
    weights: Optional[jnp.ndarray] = None,
) -> Dict[str, Any]:
    """
    Calibrate SABR to one expiry slice (same T preferred; supports arrays anyway).
    Returns params + fit stats + fitted iv vector.
    """

    K = jnp.asarray(K, dtype=jnp.float64)
    T = jnp.asarray(T, dtype=jnp.float64)
    iv_mkt = jnp.asarray(iv_mkt, dtype=jnp.float64)

    valid = (
        jnp.isfinite(K) & jnp.isfinite(T) & jnp.isfinite(iv_mkt) &
        (K > 0.0) & (T > 0.0) &
        (iv_mkt >= cfg.min_iv) & (iv_mkt <= cfg.max_iv)
    )

    K = K[valid]
    T = T[valid]
    iv_mkt = iv_mkt[valid]

    if K.shape[0] < 5:
        raise ValueError("Need at least 5 valid points for stable SABR calibration.")

    if weights is None:
        w = _default_weights(F, K, T, iv_mkt)
    else:
        w = jnp.asarray(weights, dtype=jnp.float64)[valid]

    # Seed alpha from ATM IV
    atm_idx = int(jnp.argmin(jnp.abs(K - F)))
    atm_iv = float(iv_mkt[atm_idx])
    alpha_seed = max(atm_iv * (float(F) ** (1.0 - cfg.beta)), 1e-4)

    starts = [
        (alpha_seed, cfg.beta, -0.35, 0.35),
        (alpha_seed, cfg.beta,  0.00, 0.60),
        (alpha_seed, cfg.beta,  0.35, 1.00),
        (alpha_seed * 0.7, cfg.beta, -0.15, 0.90),
        (alpha_seed * 1.3, cfg.beta,  0.15, 0.25),
    ]

    best_loss = jnp.inf
    best_u = None
    best_tuple = None

    for a0, b0, r0, n0 in starts:
        u0 = _pack_unconstrained(a0, b0, r0, n0, cfg)
        u_star, loss_star = _adam_optimize(u0, F, K, T, iv_mkt, w, cfg)
        if float(loss_star) < float(best_loss):
            best_loss = loss_star
            best_u = u_star
            best_tuple = _unpack_constrained(u_star, cfg)

    alpha, beta, rho, nu = best_tuple
    iv_fit = sabr_lognormal_iv(F, K, T, alpha, beta, rho, nu)

    rmse = jnp.sqrt(jnp.mean((iv_fit - iv_mkt) ** 2))
    mae = jnp.mean(jnp.abs(iv_fit - iv_mkt))

    return {
        "params": {
            "alpha": float(alpha),
            "beta": float(beta),
            "rho": float(rho),
            "nu": float(nu),
        },
        "fit": {
            "loss": float(best_loss),
            "rmse_iv": float(rmse),
            "mae_iv": float(mae),
            "n_points": int(K.shape[0]),
        },
        "diagnostics": {
            "weights_sum": float(jnp.sum(w)),
            "u_star": [float(x) for x in best_u],
        },
        "arrays": {
            "K": K,
            "T": T,
            "iv_market": iv_mkt,
            "iv_fit": iv_fit,
        },
    }


def calibrate_sabr_surface_from_points(
    points: List[Dict[str, Any]],
    F: float,
    beta: float = 0.7,
    min_points_per_expiry: int = 8,
) -> Dict[float, Dict[str, Any]]:
    """
    Utility for your surface_engine payload:
      points[i] has keys: K, T, iv, ...
    Returns dict keyed by T (expiry in years) with SABR fit per slice.
    """
    by_T: Dict[float, List[Dict[str, Any]]] = {}
    for p in points:
        T = float(p["T"])
        by_T.setdefault(T, []).append(p)

    out = {}
    cfg = SABRCalibConfig(beta=beta, fit_beta=False)

    for T, rows in by_T.items():
        if len(rows) < min_points_per_expiry:
            continue

        rows = sorted(rows, key=lambda x: float(x["K"]))
        K = jnp.array([float(r["K"]) for r in rows], dtype=jnp.float64)
        Tv = jnp.array([float(r["T"]) for r in rows], dtype=jnp.float64)
        iv = jnp.array([float(r["iv"]) for r in rows], dtype=jnp.float64)

        out[T] = calibrate_sabr_slice(F=F, K=K, T=Tv, iv_mkt=iv, cfg=cfg)

    return out

