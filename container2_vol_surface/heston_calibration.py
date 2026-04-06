# heston_calibration.py
# =============================================================================
# FP64 HESTON CALIBRATION (JAX)
# - Per-expiry calibration
# - Robust weighted objective (shared common layer)
# - Deterministic multi-start + JIT Adam
# - Returns params + fit stats + fitted arrays (SABR-like API)
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Any, Tuple, Optional, List

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from bs_jax import bs_price_greeks, implied_vol
from calibration_common import (
    RobustLossConfig,
    build_valid_iv_mask,
    default_slice_weights,
    robust_weighted_objective,
    rmse_mae,
)

EPS = 1e-14


@dataclass(frozen=True)
class HestonCalibConfig:
    # parameter guards
    rho_clip: float = 0.999
    kappa_floor: float = 1e-8
    theta_floor: float = 1e-8
    sigma_floor: float = 1e-8
    v0_floor: float = 1e-8

    # additional penalties / scaling
    feller_penalty: float = 5.0
    vega_floor: float = 1e-4  # used for price->iv proxy error scaling

    # integration
    n_int: int = 128
    u_max: float = 150.0

    # optimizer
    steps: int = 1400
    lr: float = 0.02

    # shared robust layer
    loss: RobustLossConfig = field(default_factory=RobustLossConfig)


@dataclass(frozen=True)
class HestonParams:
    kappa: float
    theta: float
    sigma: float
    rho: float
    v0: float


def _softplus_inv(y: jnp.ndarray) -> jnp.ndarray:
    y = jnp.maximum(y, 1e-12)
    return jnp.log(jnp.expm1(y) + 1e-12)


def _pack_unconstrained(
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
    cfg: HestonCalibConfig,
) -> jnp.ndarray:
    ku = _softplus_inv(jnp.array(max(kappa, cfg.kappa_floor), dtype=jnp.float64))
    tu = _softplus_inv(jnp.array(max(theta, cfg.theta_floor), dtype=jnp.float64))
    su = _softplus_inv(jnp.array(max(sigma, cfg.sigma_floor), dtype=jnp.float64))
    ru = jnp.arctanh(jnp.clip(jnp.array(rho, dtype=jnp.float64), -cfg.rho_clip, cfg.rho_clip))
    vu = _softplus_inv(jnp.array(max(v0, cfg.v0_floor), dtype=jnp.float64))
    return jnp.array([ku, tu, su, ru, vu], dtype=jnp.float64)


def _unpack_constrained(u: jnp.ndarray, cfg: HestonCalibConfig):
    kappa = jax.nn.softplus(u[0]) + cfg.kappa_floor
    theta = jax.nn.softplus(u[1]) + cfg.theta_floor
    sigma = jax.nn.softplus(u[2]) + cfg.sigma_floor
    rho = cfg.rho_clip * jnp.tanh(u[3])
    v0 = jax.nn.softplus(u[4]) + cfg.v0_floor
    return kappa, theta, sigma, rho, v0


@jax.jit
def _heston_cf(
    u: jnp.ndarray,  # complex or real
    F: float,
    T: jnp.ndarray,
    r: float,
    q: float,
    kappa: jnp.ndarray,
    theta: jnp.ndarray,
    sigma: jnp.ndarray,
    rho: jnp.ndarray,
    v0: jnp.ndarray,
) -> jnp.ndarray:
    """
    Characteristic function of log(S_T), Heston (Gatheral form).
    """
    i = jnp.array(1j, dtype=jnp.complex128)
    u = jnp.asarray(u, dtype=jnp.complex128)
    T = jnp.asarray(T, dtype=jnp.float64)

    x0 = jnp.log(jnp.maximum(jnp.asarray(F, dtype=jnp.float64), EPS))
    a = kappa * theta
    b = kappa - rho * sigma * i * u
    d = jnp.sqrt(b * b + sigma * sigma * (u * u + i * u))
    g = (b - d) / (b + d + EPS)

    exp_neg_dT = jnp.exp(-d * T)
    C = (
        (r - q) * i * u * T
        + (a / (sigma * sigma + EPS))
        * ((b - d) * T - 2.0 * jnp.log((1.0 - g * exp_neg_dT) / (1.0 - g + EPS)))
    )
    D = ((b - d) / (sigma * sigma + EPS)) * ((1.0 - exp_neg_dT) / (1.0 - g * exp_neg_dT + EPS))
    return jnp.exp(i * u * x0 + C + D * v0)


def _heston_call_price_scalar(
    F: float,
    K: jnp.ndarray,
    T: jnp.ndarray,
    r: float,
    q: float,
    kappa: jnp.ndarray,
    theta: jnp.ndarray,
    sigma: jnp.ndarray,
    rho: jnp.ndarray,
    v0: jnp.ndarray,
    cfg: HestonCalibConfig,
) -> jnp.ndarray:
    """
    Lewis-style integration for call price.
    """
    u = jnp.linspace(1e-6, cfg.u_max, cfg.n_int, dtype=jnp.float64)
    klog = jnp.log(jnp.maximum(K, EPS))

    phi_u = _heston_cf(u, F, T, r, q, kappa, theta, sigma, rho, v0)
    phi_u_minus_i = _heston_cf(u - 1j, F, T, r, q, kappa, theta, sigma, rho, v0)
    phi_minus_i = _heston_cf(jnp.array(-1j, dtype=jnp.complex128), F, T, r, q, kappa, theta, sigma, rho, v0)

    exp_term = jnp.exp(-1j * u * klog)

    int1 = jnp.real(exp_term * phi_u_minus_i / (1j * u * phi_minus_i + EPS))
    int2 = jnp.real(exp_term * phi_u / (1j * u + EPS))

    du = cfg.u_max / (cfg.n_int - 1)
    P1 = 0.5 + (1.0 / jnp.pi) * jnp.sum(int1) * du
    P2 = 0.5 + (1.0 / jnp.pi) * jnp.sum(int2) * du

    P1 = jnp.clip(P1, 0.0, 1.0)
    P2 = jnp.clip(P2, 0.0, 1.0)

    disc_q = jnp.exp(-q * T)
    disc_r = jnp.exp(-r * T)

    call = F * disc_q * P1 - K * disc_r * P2
    lower = jnp.maximum(0.0, F * disc_q - K * disc_r)
    upper = F * disc_q
    return jnp.clip(call, lower, upper)


@partial(jax.jit, static_argnames=("cfg",))
def _heston_call_prices(
    F: float,
    K: jnp.ndarray,
    T: jnp.ndarray,
    r: float,
    q: float,
    kappa: jnp.ndarray,
    theta: jnp.ndarray,
    sigma: jnp.ndarray,
    rho: jnp.ndarray,
    v0: jnp.ndarray,
    cfg: HestonCalibConfig,
) -> jnp.ndarray:
    fn = lambda kk, tt: _heston_call_price_scalar(
        F, kk, tt, r, q, kappa, theta, sigma, rho, v0, cfg
    )
    return jax.vmap(fn)(K, T)


def _loss_from_u(
    u: jnp.ndarray,
    F: float,
    K: jnp.ndarray,
    T: jnp.ndarray,
    cp: jnp.ndarray,
    mkt_price: jnp.ndarray,
    mkt_vega: jnp.ndarray,
    w: jnp.ndarray,
    r: float,
    q: float,
    cfg: HestonCalibConfig,
) -> jnp.ndarray:
    kappa, theta, sigma, rho, v0 = _unpack_constrained(u, cfg)

    call_model = _heston_call_prices(F, K, T, r, q, kappa, theta, sigma, rho, v0, cfg)
    put_model = call_model - F * jnp.exp(-q * T) + K * jnp.exp(-r * T)
    model_price = jnp.where(cp > 0, call_model, put_model)

    # Price residual normalized by vega -> IV-space proxy residual
    err_iv_proxy = (model_price - mkt_price) / jnp.maximum(mkt_vega, cfg.vega_floor)
    base_obj = robust_weighted_objective(err_iv_proxy, w, cfg.loss, u=u)

    # Feller condition soft penalty: 2*kappa*theta >= sigma^2
    feller_gap = sigma * sigma - 2.0 * kappa * theta
    feller_obj = cfg.feller_penalty * jnp.maximum(feller_gap, 0.0) ** 2

    return base_obj + feller_obj


@partial(jax.jit, static_argnames=("cfg",))
def _adam_optimize(
    u0: jnp.ndarray,
    F: float,
    K: jnp.ndarray,
    T: jnp.ndarray,
    cp: jnp.ndarray,
    mkt_price: jnp.ndarray,
    mkt_vega: jnp.ndarray,
    w: jnp.ndarray,
    r: float,
    q: float,
    cfg: HestonCalibConfig,
):
    b1, b2, eps = 0.9, 0.999, 1e-8
    vg = jax.value_and_grad(_loss_from_u)

    def body(carry, t):
        u, m, v = carry
        loss, g = vg(u, F, K, T, cp, mkt_price, mkt_vega, w, r, q, cfg)

        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * (g * g)

        t1 = t + 1.0
        m_hat = m / (1.0 - b1 ** t1)
        v_hat = v / (1.0 - b2 ** t1)

        u = u - cfg.lr * m_hat / (jnp.sqrt(v_hat) + eps)
        return (u, m, v), loss

    ts = jnp.arange(cfg.steps, dtype=jnp.float64)
    init = (u0, jnp.zeros_like(u0), jnp.zeros_like(u0))
    (u_star, _, _), losses = jax.lax.scan(body, init, ts)
    return u_star, losses[-1]


def calibrate_heston_slice(
    F: float,
    K: jnp.ndarray,
    T: jnp.ndarray,
    cp: jnp.ndarray,
    iv_mkt: jnp.ndarray,
    r: float,
    q: float,
    cfg: HestonCalibConfig = HestonCalibConfig(),
    weights: Optional[jnp.ndarray] = None,
) -> Dict[str, Any]:
    """
    Calibrate Heston to one expiry slice.
    API intentionally mirrors SABR calibrate_sabr_slice().
    """
    K = jnp.asarray(K, dtype=jnp.float64)
    T = jnp.asarray(T, dtype=jnp.float64)
    cp = jnp.asarray(cp, dtype=jnp.int32)
    iv_mkt = jnp.asarray(iv_mkt, dtype=jnp.float64)

    valid = build_valid_iv_mask(K, T, iv_mkt, cfg.loss) & jnp.isfinite(cp)
    K, T, cp, iv_mkt = K[valid], T[valid], cp[valid], iv_mkt[valid]

    if K.shape[0] < 5:
        raise ValueError("Need at least 5 valid points for stable Heston calibration.")

    if weights is None:
        w = default_slice_weights(F, K, T)
    else:
        w = jnp.asarray(weights, dtype=jnp.float64)[valid]

    # Convert market IV to market price once (constant for optimizer)
    S = jnp.full_like(K, float(F), dtype=jnp.float64)
    r_arr = jnp.full_like(K, float(r), dtype=jnp.float64)
    q_arr = jnp.full_like(K, float(q), dtype=jnp.float64)

    mkt_price, _, _, mkt_vega = bs_price_greeks(S, K, T, r_arr, q_arr, iv_mkt, cp)
    mkt_vega = jnp.maximum(mkt_vega, cfg.vega_floor)

    # Seeds
    atm_idx = int(jnp.argmin(jnp.abs(K - float(F))))
    atm_iv = float(iv_mkt[atm_idx])
    v0_seed = max(atm_iv * atm_iv, 1e-4)
    theta_seed = v0_seed

    starts = [
        (1.0, theta_seed, 0.40, -0.60, v0_seed),
        (1.5, theta_seed, 0.70, -0.30, v0_seed),
        (0.7, theta_seed, 0.30, -0.10, v0_seed * 1.2),
        (2.2, theta_seed, 0.90, -0.75, v0_seed * 0.8),
        (1.2, theta_seed * 1.3, 0.55, 0.00, v0_seed),
    ]

    best_loss = jnp.inf
    best_u = None
    best_tuple = None

    for k0, t0, s0, r0, v00 in starts:
        u0 = _pack_unconstrained(k0, t0, s0, r0, v00, cfg)
        u_star, loss_star = _adam_optimize(
            u0, float(F), K, T, cp, mkt_price, mkt_vega, w, float(r), float(q), cfg
        )
        if float(loss_star) < float(best_loss):
            best_loss = loss_star
            best_u = u_star
            best_tuple = _unpack_constrained(u_star, cfg)

    kappa, theta, sigma, rho, v0 = best_tuple

    # Final fitted outputs
    call_fit = _heston_call_prices(
        float(F), K, T, float(r), float(q), kappa, theta, sigma, rho, v0, cfg
    )
    put_fit = call_fit - float(F) * jnp.exp(-float(q) * T) + K * jnp.exp(-float(r) * T)
    model_price = jnp.where(cp > 0, call_fit, put_fit)

    iv_fit, ok, proj, ident = implied_vol(S, K, T, r_arr, q_arr, model_price, cp)
    iv_fit = jnp.where(ident, iv_fit, jnp.nan)

    rmse, mae = rmse_mae(iv_fit, iv_mkt)

    return {
        "params": {
            "kappa": float(kappa),
            "theta": float(theta),
            "sigma": float(sigma),
            "rho": float(rho),
            "v0": float(v0),
        },
        "fit": {
            "loss": float(best_loss),
            "rmse_iv": float(rmse),
            "mae_iv": float(mae),
            "n_points": int(K.shape[0]),
        },
        "diagnostics": {
            "weights_sum": float(jnp.sum(w)),
            "feller_lhs": float(2.0 * kappa * theta),
            "feller_rhs": float(sigma * sigma),
            "u_star": [float(x) for x in best_u],
        },
        "arrays": {
            "K": K,
            "T": T,
            "cp": cp,
            "iv_market": iv_mkt,
            "iv_fit": iv_fit,
            "price_fit": model_price,
        },
    }


def calibrate_heston_surface_from_points(
    points: List[Dict[str, Any]],
    F: float,
    r: float,
    q: float,
    min_points_per_expiry: int = 8,
    cfg: HestonCalibConfig = HestonCalibConfig(),
) -> Dict[float, Dict[str, Any]]:
    """
    Utility for surface_engine payload.
    points[i] expected keys: K, T, cp, iv
    """
    by_T: Dict[float, List[Dict[str, Any]]] = {}
    for p in points:
        t_key = round(float(p["T"]), 10)
        by_T.setdefault(t_key, []).append(p)

    out: Dict[float, Dict[str, Any]] = {}
    for T_key, rows in by_T.items():
        if len(rows) < min_points_per_expiry:
            continue

        rows = sorted(rows, key=lambda x: float(x["K"]))
        K = jnp.array([float(x["K"]) for x in rows], dtype=jnp.float64)
        T = jnp.array([float(x["T"]) for x in rows], dtype=jnp.float64)
        cp = jnp.array([int(x.get("cp", 1)) for x in rows], dtype=jnp.int32)
        iv = jnp.array([float(x["iv"]) for x in rows], dtype=jnp.float64)

        out[T_key] = calibrate_heston_slice(
            F=F, K=K, T=T, cp=cp, iv_mkt=iv, r=r, q=q, cfg=cfg
        )

    return out

