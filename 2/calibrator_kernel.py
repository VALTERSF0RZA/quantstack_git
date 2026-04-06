# calibrators_kernel.py
# =============================================================================
# PRODUCTION-GRADE BATCH CALIBRATION KERNEL (JAX)
#
# Pure functional, JIT/AOT-compatible core for SABR and Heston calibration.
# Design:
#   - Single-file, pure JAX numerical logic.
#   - Operates on static-shaped batch tensors [N, ...].
#   - FP64 precision enforced globally.
#   - All Python control-flow (loops, conditionals) replaced with JAX
#     primitives (vmap, lax.scan) where needed for deterministic XLA lowering.
#   - No side effects, I/O, orchestration, or host/device transfers.
# =============================================================================

from __future__ import annotations

import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from functools import partial
from typing import Tuple

# Hard requirement: Enforce FP64 precision globally at the top.
jax.config.update("jax_enable_x64", True)

F64 = jnp.float64
EPS = jnp.array(1e-14, dtype=F64)
CEPS = jnp.array(1e-14 + 0.0j, dtype=jnp.complex128)

# Assume bs_jax is available in the path for these critical functions
# Expected:
#   bs_price_greeks(F, K, T, r, q, sigma, cp) -> (price, delta, gamma, vega)
#   implied_vol(F, K, T, r, q, price, cp) -> array or tuple(...)
from bs_jax import bs_price_greeks, implied_vol


# =============================================================================
# Configuration Dataclasses
# =============================================================================
@dataclass(frozen=True)
class RobustLossConfig:
    huber_k: float = 0.0015
    l2_reg: float = 1e-6
    min_iv: float = 1e-6
    max_iv: float = 5.0


@dataclass(frozen=True)
class SABRCalibConfig:
    beta: float = 0.7
    fit_beta: bool = False
    beta_bounds: Tuple[float, float] = (0.0, 1.0)
    rho_clip: float = 0.999
    alpha_floor: float = 1e-8
    nu_floor: float = 1e-8
    loss: RobustLossConfig = field(default_factory=RobustLossConfig)
    optimizer_steps: int = 1200
    optimizer_lr: float = 0.03
    grad_clip: float = 100.0  # stability on noisy slices


@dataclass(frozen=True)
class HestonCalibConfig:
    rho_clip: float = 0.999
    kappa_floor: float = 1e-8
    theta_floor: float = 1e-8
    sigma_floor: float = 1e-8
    v0_floor: float = 1e-8
    feller_penalty: float = 5.0
    vega_floor: float = 1e-4
    integration_n_points: int = 128
    integration_u_max: float = 150.0
    optimizer_steps: int = 1400
    optimizer_lr: float = 0.02
    grad_clip: float = 100.0  # stability on noisy slices
    loss: RobustLossConfig = field(default_factory=RobustLossConfig)


# =============================================================================
# Shared Utilities
# =============================================================================
def _safe_log(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.log(jnp.maximum(x, EPS))


def _huber(x: jnp.ndarray, k: float) -> jnp.ndarray:
    ax = jnp.abs(x)
    kf = F64(k)
    return jnp.where(ax <= kf, F64(0.5) * x * x, kf * (ax - F64(0.5) * kf))


def _build_valid_iv_mask(K, T, iv, cfg: RobustLossConfig) -> jnp.ndarray:
    return (
        jnp.isfinite(K)
        & jnp.isfinite(T)
        & jnp.isfinite(iv)
        & (K > 0.0)
        & (T > 0.0)
        & (iv >= cfg.min_iv)
        & (iv <= cfg.max_iv)
    )


def _default_slice_weights(F: jnp.ndarray, K: jnp.ndarray, T: jnp.ndarray) -> jnp.ndarray:
    F_pos = jnp.maximum(F, EPS)
    m = _safe_log(K / F_pos)
    return F64(1.0) / (F64(1.0) + F64(8.0) * (m * m) + F64(0.2) * T)


def _robust_weighted_objective(err, w, mask, cfg: RobustLossConfig, u) -> jnp.ndarray:
    mask_f = mask.astype(F64)
    masked_w = w * mask_f
    denom = jnp.maximum(jnp.sum(masked_w), EPS)
    data = jnp.sum(masked_w * _huber(err, cfg.huber_k)) / denom
    reg = F64(cfg.l2_reg) * jnp.sum(u * u)
    return data + reg


def _softplus_inv(y: jnp.ndarray) -> jnp.ndarray:
    y = jnp.maximum(y, EPS)
    # numerically safer inverse softplus
    return jnp.where(y > F64(20.0), y, jnp.log(jnp.expm1(y) + EPS))


def _masked_mean(x: jnp.ndarray, mask: jnp.ndarray, fallback: float) -> jnp.ndarray:
    mask_f = mask.astype(F64)
    denom = jnp.sum(mask_f)
    num = jnp.sum(jnp.where(mask, x, F64(0.0)))
    return jnp.where(denom > 0.0, num / jnp.maximum(denom, EPS), F64(fallback))


def _adam_scan(vg, u0: jnp.ndarray, steps: int, lr: float, grad_clip: float):
    """Adam optimizer in lax.scan form (JIT-safe). Returns (u_star, losses)."""
    beta1 = F64(0.9)
    beta2 = F64(0.999)
    one = F64(1.0)
    lr_f = F64(lr)
    clip_f = F64(grad_clip)

    def body(carry, t):
        u, m, v = carry
        loss, g = vg(u)
        g = jnp.clip(g, -clip_f, clip_f)

        m = beta1 * m + (one - beta1) * g
        v = beta2 * v + (one - beta2) * (g * g)

        t1 = t + one
        m_hat = m / jnp.maximum(one - beta1**t1, EPS)
        v_hat = v / jnp.maximum(one - beta2**t1, EPS)

        u_new = u - lr_f * m_hat / (jnp.sqrt(v_hat) + EPS)
        return (u_new, m, v), loss

    init = (u0, jnp.zeros_like(u0), jnp.zeros_like(u0))
    (u_star, _, _), losses = jax.lax.scan(body, init, jnp.arange(steps, dtype=F64))
    return u_star, losses


def _csafe_div(num: jnp.ndarray, den: jnp.ndarray) -> jnp.ndarray:
    """Complex-safe division with tiny complex epsilon added to denominator."""
    return num / (den + CEPS)


def _coerce_implied_vol_output(out, fallback_shape):
    """
    Supports multiple bs_jax implied_vol return signatures:
    - iv
    - (iv, ...)
    - (iv, ..., ident)
    Returns: (iv, ident_bool)
    """
    if isinstance(out, tuple):
        if len(out) >= 4:
            iv = out[0]
            ident = out[3]
            ident = jnp.broadcast_to(jnp.asarray(ident, dtype=bool), jnp.shape(iv))
            return iv, ident
        elif len(out) >= 1:
            iv = out[0]
            ident = jnp.isfinite(iv)
            return iv, ident

    iv = out
    iv = jnp.broadcast_to(iv, fallback_shape)
    ident = jnp.broadcast_to(jnp.isfinite(iv), fallback_shape)
    return iv, ident


# =============================================================================
# SABR Batch Calibration
# =============================================================================
def _sabr_z_over_xz(z: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
    one_minus_rho = jnp.maximum(F64(1.0) - rho, EPS)
    sqrt_term = jnp.sqrt(jnp.maximum(F64(1.0) - F64(2.0) * rho * z + z * z, EPS * EPS))
    xz = jnp.log((sqrt_term + z - rho) / one_minus_rho)
    taylor = (
        F64(1.0)
        - F64(0.5) * rho * z
        + ((F64(2.0) - F64(3.0) * rho * rho) / F64(12.0)) * z * z
    )
    return jnp.where(jnp.abs(z) < F64(1e-7), taylor, z / jnp.maximum(xz, EPS))


def sabr_lognormal_iv(F, K, T, alpha, beta, rho, nu) -> jnp.ndarray:
    """
    Hagan-style SABR lognormal implied vol approximation.
    Shapes can be scalar/broadcastable arrays.
    """
    F = jnp.maximum(F, EPS)
    K = jnp.maximum(K, EPS)
    T = jnp.maximum(T, EPS)

    omb = F64(1.0) - beta
    logFK = _safe_log(F / K)
    logFK2 = logFK * logFK
    logFK4 = logFK2 * logFK2

    FK = F * K
    FK_omb2 = jnp.power(FK, F64(0.5) * omb)
    FK_omb = jnp.power(FK, omb)

    z = (nu / jnp.maximum(alpha, EPS)) * FK_omb2 * logFK
    zox = _sabr_z_over_xz(z, rho)

    omb2 = omb * omb
    omb4 = omb2 * omb2

    denom = FK_omb2 * (
        F64(1.0)
        + (omb2 / F64(24.0)) * logFK2
        + (omb4 / F64(1920.0)) * logFK4
    )
    A = alpha / jnp.maximum(denom, EPS)

    C = F64(1.0) + T * (
        (omb2 / F64(24.0)) * (alpha * alpha / jnp.maximum(FK_omb, EPS))
        + F64(0.25) * rho * beta * nu * alpha / jnp.maximum(FK_omb2, EPS)
        + ((F64(2.0) - F64(3.0) * rho * rho) / F64(24.0)) * (nu * nu)
    )

    sigma = A * zox * C

    # ATM special case (logFK -> 0)
    F_omb = jnp.maximum(jnp.power(F, omb), EPS)
    F_2omb = jnp.maximum(jnp.power(F, F64(2.0) * omb), EPS)
    sigma_atm = (alpha / F_omb) * (
        F64(1.0)
        + T
        * (
            (omb2 / F64(24.0)) * (alpha * alpha / F_2omb)
            + F64(0.25) * rho * beta * nu * alpha / F_omb
            + ((F64(2.0) - F64(3.0) * rho * rho) / F64(24.0)) * nu * nu
        )
    )

    return jnp.where(jnp.abs(logFK) < F64(1e-10), jnp.maximum(sigma_atm, EPS), jnp.maximum(sigma, EPS))


def _sabr_pack(p, cfg: SABRCalibConfig) -> jnp.ndarray:
    """
    Map constrained SABR params -> unconstrained optimizer vector.
    NOTE: Python if is intentional (cfg is static). Using lax.cond here is wrong
    because branches have different output shapes when fit_beta toggles.
    """
    alpha, beta, rho, nu = p[0], p[1], p[2], p[3]

    a_u = _softplus_inv(jnp.maximum(alpha, cfg.alpha_floor))
    r_u = jnp.arctanh(jnp.clip(rho, -cfg.rho_clip, cfg.rho_clip))
    n_u = _softplus_inv(jnp.maximum(nu, cfg.nu_floor))

    if cfg.fit_beta:
        lo, hi = cfg.beta_bounds
        b = jnp.clip(beta, lo + 1e-12, hi - 1e-12)
        b01 = (b - lo) / (hi - lo)
        b_u = _safe_log(b01 / (F64(1.0) - b01))
        return jnp.array([a_u, b_u, r_u, n_u], dtype=F64)

    return jnp.array([a_u, r_u, n_u], dtype=F64)


def _sabr_unpack(u: jnp.ndarray, cfg: SABRCalibConfig) -> jnp.ndarray:
    """
    Map unconstrained optimizer vector -> constrained SABR params [alpha,beta,rho,nu].
    """
    alpha = jax.nn.softplus(u[0]) + cfg.alpha_floor

    if cfg.fit_beta:
        lo, hi = cfg.beta_bounds
        beta = lo + (hi - lo) * jax.nn.sigmoid(u[1])
        rho = cfg.rho_clip * jnp.tanh(u[2])
        nu = jax.nn.softplus(u[3]) + cfg.nu_floor
        return jnp.array([alpha, beta, rho, nu], dtype=F64)

    beta = jnp.array(cfg.beta, dtype=F64)
    rho = cfg.rho_clip * jnp.tanh(u[1])
    nu = jax.nn.softplus(u[2]) + cfg.nu_floor
    return jnp.array([alpha, beta, rho, nu], dtype=F64)


@partial(jax.jit, static_argnames=("cfg",))
def sabr_calibrate_single(
    iv_slice: jnp.ndarray,
    F: jnp.ndarray,
    K_grid: jnp.ndarray,
    T_grid: jnp.ndarray,
    cfg: SABRCalibConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calibrate one surface slice batch [nT, nK] to SABR.
    Returns:
      best_params: [4] = [alpha, beta, rho, nu]
      final_iv   : [nT, nK]
    """
    K, T = jnp.meshgrid(K_grid, T_grid, indexing="xy")
    mask = _build_valid_iv_mask(K, T, iv_slice, cfg.loss)
    weights = _default_slice_weights(F, K, T)

    def loss_fn(u):
        p = _sabr_unpack(u, cfg)
        iv_model = sabr_lognormal_iv(F, K, T, p[0], p[1], p[2], p[3])
        return _robust_weighted_objective(iv_model - iv_slice, weights, mask, cfg.loss, u)

    vg = jax.value_and_grad(loss_fn)

    def solve_from_start(u0):
        u_star, losses = _adam_scan(vg, u0, cfg.optimizer_steps, cfg.optimizer_lr, cfg.grad_clip)
        return u_star, losses[-1]

    # ATM seed robust to missing/invalid points
    atm_idx = jnp.argmin(jnp.abs(K_grid - F))
    atm_col = iv_slice[:, atm_idx]
    atm_mask = mask[:, atm_idx]
    atm_iv = _masked_mean(atm_col, atm_mask, fallback=0.20)

    alpha_seed = jnp.maximum(atm_iv * (jnp.maximum(F, EPS) ** (F64(1.0) - cfg.beta)), F64(1e-4))

    starts_params = jnp.array(
        [
            (alpha_seed,       cfg.beta, -0.35, 0.35),
            (alpha_seed,       cfg.beta,  0.00, 0.60),
            (alpha_seed,       cfg.beta,  0.35, 1.00),
            (alpha_seed * 0.7, cfg.beta, -0.15, 0.90),
            (alpha_seed * 1.3, cfg.beta,  0.15, 0.25),
        ],
        dtype=F64,
    )

    u0_batch = jax.vmap(_sabr_pack, in_axes=(0, None))(starts_params, cfg)
    u_stars, losses = jax.vmap(solve_from_start)(u0_batch)

    best_idx = jnp.argmin(losses)
    best_params = _sabr_unpack(u_stars[best_idx], cfg)
    final_iv = sabr_lognormal_iv(F, K, T, best_params[0], best_params[1], best_params[2], best_params[3])

    return best_params, final_iv


# =============================================================================
# Heston Batch Calibration
# =============================================================================
def _heston_pack(p, cfg: HestonCalibConfig) -> jnp.ndarray:
    kappa, theta, sigma, rho, v0 = p[0], p[1], p[2], p[3], p[4]

    ku = _softplus_inv(jnp.maximum(kappa, cfg.kappa_floor))
    tu = _softplus_inv(jnp.maximum(theta, cfg.theta_floor))
    su = _softplus_inv(jnp.maximum(sigma, cfg.sigma_floor))
    ru = jnp.arctanh(jnp.clip(rho, -cfg.rho_clip, cfg.rho_clip))
    vu = _softplus_inv(jnp.maximum(v0, cfg.v0_floor))

    return jnp.array([ku, tu, su, ru, vu], dtype=F64)


def _heston_unpack(u: jnp.ndarray, cfg: HestonCalibConfig) -> jnp.ndarray:
    kappa = jax.nn.softplus(u[0]) + cfg.kappa_floor
    theta = jax.nn.softplus(u[1]) + cfg.theta_floor
    sigma = jax.nn.softplus(u[2]) + cfg.sigma_floor
    rho = cfg.rho_clip * jnp.tanh(u[3])
    v0 = jax.nn.softplus(u[4]) + cfg.v0_floor

    return jnp.array([kappa, theta, sigma, rho, v0], dtype=F64)


@partial(jax.jit, static_argnames=("cfg",))
def _heston_prices(
    F: jnp.ndarray,
    K: jnp.ndarray,   # 1D strike grid [nK]
    T: jnp.ndarray,   # 1D tenor grid [nT]
    r: jnp.ndarray,
    q: jnp.ndarray,
    params: jnp.ndarray,
    cfg: HestonCalibConfig,
) -> jnp.ndarray:
    """
    Heston call prices on rectangular grid [nT, nK] using characteristic function integration.
    """
    kappa, theta, sigma, rho, v0 = params[0], params[1], params[2], params[3], params[4]
    i = jnp.array(1j, dtype=jnp.complex128)

    x0 = _safe_log(F)
    a = kappa * theta
    sigma2 = jnp.maximum(sigma * sigma, EPS)

    n_pts = cfg.integration_n_points
    u_int = jnp.linspace(F64(1e-6), F64(cfg.integration_u_max), n_pts, dtype=F64)
    du = F64(cfg.integration_u_max) / jnp.maximum(F64(n_pts - 1), F64(1.0))

    # Trapezoidal weights for slightly better numerical stability
    trap_w = jnp.ones((n_pts,), dtype=F64).at[0].set(F64(0.5)).at[-1].set(F64(0.5))

    one_c = jnp.array(1.0 + 0.0j, dtype=jnp.complex128)

    def cf(u_cf, t_scalar):
        b = kappa - rho * sigma * i * u_cf
        d = jnp.sqrt(b * b + sigma * sigma * (u_cf * u_cf + i * u_cf))
        g = _csafe_div(b - d, b + d)
        exp_neg_dT = jnp.exp(-d * t_scalar)

        C = (r - q) * i * u_cf * t_scalar + (a / sigma2) * (
            (b - d) * t_scalar
            - F64(2.0) * jnp.log(_csafe_div(one_c - g * exp_neg_dT, one_c - g))
        )
        D = ((b - d) / sigma2) * _csafe_div(one_c - exp_neg_dT, one_c - g * exp_neg_dT)
        return jnp.exp(i * u_cf * x0 + C + D * v0)

    def price_one(k_scalar, t_scalar):
        t_scalar = jnp.maximum(t_scalar, EPS)
        klog = _safe_log(k_scalar)

        phi_u_m_i = cf(u_int - i, t_scalar)
        phi_u = cf(u_int, t_scalar)
        phi_m_i = cf(-i, t_scalar)

        exp_term = jnp.exp(-i * u_int * klog)

        int1 = jnp.real(_csafe_div(exp_term * phi_u_m_i, i * u_int * phi_m_i))
        int2 = jnp.real(_csafe_div(exp_term * phi_u, i * u_int))

        P1 = jnp.clip(
            F64(0.5) + (F64(1.0) / jnp.pi) * jnp.sum(trap_w * int1) * du,
            0.0,
            1.0,
        )
        P2 = jnp.clip(
            F64(0.5) + (F64(1.0) / jnp.pi) * jnp.sum(trap_w * int2) * du,
            0.0,
            1.0,
        )

        disc_q = jnp.exp(-q * t_scalar)
        disc_r = jnp.exp(-r * t_scalar)

        call = F * disc_q * P1 - k_scalar * disc_r * P2
        lower = jnp.maximum(F64(0.0), F * disc_q - k_scalar * disc_r)
        upper = F * disc_q
        return jnp.clip(call, lower, upper)

    # output shape [nT, nK]
    return jax.vmap(lambda t_s: jax.vmap(lambda k_s: price_one(k_s, t_s))(K))(T)


@partial(jax.jit, static_argnames=("cfg",))
def heston_calibrate_single(
    iv_slice: jnp.ndarray,   # [nT, nK]
    cp_slice: jnp.ndarray,   # [nT, nK], +1 call / -1 put
    F: jnp.ndarray,          # scalar
    K_grid: jnp.ndarray,     # [nK]
    T_grid: jnp.ndarray,     # [nT]
    r: jnp.ndarray,          # scalar
    q: jnp.ndarray,          # scalar
    cfg: HestonCalibConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Calibrate one surface slice [nT, nK] to Heston (price-based objective, vega-normalized).
    Returns:
      best_params   : [5] = [kappa, theta, sigma, rho, v0]
      best_loss     : scalar
      iv_fit_final  : [nT, nK]
      price_fit     : [nT, nK]
    """
    K, T = jnp.meshgrid(K_grid, T_grid, indexing="xy")

    mask_iv = _build_valid_iv_mask(K, T, iv_slice, cfg.loss)
    mask_cp = jnp.isfinite(cp_slice) & ((cp_slice > 0) | (cp_slice < 0))
    mask = mask_iv & mask_cp

    weights = _default_slice_weights(F, K, T)

    mkt_price, _, _, mkt_vega = bs_price_greeks(F, K, T, r, q, iv_slice, cp_slice)
    mkt_vega = jnp.maximum(mkt_vega, cfg.vega_floor)

    def loss_fn(u):
        params = _heston_unpack(u, cfg)

        # model calls on rectangular grid [nT, nK]
        call_model = _heston_prices(F, K_grid, T_grid, r, q, params, cfg)

        # put-call parity on mesh [nT, nK]
        put_model = call_model - F * jnp.exp(-q * T) + K * jnp.exp(-r * T)

        model_price = jnp.where(cp_slice > 0, call_model, put_model)

        # price error normalized by BS vega => approx IV-scale objective
        err_iv_proxy = (model_price - mkt_price) / mkt_vega

        base_obj = _robust_weighted_objective(err_iv_proxy, weights, mask, cfg.loss, u)

        # Penalize Feller violation: sigma^2 <= 2*kappa*theta
        feller_gap = params[2] * params[2] - F64(2.0) * params[0] * params[1]
        feller_obj = F64(cfg.feller_penalty) * jnp.maximum(feller_gap, F64(0.0)) ** F64(2.0)

        return base_obj + feller_obj

    vg = jax.value_and_grad(loss_fn)

    def solve_from_start(u0):
        u_star, losses = _adam_scan(vg, u0, cfg.optimizer_steps, cfg.optimizer_lr, cfg.grad_clip)
        return u_star, losses[-1]

    atm_idx = jnp.argmin(jnp.abs(K_grid - F))
    atm_col = iv_slice[:, atm_idx]
    atm_mask = mask[:, atm_idx]
    atm_iv = _masked_mean(atm_col, atm_mask, fallback=0.20)

    v0_seed = jnp.maximum(atm_iv * atm_iv, F64(1e-4))
    theta_seed = jnp.maximum(atm_iv * atm_iv, F64(1e-4))

    starts_params = jnp.array(
        [
            (F64(1.0), theta_seed,             F64(0.40), F64(-0.60), v0_seed),
            (F64(1.5), theta_seed,             F64(0.70), F64(-0.30), v0_seed),
            (F64(0.7), theta_seed,             F64(0.30), F64(-0.10), v0_seed * F64(1.2)),
            (F64(2.2), theta_seed,             F64(0.90), F64(-0.75), v0_seed * F64(0.8)),
            (F64(1.2), theta_seed * F64(1.3),  F64(0.55), F64(0.00),  v0_seed),
        ],
        dtype=F64,
    )

    u0_batch = jax.vmap(_heston_pack, in_axes=(0, None))(starts_params, cfg)
    u_stars, losses = jax.vmap(solve_from_start)(u0_batch)

    best_idx = jnp.argmin(losses)
    best_params = _heston_unpack(u_stars[best_idx], cfg)

    call_fit = _heston_prices(F, K_grid, T_grid, r, q, best_params, cfg)
    put_fit = call_fit - F * jnp.exp(-q * T) + K * jnp.exp(-r * T)
    price_fit = jnp.where(cp_slice > 0, call_fit, put_fit)

    # Convert fitted prices back to implied vols (for diagnostics / downstream C3 features)
    iv_out = implied_vol(F, K, T, r, q, price_fit, cp_slice)
    iv_fit, ident = _coerce_implied_vol_output(iv_out, price_fit.shape)
    iv_fit_final = jnp.where(ident, iv_fit, jnp.nan)

    return best_params, losses[best_idx], iv_fit_final, price_fit


# =============================================================================
# Unified Batch Entrypoint
# =============================================================================
@partial(jax.jit, static_argnames=("sabr_cfg", "heston_cfg"))
def calibrate_batch_kernel(
    iv: jnp.ndarray,         # [N, nT, nK]
    cp: jnp.ndarray,         # [N, nT, nK]
    F: jnp.ndarray,          # [N]
    K_grid: jnp.ndarray,     # [nK]
    T_grid: jnp.ndarray,     # [nT]
    r: jnp.ndarray,          # [N]
    q: jnp.ndarray,          # [N]
    sabr_cfg: SABRCalibConfig,
    heston_cfg: HestonCalibConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Pure JAX kernel for batch calibration of SABR and Heston models.

    Returns:
      sabr_params      : [N, 4]
      sabr_iv_fit      : [N, nT, nK]
      heston_params    : [N, 5]
      heston_loss      : [N]
      heston_iv_fit    : [N, nT, nK]
      heston_price_fit : [N, nT, nK]
    """
    # Vectorize the single-surface calibration functions across batch dimension N
    sabr_batch_fn = jax.vmap(
        sabr_calibrate_single,
        in_axes=(0, 0, None, None, None),
    )
    heston_batch_fn = jax.vmap(
        heston_calibrate_single,
        in_axes=(0, 0, 0, None, None, 0, 0, None),
    )

    sabr_params, sabr_iv_fit = sabr_batch_fn(iv, F, K_grid, T_grid, sabr_cfg)
    heston_params, heston_loss, heston_iv_fit, heston_price_fit = heston_batch_fn(
        iv, cp, F, K_grid, T_grid, r, q, heston_cfg
    )

    return sabr_params, sabr_iv_fit, heston_params, heston_loss, heston_iv_fit, heston_price_fit
