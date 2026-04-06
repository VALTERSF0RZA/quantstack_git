# container2_vol_surface/geometry_instability_measures.py
# =============================================================================
# CATEGORY 3 — Geometry Instability Measures (21..30)
# FP64 • JAX • XLA • JIT/AOT • Static-shape safe
#
# Metrics:
# 21) Skew Acceleration
# 22) Convexity Acceleration
# 23) Curvature Asymmetry (left wing - right wing)
# 24) Term Curvature Slope
# 25) Surface Energy
# 26) Surface Entropy
# 27) Surface Roughness (mean abs Laplacian)
# 28) ATM vs OTM Vol Spread
# 29) PCA Reconstruction Error
# 30) Local Instability Index
#
# Inputs:
#   sigma_t       [N,K,T]   current surface
#   sigma_t1      [N,K,T]   previous surface
#   sigma_t2      [N,K,T]   two-steps-back surface
#   log_m         [K]
#   tau           [T]
#   dt            []        time step (same units as t,t1,t2)
#   pca_mean      [K*T]     PCA mean of flattened surfaces
#   pca_basis     [K*T,P]   PCA basis (columns are modes)
#   otm_abs_logm  []        deep OTM threshold in |log-moneyness|
#   eps           []        numerical floor
#
# Outputs:
#   metric_21..metric_30 (mostly [N], plus summary vector [10])
# =============================================================================

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
F64 = jnp.float64


# -----------------------------------------------------------------------------
# Finite-difference helpers (static-shape)
# -----------------------------------------------------------------------------
@jax.jit
def _first_diff_axis1(x: jnp.ndarray, dm: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
    """First derivative along strike axis K (axis=1), [N,K,T]."""
    dm_safe = jnp.maximum(dm, eps)
    left = (x[:, 1:2, :] - x[:, 0:1, :]) / dm_safe
    mid = (x[:, 2:, :] - x[:, :-2, :]) / (2.0 * dm_safe)
    right = (x[:, -1:, :] - x[:, -2:-1, :]) / dm_safe
    return jnp.concatenate([left, mid, right], axis=1)


@jax.jit
def _second_diff_axis1(x: jnp.ndarray, dm: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
    """Second derivative along strike axis K (axis=1), [N,K,T]."""
    dm2 = jnp.maximum(dm * dm, eps)
    mid = (x[:, 2:, :] - 2.0 * x[:, 1:-1, :] + x[:, :-2, :]) / dm2
    left = mid[:, 0:1, :]
    right = mid[:, -1:, :]
    return jnp.concatenate([left, mid, right], axis=1)


@jax.jit
def _second_diff_axis2(x: jnp.ndarray, dtau: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
    """Second derivative along tenor axis T (axis=2), [N,K,T]."""
    dt2 = jnp.maximum(dtau * dtau, eps)
    mid = (x[:, :, 2:] - 2.0 * x[:, :, 1:-1] + x[:, :, :-2]) / dt2
    left = mid[:, :, 0:1]
    right = mid[:, :, -1:]
    return jnp.concatenate([left, mid, right], axis=2)


@jax.jit
def _first_diff_last_axis_2d(x: jnp.ndarray, dx: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
    """First derivative along last axis for [N,T] tensors."""
    dx_safe = jnp.maximum(dx, eps)
    left = (x[:, 1:2] - x[:, 0:1]) / dx_safe
    mid = (x[:, 2:] - x[:, :-2]) / (2.0 * dx_safe)
    right = (x[:, -1:] - x[:, -2:-1]) / dx_safe
    return jnp.concatenate([left, mid, right], axis=1)


@jax.jit
def _laplacian_axis1(x: jnp.ndarray) -> jnp.ndarray:
    """Discrete Laplacian along strike K (edge replication), [N,K,T]."""
    xl = jnp.concatenate([x[:, 0:1, :], x[:, :-1, :]], axis=1)
    xr = jnp.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
    return xl - 2.0 * x + xr


@jax.jit
def _laplacian_axis2(x: jnp.ndarray) -> jnp.ndarray:
    """Discrete Laplacian along tenor T (edge replication), [N,K,T]."""
    xl = jnp.concatenate([x[:, :, 0:1], x[:, :, :-1]], axis=2)
    xr = jnp.concatenate([x[:, :, 1:], x[:, :, -1:]], axis=2)
    return xl - 2.0 * x + xr


# -----------------------------------------------------------------------------
# Main kernel
# -----------------------------------------------------------------------------
@jax.jit
def category3_geometry_instability_metrics(
    sigma_t: jnp.ndarray,      # [N,K,T]
    sigma_t1: jnp.ndarray,     # [N,K,T]
    sigma_t2: jnp.ndarray,     # [N,K,T]
    log_m: jnp.ndarray,        # [K]
    tau: jnp.ndarray,          # [T]
    dt: jnp.ndarray,           # []
    pca_mean: jnp.ndarray,     # [K*T]
    pca_basis: jnp.ndarray,    # [K*T,P]
    otm_abs_logm: jnp.ndarray, # []
    eps: jnp.ndarray,          # []
):
    eps = jnp.asarray(eps, dtype=F64)
    dt = jnp.maximum(jnp.asarray(dt, dtype=F64), eps)

    s0 = jnp.asarray(sigma_t, dtype=F64)
    s1 = jnp.asarray(sigma_t1, dtype=F64)
    s2 = jnp.asarray(sigma_t2, dtype=F64)

    log_m = jnp.asarray(log_m, dtype=F64)
    tau = jnp.asarray(tau, dtype=F64)
    pca_mean = jnp.asarray(pca_mean, dtype=F64)
    pca_basis = jnp.asarray(pca_basis, dtype=F64)
    otm_abs_logm = jnp.asarray(otm_abs_logm, dtype=F64)

    N, K, T = s0.shape
    M = K * T

    dm = jnp.maximum(jnp.mean(jnp.diff(log_m)), eps)
    dtau = jnp.maximum(jnp.mean(jnp.diff(tau)), eps)
    atm_idx = jnp.argmin(jnp.abs(log_m))

    # Derivatives at t, t-1, t-2
    dsdm_0 = _first_diff_axis1(s0, dm, eps)
    d2sdm2_0 = _second_diff_axis1(s0, dm, eps)
    d2sdT2_0 = _second_diff_axis2(s0, dtau, eps)

    dsdm_1 = _first_diff_axis1(s1, dm, eps)
    d2sdm2_1 = _second_diff_axis1(s1, dm, eps)

    dsdm_2 = _first_diff_axis1(s2, dm, eps)
    d2sdm2_2 = _second_diff_axis1(s2, dm, eps)

    # Factor proxies for acceleration metrics
    skew_0 = jnp.mean(dsdm_0[:, atm_idx, :], axis=1)       # [N]
    skew_1 = jnp.mean(dsdm_1[:, atm_idx, :], axis=1)
    skew_2 = jnp.mean(dsdm_2[:, atm_idx, :], axis=1)

    conv_0 = jnp.mean(d2sdm2_0[:, atm_idx, :], axis=1)     # [N]
    conv_1 = jnp.mean(d2sdm2_1[:, atm_idx, :], axis=1)
    conv_2 = jnp.mean(d2sdm2_2[:, atm_idx, :], axis=1)

    # 21) Skew Acceleration
    metric_21_skew_acceleration = (skew_0 - 2.0 * skew_1 + skew_2) / jnp.maximum(dt * dt, eps)

    # 22) Convexity Acceleration
    metric_22_convexity_acceleration = (conv_0 - 2.0 * conv_1 + conv_2) / jnp.maximum(dt * dt, eps)

    # 23) Curvature Asymmetry (left - right wing)
    left_mask = (log_m < 0.0).astype(F64)   # [K]
    right_mask = (log_m > 0.0).astype(F64)  # [K]
    left_den = jnp.maximum(jnp.sum(left_mask) * jnp.asarray(T, dtype=F64), 1.0)
    right_den = jnp.maximum(jnp.sum(right_mask) * jnp.asarray(T, dtype=F64), 1.0)

    left_curv = jnp.sum(d2sdm2_0 * left_mask[None, :, None], axis=(1, 2)) / left_den
    right_curv = jnp.sum(d2sdm2_0 * right_mask[None, :, None], axis=(1, 2)) / right_den
    metric_23_curvature_asymmetry = left_curv - right_curv  # [N]

    # 24) Term Curvature Slope (derivative across tenor of ATM curvature)
    atm_curv_tau = d2sdm2_0[:, atm_idx, :]                   # [N,T]
    dcurv_dtau = _first_diff_last_axis_2d(atm_curv_tau, dtau, eps)
    metric_24_term_curvature_slope = jnp.mean(dcurv_dtau, axis=1)  # [N]

    # 25) Surface Energy
    metric_25_surface_energy = jnp.mean(d2sdm2_0 * d2sdm2_0 + d2sdT2_0 * d2sdT2_0, axis=(1, 2))  # [N]

    # 26) Surface Entropy (normalized entropy of positive-normalized surface values)
    s_flat = s0.reshape(N, M)  # [N,M]
    s_shift = s_flat - jnp.min(s_flat, axis=1, keepdims=True)
    p = s_shift + eps
    p = p / jnp.maximum(jnp.sum(p, axis=1, keepdims=True), eps)
    entropy = -jnp.sum(p * jnp.log(jnp.maximum(p, eps)), axis=1)
    metric_26_surface_entropy = entropy / jnp.log(jnp.maximum(jnp.asarray(M, dtype=F64), 2.0))  # [N]

    # 27) Surface Roughness (mean absolute 2D Laplacian)
    lap2d = _laplacian_axis1(s0) + _laplacian_axis2(s0)
    metric_27_surface_roughness = jnp.mean(jnp.abs(lap2d), axis=(1, 2))  # [N]

    # 28) ATM vs OTM Vol Spread
    atm_vol = jnp.mean(s0[:, atm_idx, :], axis=1)  # [N]
    otm_mask = (jnp.abs(log_m) >= otm_abs_logm).astype(F64)  # [K]
    otm_den = jnp.maximum(jnp.sum(otm_mask) * jnp.asarray(T, dtype=F64), 1.0)
    otm_vol = jnp.sum(s0 * otm_mask[None, :, None], axis=(1, 2)) / otm_den
    metric_28_atm_vs_otm_vol_spread = atm_vol - otm_vol  # [N]

    # 29) PCA Reconstruction Error: ||surface - recon||
    # pca_mean: [M], pca_basis: [M,P]
    x = s_flat - pca_mean[None, :]          # [N,M]
    coeff = x @ pca_basis                    # [N,P]
    recon = coeff @ pca_basis.T + pca_mean[None, :]  # [N,M]
    resid = s_flat - recon
    metric_29_pca_reconstruction_error = jnp.sqrt(jnp.mean(resid * resid, axis=1) + eps)  # [N]

    # 30) Local Instability Index (norm of second-order derivatives)
    metric_30_local_instability_index = jnp.sqrt(
        jnp.mean(d2sdm2_0 * d2sdm2_0 + d2sdT2_0 * d2sdT2_0, axis=(1, 2)) + eps
    )  # [N]

    # Compact summary [10] (cross-asset mean of each metric)
    summary_10 = jnp.array(
        [
            jnp.mean(metric_21_skew_acceleration),
            jnp.mean(metric_22_convexity_acceleration),
            jnp.mean(metric_23_curvature_asymmetry),
            jnp.mean(metric_24_term_curvature_slope),
            jnp.mean(metric_25_surface_energy),
            jnp.mean(metric_26_surface_entropy),
            jnp.mean(metric_27_surface_roughness),
            jnp.mean(metric_28_atm_vs_otm_vol_spread),
            jnp.mean(metric_29_pca_reconstruction_error),
            jnp.mean(metric_30_local_instability_index),
        ],
        dtype=F64,
    )

    return {
        "metric_21_skew_acceleration": metric_21_skew_acceleration,                 # [N]
        "metric_22_convexity_acceleration": metric_22_convexity_acceleration,       # [N]
        "metric_23_curvature_asymmetry": metric_23_curvature_asymmetry,             # [N]
        "metric_24_term_curvature_slope": metric_24_term_curvature_slope,           # [N]
        "metric_25_surface_energy": metric_25_surface_energy,                       # [N]
        "metric_26_surface_entropy": metric_26_surface_entropy,                     # [N]
        "metric_27_surface_roughness": metric_27_surface_roughness,                 # [N]
        "metric_28_atm_vs_otm_vol_spread": metric_28_atm_vs_otm_vol_spread,         # [N]
        "metric_29_pca_reconstruction_error": metric_29_pca_reconstruction_error,   # [N]
        "metric_30_local_instability_index": metric_30_local_instability_index,     # [N]
        "category3_summary_10": summary_10,                                         # [10]

        # Useful internals for audit/debug
        "d2sdm2": d2sdm2_0,   # [N,K,T]
        "d2sdT2": d2sdT2_0,   # [N,K,T]
        "lap2d": lap2d,       # [N,K,T]
    }


def run_category3_geometry_instability(
    sigma_t: jnp.ndarray,
    sigma_t1: jnp.ndarray,
    sigma_t2: jnp.ndarray,
    log_m: jnp.ndarray,
    tau: jnp.ndarray,
    dt: float,
    pca_mean: jnp.ndarray,
    pca_basis: jnp.ndarray,
    otm_abs_logm: float = 0.20,
    eps: float = 1e-12,
):
    out = category3_geometry_instability_metrics(
        sigma_t=sigma_t,
        sigma_t1=sigma_t1,
        sigma_t2=sigma_t2,
        log_m=log_m,
        tau=tau,
        dt=jnp.asarray(dt, dtype=F64),
        pca_mean=pca_mean,
        pca_basis=pca_basis,
        otm_abs_logm=jnp.asarray(otm_abs_logm, dtype=F64),
        eps=jnp.asarray(eps, dtype=F64),
    )
    _ = out["category3_summary_10"].block_until_ready()
    return out


def aot_compile_category3_geometry_instability(
    n_assets: int = 200,
    n_strikes: int = 64,
    n_tenors: int = 32,
    pca_rank: int = 12,
):
    sigma_aval = jax.ShapeDtypeStruct((n_assets, n_strikes, n_tenors), F64)
    logm_aval = jax.ShapeDtypeStruct((n_strikes,), F64)
    tau_aval = jax.ShapeDtypeStruct((n_tenors,), F64)
    pca_mean_aval = jax.ShapeDtypeStruct((n_strikes * n_tenors,), F64)
    pca_basis_aval = jax.ShapeDtypeStruct((n_strikes * n_tenors, pca_rank), F64)

    scalar = jax.ShapeDtypeStruct((), F64)

    lowered = category3_geometry_instability_metrics.lower(
        sigma_aval,      # sigma_t
        sigma_aval,      # sigma_t1
        sigma_aval,      # sigma_t2
        logm_aval,
        tau_aval,
        scalar,          # dt
        pca_mean_aval,
        pca_basis_aval,
        scalar,          # otm_abs_logm
        scalar,          # eps
    )
    return lowered.compile()


if __name__ == "__main__":
    # Smoke test
    key = jax.random.PRNGKey(7)
    N, K, T, P = 200, 64, 32, 12

    log_m = jnp.linspace(-0.5, 0.5, K, dtype=F64)
    tau = jnp.linspace(1.0 / 365.0, 2.0, T, dtype=F64)

    # Synthetic surfaces
    base = 0.18 + 0.05 * jnp.exp(-2.0 * tau)[None, None, :] + 0.03 * (log_m[None, :, None] ** 2)
    shifts = jnp.linspace(-0.02, 0.02, N, dtype=F64)[:, None, None]
    sigma_t = jnp.maximum(base + shifts, 0.04)
    sigma_t1 = jnp.maximum(sigma_t * 0.997 + 0.0005, 0.04)
    sigma_t2 = jnp.maximum(sigma_t1 * 0.997 + 0.0005, 0.04)

    # Dummy orthonormal PCA basis
    k1 = jax.random.PRNGKey(123)
    rand_mat = jax.random.normal(k1, (K * T, P), dtype=F64)
    q, _ = jnp.linalg.qr(rand_mat)  # [K*T,P]
    pca_basis = q
    pca_mean = jnp.mean(sigma_t.reshape(N, K * T), axis=0)

    out = run_category3_geometry_instability(
        sigma_t=sigma_t,
        sigma_t1=sigma_t1,
        sigma_t2=sigma_t2,
        log_m=log_m,
        tau=tau,
        dt=1.0 / (252.0 * 390.0),  # example intraday step in years
        pca_mean=pca_mean,
        pca_basis=pca_basis,
        otm_abs_logm=0.20,
        eps=1e-12,
    )
    print("metric_21 shape:", out["metric_21_skew_acceleration"].shape, out["metric_21_skew_acceleration"].dtype)
    print("metric_29 shape:", out["metric_29_pca_reconstruction_error"].shape, out["metric_29_pca_reconstruction_error"].dtype)
    print("summary shape:", out["category3_summary_10"].shape, out["category3_summary_10"].dtype)

    compiled = aot_compile_category3_geometry_instability(N, K, T, P)
    out2 = compiled(
        sigma_t,
        sigma_t1,
        sigma_t2,
        log_m,
        tau,
        jnp.asarray(1.0 / (252.0 * 390.0), dtype=F64),
        pca_mean,
        pca_basis,
        jnp.asarray(0.20, dtype=F64),
        jnp.asarray(1e-12, dtype=F64),
    )
    _ = out2["category3_summary_10"].block_until_ready()
    print("AOT compiled run OK")

