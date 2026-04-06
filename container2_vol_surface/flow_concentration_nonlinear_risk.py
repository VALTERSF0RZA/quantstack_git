# container2_vol_surface/flow_concentration_nonlinear_risk.py
# =============================================================================
# CATEGORY 2 — Flow Concentration & Nonlinear Risk (11..20)
# FP64 • JAX • XLA • JIT/AOT • Static-shape friendly
#
# Metrics implemented:
# 11) GEX Front/Back Spread
# 12) Vega Term Concentration
# 13) Gamma Zero Distance (to ATM)
# 14) Gamma Convexity (2nd diff across strike)
# 15) Vanna Skew Tilt (left wing - right wing)
# 16) Flow Entropy (across strikes)
# 17) Flow Skew Imbalance (call vs put weighted flow)
# 18) GEX Acceleration
# 19) Cross-Asset Vega Correlation (matrix + offdiag mean)
# 20) Vega Dispersion Index
#
# Inputs (static shapes):
#   gex_grid     [N,K,T]   signed dealer gamma exposure grid
#   vanna_grid   [N,K,T]   signed dealer vanna exposure grid
#   vega_grid    [N,K,T]   signed dealer vega exposure grid
#   log_m        [K]       log-moneyness grid (ATM ~ 0)
#   tau          [T]       tenor grid (years), sorted ascending
#   is_call      [N,K,T]   bool mask
#   gex_prev     [N]       previous-step asset-level GEX
#   gex_prev2    [N]       two-steps-back asset-level GEX
#   dt           []        time step (years or consistent unit)
#
# Scalars:
#   front_frac, back_frac, w_gex, w_vanna, w_vega, eps
# =============================================================================

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
F64 = jnp.float64


@jax.jit
def _safe_std(x: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(jnp.std(x), eps)


@jax.jit
def _cov_to_corr(cov: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
    d = jnp.sqrt(jnp.maximum(jnp.diag(cov), eps))
    denom = d[:, None] * d[None, :]
    return cov / jnp.maximum(denom, eps)


@jax.jit
def _second_diff_strike(x: jnp.ndarray, dm: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
    """
    x: [N,K]
    returns second derivative across K with edge replication -> [N,K]
    """
    dm2 = jnp.maximum(dm * dm, eps)
    mid = (x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]) / dm2
    left = mid[:, 0:1]
    right = mid[:, -1:]
    return jnp.concatenate([left, mid, right], axis=1)


def _zero_gamma_distance_one(
    g_by_k: jnp.ndarray,   # [K]
    log_m: jnp.ndarray,    # [K]
    eps: jnp.ndarray,      # []
) -> jnp.ndarray:
    """
    Distance (abs log-moneyness) to nearest zero crossing of gamma profile.
    If no crossing, fallback to strike with minimum |gamma|.
    """
    g0 = g_by_k[:-1]
    g1 = g_by_k[1:]
    x0 = log_m[:-1]
    x1 = log_m[1:]

    # Crossing when sign flips or one endpoint is exactly zero
    cross = (g0 == 0.0) | (g1 == 0.0) | ((g0 > 0.0) & (g1 < 0.0)) | ((g0 < 0.0) & (g1 > 0.0))

    denom = g1 - g0
    safe_denom = jnp.where(
        jnp.abs(denom) < eps,
        jnp.where(denom >= 0.0, eps, -eps),
        denom,
    )
    # Linear interpolation to crossing point on each segment
    x_star = x0 - g0 * (x1 - x0) / safe_denom

    dist_candidates = jnp.where(cross, jnp.abs(x_star), jnp.inf)
    min_dist = jnp.min(dist_candidates)

    fallback_idx = jnp.argmin(jnp.abs(g_by_k))
    fallback_dist = jnp.abs(log_m[fallback_idx])

    return jnp.where(jnp.isfinite(min_dist), min_dist, fallback_dist)


# Vectorized across assets
_zero_gamma_distance_vmapped = jax.jit(jax.vmap(_zero_gamma_distance_one, in_axes=(0, None, None)))


@jax.jit
def category2_flow_metrics(
    gex_grid: jnp.ndarray,    # [N,K,T]
    vanna_grid: jnp.ndarray,  # [N,K,T]
    vega_grid: jnp.ndarray,   # [N,K,T]
    log_m: jnp.ndarray,       # [K]
    tau: jnp.ndarray,         # [T]
    is_call: jnp.ndarray,     # [N,K,T] bool
    gex_prev: jnp.ndarray,    # [N]
    gex_prev2: jnp.ndarray,   # [N]
    dt: jnp.ndarray,          # []
    front_frac: jnp.ndarray,  # []
    back_frac: jnp.ndarray,   # []
    w_gex: jnp.ndarray,       # []
    w_vanna: jnp.ndarray,     # []
    w_vega: jnp.ndarray,      # []
    eps: jnp.ndarray,         # []
):
    """
    Full Category-2 metrics kernel (11..20).
    Returns dict of metric tensors (JAX pytree).
    """
    # -------------------------------------------------------------------------
    # Cast + safety
    # -------------------------------------------------------------------------
    gex = jnp.asarray(gex_grid, dtype=F64)
    vanna = jnp.asarray(vanna_grid, dtype=F64)
    vega = jnp.asarray(vega_grid, dtype=F64)
    log_m = jnp.asarray(log_m, dtype=F64)
    tau = jnp.asarray(tau, dtype=F64)
    is_call_f = jnp.asarray(is_call, dtype=F64)
    gex_prev = jnp.asarray(gex_prev, dtype=F64)
    gex_prev2 = jnp.asarray(gex_prev2, dtype=F64)

    dt = jnp.maximum(jnp.asarray(dt, dtype=F64), eps)
    front_frac = jnp.asarray(front_frac, dtype=F64)
    back_frac = jnp.asarray(back_frac, dtype=F64)
    w_gex = jnp.asarray(w_gex, dtype=F64)
    w_vanna = jnp.asarray(w_vanna, dtype=F64)
    w_vega = jnp.asarray(w_vega, dtype=F64)
    eps = jnp.asarray(eps, dtype=F64)

    N, K, T = gex.shape

    # -------------------------------------------------------------------------
    # Shared projections
    # -------------------------------------------------------------------------
    # asset-level
    gex_asset = jnp.sum(gex, axis=(1, 2))         # [N]
    vanna_asset = jnp.sum(vanna, axis=(1, 2))     # [N]
    vega_asset = jnp.sum(vega, axis=(1, 2))       # [N]

    # tenor-level [N,T]
    gex_tenor = jnp.sum(gex, axis=1)
    vega_tenor_signed = jnp.sum(vega, axis=1)
    vega_tenor_abs = jnp.sum(jnp.abs(vega), axis=1)

    # strike-level [N,K]
    gex_strike = jnp.sum(gex, axis=2)
    vanna_strike = jnp.sum(vanna, axis=2)

    # front/back tenor masks by index (static-shape, no dynamic slicing)
    idx_t = jnp.arange(T, dtype=jnp.int32)
    T_f = jnp.asarray(T, dtype=F64)
    n_front = jnp.maximum(jnp.int32(1), jnp.int32(jnp.floor(front_frac * T_f)))
    n_back = jnp.maximum(jnp.int32(1), jnp.int32(jnp.floor(back_frac * T_f)))

    front_mask = (idx_t < n_front).astype(F64)                  # [T]
    back_mask = (idx_t >= (T - n_back)).astype(F64)             # [T]
    front_den = jnp.maximum(jnp.sum(front_mask), F64(1.0))
    back_den = jnp.maximum(jnp.sum(back_mask), F64(1.0))

    # -------------------------------------------------------------------------
    # 11) GEX Front/Back Spread
    # -------------------------------------------------------------------------
    gex_front = jnp.sum(gex_tenor * front_mask[None, :], axis=1) / front_den
    gex_back = jnp.sum(gex_tenor * back_mask[None, :], axis=1) / back_den
    metric_11_gex_front_back_spread = gex_front - gex_back  # [N]

    # -------------------------------------------------------------------------
    # 12) Vega Term Concentration (front 20% share)
    # -------------------------------------------------------------------------
    vega_front = jnp.sum(vega_tenor_abs * front_mask[None, :], axis=1)
    vega_total = jnp.maximum(jnp.sum(vega_tenor_abs, axis=1), eps)
    metric_12_vega_term_concentration = vega_front / vega_total  # [N]

    # -------------------------------------------------------------------------
    # 13) Gamma Zero Distance (to ATM)
    # -------------------------------------------------------------------------
    metric_13_gamma_zero_distance = _zero_gamma_distance_vmapped(gex_strike, log_m, eps)  # [N]

    # -------------------------------------------------------------------------
    # 14) Gamma Convexity (2nd diff across strike)
    # -------------------------------------------------------------------------
    dm = jnp.maximum(jnp.mean(jnp.diff(log_m)), eps)
    gex_d2k = _second_diff_strike(gex_strike, dm, eps)  # [N,K]
    metric_14_gamma_convexity = jnp.mean(jnp.abs(gex_d2k), axis=1)  # [N]

    # -------------------------------------------------------------------------
    # 15) Vanna Skew Tilt (left wing - right wing)
    # -------------------------------------------------------------------------
    left_mask = (log_m < 0.0).astype(F64)   # [K]
    right_mask = (log_m > 0.0).astype(F64)  # [K]
    left_den = jnp.maximum(jnp.sum(left_mask), F64(1.0))
    right_den = jnp.maximum(jnp.sum(right_mask), F64(1.0))

    left_vanna = jnp.sum(vanna_strike * left_mask[None, :], axis=1) / left_den
    right_vanna = jnp.sum(vanna_strike * right_mask[None, :], axis=1) / right_den
    metric_15_vanna_skew_tilt = left_vanna - right_vanna  # [N]

    # -------------------------------------------------------------------------
    # 16) Flow Entropy (across strikes)
    # -------------------------------------------------------------------------
    flow_abs_strike = jnp.sum(jnp.abs(gex) + jnp.abs(vanna) + jnp.abs(vega), axis=2)  # [N,K]
    p = flow_abs_strike / jnp.maximum(jnp.sum(flow_abs_strike, axis=1, keepdims=True), eps)
    entropy = -jnp.sum(p * jnp.log(jnp.maximum(p, eps)), axis=1)
    entropy_norm = jnp.log(jnp.maximum(jnp.asarray(K, dtype=F64), F64(2.0)))
    metric_16_flow_entropy = entropy / entropy_norm  # [N], normalized to [0,1] range-ish

    # -------------------------------------------------------------------------
    # 17) Flow Skew Imbalance (call vs put weighted signed exposure)
    # -------------------------------------------------------------------------
    flow_signed = w_gex * gex + w_vanna * vanna + w_vega * vega  # [N,K,T]
    call_sum = jnp.sum(flow_signed * is_call_f, axis=(1, 2))
    put_sum = jnp.sum(flow_signed * (1.0 - is_call_f), axis=(1, 2))
    metric_17_flow_skew_imbalance = (call_sum - put_sum) / (jnp.abs(call_sum) + jnp.abs(put_sum) + eps)  # [N]

    # -------------------------------------------------------------------------
    # 18) GEX Acceleration
    # -------------------------------------------------------------------------
    metric_18_gex_acceleration = (gex_asset - 2.0 * gex_prev + gex_prev2) / jnp.maximum(dt * dt, eps)  # [N]

    # -------------------------------------------------------------------------
    # 19) Cross-Asset Vega Correlation
    # -------------------------------------------------------------------------
    # Corr across assets using tenor vectors [N,T]
    X = vega_tenor_signed - jnp.mean(vega_tenor_signed, axis=1, keepdims=True)
    cov = (X @ X.T) / jnp.maximum(jnp.asarray(T - 1, dtype=F64), F64(1.0))  # [N,N]
    metric_19_vega_corr_matrix = _cov_to_corr(cov, eps)

    eye = jnp.eye(N, dtype=F64)
    offdiag_mask = 1.0 - eye
    offdiag_den = jnp.maximum(jnp.sum(offdiag_mask), F64(1.0))
    metric_19_vega_corr_offdiag_mean = jnp.sum(metric_19_vega_corr_matrix * offdiag_mask) / offdiag_den  # scalar

    # -------------------------------------------------------------------------
    # 20) Vega Dispersion Index
    # -------------------------------------------------------------------------
    metric_20_vega_dispersion_index = _safe_std(vega_asset, eps)  # scalar

    # -------------------------------------------------------------------------
    # Optional compact summary vector [10]
    # -------------------------------------------------------------------------
    summary_10 = jnp.array(
        [
            jnp.mean(metric_11_gex_front_back_spread),
            jnp.mean(metric_12_vega_term_concentration),
            jnp.mean(metric_13_gamma_zero_distance),
            jnp.mean(metric_14_gamma_convexity),
            jnp.mean(metric_15_vanna_skew_tilt),
            jnp.mean(metric_16_flow_entropy),
            jnp.mean(metric_17_flow_skew_imbalance),
            jnp.mean(metric_18_gex_acceleration),
            metric_19_vega_corr_offdiag_mean,
            metric_20_vega_dispersion_index,
        ],
        dtype=F64,
    )

    return {
        # raw exposures (useful downstream)
        "gex_asset": gex_asset,                 # [N]
        "vanna_asset": vanna_asset,             # [N]
        "vega_asset": vega_asset,               # [N]

        # Category 2 metrics
        "metric_11_gex_front_back_spread": metric_11_gex_front_back_spread,      # [N]
        "metric_12_vega_term_concentration": metric_12_vega_term_concentration,  # [N]
        "metric_13_gamma_zero_distance": metric_13_gamma_zero_distance,          # [N]
        "metric_14_gamma_convexity": metric_14_gamma_convexity,                  # [N]
        "metric_15_vanna_skew_tilt": metric_15_vanna_skew_tilt,                  # [N]
        "metric_16_flow_entropy": metric_16_flow_entropy,                        # [N]
        "metric_17_flow_skew_imbalance": metric_17_flow_skew_imbalance,          # [N]
        "metric_18_gex_acceleration": metric_18_gex_acceleration,                # [N]
        "metric_19_vega_corr_matrix": metric_19_vega_corr_matrix,                # [N,N]
        "metric_19_vega_corr_offdiag_mean": metric_19_vega_corr_offdiag_mean,    # []
        "metric_20_vega_dispersion_index": metric_20_vega_dispersion_index,      # []

        # convenience
        "category2_summary_10": summary_10,                                      # [10]
    }


def run_category2_flow_metrics(
    gex_grid: jnp.ndarray,
    vanna_grid: jnp.ndarray,
    vega_grid: jnp.ndarray,
    log_m: jnp.ndarray,
    tau: jnp.ndarray,
    is_call: jnp.ndarray,
    gex_prev: jnp.ndarray,
    gex_prev2: jnp.ndarray,
    dt: float,
    front_frac: float = 0.20,
    back_frac: float = 0.20,
    w_gex: float = 1.0,
    w_vanna: float = 1.0,
    w_vega: float = 1.0,
    eps: float = 1e-12,
):
    """
    Runtime wrapper (stable scalar dtypes + execution barrier).
    """
    out = category2_flow_metrics(
        gex_grid=gex_grid,
        vanna_grid=vanna_grid,
        vega_grid=vega_grid,
        log_m=log_m,
        tau=tau,
        is_call=is_call,
        gex_prev=gex_prev,
        gex_prev2=gex_prev2,
        dt=jnp.asarray(dt, dtype=F64),
        front_frac=jnp.asarray(front_frac, dtype=F64),
        back_frac=jnp.asarray(back_frac, dtype=F64),
        w_gex=jnp.asarray(w_gex, dtype=F64),
        w_vanna=jnp.asarray(w_vanna, dtype=F64),
        w_vega=jnp.asarray(w_vega, dtype=F64),
        eps=jnp.asarray(eps, dtype=F64),
    )
    _ = out["category2_summary_10"].block_until_ready()
    return out


def aot_compile_category2_flow_metrics(
    n_assets: int = 200,
    n_strikes: int = 64,
    n_tenors: int = 32,
):
    """
    XLA AOT compile hook for fixed contracts.
    """
    gex_aval = jax.ShapeDtypeStruct((n_assets, n_strikes, n_tenors), F64)
    vanna_aval = jax.ShapeDtypeStruct((n_assets, n_strikes, n_tenors), F64)
    vega_aval = jax.ShapeDtypeStruct((n_assets, n_strikes, n_tenors), F64)
    logm_aval = jax.ShapeDtypeStruct((n_strikes,), F64)
    tau_aval = jax.ShapeDtypeStruct((n_tenors,), F64)
    is_call_aval = jax.ShapeDtypeStruct((n_assets, n_strikes, n_tenors), jnp.bool_)
    gex_prev_aval = jax.ShapeDtypeStruct((n_assets,), F64)
    gex_prev2_aval = jax.ShapeDtypeStruct((n_assets,), F64)

    scalar_f64 = jax.ShapeDtypeStruct((), F64)

    lowered = category2_flow_metrics.lower(
        gex_aval,
        vanna_aval,
        vega_aval,
        logm_aval,
        tau_aval,
        is_call_aval,
        gex_prev_aval,
        gex_prev2_aval,
        scalar_f64,  # dt
        scalar_f64,  # front_frac
        scalar_f64,  # back_frac
        scalar_f64,  # w_gex
        scalar_f64,  # w_vanna
        scalar_f64,  # w_vega
        scalar_f64,  # eps
    )
    return lowered.compile()


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Smoke test
    # -------------------------------------------------------------------------
    N, K, T = 200, 64, 32
    key = jax.random.PRNGKey(42)

    log_m = jnp.linspace(-0.5, 0.5, K, dtype=F64)
    tau = jnp.linspace(1.0 / 365.0, 2.0, T, dtype=F64)

    k1, k2, k3, k4 = jax.random.split(key, 4)
    gex = 1e6 * jax.random.normal(k1, (N, K, T), dtype=F64)
    vanna = 1e5 * jax.random.normal(k2, (N, K, T), dtype=F64)
    vega = 1e5 * jax.random.normal(k3, (N, K, T), dtype=F64)
    is_call = jax.random.bernoulli(k4, p=0.5, shape=(N, K, T))

    gex_prev = jnp.zeros((N,), dtype=F64)
    gex_prev2 = jnp.zeros((N,), dtype=F64)

    out = run_category2_flow_metrics(
        gex_grid=gex,
        vanna_grid=vanna,
        vega_grid=vega,
        log_m=log_m,
        tau=tau,
        is_call=is_call,
        gex_prev=gex_prev,
        gex_prev2=gex_prev2,
        dt=1.0 / (252.0 * 390.0),  # example intraday step in years
    )

    print("m11:", out["metric_11_gex_front_back_spread"].shape, out["metric_11_gex_front_back_spread"].dtype)
    print("m19:", out["metric_19_vega_corr_matrix"].shape, out["metric_19_vega_corr_matrix"].dtype)
    print("summary:", out["category2_summary_10"].shape, out["category2_summary_10"].dtype)

    compiled = aot_compile_category2_flow_metrics(N, K, T)
    out2 = compiled(
        gex, vanna, vega, log_m, tau, is_call, gex_prev, gex_prev2,
        jnp.asarray(1.0 / (252.0 * 390.0), dtype=F64),
        jnp.asarray(0.20, dtype=F64),
        jnp.asarray(0.20, dtype=F64),
        jnp.asarray(1.0, dtype=F64),
        jnp.asarray(1.0, dtype=F64),
        jnp.asarray(1.0, dtype=F64),
        jnp.asarray(1e-12, dtype=F64),
    )
    _ = out2["category2_summary_10"].block_until_ready()
    print("AOT compiled run OK")

