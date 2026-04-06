# container2_vol_surface/cross_regime_allocation_signals.py
# =============================================================================
# CATEGORY 4 — Cross-Regime & Allocation Signals (31..40)
# FP64 • JAX • XLA • JIT/AOT • Static-shape safe
#
# 31) Regime Transition Probability Proxy
# 32) Regime Instability Score
# 33) Factor Drift Magnitude
# 34) Factor Volatility (window std)
# 35) Market Compression Index
# 36) Convexity Crowding Index
# 37) Skew Crowding Index
# 38) Factor Covariance Determinant (and logdet)
# 39) Cross-Asset Skew Beta to PC1
# 40) Cross-Asset Convexity Beta to PC1
#
# Inputs:
#   features_z       [N,F]      current normalized factors
#   features_window  [W,N,F]    rolling history (oldest -> newest)
#   pca_scores       [N,P]      current PCA scores
#   pca_eigvals      [P]        current PCA eigenvalues (descending)
#   asset_corr       [N,N]      cross-asset correlation/similarity matrix
#   regime_id        [N] int32  deterministic regime labels
#   z_thr            []         threshold used by regime rules (e.g. 1.0)
#   dt               []         time step between window slices
#   eps              []         numerical floor
#
# Output:
#   dict with metrics + per_asset_signals_N10 [N,10] for C3/C4 consumption
# =============================================================================

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
F64 = jnp.float64


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@jax.jit
def _offdiag_mean_std(M: jnp.ndarray, eps: jnp.ndarray):
    """
    Off-diagonal mean/std for square matrix M [N,N], static-shape safe.
    """
    n = M.shape[0]
    n_f = jnp.asarray(n, dtype=F64)
    denom = jnp.maximum(n_f * (n_f - 1.0), 1.0)

    total = jnp.sum(M)
    diag = jnp.trace(M)
    total2 = jnp.sum(M * M)
    diag2 = jnp.sum(jnp.diag(M) ** 2)

    mean_off = (total - diag) / denom
    second_off = (total2 - diag2) / denom
    var_off = jnp.maximum(second_off - mean_off * mean_off, 0.0)
    std_off = jnp.sqrt(var_off + eps)
    return mean_off, std_off


@jax.jit
def _scalar_pairwise_similarity_index(v: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
    """
    Crowding/alignment index from scalar per-asset signal v [N].
    Returns off-diagonal mean cosine-like similarity in [-1,1].
    """
    v = jnp.asarray(v, dtype=F64)
    v_norm = v / jnp.maximum(jnp.linalg.norm(v), eps)  # [N]
    S = v_norm[:, None] * v_norm[None, :]              # [N,N]
    mean_off, _ = _offdiag_mean_std(S, eps)
    return mean_off


@jax.jit
def _regime_boundary_distance_proxy(
    level_z: jnp.ndarray,      # [N]
    skew_z: jnp.ndarray,       # [N]
    conv_z: jnp.ndarray,       # [N]
    tslope_z: jnp.ndarray,     # [N]
    regime_id: jnp.ndarray,    # [N]
    thr: jnp.ndarray,          # []
    eps: jnp.ndarray,          # []
):
    """
    Distance-to-boundary proxy consistent with deterministic C2 regime rules.
    Larger transition probability when closer to boundary.
    """

    # Distances to each rule boundary set (always non-negative)
    d1 = jnp.minimum(
        jnp.abs(thr - skew_z),          # skew < -thr boundary at skew = -thr => |skew+thr|
        jnp.abs(conv_z - thr),          # conv > thr boundary
    )
    d1 = jnp.minimum(d1, jnp.abs(skew_z + thr))  # ensure exact boundary term included

    d2 = jnp.minimum(
        jnp.abs(level_z - thr),         # level > thr boundary
        jnp.abs(tslope_z + thr),        # tslope < -thr boundary
    )

    d3 = jnp.minimum(
        jnp.minimum(
            jnp.abs(level_z + thr),     # level < -thr boundary
            jnp.abs(skew_z - thr),      # skew > thr boundary
        ),
        jnp.abs(conv_z),                # conv < 0 boundary
    )

    d4 = jnp.minimum(
        jnp.minimum(
            jnp.abs(jnp.abs(skew_z) - 0.5 * thr),
            jnp.abs(jnp.abs(conv_z) - 0.5 * thr),
        ),
        jnp.abs(jnp.abs(level_z) - 0.75 * thr),
    )

    # Default/mixed regime -> nearest among all major boundaries
    d0 = jnp.min(
        jnp.stack(
            [
                jnp.abs(skew_z + thr),
                jnp.abs(conv_z - thr),
                jnp.abs(level_z - thr),
                jnp.abs(tslope_z + thr),
                jnp.abs(level_z + thr),
                jnp.abs(skew_z - thr),
                jnp.abs(conv_z),
                jnp.abs(jnp.abs(skew_z) - 0.5 * thr),
                jnp.abs(jnp.abs(conv_z) - 0.5 * thr),
                jnp.abs(jnp.abs(level_z) - 0.75 * thr),
            ],
            axis=1,
        ),
        axis=1,
    )

    dist = jnp.where(
        regime_id == 1, d1,
        jnp.where(
            regime_id == 2, d2,
            jnp.where(regime_id == 3, d3, jnp.where(regime_id == 4, d4, d0)),
        ),
    )
    dist = jnp.maximum(dist, eps)

    # Transition proxy: high near boundary
    proxy = jnp.exp(-dist)  # scale=1.0 in z-space
    return dist, proxy


# -----------------------------------------------------------------------------
# Main kernel
# -----------------------------------------------------------------------------
@jax.jit
def category4_cross_regime_allocation_signals(
    features_z: jnp.ndarray,       # [N,F]
    features_window: jnp.ndarray,  # [W,N,F]
    pca_scores: jnp.ndarray,       # [N,P]
    pca_eigvals: jnp.ndarray,      # [P]
    asset_corr: jnp.ndarray,       # [N,N]
    regime_id: jnp.ndarray,        # [N]
    z_thr: jnp.ndarray,            # []
    dt: jnp.ndarray,               # []
    eps: jnp.ndarray,              # []
):
    eps = jnp.asarray(eps, dtype=F64)
    dt = jnp.maximum(jnp.asarray(dt, dtype=F64), eps)
    thr = jnp.asarray(z_thr, dtype=F64)

    X = jnp.asarray(features_z, dtype=F64)
    W = jnp.asarray(features_window, dtype=F64)
    pcs = jnp.asarray(pca_scores, dtype=F64)
    lam = jnp.asarray(pca_eigvals, dtype=F64)
    A = jnp.asarray(asset_corr, dtype=F64)
    r = jnp.asarray(regime_id, dtype=jnp.int32)

    # Factor columns (aligned to your C2 contract)
    # 0: level, 1: skew, 2: convexity, 3: term_slope
    level_z = X[:, 0]
    skew_z = X[:, 1]
    conv_z = X[:, 2]
    tslope_z = X[:, 3]

    # -------------------------------------------------------------------------
    # 31) Regime Transition Probability Proxy
    # -------------------------------------------------------------------------
    boundary_dist, metric_31_transition_prob_proxy = _regime_boundary_distance_proxy(
        level_z, skew_z, conv_z, tslope_z, r, thr, eps
    )  # [N], [N]

    # -------------------------------------------------------------------------
    # 32) Regime Instability Score
    #     "near threshold" norm in z-space -> high score near boundaries
    # -------------------------------------------------------------------------
    near_thr_vec = jnp.stack(
        [
            jnp.abs(jnp.abs(level_z) - thr),
            jnp.abs(jnp.abs(skew_z) - thr),
            jnp.abs(jnp.abs(conv_z) - thr),
            jnp.abs(jnp.abs(tslope_z) - thr),
        ],
        axis=1,
    )  # [N,4]
    near_thr_norm = jnp.linalg.norm(near_thr_vec, axis=1)  # [N]
    metric_32_regime_instability = 1.0 / (1.0 + near_thr_norm)

    # -------------------------------------------------------------------------
    # 33) Factor Drift Magnitude
    # -------------------------------------------------------------------------
    # W: [T,N,F]
    dW = (W[1:, :, :] - W[:-1, :, :]) / dt  # [T-1,N,F]
    metric_33_factor_drift_magnitude = jnp.mean(jnp.linalg.norm(dW, axis=2), axis=0)  # [N]

    # -------------------------------------------------------------------------
    # 34) Factor Volatility (std over window)
    # -------------------------------------------------------------------------
    metric_34_factor_volatility = jnp.mean(jnp.std(W, axis=0), axis=1)  # [N]

    # -------------------------------------------------------------------------
    # 35) Market Compression Index
    #     high when PCA concentration is high and cross-asset dispersion is low
    # -------------------------------------------------------------------------
    _, corr_dispersion = _offdiag_mean_std(A, eps)  # low => compressed
    lambda_sum = jnp.maximum(jnp.sum(jnp.maximum(lam, 0.0)), eps)
    lambda1 = lam[0]
    pca_concentration = lambda1 / lambda_sum
    metric_35_market_compression = pca_concentration / jnp.maximum(corr_dispersion, eps)

    # -------------------------------------------------------------------------
    # 36) Convexity Crowding Index
    # -------------------------------------------------------------------------
    metric_36_convexity_crowding = _scalar_pairwise_similarity_index(conv_z, eps)

    # -------------------------------------------------------------------------
    # 37) Skew Crowding Index
    # -------------------------------------------------------------------------
    metric_37_skew_crowding = _scalar_pairwise_similarity_index(skew_z, eps)

    # -------------------------------------------------------------------------
    # 38) Factor Covariance Determinant (and logdet)
    # -------------------------------------------------------------------------
    Xc = X - jnp.mean(X, axis=0, keepdims=True)
    n = X.shape[0]
    cov = (Xc.T @ Xc) / jnp.maximum(jnp.asarray(n - 1, dtype=F64), 1.0)  # [F,F]
    cov_reg = cov + eps * jnp.eye(cov.shape[0], dtype=F64)
    sign, logdet = jnp.linalg.slogdet(cov_reg)
    metric_38_factor_cov_logdet = jnp.where(sign > 0, logdet, -jnp.inf)
    metric_38_factor_cov_determinant = jnp.where(sign > 0, jnp.exp(logdet), 0.0)

    # -------------------------------------------------------------------------
    # 39) Cross-Asset Skew Beta (to PC1)
    # 40) Convexity Beta (to PC1)
    # -------------------------------------------------------------------------
    pc1 = pcs[:, 0]
    pc1c = pc1 - jnp.mean(pc1)
    var_pc1 = jnp.maximum(jnp.mean(pc1c * pc1c), eps)

    skew_c = skew_z - jnp.mean(skew_z)
    conv_c = conv_z - jnp.mean(conv_z)

    metric_39_skew_beta_pc1 = jnp.mean(skew_c * pc1c) / var_pc1
    metric_40_convexity_beta_pc1 = jnp.mean(conv_c * pc1c) / var_pc1

    # -------------------------------------------------------------------------
    # Per-asset packed [N,10] for direct C3/C4 ingestion
    # Global metrics are broadcast across assets for static contract simplicity.
    # -------------------------------------------------------------------------
    global6 = jnp.array(
        [
            metric_35_market_compression,
            metric_36_convexity_crowding,
            metric_37_skew_crowding,
            metric_38_factor_cov_determinant,
            metric_39_skew_beta_pc1,
            metric_40_convexity_beta_pc1,
        ],
        dtype=F64,
    )  # [6]
    global6_N = jnp.broadcast_to(global6[None, :], (X.shape[0], 6))  # [N,6]

    per_asset_signals_N10 = jnp.concatenate(
        [
            metric_31_transition_prob_proxy[:, None],   # [N,1]
            metric_32_regime_instability[:, None],      # [N,1]
            metric_33_factor_drift_magnitude[:, None],  # [N,1]
            metric_34_factor_volatility[:, None],       # [N,1]
            global6_N,                                  # [N,6]
        ],
        axis=1,
    )  # [N,10]

    # Summary [10] (for monitoring/telemetry)
    category4_summary_10 = jnp.array(
        [
            jnp.mean(metric_31_transition_prob_proxy),
            jnp.mean(metric_32_regime_instability),
            jnp.mean(metric_33_factor_drift_magnitude),
            jnp.mean(metric_34_factor_volatility),
            metric_35_market_compression,
            metric_36_convexity_crowding,
            metric_37_skew_crowding,
            metric_38_factor_cov_determinant,
            metric_39_skew_beta_pc1,
            metric_40_convexity_beta_pc1,
        ],
        dtype=F64,
    )

    return {
        # per-asset
        "metric_31_transition_prob_proxy": metric_31_transition_prob_proxy,     # [N]
        "metric_32_regime_instability": metric_32_regime_instability,           # [N]
        "metric_33_factor_drift_magnitude": metric_33_factor_drift_magnitude,   # [N]
        "metric_34_factor_volatility": metric_34_factor_volatility,             # [N]

        # global/systemic
        "metric_35_market_compression": metric_35_market_compression,           # []
        "metric_36_convexity_crowding": metric_36_convexity_crowding,           # []
        "metric_37_skew_crowding": metric_37_skew_crowding,                     # []
        "metric_38_factor_cov_determinant": metric_38_factor_cov_determinant,   # []
        "metric_38_factor_cov_logdet": metric_38_factor_cov_logdet,             # []
        "metric_39_skew_beta_pc1": metric_39_skew_beta_pc1,                     # []
        "metric_40_convexity_beta_pc1": metric_40_convexity_beta_pc1,           # []

        # packed contracts
        "per_asset_signals_N10": per_asset_signals_N10,                         # [N,10]
        "category4_summary_10": category4_summary_10,                           # [10]

        # diagnostics
        "boundary_distance": boundary_dist,                                     # [N]
        "factor_cov": cov_reg,                                                  # [F,F]
    }


def run_category4_cross_regime_allocation_signals(
    features_z: jnp.ndarray,
    features_window: jnp.ndarray,
    pca_scores: jnp.ndarray,
    pca_eigvals: jnp.ndarray,
    asset_corr: jnp.ndarray,
    regime_id: jnp.ndarray,
    z_thr: float = 1.0,
    dt: float = 1.0,
    eps: float = 1e-12,
):
    out = category4_cross_regime_allocation_signals(
        features_z=features_z,
        features_window=features_window,
        pca_scores=pca_scores,
        pca_eigvals=pca_eigvals,
        asset_corr=asset_corr,
        regime_id=regime_id,
        z_thr=jnp.asarray(z_thr, dtype=F64),
        dt=jnp.asarray(dt, dtype=F64),
        eps=jnp.asarray(eps, dtype=F64),
    )
    _ = out["per_asset_signals_N10"].block_until_ready()
    return out


def aot_compile_category4_cross_regime_allocation_signals(
    n_assets: int = 200,
    n_factors: int = 16,
    window: int = 32,
    pca_components: int = 5,
):
    features_aval = jax.ShapeDtypeStruct((n_assets, n_factors), F64)
    window_aval = jax.ShapeDtypeStruct((window, n_assets, n_factors), F64)
    pca_scores_aval = jax.ShapeDtypeStruct((n_assets, pca_components), F64)
    eigvals_aval = jax.ShapeDtypeStruct((pca_components,), F64)
    corr_aval = jax.ShapeDtypeStruct((n_assets, n_assets), F64)
    regime_aval = jax.ShapeDtypeStruct((n_assets,), jnp.int32)
    scalar_aval = jax.ShapeDtypeStruct((), F64)

    lowered = category4_cross_regime_allocation_signals.lower(
        features_aval,
        window_aval,
        pca_scores_aval,
        eigvals_aval,
        corr_aval,
        regime_aval,
        scalar_aval,  # z_thr
        scalar_aval,  # dt
        scalar_aval,  # eps
    )
    return lowered.compile()


if __name__ == "__main__":
    # ---------------------------
    # Smoke test
    # ---------------------------
    key = jax.random.PRNGKey(123)
    N, F, WN, P = 200, 16, 32, 5

    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

    features_z = jax.random.normal(k1, (N, F), dtype=F64)
    features_window = jax.random.normal(k2, (WN, N, F), dtype=F64) * 0.1 + features_z[None, :, :]
    pca_scores = jax.random.normal(k3, (N, P), dtype=F64)
    pca_eigvals = jnp.sort(jax.random.uniform(k4, (P,), minval=0.1, maxval=2.0, dtype=F64))[::-1]

    A = jax.random.normal(k5, (N, N), dtype=F64)
    asset_corr = (A + A.T) * 0.5
    d = jnp.sqrt(jnp.maximum(jnp.diag(asset_corr @ asset_corr.T), 1e-12))
    asset_corr = (asset_corr @ asset_corr.T) / (d[:, None] * d[None, :] + 1e-12)

    regime_id = jax.random.randint(k6, (N,), minval=0, maxval=5, dtype=jnp.int32)

    out = run_category4_cross_regime_allocation_signals(
        features_z=features_z,
        features_window=features_window,
        pca_scores=pca_scores,
        pca_eigvals=pca_eigvals,
        asset_corr=asset_corr,
        regime_id=regime_id,
        z_thr=1.0,
        dt=1.0,
        eps=1e-12,
    )

    print("per_asset_signals_N10:", out["per_asset_signals_N10"].shape, out["per_asset_signals_N10"].dtype)
    print("category4_summary_10:", out["category4_summary_10"].shape, out["category4_summary_10"].dtype)
    print("metric_35_market_compression:", out["metric_35_market_compression"])
    print("metric_38_factor_cov_logdet:", out["metric_38_factor_cov_logdet"])

    compiled = aot_compile_category4_cross_regime_allocation_signals(
        n_assets=N,
        n_factors=F,
        window=WN,
        pca_components=P,
    )
    out2 = compiled(
        features_z,
        features_window,
        pca_scores,
        pca_eigvals,
        asset_corr,
        regime_id,
        jnp.asarray(1.0, dtype=F64),
        jnp.asarray(1.0, dtype=F64),
        jnp.asarray(1e-12, dtype=F64),
    )
    _ = out2["per_asset_signals_N10"].block_until_ready()
    print("AOT compiled run OK")

