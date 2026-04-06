# container2_vol_surface/cross_asset_structure_extended.py
# =============================================================================
# C2-ONLY (single-snapshot) CATEGORY 1 EXTENDED
# JAX + XLA + FP64, static-shape safe
#
# Implements:
# 1  Eigenvalue Gap Ladder
# 2  Effective Dimensionality (Participation Ratio)
# 3  Eigenvalue Entropy
# 6  Average Pairwise Correlation
# 7  Correlation Dispersion
# 8  Correlation Skewness
# 9  Correlation Kurtosis
# 10 Correlation Regime Distance (vs fixed baseline)
# 11 Correlation Eigenvector Centrality
# 12 Correlation Graph Density
# 13 Network Clustering Coefficient
# 14 Skew Alignment Index
# 15 Convexity Alignment Index
# 16 Term Structure Alignment
# 17 Surface Curvature Correlation
# 18 ATM Vol Synchronization
# 19 Wing Stress Synchronization
# 20 Surface Shape Dispersion
# 21 Cross-GEX Correlation Matrix
# 22 Systemic Gamma Concentration
# 23 Vanna Synchronization
# 24 Vega Network Density
# 25 Flow Concentration Entropy
# 26 Flow Crowding Index
# 27 Risk Dominance Index
# 28 Diversification Capacity
# 29 Fragility Score
# 30 Systemic Leverage Proxy
# 31 Hedging Efficiency Metric
# 32 Orthogonality Score
# 33 Allocation Instability Indicator (snapshot proxy)
# =============================================================================

from __future__ import annotations

from functools import partial
from typing import Dict

import jax
import jax.numpy as jnp

from .config import C2Config, F64

jax.config.update("jax_enable_x64", True)


def _symmetrize(x: jnp.ndarray) -> jnp.ndarray:
    return F64(0.5) * (x + x.T)


def _safe_norm_rows(x: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
    nrm = jnp.sqrt(jnp.maximum(jnp.sum(x * x, axis=1, keepdims=True), eps))
    return x / nrm


def _cosine_matrix(x: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
    xn = _safe_norm_rows(x, eps)
    c = xn @ xn.T
    n = c.shape[0]
    eye = jnp.eye(n, dtype=c.dtype)
    return jnp.clip(c * (F64(1.0) - eye) + eye, F64(-1.0), F64(1.0))


def _corr_from_profiles(x: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
    xc = x - jnp.mean(x, axis=1, keepdims=True)
    xn = _safe_norm_rows(xc, eps)
    c = xn @ xn.T
    n = c.shape[0]
    eye = jnp.eye(n, dtype=c.dtype)
    return jnp.clip(c * (F64(1.0) - eye) + eye, F64(-1.0), F64(1.0))


def _offdiag_count(n: int, dtype) -> jnp.ndarray:
    return jnp.asarray(n * n - n, dtype=dtype)


def _offdiag_mean(x: jnp.ndarray) -> jnp.ndarray:
    n = x.shape[0]
    eye = jnp.eye(n, dtype=x.dtype)
    off = F64(1.0) - eye
    cnt = _offdiag_count(n, x.dtype)
    return jnp.sum(x * off) / jnp.maximum(cnt, F64(1.0))


def _offdiag_abs_mean(x: jnp.ndarray) -> jnp.ndarray:
    n = x.shape[0]
    eye = jnp.eye(n, dtype=x.dtype)
    off = F64(1.0) - eye
    cnt = _offdiag_count(n, x.dtype)
    return jnp.sum(jnp.abs(x) * off) / jnp.maximum(cnt, F64(1.0))


def _offdiag_std(x: jnp.ndarray, mean_off: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
    n = x.shape[0]
    eye = jnp.eye(n, dtype=x.dtype)
    off = F64(1.0) - eye
    cnt = _offdiag_count(n, x.dtype)
    d = (x - mean_off) * off
    var = jnp.sum(d * d) / jnp.maximum(cnt, F64(1.0))
    return jnp.sqrt(jnp.maximum(var, eps))


def _offdiag_skew_kurt(
    x: jnp.ndarray, mean_off: jnp.ndarray, std_off: jnp.ndarray, eps: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    n = x.shape[0]
    eye = jnp.eye(n, dtype=x.dtype)
    off = F64(1.0) - eye
    cnt = _offdiag_count(n, x.dtype)
    z = ((x - mean_off) / jnp.maximum(std_off, eps)) * off
    skew = jnp.sum(z**3) / jnp.maximum(cnt, F64(1.0))
    kurt = jnp.sum(z**4) / jnp.maximum(cnt, F64(1.0))
    return skew, kurt


def _graph_density(corr: jnp.ndarray, thr: jnp.ndarray) -> jnp.ndarray:
    n = corr.shape[0]
    eye = jnp.eye(n, dtype=corr.dtype)
    off = F64(1.0) - eye
    a = (jnp.abs(corr) > thr).astype(corr.dtype) * off
    cnt = _offdiag_count(n, corr.dtype)
    return jnp.sum(a) / jnp.maximum(cnt, F64(1.0))


def _clustering_coeff(corr: jnp.ndarray, thr: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
    n = corr.shape[0]
    eye = jnp.eye(n, dtype=corr.dtype)
    off = F64(1.0) - eye
    a = (jnp.abs(corr) > thr).astype(corr.dtype) * off
    deg = jnp.sum(a, axis=1)
    a2 = a @ a
    a3 = a2 @ a
    tri = jnp.diag(a3)
    denom = deg * (deg - F64(1.0)) + eps
    c_i = tri / denom
    valid = (deg > F64(1.0)).astype(corr.dtype)
    return jnp.sum(c_i * valid) / jnp.maximum(jnp.sum(valid), F64(1.0))


def _eig_sorted_desc_sym(x: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
    xs = _symmetrize(x)
    evals = jnp.linalg.eigvalsh(xs)
    evals = jnp.sort(jnp.maximum(evals, eps))[::-1]
    return evals


@partial(jax.jit, static_argnames=("cfg",))
def cross_asset_structure_extended(
    *,
    # core snapshot tensors
    features_z: jnp.ndarray,        # [N,F]
    pca_eigvals: jnp.ndarray,       # [P]
    asset_corr: jnp.ndarray,        # [N,N]
    corr_baseline: jnp.ndarray,     # [N,N]

    # geometry profiles (per-asset vectors over a fixed profile axis M)
    skew_profile: jnp.ndarray,      # [N,M]
    convexity_profile: jnp.ndarray, # [N,M]
    term_profile: jnp.ndarray,      # [N,M]
    curvature_profile: jnp.ndarray, # [N,M]
    atm_profile: jnp.ndarray,       # [N,M]
    wing_profile: jnp.ndarray,      # [N,M]
    shape_factors: jnp.ndarray,     # [N,S]

    # dealer-flow profiles (fixed axis G, e.g., tenors or strike-buckets)
    gex_profile: jnp.ndarray,       # [N,G]
    vanna_profile: jnp.ndarray,     # [N,G]
    vega_profile: jnp.ndarray,      # [N,G]

    cfg: C2Config,
) -> Dict[str, jnp.ndarray]:
    eps = F64(cfg.eps)
    corr_thr = F64(getattr(cfg, "corr_graph_threshold", 0.40))
    flow_thr = F64(getattr(cfg, "flow_graph_threshold", 0.35))
    topk_cfg = int(getattr(cfg, "flow_topk", 10))

    # Cast + sanitize
    features_z = jnp.asarray(features_z, dtype=F64)
    pca_eigvals = jnp.asarray(pca_eigvals, dtype=F64)
    asset_corr = jnp.clip(_symmetrize(jnp.asarray(asset_corr, dtype=F64)), F64(-1.0), F64(1.0))
    corr_baseline = jnp.clip(_symmetrize(jnp.asarray(corr_baseline, dtype=F64)), F64(-1.0), F64(1.0))

    skew_profile = jnp.asarray(skew_profile, dtype=F64)
    convexity_profile = jnp.asarray(convexity_profile, dtype=F64)
    term_profile = jnp.asarray(term_profile, dtype=F64)
    curvature_profile = jnp.asarray(curvature_profile, dtype=F64)
    atm_profile = jnp.asarray(atm_profile, dtype=F64)
    wing_profile = jnp.asarray(wing_profile, dtype=F64)
    shape_factors = jnp.asarray(shape_factors, dtype=F64)

    gex_profile = jnp.asarray(gex_profile, dtype=F64)
    vanna_profile = jnp.asarray(vanna_profile, dtype=F64)
    vega_profile = jnp.asarray(vega_profile, dtype=F64)

    n_assets = features_z.shape[0]
    k_top = min(topk_cfg, int(n_assets))

    # -------------------------------------------------------------------------
    # 1..3: PCA-spectrum structure
    # -------------------------------------------------------------------------
    lam = jnp.maximum(pca_eigvals, eps)
    lam_sum = jnp.maximum(jnp.sum(lam), eps)

    eigenvalue_gap_ladder = jnp.maximum(lam[:-1] - lam[1:], F64(0.0))
    effective_dimensionality = (lam_sum * lam_sum) / jnp.maximum(jnp.sum(lam * lam), eps)
    p_lam = lam / lam_sum
    eigenvalue_entropy = -jnp.sum(p_lam * jnp.log(jnp.maximum(p_lam, eps)))

    # -------------------------------------------------------------------------
    # 6..13: Correlation geometry
    # -------------------------------------------------------------------------
    average_pairwise_correlation = _offdiag_mean(asset_corr)
    correlation_dispersion = _offdiag_std(asset_corr, average_pairwise_correlation, eps)
    correlation_skewness, correlation_kurtosis = _offdiag_skew_kurt(
        asset_corr, average_pairwise_correlation, correlation_dispersion, eps
    )
    correlation_regime_distance = jnp.linalg.norm(asset_corr - corr_baseline, ord="fro")

    graph_for_centrality = jnp.abs(asset_corr)
    _, evecs = jnp.linalg.eigh(_symmetrize(graph_for_centrality))
    centrality = jnp.abs(evecs[:, -1])
    correlation_eigenvector_centrality = centrality / jnp.maximum(jnp.sum(centrality), eps)

    correlation_graph_density = _graph_density(asset_corr, corr_thr)
    network_clustering_coefficient = _clustering_coeff(asset_corr, corr_thr, eps)

    # -------------------------------------------------------------------------
    # 14..20: Surface alignment / synchronization
    # -------------------------------------------------------------------------
    skew_cos = _cosine_matrix(skew_profile, eps)
    convexity_cos = _cosine_matrix(convexity_profile, eps)
    term_cos = _cosine_matrix(term_profile, eps)
    curvature_corr = _corr_from_profiles(curvature_profile, eps)
    atm_corr = _corr_from_profiles(atm_profile, eps)
    wing_corr = _corr_from_profiles(wing_profile, eps)

    skew_alignment_index = _offdiag_mean(skew_cos)
    convexity_alignment_index = _offdiag_mean(convexity_cos)
    term_structure_alignment = _offdiag_mean(term_cos)
    surface_curvature_correlation = _offdiag_mean(curvature_corr)
    atm_vol_synchronization = _offdiag_mean(atm_corr)
    wing_stress_synchronization = _offdiag_mean(wing_corr)

    shape_centered = shape_factors - jnp.mean(shape_factors, axis=0, keepdims=True)
    surface_shape_dispersion = jnp.mean(jnp.var(shape_centered, axis=0))

    # -------------------------------------------------------------------------
    # 21..26: Dealer-flow coupling
    # -------------------------------------------------------------------------
    cross_gex_correlation_matrix = _corr_from_profiles(gex_profile, eps)  # [N,N]
    cross_vanna_corr = _corr_from_profiles(vanna_profile, eps)            # [N,N]
    cross_vega_corr = _corr_from_profiles(vega_profile, eps)              # [N,N]

    gex_asset = jnp.sum(gex_profile, axis=1)  # [N]
    abs_gex = jnp.abs(gex_asset)
    total_abs_gex = jnp.maximum(jnp.sum(abs_gex), eps)
    topk_vals, _ = jax.lax.top_k(abs_gex, k_top)
    systemic_gamma_concentration = jnp.sum(topk_vals) / total_abs_gex

    vanna_synchronization = _offdiag_mean(cross_vanna_corr)
    vega_network_density = _graph_density(cross_vega_corr, flow_thr)

    gex_share = abs_gex / total_abs_gex
    flow_concentration_entropy = -jnp.sum(gex_share * jnp.log(jnp.maximum(gex_share, eps)))

    cross_gex_abs_mean = _offdiag_abs_mean(cross_gex_correlation_matrix)
    flow_crowding_index = systemic_gamma_concentration * cross_gex_abs_mean

    # -------------------------------------------------------------------------
    # 27..33: Allocation-level structure
    # -------------------------------------------------------------------------
    corr_eigs_desc = _eig_sorted_desc_sym(asset_corr, eps)
    risk_dominance_index = corr_eigs_desc[0] / jnp.maximum(jnp.sum(corr_eigs_desc), eps)

    avg_pairwise_abs_corr = _offdiag_abs_mean(asset_corr)
    diversification_capacity = F64(1.0) / jnp.maximum(avg_pairwise_abs_corr, eps)

    eye_n = jnp.eye(asset_corr.shape[0], dtype=F64)
    corr_reg = _symmetrize(asset_corr) + eps * eye_n
    sign_det, log_det = jnp.linalg.slogdet(corr_reg)
    fragility_score = jnp.where(
        sign_det > 0, jnp.exp(jnp.clip(log_det, F64(-50.0), F64(50.0))), F64(0.0)
    )

    systemic_leverage_proxy = jnp.linalg.norm(corr_reg, ord="fro")
    hedging_efficiency_metric = risk_dominance_index

    # factor orthogonality from cross-factor correlation of features
    fz = features_z - jnp.mean(features_z, axis=0, keepdims=True)
    factor_cov = (fz.T @ fz) / jnp.maximum(F64(fz.shape[0] - 1), F64(1.0))
    d = jnp.sqrt(jnp.maximum(jnp.diag(factor_cov), eps))
    denom = d[:, None] * d[None, :]
    factor_corr = _symmetrize(factor_cov / jnp.maximum(denom, eps))
    orthogonality_score = _offdiag_abs_mean(factor_corr)

    # snapshot proxy: conditioning * regime displacement
    corr_eigs_safe = jnp.maximum(corr_eigs_desc, eps)
    corr_condition_number = corr_eigs_safe[0] / corr_eigs_safe[-1]
    allocation_instability_indicator = corr_condition_number * correlation_regime_distance

    return {
        # 1..3
        "eigenvalue_gap_ladder": eigenvalue_gap_ladder,                        # [P-1]
        "effective_dimensionality": effective_dimensionality,                  # scalar
        "eigenvalue_entropy": eigenvalue_entropy,                              # scalar

        # 6..13
        "average_pairwise_correlation": average_pairwise_correlation,          # scalar
        "correlation_dispersion": correlation_dispersion,                      # scalar
        "correlation_skewness": correlation_skewness,                          # scalar
        "correlation_kurtosis": correlation_kurtosis,                          # scalar
        "correlation_regime_distance": correlation_regime_distance,            # scalar
        "correlation_eigenvector_centrality": correlation_eigenvector_centrality,  # [N]
        "correlation_graph_density": correlation_graph_density,                # scalar
        "network_clustering_coefficient": network_clustering_coefficient,      # scalar

        # 14..20
        "skew_alignment_index": skew_alignment_index,                          # scalar
        "convexity_alignment_index": convexity_alignment_index,                # scalar
        "term_structure_alignment": term_structure_alignment,                  # scalar
        "surface_curvature_correlation": surface_curvature_correlation,        # scalar
        "atm_vol_synchronization": atm_vol_synchronization,                    # scalar
        "wing_stress_synchronization": wing_stress_synchronization,            # scalar
        "surface_shape_dispersion": surface_shape_dispersion,                  # scalar

        # 21..26
        "cross_gex_correlation_matrix": cross_gex_correlation_matrix,          # [N,N]
        "systemic_gamma_concentration": systemic_gamma_concentration,          # scalar
        "vanna_synchronization": vanna_synchronization,                        # scalar
        "vega_network_density": vega_network_density,                          # scalar
        "flow_concentration_entropy": flow_concentration_entropy,              # scalar
        "flow_crowding_index": flow_crowding_index,                            # scalar

        # 27..33
        "risk_dominance_index": risk_dominance_index,                          # scalar
        "diversification_capacity": diversification_capacity,                  # scalar
        "fragility_score": fragility_score,                                    # scalar
        "systemic_leverage_proxy": systemic_leverage_proxy,                    # scalar
        "hedging_efficiency_metric": hedging_efficiency_metric,                # scalar
        "orthogonality_score": orthogonality_score,                            # scalar
        "allocation_instability_indicator": allocation_instability_indicator,  # scalar

        # diagnostics
        "corr_condition_number": corr_condition_number,                        # scalar
        "corr_eigenvalues_desc": corr_eigs_desc,                               # [N]
        "factor_corr": factor_corr,                                            # [F,F]
        "cross_vanna_corr": cross_vanna_corr,                                  # [N,N]
        "cross_vega_corr": cross_vega_corr,                                    # [N,N]
    }


def run_cross_asset_structure_extended(
    *,
    features_z: jnp.ndarray,
    pca_eigvals: jnp.ndarray,
    asset_corr: jnp.ndarray,
    corr_baseline: jnp.ndarray,
    skew_profile: jnp.ndarray,
    convexity_profile: jnp.ndarray,
    term_profile: jnp.ndarray,
    curvature_profile: jnp.ndarray,
    atm_profile: jnp.ndarray,
    wing_profile: jnp.ndarray,
    shape_factors: jnp.ndarray,
    gex_profile: jnp.ndarray,
    vanna_profile: jnp.ndarray,
    vega_profile: jnp.ndarray,
    cfg: C2Config,
) -> Dict[str, jnp.ndarray]:
    out = cross_asset_structure_extended(
        features_z=features_z,
        pca_eigvals=pca_eigvals,
        asset_corr=asset_corr,
        corr_baseline=corr_baseline,
        skew_profile=skew_profile,
        convexity_profile=convexity_profile,
        term_profile=term_profile,
        curvature_profile=curvature_profile,
        atm_profile=atm_profile,
        wing_profile=wing_profile,
        shape_factors=shape_factors,
        gex_profile=gex_profile,
        vanna_profile=vanna_profile,
        vega_profile=vega_profile,
        cfg=cfg,
    )
    _ = out["risk_dominance_index"].block_until_ready()
    return out


def aot_compile_cross_asset_structure_extended(
    cfg: C2Config,
    feature_dim: int,
    pca_dim: int,
    profile_dim: int,
    shape_dim: int,
):
    """
    AOT compile for fixed static contracts.
    """
    n = cfg.n_assets

    features_aval = jax.ShapeDtypeStruct((n, feature_dim), F64)
    pca_eval_aval = jax.ShapeDtypeStruct((pca_dim,), F64)
    corr_aval = jax.ShapeDtypeStruct((n, n), F64)

    prof_aval = jax.ShapeDtypeStruct((n, profile_dim), F64)
    shape_aval = jax.ShapeDtypeStruct((n, shape_dim), F64)

    lowered = cross_asset_structure_extended.lower(
        features_z=features_aval,
        pca_eigvals=pca_eval_aval,
        asset_corr=corr_aval,
        corr_baseline=corr_aval,
        skew_profile=prof_aval,
        convexity_profile=prof_aval,
        term_profile=prof_aval,
        curvature_profile=prof_aval,
        atm_profile=prof_aval,
        wing_profile=prof_aval,
        shape_factors=shape_aval,
        gex_profile=prof_aval,
        vanna_profile=prof_aval,
        vega_profile=prof_aval,
        vega_profile=prof_aval,
        cfg=cfg,
    )
    return lowered.compile()

