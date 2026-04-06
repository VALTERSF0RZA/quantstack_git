# c2/c2a_fused.py
# =============================================================================
# C2A Fused Pass: factor_engine + normalization + regime_tags
# FP64 • JAX • XLA • JIT
#
# Input:
#   sigma      [N,K,T]
#   dsdt       [N,K,T]
#   log_m      [K]
#   tau        [T]
#
# Output:
#   features_raw      [N,F]
#   features_z        [N,F]
#   feature_mu        [F]
#   feature_std       [F]
#   pca_scores        [N,p]
#   pca_eigvecs       [F,p]
#   pca_eigvals       [p]
#   pca_explained     [p]
#   regime_id         [N] int32
#   regime_onehot     [N,R]
#   cluster_id        [N] int32
#   dsdm,d2sdm2,dsdT,d2sdT2,lapK,lapT  [N,K,T]
# =============================================================================

from __future__ import annotations

from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from .config import C2Config, F64
from .numerics import geometry_ops_bundle

jax.config.update("jax_enable_x64", True)


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _safe_std(x: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
    s = jnp.std(x, axis=0)
    return jnp.maximum(s, eps)


def _pca_from_z(features_z: jnp.ndarray, p: int, eps: jnp.ndarray):
    """
    PCA on [N,F] normalized features.
    returns:
      scores [N,p], eigvecs [F,p], eigvals [p], explained [p]
    """
    X = jnp.asarray(features_z, dtype=F64)
    n = X.shape[0]

    Xc = X - jnp.mean(X, axis=0, keepdims=True)       # [N,F]
    cov = (Xc.T @ Xc) / jnp.maximum(F64(n - 1), 1.0)  # [F,F]

    eigvals, eigvecs = jnp.linalg.eigh(cov)           # ascending
    idx = jnp.argsort(eigvals)[::-1]                  # descending
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    eigvals_p = eigvals[:p]
    eigvecs_p = eigvecs[:, :p]
    scores = Xc @ eigvecs_p

    explained = eigvals_p / jnp.maximum(jnp.sum(eigvals), eps)
    return scores, eigvecs_p, eigvals_p, explained


def _regime_rules(features_z: jnp.ndarray, thr: jnp.ndarray) -> jnp.ndarray:
    """
    deterministic regimes from factor z-scores.
    IDs:
      0 default/mixed
      1 crash-convexity
      2 front stress
      3 calm/reflation
      4 neutral compressed
    """
    z = features_z
    level_z = z[:, 0]
    skew_z = z[:, 1]
    conv_z = z[:, 2]
    tslope_z = z[:, 3]

    r = jnp.zeros((z.shape[0],), dtype=jnp.int32)
    r = jnp.where((skew_z < -thr) & (conv_z > thr), jnp.int32(1), r)
    r = jnp.where((level_z > thr) & (tslope_z < -thr), jnp.int32(2), r)
    r = jnp.where((level_z < -thr) & (skew_z > thr) & (conv_z < 0.0), jnp.int32(3), r)
    r = jnp.where(
        (jnp.abs(skew_z) < 0.5 * thr)
        & (jnp.abs(conv_z) < 0.5 * thr)
        & (jnp.abs(level_z) < 0.75 * thr),
        jnp.int32(4),
        r,
    )
    return r


def _cluster_pc12(pca_scores: jnp.ndarray, pca_components: int) -> jnp.ndarray:
    """
    nearest static centroid assignment on PC1/PC2.
    """
    if pca_components >= 2:
        pc2 = pca_scores[:, :2]
    else:
        pad = jnp.zeros((pca_scores.shape[0], 1), dtype=F64)
        pc2 = jnp.concatenate([pca_scores, pad], axis=1)

    centroids = jnp.array(
        [[-1.5, -1.5], [-1.5, 1.5], [1.5, -1.5], [1.5, 1.5]],
        dtype=F64,
    )  # [4,2]

    d2 = jnp.sum((pc2[:, None, :] - centroids[None, :, :]) ** 2, axis=2)  # [N,4]
    return jnp.argmin(d2, axis=1).astype(jnp.int32)


# -----------------------------------------------------------------------------
# fused kernel
# -----------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("cfg",))
def c2a_fused_factors_norm_regimes(
    sigma: jnp.ndarray,   # [N,K,T]
    dsdt: jnp.ndarray,    # [N,K,T]
    log_m: jnp.ndarray,   # [K]
    tau: jnp.ndarray,     # [T]
    cfg: C2Config,
) -> Dict[str, jnp.ndarray]:
    eps = jnp.asarray(cfg.eps, dtype=F64)

    sigma = jnp.asarray(sigma, dtype=F64)
    dsdt = jnp.asarray(dsdt, dtype=F64)
    log_m = jnp.asarray(log_m, dtype=F64)
    tau = jnp.asarray(tau, dtype=F64)

    # -------- 1) geometry (fused numerics call) --------
    dm = jnp.maximum(jnp.mean(jnp.diff(log_m)), eps)
    dT = jnp.maximum(jnp.mean(jnp.diff(tau)), eps)

    dsdm, d2sdm2, dsdT, d2sdT2, lapK, lapT = geometry_ops_bundle(
        sigma, dm, dT, cfg.eps
    )

    # -------- 2) factor extraction --------
    atm_idx = jnp.argmin(jnp.abs(log_m))
    short_idx = 0
    long_idx = tau.shape[0] - 1

    level = jnp.mean(sigma, axis=(1, 2))
    skew = jnp.mean(dsdm[:, atm_idx, :], axis=1)
    convexity = jnp.mean(d2sdm2[:, atm_idx, :], axis=1)
    term_slope = jnp.mean(dsdT[:, atm_idx, :], axis=1)
    term_curvature = jnp.mean(d2sdT2[:, atm_idx, :], axis=1)
    velocity = jnp.mean(dsdt, axis=(1, 2))
    vov_proxy = jnp.std(sigma[:, atm_idx, :], axis=1)
    term_spread = sigma[:, atm_idx, long_idx] - sigma[:, atm_idx, short_idx]

    features_raw = jnp.column_stack(
        [level, skew, convexity, term_slope, term_curvature, velocity, vov_proxy, term_spread]
    )  # [N,F=8]

    # -------- 3) cross-asset normalization --------
    feature_mu = jnp.mean(features_raw, axis=0)             # [F]
    feature_std = _safe_std(features_raw, eps)              # [F]
    features_z = (features_raw - feature_mu[None, :]) / feature_std[None, :]  # [N,F]

    # -------- 4) PCA --------
    p = cfg.pca_components
    pca_scores, pca_eigvecs, pca_eigvals, pca_explained = _pca_from_z(features_z, p, eps)

    # -------- 5) deterministic regimes --------
    thr = jnp.asarray(cfg.z_thr, dtype=F64)
    regime_id = _regime_rules(features_z, thr)                          # [N]
    regime_onehot = jax.nn.one_hot(regime_id, cfg.num_regimes, dtype=F64)
    cluster_id = _cluster_pc12(pca_scores, cfg.pca_components)          # [N]

    return {
        # factors
        "features_raw": features_raw,
        "features_z": features_z,
        "feature_mu": feature_mu,
        "feature_std": feature_std,

        # PCA
        "pca_scores": pca_scores,
        "pca_eigvecs": pca_eigvecs,
        "pca_eigvals": pca_eigvals,
        "pca_explained": pca_explained,

        # regimes
        "regime_id": regime_id,
        "regime_onehot": regime_onehot,
        "cluster_id": cluster_id,

        # derivatives for audit/debug
        "dsdm": dsdm,
        "d2sdm2": d2sdm2,
        "dsdT": dsdT,
        "d2sdT2": d2sdT2,
        "lapK": lapK,
        "lapT": lapT,
    }


# -----------------------------------------------------------------------------
# runtime wrapper
# -----------------------------------------------------------------------------
def run_c2a_fused(
    sigma: jnp.ndarray,
    dsdt: jnp.ndarray,
    log_m: jnp.ndarray,
    tau: jnp.ndarray,
    cfg: C2Config,
) -> Dict[str, jnp.ndarray]:
    out = c2a_fused_factors_norm_regimes(sigma, dsdt, log_m, tau, cfg)
    _ = out["features_z"].block_until_ready()  # observability barrier
    return out


# -----------------------------------------------------------------------------
# AOT compile helper
# -----------------------------------------------------------------------------
def aot_compile_c2a_fused(cfg: C2Config):
    sigma_aval = jax.ShapeDtypeStruct((cfg.n_assets, cfg.n_strikes, cfg.n_tenors), F64)
    dsdt_aval = jax.ShapeDtypeStruct((cfg.n_assets, cfg.n_strikes, cfg.n_tenors), F64)
    logm_aval = jax.ShapeDtypeStruct((cfg.n_strikes,), F64)
    tau_aval = jax.ShapeDtypeStruct((cfg.n_tenors,), F64)

    lowered = c2a_fused_factors_norm_regimes.lower(sigma_aval, dsdt_aval, logm_aval, tau_aval, cfg)
    return lowered.compile()


if __name__ == "__main__":
    # smoke test
    cfg = C2Config(
        n_assets=200,
        n_strikes=64,
        n_tenors=32,
        pca_components=5,
        num_regimes=5,
    )

    N, K, T = cfg.n_assets, cfg.n_strikes, cfg.n_tenors
    log_m = jnp.linspace(-0.5, 0.5, K, dtype=F64)
    tau = jnp.linspace(1.0 / 365.0, 2.0, T, dtype=F64)

    base = 0.18 + 0.04 * jnp.exp(-2.0 * tau)[None, None, :] + 0.03 * (log_m[None, :, None] ** 2)
    asset_shift = jnp.linspace(-0.025, 0.025, N, dtype=F64)[:, None, None]
    sigma = jnp.maximum(base + asset_shift, 0.03)

    dsdt = jnp.zeros_like(sigma)

    out = run_c2a_fused(sigma, dsdt, log_m, tau, cfg)
    print("features_z:", out["features_z"].shape, out["features_z"].dtype)
    print("pca_scores:", out["pca_scores"].shape, out["pca_scores"].dtype)
    print("regime_id:", out["regime_id"].shape, out["regime_id"].dtype)
    print("cluster_id:", out["cluster_id"].shape, out["cluster_id"].dtype)

