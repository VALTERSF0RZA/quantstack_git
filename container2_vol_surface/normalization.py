# Normalization.py
# =============================================================================
# Cross-Asset Normalization + PCA
# FP64 • JAX • XLA • FULL JIT
#
# Expected usage:
#   called INSIDE C2 orchestrator (single compilation boundary)
# =============================================================================

from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from .config import C2Config, F64


# -----------------------------------------------------------------------------
# 1) Cross-Asset Factor Normalization (FULL JIT)
# -----------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("cfg",))
def normalize_cross_asset(
    features_raw: jnp.ndarray,   # [N,F]
    cfg: C2Config,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Cross-asset z-score normalization.

    Inputs:
        features_raw : [N,F]

    Outputs:
        features_z : [N,F]
        mu         : [F]
        std        : [F]
    """
    eps = F64(cfg.eps)

    # Ensure FP64
    X = jnp.asarray(features_raw, F64)

    # Mean across assets
    mu = jnp.mean(X, axis=0)

    # Std across assets
    std = jnp.std(X, axis=0)

    # Numerical stability
    std = jnp.maximum(std, eps)

    # Z-score normalization
    z = (X - mu[None, :]) / std[None, :]

    return z, mu, std


# -----------------------------------------------------------------------------
# 2) PCA / Eigenmode Decomposition (FULL JIT)
# -----------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("cfg",))
def pca_factors(
    features_z: jnp.ndarray,   # [N,F]
    cfg: C2Config,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    PCA on normalized cross-asset factors.

    Outputs:
        scores     : [N,k]
        eigvecs_k  : [F,k]
        eigvals_k  : [k]
        explained  : [k]
    """
    eps = F64(cfg.eps)

    X = jnp.asarray(features_z, F64)
    n = X.shape[0]

    # -------------------------------------------------------------------------
    # Center features
    # -------------------------------------------------------------------------
    Xc = X - jnp.mean(X, axis=0, keepdims=True)

    # Covariance matrix
    cov = (Xc.T @ Xc) / jnp.maximum(F64(n - 1), F64(1.0))

    # -------------------------------------------------------------------------
    # Eigen decomposition (XLA-safe)
    # eigh returns ascending eigenvalues
    # -------------------------------------------------------------------------
    eigvals, eigvecs = jnp.linalg.eigh(cov)

    # Reverse instead of argsort (faster + static)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    # -------------------------------------------------------------------------
    # Truncate components
    # -------------------------------------------------------------------------
    k = cfg.pca_components

    eigvals_k = eigvals[:k]
    eigvecs_k = eigvecs[:, :k]

    # Project into PCA space
    scores = Xc @ eigvecs_k

    # Explained variance ratio
    explained = eigvals_k / jnp.maximum(jnp.sum(eigvals), eps)

    return scores, eigvecs_k, eigvals_k, explained


# -----------------------------------------------------------------------------
# Optional fused helper (MOST EFFICIENT path)
# -----------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("cfg",))
def normalize_and_pca(
    features_raw: jnp.ndarray,
    cfg: C2Config,
):
    """
    Single fused normalization + PCA pass.

    Use this if you want maximum XLA fusion.
    """
    z, mu, std = normalize_cross_asset(features_raw, cfg)
    scores, vecs, vals, exp = pca_factors(z, cfg)

    return z, mu, std, scores, vecs, vals, exp

