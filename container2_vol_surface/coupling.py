# c2/coupling.py
# =============================================================================
# Cross-Asset Coupling Tensor
# FP64 • JAX • XLA • JIT
#
# Produces:
#   factor_cov    [F,F]
#   factor_corr   [F,F]
#   asset_corr    [N,N]
#   coupling_NN3  [N,N,3]
#
# Input:
#   features_z : [N,F]   (cross-asset normalized factors)
# =============================================================================

from __future__ import annotations

from functools import partial
from typing import Dict

import jax
import jax.numpy as jnp

from .config import C2Config, F64
from .math_utils import cov_to_corr

# Explicit FP64 enable
jax.config.update("jax_enable_x64", True)


# -----------------------------------------------------------------------------
# Core coupling kernel (single XLA graph)
# -----------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("cfg",))
def cross_asset_coupling(
    features_z: jnp.ndarray,
    cfg: C2Config,
) -> Dict[str, jnp.ndarray]:
    """
    Cross-asset coupling outputs:
      factor_cov    [F,F]
      factor_corr   [F,F]
      asset_corr    [N,N] (cosine similarity)
      coupling_NN3  [N,N,3] channels:
          0 level
          1 skew
          2 convexity
    """
    eps = jnp.asarray(cfg.eps, dtype=F64)

    # FP64 cast
    X = jnp.asarray(features_z, dtype=F64)  # [N,F]
    n = X.shape[0]

    # ------------------------------------------------------------------
    # Factor covariance / correlation (factor space)
    # ------------------------------------------------------------------
    # X already normalized cross-sectionally in previous stage.
    factor_cov = (X.T @ X) / jnp.maximum(jnp.asarray(n - 1, dtype=F64), 1.0)

    factor_corr = cov_to_corr(factor_cov, cfg.eps)

    # ------------------------------------------------------------------
    # Asset similarity matrix (cosine-like)
    # ------------------------------------------------------------------
    norms = jnp.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / jnp.maximum(norms, eps)

    asset_corr = Xn @ Xn.T

    # ------------------------------------------------------------------
    # Level/skew/convexity coupling tensor
    # ------------------------------------------------------------------
    # assumes factor ordering from factor_engine:
    # [level, skew, convexity, ...]
    sel = X[:, :3]  # [N,3]

    # outer product per channel
    coupling_NN3 = sel[:, None, :] * sel[None, :, :]  # [N,N,3]

    return {
        "factor_cov": factor_cov,
        "factor_corr": factor_corr,
        "asset_corr": asset_corr,
        "coupling_NN3": coupling_NN3,
    }


# -----------------------------------------------------------------------------
# AOT compile helper
# -----------------------------------------------------------------------------
def aot_compile_cross_asset_coupling(cfg: C2Config):
    """
    XLA AOT compile for fixed shape contract.
    """
    # F determined by factor_engine (currently 8)
    f_dim = 8

    X_aval = jax.ShapeDtypeStruct(
        (cfg.n_assets, f_dim),
        F64,
    )

    lowered = cross_asset_coupling.lower(X_aval, cfg)
    return lowered.compile()


# -----------------------------------------------------------------------------
# Optional smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = C2Config(
        n_assets=200,
        pca_components=5,
    )

    N = cfg.n_assets
    F = 8  # must match factor_engine output

    # Synthetic normalized factors
    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, (N, F), dtype=F64)

    out = cross_asset_coupling(X, cfg)
    _ = out["factor_cov"].block_until_ready()

    print("factor_cov   :", out["factor_cov"].shape, out["factor_cov"].dtype)
    print("factor_corr  :", out["factor_corr"].shape, out["factor_corr"].dtype)
    print("asset_corr   :", out["asset_corr"].shape, out["asset_corr"].dtype)
    print("coupling_NN3 :", out["coupling_NN3"].shape, out["coupling_NN3"].dtype)

