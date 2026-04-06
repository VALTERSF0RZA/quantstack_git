# c2/regime_tags.py
# =============================================================================
# C2 Deterministic Regime Classification
# FP64 • JAX • XLA • JIT
#
# Produces:
#   regime_id   [N] int32
#   regime_oh   [N,R]
#   cluster_id  [N] int32   (PCA centroid assignment)
#
# Input:
#   features_z : [N,F]
#   pca_scores : [N,k]
# =============================================================================

from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from .config import C2Config, F64

# Explicit FP64
jax.config.update("jax_enable_x64", True)


# -----------------------------------------------------------------------------
# Main deterministic regime classifier
# -----------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("cfg",))
def classify_regimes_c2(
    features_z: jnp.ndarray,  # [N,F]
    pca_scores: jnp.ndarray,  # [N,k]
    cfg: C2Config,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Deterministic C2 regime labels (non-stochastic).

    Returns:
      regime_id   [N] int32
      regime_oh   [N,R] FP64
      cluster_id  [N] int32 (nearest static centroid in PC1/PC2)
    """

    # ------------------------------------------------------------------
    # FP64 cast
    # ------------------------------------------------------------------
    z = jnp.asarray(features_z, dtype=F64)
    pc = jnp.asarray(pca_scores, dtype=F64)
    thr = jnp.asarray(cfg.z_thr, dtype=F64)

    # Factor indices (must match factor_engine ordering)
    IDX_LEVEL = 0
    IDX_SKEW = 1
    IDX_CONV = 2
    IDX_TSLOPE = 3

    level_z = z[:, IDX_LEVEL]
    skew_z = z[:, IDX_SKEW]
    conv_z = z[:, IDX_CONV]
    tslope_z = z[:, IDX_TSLOPE]

    # ------------------------------------------------------------------
    # Rule-based deterministic regimes
    # ------------------------------------------------------------------
    # regime IDs:
    # 0 = default / mixed
    # 1 = crash-convexity
    # 2 = front stress
    # 3 = calm / reflation
    # 4 = neutral compressed
    # ------------------------------------------------------------------

    r = jnp.zeros((z.shape[0],), dtype=jnp.int32)

    # 1) crash-convexity
    cond1 = (skew_z < -thr) & (conv_z > thr)
    r = jnp.where(cond1, jnp.int32(1), r)

    # 2) front stress
    cond2 = (level_z > thr) & (tslope_z < -thr)
    r = jnp.where(cond2, jnp.int32(2), r)

    # 3) calm / reflation
    cond3 = (level_z < -thr) & (skew_z > thr) & (conv_z < 0.0)
    r = jnp.where(cond3, jnp.int32(3), r)

    # 4) neutral compressed
    cond4 = (
        (jnp.abs(skew_z) < 0.5 * thr)
        & (jnp.abs(conv_z) < 0.5 * thr)
        & (jnp.abs(level_z) < 0.75 * thr)
    )
    r = jnp.where(cond4, jnp.int32(4), r)

    # one-hot encoding (FP64 for downstream consistency)
    regime_oh = jax.nn.one_hot(
        r,
        cfg.num_regimes,
        dtype=F64,
    )

    # ------------------------------------------------------------------
    # PCA centroid clustering (PC1 / PC2)
    # ------------------------------------------------------------------
    # Ensure at least 2 dimensions
    def build_pc2():
        if cfg.pca_components >= 2:
            return pc[:, :2]
        pad = jnp.zeros((pc.shape[0], 1), dtype=F64)
        return jnp.concatenate([pc, pad], axis=1)

    pc2 = build_pc2()

    # static quadrant centroids
    centroids = jnp.array(
        [
            [-1.5, -1.5],
            [-1.5,  1.5],
            [ 1.5, -1.5],
            [ 1.5,  1.5],
        ],
        dtype=F64,
    )  # [4,2]

    # distance squared: [N,4]
    diff = pc2[:, None, :] - centroids[None, :, :]
    dist2 = jnp.sum(diff * diff, axis=2)

    cluster_id = jnp.argmin(dist2, axis=1).astype(jnp.int32)

    return r, regime_oh, cluster_id


# -----------------------------------------------------------------------------
# AOT compile helper
# -----------------------------------------------------------------------------
def aot_compile_classify_regimes_c2(cfg: C2Config):
    """
    XLA AOT compile helper for fixed shape contract.
    """

    # factor engine output size (must match your pipeline)
    F = 8
    k = cfg.pca_components

    feats_aval = jax.ShapeDtypeStruct(
        (cfg.n_assets, F),
        F64,
    )

    pca_aval = jax.ShapeDtypeStruct(
        (cfg.n_assets, k),
        F64,
    )

    lowered = classify_regimes_c2.lower(
        feats_aval,
        pca_aval,
        cfg,
    )
    return lowered.compile()


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = C2Config(
        n_assets=200,
        pca_components=5,
        num_regimes=5,
    )

    N = cfg.n_assets
    F = 8
    K = cfg.pca_components

    key1 = jax.random.PRNGKey(0)
    key2 = jax.random.PRNGKey(1)

    feats = jax.random.normal(key1, (N, F), dtype=F64)
    pca = jax.random.normal(key2, (N, K), dtype=F64)

    rid, roh, cid = classify_regimes_c2(feats, pca, cfg)
    _ = roh.block_until_ready()

    print("regime_id :", rid.shape, rid.dtype)
    print("regime_oh :", roh.shape, roh.dtype)
    print("cluster_id:", cid.shape, cid.dtype)

