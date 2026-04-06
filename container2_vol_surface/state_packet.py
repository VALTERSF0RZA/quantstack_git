# c2/state_packet.py
# =============================================================================
# Canonical C2 -> C3 packet builder
# FP64 • JAX • XLA/JIT-safe
#
# Notes:
# - This is a pure packaging boundary (no heavy math).
# - JIT here is optional; included for consistency and to keep one traced path.
# - Dict[str, Array] is a valid JAX PyTree and can be returned from jitted funcs.
# =============================================================================

from __future__ import annotations

from functools import partial
from typing import Dict

import jax
import jax.numpy as jnp

from .config import F64

# Global FP64 consistency
jax.config.update("jax_enable_x64", True)


def _as_fp64(x: jnp.ndarray) -> jnp.ndarray:
    """Cast floating tensors to FP64; keep integer tensors unchanged."""
    x = jnp.asarray(x)
    return x.astype(F64) if jnp.issubdtype(x.dtype, jnp.floating) else x


@partial(jax.jit, static_argnames=())
def build_c2_state_packet(
    *,
    sigma_noarb: jnp.ndarray,      # [N,K,T] float64
    total_var_noarb: jnp.ndarray,  # [N,K,T] float64
    arb_metrics: jnp.ndarray,      # [4] float64

    features_raw: jnp.ndarray,     # [N,F] float64
    features_z: jnp.ndarray,       # [N,F] float64
    feature_mu: jnp.ndarray,       # [F] float64
    feature_std: jnp.ndarray,      # [F] float64

    dsdm: jnp.ndarray,             # [N,K,T] float64
    d2sdm2: jnp.ndarray,           # [N,K,T] float64
    dsdT: jnp.ndarray,             # [N,K,T] float64
    d2sdT2: jnp.ndarray,           # [N,K,T] float64

    pca_scores: jnp.ndarray,       # [N,p] float64
    pca_eigvecs: jnp.ndarray,      # [F,p] float64
    pca_eigvals: jnp.ndarray,      # [p] float64
    pca_explained: jnp.ndarray,    # [p] float64

    regime_id: jnp.ndarray,        # [N] int32
    regime_onehot: jnp.ndarray,    # [N,R] float64
    cluster_id: jnp.ndarray,       # [N] int32

    factor_cov: jnp.ndarray,       # [F,F] float64
    factor_corr: jnp.ndarray,      # [F,F] float64
    asset_corr: jnp.ndarray,       # [N,N] float64
    coupling_NN3: jnp.ndarray,     # [N,N,3] float64
) -> Dict[str, jnp.ndarray]:
    """
    Canonical C2 -> C3 packet contract.
    Returns a JAX PyTree dictionary suitable for downstream JIT/XLA pipelines.
    """

    # Enforce deterministic dtype contract at boundary
    sigma_noarb = _as_fp64(sigma_noarb)
    total_var_noarb = _as_fp64(total_var_noarb)
    arb_metrics = _as_fp64(arb_metrics)

    features_raw = _as_fp64(features_raw)
    features_z = _as_fp64(features_z)
    feature_mu = _as_fp64(feature_mu)
    feature_std = _as_fp64(feature_std)

    dsdm = _as_fp64(dsdm)
    d2sdm2 = _as_fp64(d2sdm2)
    dsdT = _as_fp64(dsdT)
    d2sdT2 = _as_fp64(d2sdT2)

    pca_scores = _as_fp64(pca_scores)
    pca_eigvecs = _as_fp64(pca_eigvecs)
    pca_eigvals = _as_fp64(pca_eigvals)
    pca_explained = _as_fp64(pca_explained)

    # Keep IDs integer, cast onehot + couplings to FP64
    regime_id = jnp.asarray(regime_id, dtype=jnp.int32)
    regime_onehot = _as_fp64(regime_onehot)
    cluster_id = jnp.asarray(cluster_id, dtype=jnp.int32)

    factor_cov = _as_fp64(factor_cov)
    factor_corr = _as_fp64(factor_corr)
    asset_corr = _as_fp64(asset_corr)
    coupling_NN3 = _as_fp64(coupling_NN3)

    return {
        "sigma_noarb": sigma_noarb,
        "total_var_noarb": total_var_noarb,
        "arb_metrics": arb_metrics,

        "features_raw": features_raw,
        "features_z": features_z,
        "feature_mu": feature_mu,
        "feature_std": feature_std,

        "dsdm": dsdm,
        "d2sdm2": d2sdm2,
        "dsdT": dsdT,
        "d2sdT2": d2sdT2,

        "pca_scores": pca_scores,
        "pca_eigvecs": pca_eigvecs,
        "pca_eigvals": pca_eigvals,
        "pca_explained": pca_explained,

        "regime_id": regime_id,
        "regime_onehot": regime_onehot,
        "cluster_id": cluster_id,

        "factor_cov": factor_cov,
        "factor_corr": factor_corr,
        "asset_corr": asset_corr,
        "coupling_NN3": coupling_NN3,
    }


def aot_compile_build_c2_state_packet(
    *,
    n_assets: int,
    n_strikes: int,
    n_tenors: int,
    n_factors: int,
    pca_components: int,
    num_regimes: int,
):
    """
    Optional AOT-style XLA compile helper for static contracts.
    Returns compiled callable for this packet builder.
    """
    sigma_aval = jax.ShapeDtypeStruct((n_assets, n_strikes, n_tenors), F64)
    w_aval = jax.ShapeDtypeStruct((n_assets, n_strikes, n_tenors), F64)
    arb_aval = jax.ShapeDtypeStruct((4,), F64)

    fr_aval = jax.ShapeDtypeStruct((n_assets, n_factors), F64)
    fz_aval = jax.ShapeDtypeStruct((n_assets, n_factors), F64)
    mu_aval = jax.ShapeDtypeStruct((n_factors,), F64)
    sd_aval = jax.ShapeDtypeStruct((n_factors,), F64)

    d_aval = jax.ShapeDtypeStruct((n_assets, n_strikes, n_tenors), F64)

    pca_s_aval = jax.ShapeDtypeStruct((n_assets, pca_components), F64)
    pca_v_aval = jax.ShapeDtypeStruct((n_factors, pca_components), F64)
    pca_l_aval = jax.ShapeDtypeStruct((pca_components,), F64)
    pca_e_aval = jax.ShapeDtypeStruct((pca_components,), F64)

    rid_aval = jax.ShapeDtypeStruct((n_assets,), jnp.int32)
    roh_aval = jax.ShapeDtypeStruct((n_assets, num_regimes), F64)
    cid_aval = jax.ShapeDtypeStruct((n_assets,), jnp.int32)

    fcov_aval = jax.ShapeDtypeStruct((n_factors, n_factors), F64)
    fcorr_aval = jax.ShapeDtypeStruct((n_factors, n_factors), F64)
    acorr_aval = jax.ShapeDtypeStruct((n_assets, n_assets), F64)
    cnn3_aval = jax.ShapeDtypeStruct((n_assets, n_assets, 3), F64)

    lowered = build_c2_state_packet.lower(
        sigma_noarb=sigma_aval,
        total_var_noarb=w_aval,
        arb_metrics=arb_aval,
        features_raw=fr_aval,
        features_z=fz_aval,
        feature_mu=mu_aval,
        feature_std=sd_aval,
        dsdm=d_aval,
        d2sdm2=d_aval,
        dsdT=d_aval,
        d2sdT2=d_aval,
        pca_scores=pca_s_aval,
        pca_eigvecs=pca_v_aval,
        pca_eigvals=pca_l_aval,
        pca_explained=pca_e_aval,
        regime_id=rid_aval,
        regime_onehot=roh_aval,
        cluster_id=cid_aval,
        factor_cov=fcov_aval,
        factor_corr=fcorr_aval,
        asset_corr=acorr_aval,
        coupling_NN3=cnn3_aval,
    )
    return lowered.compile()

