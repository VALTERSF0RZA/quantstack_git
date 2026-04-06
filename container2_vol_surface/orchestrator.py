# container2_vol_surface/orchestrator.py
from __future__ import annotations

from functools import partial
from typing import Dict

import jax
import jax.numpy as jnp

from .config import C2Config, F64, cast_inputs_fp64
from .arbitrage_constraints import enforce_surface_noarb
from .factor_engine import extract_surface_factors
from .normalization import normalize_cross_asset, pca_factors
from .regime_tags import classify_regimes_c2
from .coupling import cross_asset_coupling
from .cross_asset_structure import compute_c2_only_category1_extended
from .state_packet import build_c2_state_packet


jax.config.update("jax_enable_x64", True)


# ============================================================
# CORE C2 PIPELINE (single JIT boundary)
# ============================================================

@partial(jax.jit, static_argnames=("cfg",))
def c2_state_core(
    sigma_raw: jnp.ndarray,   # [N,K,T]
    log_m: jnp.ndarray,       # [K]
    tau: jnp.ndarray,         # [T]
    dsdt_raw: jnp.ndarray,    # [N,K,T]
    cfg: C2Config,
) -> Dict[str, jnp.ndarray]:

    # --------------------------------------------------------
    # 0) FP64 casting
    # --------------------------------------------------------
    sigma, log_m, tau, dsdt, const = cast_inputs_fp64(
        sigma_raw=sigma_raw,
        log_m=log_m,
        tau=tau,
        dsdt_raw=dsdt_raw,
        cfg=cfg,
    )
    _ = const

    # --------------------------------------------------------
    # 1) No-arbitrage cleanup
    # --------------------------------------------------------
    sigma_noarb, total_var_noarb, arb_metrics = enforce_surface_noarb(
        sigma_raw=sigma,
        tau=tau,
        cfg=cfg,
    )

    # --------------------------------------------------------
    # 2) Geometric factor extraction
    # --------------------------------------------------------
    features_raw, geom = extract_surface_factors(
        sigma=sigma_noarb,
        dsdt=dsdt,
        log_m=log_m,
        tau=tau,
        cfg=cfg,
    )

    # --------------------------------------------------------
    # 3) Cross-asset normalization
    # --------------------------------------------------------
    features_z, feature_mu, feature_std = normalize_cross_asset(
        features_raw=features_raw,
        cfg=cfg,
    )

    # --------------------------------------------------------
    # 4) PCA / eigenmodes
    # --------------------------------------------------------
    pca_scores, pca_eigvecs, pca_eigvals, pca_explained = pca_factors(
        features_z=features_z,
        cfg=cfg,
    )

    # --------------------------------------------------------
    # 5) Regime classification
    # --------------------------------------------------------
    regime_id, regime_onehot, cluster_id = classify_regimes_c2(
        features_z=features_z,
        pca_scores=pca_scores,
        cfg=cfg,
    )

    # --------------------------------------------------------
    # 6) Cross-sectional coupling
    # --------------------------------------------------------
    coupling = cross_asset_coupling(
        features_z=features_z,
        cfg=cfg,
    )

    # --------------------------------------------------------
    # 7) CATEGORY-1 EXTENDED (C2-only enrichment)
    # --------------------------------------------------------
    dealer_flow_raw = jnp.zeros((features_z.shape[0], 3), dtype=F64)  # [N,3]
    corr_baseline = jnp.eye(features_z.shape[0], dtype=F64)           # [N,N]

    ext = compute_c2_only_category1_extended(
        features_z=features_z,
        pca_scores=pca_scores,
        pca_eigvals=pca_eigvals,
        pca_eigvecs=pca_eigvecs,
        asset_corr=coupling["asset_corr"],
        factor_cov=coupling["factor_cov"],
        factor_corr=coupling["factor_corr"],
        coupling_NN3=coupling["coupling_NN3"],
        dealer_flow_raw=dealer_flow_raw,
        corr_baseline=corr_baseline,
        cfg=cfg,
    )

    # C3-ready enriched state [N, F+E]
    features_z_plus = jnp.concatenate(
        [features_z, ext["ext_z_per_asset"]],
        axis=1,
    )

    # --------------------------------------------------------
    # 8) Canonical state packet
    # --------------------------------------------------------
    packet = build_c2_state_packet(
        sigma_noarb=sigma_noarb,
        total_var_noarb=total_var_noarb,
        arb_metrics=arb_metrics,

        features_raw=features_raw,
        features_z=features_z,
        feature_mu=feature_mu,
        feature_std=feature_std,

        dsdm=geom["dsdm"],
        d2sdm2=geom["d2sdm2"],
        dsdT=geom["dsdT"],
        d2sdT2=geom["d2sdT2"],

        pca_scores=pca_scores,
        pca_eigvecs=pca_eigvecs,
        pca_eigvals=pca_eigvals,
        pca_explained=pca_explained,

        regime_id=regime_id,
        regime_onehot=regime_onehot,
        cluster_id=cluster_id,

        factor_cov=coupling["factor_cov"],
        factor_corr=coupling["factor_corr"],
        asset_corr=coupling["asset_corr"],
        coupling_NN3=coupling["coupling_NN3"],
    )

    # append enriched channels
    return {
        **packet,
        "features_z_plus": features_z_plus,
        "ext_raw_per_asset": ext["ext_raw_per_asset"],
        "ext_z_per_asset": ext["ext_z_per_asset"],
        "ext_global_raw": ext["ext_global_raw"],
        "ext_global_z": ext["ext_global_z"],
    }


# ============================================================
# RUNTIME ENTRYPOINTS
# ============================================================

def run_c2_state_packet(
    sigma_raw: jnp.ndarray,
    log_m: jnp.ndarray,
    tau: jnp.ndarray,
    dsdt_raw: jnp.ndarray,
    state_id_ns: int,
    cfg: C2Config,
) -> Dict[str, jnp.ndarray]:

    packet = c2_state_core(sigma_raw, log_m, tau, dsdt_raw, cfg)

    # barrier (moved to enriched tensor)
    _ = packet["features_z_plus"].block_until_ready()

    packet["state_id_ns"] = jnp.asarray(state_id_ns, dtype=jnp.int64)
    return packet


def aot_compile_c2_state_core(cfg: C2Config):
    sigma_aval = jax.ShapeDtypeStruct(
        (cfg.n_assets, cfg.n_strikes, cfg.n_tenors), F64
    )
    logm_aval = jax.ShapeDtypeStruct((cfg.n_strikes,), F64)
    tau_aval = jax.ShapeDtypeStruct((cfg.n_tenors,), F64)
    dsdt_aval = jax.ShapeDtypeStruct(
        (cfg.n_assets, cfg.n_strikes, cfg.n_tenors), F64
    )

    lowered = c2_state_core.lower(
        sigma_aval,
        logm_aval,
        tau_aval,
        dsdt_aval,
        cfg,
    )
    return lowered.compile()


def run_c2_state_packet_compiled(
    compiled_c2,
    sigma_raw: jnp.ndarray,
    log_m: jnp.ndarray,
    tau: jnp.ndarray,
    dsdt_raw: jnp.ndarray,
    state_id_ns: int,
) -> Dict[str, jnp.ndarray]:

    packet = compiled_c2(sigma_raw, log_m, tau, dsdt_raw)

    _ = packet["features_z_plus"].block_until_ready()

    packet["state_id_ns"] = jnp.asarray(state_id_ns, dtype=jnp.int64)
    return packet


# ============================================================
# LOCAL SMOKE TEST
# ============================================================

if __name__ == "__main__":
    cfg = C2Config(
        n_assets=200,
        n_strikes=64,
        n_tenors=32,
        pca_components=5,
    )

    N, K, T = cfg.n_assets, cfg.n_strikes, cfg.n_tenors

    log_m = jnp.linspace(-0.5, 0.5, K, dtype=F64)
    tau = jnp.linspace(1.0 / 365.0, 2.0, T, dtype=F64)

    base = 0.18 + 0.05 * jnp.exp(-2.5 * tau)[None, None, :] \
        + 0.04 * (log_m[None, :, None] ** 2)

    shift = jnp.linspace(-0.03, 0.03, N, dtype=F64)[:, None, None]
    sigma_raw = jnp.maximum(base + shift, 0.03)

    dsdt_raw = jnp.zeros((N, K, T), dtype=F64)

    out = run_c2_state_packet(
        sigma_raw=sigma_raw,
        log_m=log_m,
        tau=tau,
        dsdt_raw=dsdt_raw,
        state_id_ns=1730000000000000000,
        cfg=cfg,
    )

    print("JIT shapes:")
    print("  features_raw :", out["features_raw"].shape)
    print("  features_z   :", out["features_z"].shape)
    print("  features_z+  :", out["features_z_plus"].shape)
    print("  pca_scores   :", out["pca_scores"].shape)
    print("  asset_corr   :", out["asset_corr"].shape)

    compiled = aot_compile_c2_state_core(cfg)
    out2 = run_c2_state_packet_compiled(
        compiled_c2=compiled,
        sigma_raw=sigma_raw,
        log_m=log_m,
        tau=tau,
        dsdt_raw=dsdt_raw,
        state_id_ns=1730000000000000001,
    )

    print("AOT shapes:")
    print("  features_raw :", out2["features_raw"].shape)
    print("  features_z   :", out2["features_z"].shape)
    print("  features_z+  :", out2["features_z_plus"].shape)

