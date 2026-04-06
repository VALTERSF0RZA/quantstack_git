# c2/factor_engine.py
# =============================================================================
# C2 Factor Engine (FP64 • JAX • XLA • JIT)
#
# Consumes geometry_ops_bundle() from c2/numerics.py to produce:
#   - canonical per-asset factors [N,F]
#   - derivative tensors for audit/observability
#
# Shape contract:
#   sigma: [N,K,T]
#   dsdt : [N,K,T]
#   log_m: [K]
#   tau  : [T]
# =============================================================================

from __future__ import annotations

from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from .config import C2Config, F64
from .numerics import geometry_ops_bundle

# explicit FP64 enable
jax.config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnames=("cfg",))
def extract_surface_factors(
    sigma: jnp.ndarray,   # [N,K,T]
    dsdt: jnp.ndarray,    # [N,K,T]
    log_m: jnp.ndarray,   # [K]
    tau: jnp.ndarray,     # [T]
    cfg: C2Config,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Produces canonical geometric factors per asset and derivative tensors.

    Returns:
      features_raw: [N,F]
        F columns (fixed order):
          0 level
          1 skew_atm
          2 convexity_atm
          3 term_slope_atm
          4 term_curvature_atm
          5 velocity
          6 vov_proxy
          7 term_spread_atm
      geom: dict of derivative tensors [N,K,T]
        dsdm, d2sdm2, dsdT, d2sdT2, lapK, lapT
    """
    eps = jnp.asarray(cfg.eps, dtype=F64)

    # FP64 cast
    sigma = jnp.asarray(sigma, dtype=F64)
    dsdt = jnp.asarray(dsdt, dtype=F64)
    log_m = jnp.asarray(log_m, dtype=F64)
    tau = jnp.asarray(tau, dtype=F64)

    # mean grid spacing (scalar) for static-compile-friendly derivatives
    dm = jnp.maximum(jnp.mean(jnp.diff(log_m)), eps)
    dT = jnp.maximum(jnp.mean(jnp.diff(tau)), eps)

    # Fused derivatives + laplacians from one JIT/XLA call
    dsdm, d2sdm2, dsdT, d2sdT2, lapK, lapT = geometry_ops_bundle(
        sigma, dm, dT, cfg.eps
    )

    # ATM strike index from log-moneyness grid
    atm_idx = jnp.argmin(jnp.abs(log_m))

    # terminal tenor indices
    short_idx = jnp.int32(0)
    long_idx = jnp.int32(tau.shape[0] - 1)

    # ---- Canonical factors [N] ----
    # 0) level
    level = jnp.mean(sigma, axis=(1, 2))

    # 1) skew at ATM, averaged across tenor
    skew_atm = jnp.mean(dsdm[:, atm_idx, :], axis=1)

    # 2) convexity at ATM, averaged across tenor
    convexity_atm = jnp.mean(d2sdm2[:, atm_idx, :], axis=1)

    # 3) term slope at ATM, averaged across tenor
    term_slope_atm = jnp.mean(dsdT[:, atm_idx, :], axis=1)

    # 4) term curvature at ATM, averaged across tenor
    term_curvature_atm = jnp.mean(d2sdT2[:, atm_idx, :], axis=1)

    # 5) surface velocity proxy from incoming dsdt channel
    velocity = jnp.mean(dsdt, axis=(1, 2))

    # 6) vol-of-vol proxy (ATM term-structure dispersion)
    vov_proxy = jnp.std(sigma[:, atm_idx, :], axis=1)

    # 7) ATM term spread (long tenor - short tenor)
    term_spread_atm = sigma[:, atm_idx, long_idx] - sigma[:, atm_idx, short_idx]

    # Stack to [N,F] in deterministic order
    features_raw = jnp.column_stack(
        [
            level,               # 0
            skew_atm,            # 1
            convexity_atm,       # 2
            term_slope_atm,      # 3
            term_curvature_atm,  # 4
            velocity,            # 5
            vov_proxy,           # 6
            term_spread_atm,     # 7
        ]
    )

    geom = {
        "dsdm": dsdm,
        "d2sdm2": d2sdm2,
        "dsdT": dsdT,
        "d2sdT2": d2sdT2,
        "lapK": lapK,
        "lapT": lapT,
    }

    return features_raw, geom


# -----------------------------------------------------------------------------
# Optional AOT compile helper for extract_surface_factors
# -----------------------------------------------------------------------------
def aot_compile_extract_surface_factors(cfg: C2Config):
    """
    AOT compile helper for fixed shape contract.
    """
    sigma_aval = jax.ShapeDtypeStruct((cfg.n_assets, cfg.n_strikes, cfg.n_tenors), F64)
    dsdt_aval = jax.ShapeDtypeStruct((cfg.n_assets, cfg.n_strikes, cfg.n_tenors), F64)
    logm_aval = jax.ShapeDtypeStruct((cfg.n_strikes,), F64)
    tau_aval = jax.ShapeDtypeStruct((cfg.n_tenors,), F64)

    lowered = extract_surface_factors.lower(sigma_aval, dsdt_aval, logm_aval, tau_aval, cfg)
    return lowered.compile()


if __name__ == "__main__":
    # Smoke test
    cfg = C2Config(
        n_assets=200,
        n_strikes=64,
        n_tenors=32,
    )

    N, K, T = cfg.n_assets, cfg.n_strikes, cfg.n_tenors
    log_m = jnp.linspace(-0.5, 0.5, K, dtype=F64)
    tau = jnp.linspace(1.0 / 365.0, 2.0, T, dtype=F64)

    # synthetic smooth surface
    base = 0.18 + 0.05 * jnp.exp(-2.0 * tau)[None, None, :] + 0.03 * (log_m[None, :, None] ** 2)
    shift = jnp.linspace(-0.02, 0.02, N, dtype=F64)[:, None, None]
    sigma = jnp.maximum(base + shift, 0.03)

    dsdt = jnp.zeros_like(sigma, dtype=F64)

    feats, geom = extract_surface_factors(sigma, dsdt, log_m, tau, cfg)
    _ = feats.block_until_ready()

    print("features_raw:", feats.shape, feats.dtype)
    print("dsdm:", geom["dsdm"].shape, geom["dsdm"].dtype)
    print("d2sdm2:", geom["d2sdm2"].shape, geom["d2sdm2"].dtype)
    print("dsdT:", geom["dsdT"].shape, geom["dsdT"].dtype)
    print("d2sdT2:", geom["d2sdT2"].shape, geom["d2sdT2"].dtype)
    print("lapK:", geom["lapK"].shape, geom["lapK"].dtype)
    print("lapT:", geom["lapT"].shape, geom["lapT"].dtype)

