# c2/config.py
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Tuple

import jax

# Must be set before any meaningful array creation/compilation
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

F64 = jnp.float64


@dataclass(frozen=True, slots=True)
class C2Config:
    """
    Static-shape + numerical configuration for C2 geometric compiler.
    Passed as static arg into jitted kernels.
    """
    # Static shape contract
    n_assets: int = 200
    n_strikes: int = 64
    n_tenors: int = 32

    # Numeric safety
    eps: float = 1e-12

    # Surface cleanup / no-arb projection controls
    smooth_iters: int = 8
    smooth_lambda_k: float = 0.08
    smooth_lambda_t: float = 0.08
    convexify_iters: int = 10
    convexify_step: float = 0.45

    # PCA
    pca_components: int = 5

    # Deterministic regime tagging
    z_thr: float = 1.0
    num_regimes: int = 5  # IDs: 0..4

    def __post_init__(self) -> None:
        if self.n_assets <= 0 or self.n_strikes <= 1 or self.n_tenors <= 1:
            raise ValueError("Invalid static dimensions.")
        if self.eps <= 0.0:
            raise ValueError("eps must be > 0.")
        if self.pca_components <= 0:
            raise ValueError("pca_components must be > 0.")
        if self.num_regimes <= 1:
            raise ValueError("num_regimes must be > 1.")
        if self.smooth_iters < 0 or self.convexify_iters < 0:
            raise ValueError("Iteration counts must be >= 0.")


def as_f64(x) -> jnp.ndarray:
    """Canonical FP64 cast helper."""
    return jnp.asarray(x, dtype=F64)


def make_shape_contract(cfg: C2Config):
    """
    Static AOT/JIT shape contract for C2 entrypoint.
    """
    sigma_aval = jax.ShapeDtypeStruct((cfg.n_assets, cfg.n_strikes, cfg.n_tenors), F64)
    logm_aval = jax.ShapeDtypeStruct((cfg.n_strikes,), F64)
    tau_aval = jax.ShapeDtypeStruct((cfg.n_tenors,), F64)
    dsdt_aval = jax.ShapeDtypeStruct((cfg.n_assets, cfg.n_strikes, cfg.n_tenors), F64)
    return sigma_aval, logm_aval, tau_aval, dsdt_aval


@partial(jax.jit, static_argnames=("cfg",))
def cast_inputs_fp64(
    sigma_raw: jnp.ndarray,  # [N,K,T]
    log_m: jnp.ndarray,      # [K]
    tau: jnp.ndarray,        # [T]
    dsdt_raw: jnp.ndarray,   # [N,K,T]
    cfg: C2Config,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    JIT-compiled FP64 cast + numeric constants pack.
    Useful as first stage in c2_state_core.
    """
    sigma = jnp.asarray(sigma_raw, dtype=F64)
    lm = jnp.asarray(log_m, dtype=F64)
    t = jnp.asarray(tau, dtype=F64)
    dsdt = jnp.asarray(dsdt_raw, dtype=F64)

    const = jnp.array(
        [
            cfg.eps,
            cfg.smooth_lambda_k,
            cfg.smooth_lambda_t,
            cfg.convexify_step,
            cfg.z_thr,
        ],
        dtype=F64,
    )
    return sigma, lm, t, dsdt, const


def aot_compile_cast_inputs(cfg: C2Config):
    """
    Example AOT compile helper for this module.
    """
    sigma_aval, logm_aval, tau_aval, dsdt_aval = make_shape_contract(cfg)
    lowered = cast_inputs_fp64.lower(sigma_aval, logm_aval, tau_aval, dsdt_aval, cfg)
    return lowered.compile()

