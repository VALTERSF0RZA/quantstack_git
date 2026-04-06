# c2/math_utils.py
# =============================================================================
# FP64 • JAX • XLA • JIT-ready numerical operators for C2
#
# Provides:
# - first/second finite differences on axis=1 (strike) and axis=2 (tenor)
# - discrete laplacians with edge replication
# - cumulative max (JAX-safe) along arbitrary axis
# - covariance -> correlation conversion
#
# Shape conventions:
#   x: [N, K, T]
# =============================================================================

from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

# Global FP64 enablement
jax.config.update("jax_enable_x64", True)
F64 = jnp.float64


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
@jax.jit
def _as_f64(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.asarray(x, dtype=F64)


@jax.jit
def _safe_scalar_step(step, eps: float) -> jnp.ndarray:
    """
    Accepts scalar-like step, returns FP64 scalar >= eps.
    """
    return jnp.maximum(jnp.asarray(step, dtype=F64), jnp.asarray(eps, dtype=F64))


# -----------------------------------------------------------------------------
# Finite differences along strike axis (axis=1)
# -----------------------------------------------------------------------------
@partial(jax.jit, static_argnames=())
def first_diff_axis1(x: jnp.ndarray, dx, eps: float) -> jnp.ndarray:
    """
    First derivative along strike axis K (axis=1), centered with one-sided edges.
    x: [N,K,T]
    returns: [N,K,T]
    """
    x = _as_f64(x)
    dx_safe = _safe_scalar_step(dx, eps)

    left = (x[:, 1:2, :] - x[:, 0:1, :]) / dx_safe
    mid = (x[:, 2:, :] - x[:, :-2, :]) / (2.0 * dx_safe)
    right = (x[:, -1:, :] - x[:, -2:-1, :]) / dx_safe
    return jnp.concatenate([left, mid, right], axis=1)


@partial(jax.jit, static_argnames=())
def second_diff_axis1(x: jnp.ndarray, dx, eps: float) -> jnp.ndarray:
    """
    Second derivative along strike axis K (axis=1).
    x: [N,K,T]
    returns: [N,K,T]
    """
    x = _as_f64(x)
    dx_safe = _safe_scalar_step(dx, eps)
    dx2 = jnp.maximum(dx_safe * dx_safe, jnp.asarray(eps, dtype=F64))

    mid = (x[:, 2:, :] - 2.0 * x[:, 1:-1, :] + x[:, :-2, :]) / dx2
    left = mid[:, 0:1, :]
    right = mid[:, -1:, :]
    return jnp.concatenate([left, mid, right], axis=1)


# -----------------------------------------------------------------------------
# Finite differences along tenor axis (axis=2)
# -----------------------------------------------------------------------------
@partial(jax.jit, static_argnames=())
def first_diff_axis2(x: jnp.ndarray, dt, eps: float) -> jnp.ndarray:
    """
    First derivative along tenor axis T (axis=2), centered with one-sided edges.
    x: [N,K,T]
    returns: [N,K,T]
    """
    x = _as_f64(x)
    dt_safe = _safe_scalar_step(dt, eps)

    left = (x[:, :, 1:2] - x[:, :, 0:1]) / dt_safe
    mid = (x[:, :, 2:] - x[:, :, :-2]) / (2.0 * dt_safe)
    right = (x[:, :, -1:] - x[:, :, -2:-1]) / dt_safe
    return jnp.concatenate([left, mid, right], axis=2)


@partial(jax.jit, static_argnames=())
def second_diff_axis2(x: jnp.ndarray, dt, eps: float) -> jnp.ndarray:
    """
    Second derivative along tenor axis T (axis=2).
    x: [N,K,T]
    returns: [N,K,T]
    """
    x = _as_f64(x)
    dt_safe = _safe_scalar_step(dt, eps)
    dt2 = jnp.maximum(dt_safe * dt_safe, jnp.asarray(eps, dtype=F64))

    mid = (x[:, :, 2:] - 2.0 * x[:, :, 1:-1] + x[:, :, :-2]) / dt2
    left = mid[:, :, 0:1]
    right = mid[:, :, -1:]
    return jnp.concatenate([left, mid, right], axis=2)


# -----------------------------------------------------------------------------
# Laplacians
# -----------------------------------------------------------------------------
@partial(jax.jit, static_argnames=())
def laplacian_axis1(x: jnp.ndarray) -> jnp.ndarray:
    """
    Discrete laplacian along strike K with edge replication.
    x: [N,K,T]
    returns: [N,K,T]
    """
    x = _as_f64(x)
    xl = jnp.concatenate([x[:, 0:1, :], x[:, :-1, :]], axis=1)
    xr = jnp.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
    return xl - 2.0 * x + xr


@partial(jax.jit, static_argnames=())
def laplacian_axis2(x: jnp.ndarray) -> jnp.ndarray:
    """
    Discrete laplacian along tenor T with edge replication.
    x: [N,K,T]
    returns: [N,K,T]
    """
    x = _as_f64(x)
    xl = jnp.concatenate([x[:, :, 0:1], x[:, :, :-1]], axis=2)
    xr = jnp.concatenate([x[:, :, 1:], x[:, :, -1:]], axis=2)
    return xl - 2.0 * x + xr


# -----------------------------------------------------------------------------
# Cumulative max (JAX-safe)
# -----------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("axis",))
def cummax_axis(x: jnp.ndarray, axis: int) -> jnp.ndarray:
    """
    JAX-safe cumulative max along any axis.
    """
    x = _as_f64(x)
    xm = jnp.moveaxis(x, axis, -1)
    ym = jax.lax.associative_scan(jnp.maximum, xm, axis=-1)
    return jnp.moveaxis(ym, -1, axis)


# -----------------------------------------------------------------------------
# Covariance -> Correlation
# -----------------------------------------------------------------------------
@partial(jax.jit, static_argnames=())
def cov_to_corr(cov: jnp.ndarray, eps: float) -> jnp.ndarray:
    """
    Converts covariance matrix to correlation matrix with epsilon safety.
    cov: [F,F]
    returns: [F,F]
    """
    cov = _as_f64(cov)
    eps64 = jnp.asarray(eps, dtype=F64)

    d = jnp.sqrt(jnp.maximum(jnp.diag(cov), eps64))
    denom = d[:, None] * d[None, :]
    corr = cov / jnp.maximum(denom, eps64)

    # Optional: force exact 1.0 diagonal for numerical cleanliness
    corr = corr.at[jnp.diag_indices(corr.shape[0])].set(jnp.asarray(1.0, dtype=F64))
    return corr


# -----------------------------------------------------------------------------
# Optional fused operator bundle (single JIT call for common C2 geometry pass)
# -----------------------------------------------------------------------------
@partial(jax.jit, static_argnames=())
def geometry_ops_bundle(x: jnp.ndarray, dx, dt, eps: float) -> Tuple[jnp.ndarray, ...]:
    """
    Convenience fused bundle for one-pass geometry extraction:
      d/dm, d2/dm2, d/dT, d2/dT2, lapK, lapT
    """
    d1k = first_diff_axis1(x, dx, eps)
    d2k = second_diff_axis1(x, dx, eps)
    d1t = first_diff_axis2(x, dt, eps)
    d2t = second_diff_axis2(x, dt, eps)
    lk = laplacian_axis1(x)
    lt = laplacian_axis2(x)
    return d1k, d2k, d1t, d2t, lk, lt


# -----------------------------------------------------------------------------
# AOT compile helper (for fixed shapes/contracts)
# -----------------------------------------------------------------------------
def aot_compile_geometry_ops(n_assets: int = 200, n_strikes: int = 64, n_tenors: int = 32):
    """
    XLA AOT-style compile for geometry_ops_bundle at fixed static shapes.
    """
    x_aval = jax.ShapeDtypeStruct((n_assets, n_strikes, n_tenors), F64)
    dx_aval = jax.ShapeDtypeStruct((), F64)
    dt_aval = jax.ShapeDtypeStruct((), F64)
    eps_aval = jax.ShapeDtypeStruct((), F64)

    lowered = geometry_ops_bundle.lower(x_aval, dx_aval, dt_aval, eps_aval)
    return lowered.compile()


if __name__ == "__main__":
    # Smoke test
    N, K, T = 200, 64, 32
    x = jnp.ones((N, K, T), dtype=F64) * 0.2
    dx = jnp.asarray(0.01, dtype=F64)
    dt = jnp.asarray(1.0 / 365.0, dtype=F64)
    eps = 1e-12

    d1k, d2k, d1t, d2t, lk, lt = geometry_ops_bundle(x, dx, dt, eps)
    _ = d1k.block_until_ready()

    print("d1k:", d1k.shape, d1k.dtype)
    print("d2k:", d2k.shape, d2k.dtype)
    print("d1t:", d1t.shape, d1t.dtype)
    print("d2t:", d2t.shape, d2t.dtype)
    print("lk :", lk.shape, lk.dtype)
    print("lt :", lt.shape, lt.dtype)

