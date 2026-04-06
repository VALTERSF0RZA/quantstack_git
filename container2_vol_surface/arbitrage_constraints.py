# c2/arbitrage_constraints.py
# =============================================================================
# C2 Surface-Level No-Arbitrage Enforcement
# FP64 • JAX • XLA • JIT
#
# Implements deterministic proxy constraints on total variance:
#   - positivity
#   - calendar monotonicity
#   - strike convexity proxy
#   - laplacian smoothing
#
# Shape contract:
#   sigma_raw : [N,K,T]
#   tau       : [T]
# =============================================================================

from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from .config import C2Config, F64
from .math_utils import laplacian_axis1, laplacian_axis2, cummax_axis

# Ensure FP64 everywhere
jax.config.update("jax_enable_x64", True)


# -----------------------------------------------------------------------------
# Main no-arb enforcement kernel
# -----------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("cfg",))
def enforce_surface_noarb(
    sigma_raw: jnp.ndarray,  # [N,K,T]
    tau: jnp.ndarray,        # [T]
    cfg: C2Config,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Surface-level no-arbitrage cleanup proxy on total variance.

    Steps:
      1) positivity enforcement
      2) laplacian smoothing
      3) calendar monotonicity (cummax in maturity)
      4) strike convexity projection proxy
      5) re-smoothing and re-calendarization

    Returns:
      sigma_noarb      [N,K,T]
      total_var_noarb  [N,K,T]
      arb_metrics      [4]
          [cal_pre, cal_post, bf_pre, bf_post]
    """

    eps = jnp.asarray(cfg.eps, dtype=F64)

    # ------------------------------------------------------------------
    # 0) Convert to total variance
    # ------------------------------------------------------------------
    tau_safe = jnp.maximum(jnp.asarray(tau, dtype=F64), eps)
    sigma = jnp.maximum(jnp.asarray(sigma_raw, dtype=F64), eps)

    # total variance w = sigma^2 * tau
    w0 = sigma * sigma * tau_safe[None, None, :]  # [N,K,T]

    # ------------------------------------------------------------------
    # 1) Pre diagnostics
    # ------------------------------------------------------------------
    cal_pre = jnp.mean(
        jnp.maximum(-(w0[:, :, 1:] - w0[:, :, :-1]), 0.0)
    )

    bf_pre = jnp.mean(
        jnp.maximum(
            -(w0[:, 2:, :] - 2.0 * w0[:, 1:-1, :] + w0[:, :-2, :]),
            0.0,
        )
    )

    # ------------------------------------------------------------------
    # 2) Laplacian smoothing
    # ------------------------------------------------------------------
    lam_k = jnp.asarray(cfg.smooth_lambda_k, dtype=F64)
    lam_t = jnp.asarray(cfg.smooth_lambda_t, dtype=F64)

    def smooth_body(_, w):
        w_new = (
            w
            + lam_k * laplacian_axis1(w)
            + lam_t * laplacian_axis2(w)
        )
        return jnp.maximum(w_new, eps)

    w = jax.lax.fori_loop(
        0,
        cfg.smooth_iters,
        smooth_body,
        w0,
    )

    # ------------------------------------------------------------------
    # 3) Calendar monotonicity
    # total variance must be nondecreasing in maturity
    # ------------------------------------------------------------------
    w = cummax_axis(w, axis=2)

    # ------------------------------------------------------------------
    # 4) Strike convexity proxy
    # ------------------------------------------------------------------
    step = jnp.asarray(cfg.convexify_step, dtype=F64)

    def convexify_body(_, w_curr):
        d2k = (
            w_curr[:, 2:, :]
            - 2.0 * w_curr[:, 1:-1, :]
            + w_curr[:, :-2, :]
        )

        neg = jnp.minimum(d2k, 0.0)
        adjust = -step * neg

        w_mid = w_curr[:, 1:-1, :] + adjust
        w_next = w_curr.at[:, 1:-1, :].set(w_mid)

        return jnp.maximum(w_next, eps)

    w = jax.lax.fori_loop(
        0,
        cfg.convexify_iters,
        convexify_body,
        w,
    )

    # ------------------------------------------------------------------
    # 5) Light re-smooth + re-calendarization
    # ------------------------------------------------------------------
    resmooth_iters = max(1, cfg.smooth_iters // 2)

    w = jax.lax.fori_loop(
        0,
        resmooth_iters,
        smooth_body,
        w,
    )

    w = cummax_axis(w, axis=2)

    # ------------------------------------------------------------------
    # 6) Convert back to sigma
    # ------------------------------------------------------------------
    sigma_noarb = jnp.sqrt(
        jnp.maximum(w, eps) / tau_safe[None, None, :]
    )

    # ------------------------------------------------------------------
    # 7) Post diagnostics
    # ------------------------------------------------------------------
    cal_post = jnp.mean(
        jnp.maximum(-(w[:, :, 1:] - w[:, :, :-1]), 0.0)
    )

    bf_post = jnp.mean(
        jnp.maximum(
            -(w[:, 2:, :] - 2.0 * w[:, 1:-1, :] + w[:, :-2, :]),
            0.0,
        )
    )

    arb_metrics = jnp.array(
        [cal_pre, cal_post, bf_pre, bf_post],
        dtype=F64,
    )

    return sigma_noarb, w, arb_metrics


# -----------------------------------------------------------------------------
# AOT compile helper
# -----------------------------------------------------------------------------
def aot_compile_enforce_surface_noarb(cfg: C2Config):
    """
    XLA AOT compile path for static-shape deployment.
    """
    sigma_aval = jax.ShapeDtypeStruct(
        (cfg.n_assets, cfg.n_strikes, cfg.n_tenors),
        F64,
    )
    tau_aval = jax.ShapeDtypeStruct(
        (cfg.n_tenors,),
        F64,
    )

    lowered = enforce_surface_noarb.lower(
        sigma_aval,
        tau_aval,
        cfg,
    )
    return lowered.compile()


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = C2Config(
        n_assets=200,
        n_strikes=64,
        n_tenors=32,
    )

    N, K, T = cfg.n_assets, cfg.n_strikes, cfg.n_tenors

    tau = jnp.linspace(1.0 / 365.0, 2.0, T, dtype=F64)

    # synthetic slightly noisy surface
    m = jnp.linspace(-0.5, 0.5, K, dtype=F64)
    base = (
        0.18
        + 0.05 * jnp.exp(-2.0 * tau)[None, None, :]
        + 0.03 * (m[None, :, None] ** 2)
    )

    shift = jnp.linspace(-0.03, 0.03, N, dtype=F64)[:, None, None]
    noise = 0.005 * jax.random.normal(
        jax.random.PRNGKey(0),
        (N, K, T),
        dtype=F64,
    )

    sigma_raw = jnp.maximum(base + shift + noise, 0.02)

    sigma_clean, w_clean, metrics = enforce_surface_noarb(
        sigma_raw,
        tau,
        cfg,
    )
    _ = sigma_clean.block_until_ready()

    print("sigma_clean:", sigma_clean.shape, sigma_clean.dtype)
    print("total_var :", w_clean.shape, w_clean.dtype)
    print("arb_metrics:", metrics)

