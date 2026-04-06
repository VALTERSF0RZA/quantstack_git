# ===============================================================
# CONTAINER 2 — DESK EXECUTION ENGINE
# Fully fused
# Static 512 contracts
# FP64
# AOT-ready
# ===============================================================

import jax
import jax.numpy as jnp
from functools import partial

from bs_jax import implied_vol_solver, bs_delta_gamma_vega
from SABR_calibration import calibrate_sabr_slice
from heston_calibration import calibrate_heston_slice
from surface_dynamics import compute_surface_geometry

jax.config.update("jax_enable_x64", True)

N = 512
DTYPE = jnp.float64


# ===============================================================
# FUSED EXECUTION BOUNDARY
# ===============================================================

@partial(jax.jit, static_argnums=())
def container2_kernel(
    S: DTYPE,
    K: jnp.ndarray,          # (512,)
    T: jnp.ndarray,          # (512,)
    r: DTYPE,
    q: DTYPE,
    price: jnp.ndarray,      # (512,)
    is_call: jnp.ndarray,    # (512,)
    sabr_cfg,
    heston_cfg,
):
    # -----------------------------
    # 1. Implied Vol Inversion
    # -----------------------------
    iv = implied_vol_solver(price, S, K, T, r, q, is_call)

    # -----------------------------
    # 2. SABR Calibration
    # -----------------------------
    sabr_params = calibrate_sabr_slice(K, T, iv, S, sabr_cfg)

    # -----------------------------
    # 3. Heston Calibration
    # -----------------------------
    heston_params = calibrate_heston_slice(K, T, iv, S, heston_cfg)

    # -----------------------------
    # 4. Surface Geometry
    # -----------------------------
    sigma_surface, skew, curvature, term_slope = \
        compute_surface_geometry(K, T, sabr_params, heston_params)

    # -----------------------------
    # 5. Greeks (Vectorized)
    # -----------------------------
    delta, gamma, vega = bs_delta_gamma_vega(
        S, K, T, r, q, sigma_surface, is_call
    )

    # -----------------------------
    # 6. Flat Tuple Return (AOT safe)
    # -----------------------------
    return (
        iv,
        sabr_params,
        heston_params,
        sigma_surface,
        skew,
        curvature,
        term_slope,
        delta,
        gamma,
        vega,
    )

}
