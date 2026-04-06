# surface_dynamics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Robust import fallback for your SABR implementation
try:
    from sabr_calibration import sabr_lognormal_iv
except Exception:
    # If your SABR module uses another function name, alias it here.
    from sabr_calibration import sabr_iv_hagan as sabr_lognormal_iv  # type: ignore

from surface_state import SurfaceSnapshot, SurfaceStateStore, dt_seconds


@dataclass(frozen=True)
class DynamicsConfig:
    dt_floor_seconds: float = 1e-3
    year_seconds: float = 365.0 * 24.0 * 3600.0
    min_sigma: float = 1e-6
    max_sigma: float = 5.0
    m_bandwidth: float = 0.08     # for smooth Heston IV reconstruction
    T_bandwidth: float = 0.12     # for smooth Heston IV reconstruction
    eps: float = 1e-14
    velocity_clip: float = 25.0   # hard clip for stability


# -------------------------
# Helpers
# -------------------------

def _interp1d_linear(x: jnp.ndarray, xp: jnp.ndarray, fp: jnp.ndarray, eps: float) -> jnp.ndarray:
    """
    Piecewise-linear interpolation with endpoint clamp.
    Works with x as scalar or vector.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    xp = jnp.asarray(xp, dtype=jnp.float64)
    fp = jnp.asarray(fp, dtype=jnp.float64)

    if xp.shape[0] == 0:
        return jnp.full_like(x, jnp.nan, dtype=jnp.float64)
    if xp.shape[0] == 1:
        return jnp.full_like(x, fp[0], dtype=jnp.float64)

    x_clip = jnp.clip(x, xp[0], xp[-1])
    idx = jnp.searchsorted(xp, x_clip, side="right") - 1
    idx = jnp.clip(idx, 0, xp.shape[0] - 2)

    x0 = xp[idx]
    x1 = xp[idx + 1]
    y0 = fp[idx]
    y1 = fp[idx + 1]

    w = (x_clip - x0) / jnp.maximum(x1 - x0, eps)
    return y0 + w * (y1 - y0)


def _extract_param_curves(
    fits_by_T: Dict[float, Dict[str, Any]],
    param_names: List[str],
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    From fits dict keyed by expiry T -> sorted nodes:
      T_nodes, {param_name: values_at_nodes}
    """
    rows: List[Tuple[float, List[float]]] = []
    for T_key, fit in fits_by_T.items():
        params = fit.get("params", {})
        if all(name in params for name in param_names):
            rows.append((float(T_key), [float(params[name]) for name in param_names]))

    rows.sort(key=lambda x: x[0])

    if not rows:
        return (
            jnp.zeros((0,), dtype=jnp.float64),
            {name: jnp.zeros((0,), dtype=jnp.float64) for name in param_names},
        )

    T_nodes = jnp.array([r[0] for r in rows], dtype=jnp.float64)
    curves = {
        name: jnp.array([r[1][i] for r in rows], dtype=jnp.float64)
        for i, name in enumerate(param_names)
    }
    return T_nodes, curves


def _collect_iv_nodes_from_fits(
    fits_by_T: Dict[float, Dict[str, Any]],
    F: float,
    min_sigma: float,
    max_sigma: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Collect (m, T, iv_fit) nodes from model fit arrays for smooth reconstruction.
    """
    m_nodes: List[float] = []
    T_nodes: List[float] = []
    sigma_nodes: List[float] = []

    for _, fit in sorted(fits_by_T.items(), key=lambda kv: float(kv[0])):
        arrays = fit.get("arrays", {})
        K = arrays.get("K", None)
        T = arrays.get("T", None)
        iv_fit = arrays.get("iv_fit", None)
        if K is None or T is None or iv_fit is None:
            continue

        K_np = jnp.asarray(K, dtype=jnp.float64)
        T_np = jnp.asarray(T, dtype=jnp.float64)
        iv_np = jnp.asarray(iv_fit, dtype=jnp.float64)

        m_np = jnp.log(jnp.maximum(K_np, 1e-14) / float(F))
        valid = (
            jnp.isfinite(m_np)
            & jnp.isfinite(T_np)
            & jnp.isfinite(iv_np)
            & (iv_np >= min_sigma)
            & (iv_np <= max_sigma)
            & (T_np > 0.0)
        )
        if int(jnp.sum(valid)) == 0:
            continue

        m_nodes.extend([float(x) for x in m_np[valid]])
        T_nodes.extend([float(x) for x in T_np[valid]])
        sigma_nodes.extend([float(x) for x in iv_np[valid]])

    if len(m_nodes) == 0:
        return (
            jnp.zeros((0,), dtype=jnp.float64),
            jnp.zeros((0,), dtype=jnp.float64),
            jnp.zeros((0,), dtype=jnp.float64),
        )

    return (
        jnp.array(m_nodes, dtype=jnp.float64),
        jnp.array(T_nodes, dtype=jnp.float64),
        jnp.array(sigma_nodes, dtype=jnp.float64),
    )


def _build_sabr_sigma_fn(
    F: float,
    sabr_fits: Dict[float, Dict[str, Any]],
    cfg: DynamicsConfig,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Differentiable sigma(m, T) from SABR param curves across expiry.
    """
    T_nodes, curves = _extract_param_curves(sabr_fits, ["alpha", "beta", "rho", "nu"])
    if T_nodes.shape[0] == 0:
        raise ValueError("SABR fits are empty or missing params.")

    def sigma_fn(m: jnp.ndarray, T: jnp.ndarray) -> jnp.ndarray:
        T = jnp.maximum(jnp.asarray(T, dtype=jnp.float64), cfg.eps)
        m = jnp.asarray(m, dtype=jnp.float64)

        alpha = _interp1d_linear(T, T_nodes, curves["alpha"], cfg.eps)
        beta = _interp1d_linear(T, T_nodes, curves["beta"], cfg.eps)
        rho = _interp1d_linear(T, T_nodes, curves["rho"], cfg.eps)
        nu = _interp1d_linear(T, T_nodes, curves["nu"], cfg.eps)

        K = float(F) * jnp.exp(m)
        sigma = sabr_lognormal_iv(float(F), K, T, alpha, beta, rho, nu)
        return jnp.clip(sigma, cfg.min_sigma, cfg.max_sigma)

    return sigma_fn


def _build_heston_sigma_fn(
    F: float,
    heston_fits: Dict[float, Dict[str, Any]],
    cfg: DynamicsConfig,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Differentiable sigma(m, T) from fitted Heston IV nodes via smooth kernel regression.
    (Good for AD derivatives / velocities while keeping Heston params as features.)
    """
    m_nodes, T_nodes, iv_nodes = _collect_iv_nodes_from_fits(
        heston_fits, F=float(F), min_sigma=cfg.min_sigma, max_sigma=cfg.max_sigma
    )
    if m_nodes.shape[0] == 0:
        raise ValueError("Heston fits missing arrays.iv_fit; cannot build smooth sigma surface.")

    def sigma_fn(m: jnp.ndarray, T: jnp.ndarray) -> jnp.ndarray:
        m = jnp.asarray(m, dtype=jnp.float64)
        T = jnp.maximum(jnp.asarray(T, dtype=jnp.float64), cfg.eps)

        dm = (m - m_nodes) / cfg.m_bandwidth
        dT = (T - T_nodes) / cfg.T_bandwidth
        w = jnp.exp(-0.5 * (dm * dm + dT * dT))
        sigma = jnp.sum(w * iv_nodes) / jnp.maximum(jnp.sum(w), cfg.eps)
        return jnp.clip(sigma, cfg.min_sigma, cfg.max_sigma)

    return sigma_fn


def _build_param_interpolator(
    fits_by_T: Dict[float, Dict[str, Any]],
    param_names: List[str],
    cfg: DynamicsConfig,
) -> Callable[[jnp.ndarray], Dict[str, jnp.ndarray]]:
    T_nodes, curves = _extract_param_curves(fits_by_T, param_names)

    def interp(T_grid: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        T_grid = jnp.asarray(T_grid, dtype=jnp.float64)
        if T_nodes.shape[0] == 0:
            return {name: jnp.full_like(T_grid, jnp.nan, dtype=jnp.float64) for name in param_names}

        return {
            name: _interp1d_linear(T_grid, T_nodes, curves[name], cfg.eps)
            for name in param_names
        }

    return interp


def _broadcast_T_curve_to_surface(curve_T: jnp.ndarray, n_m: int) -> jnp.ndarray:
    return jnp.repeat(curve_T[:, None], repeats=n_m, axis=1)


def _eval_surface_partials(
    sigma_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    m2d: jnp.ndarray,
    T2d: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """
    AD stack:
      sigma, dσ/dm, d²σ/dm², dσ/dT, d²σ/(dm dT), d²σ/dT²
    """
    d_sigma_dm_fn = jax.grad(sigma_fn, argnums=0)
    d2_sigma_dm2_fn = jax.grad(d_sigma_dm_fn, argnums=0)
    d_sigma_dT_fn = jax.grad(sigma_fn, argnums=1)
    d2_sigma_dmdT_fn = jax.grad(d_sigma_dm_fn, argnums=1)
    d2_sigma_dT2_fn = jax.grad(d_sigma_dT_fn, argnums=1)

    m_flat = m2d.reshape(-1)
    T_flat = T2d.reshape(-1)

    v_sigma = jax.jit(jax.vmap(lambda m, t: sigma_fn(m, t)))
    v_dmd = jax.jit(jax.vmap(lambda m, t: d_sigma_dm_fn(m, t)))
    v_d2md2 = jax.jit(jax.vmap(lambda m, t: d2_sigma_dm2_fn(m, t)))
    v_dTd = jax.jit(jax.vmap(lambda m, t: d_sigma_dT_fn(m, t)))
    v_dmdT = jax.jit(jax.vmap(lambda m, t: d2_sigma_dmdT_fn(m, t)))
    v_dT2 = jax.jit(jax.vmap(lambda m, t: d2_sigma_dT2_fn(m, t)))

    sigma = v_sigma(m_flat, T_flat).reshape(m2d.shape)
    d_sigma_dm = v_dmd(m_flat, T_flat).reshape(m2d.shape)
    d2_sigma_dm2 = v_d2md2(m_flat, T_flat).reshape(m2d.shape)
    d_sigma_dT = v_dTd(m_flat, T_flat).reshape(m2d.shape)
    d2_sigma_dmdT = v_dmdT(m_flat, T_flat).reshape(m2d.shape)
    d2_sigma_dT2 = v_dT2(m_flat, T_flat).reshape(m2d.shape)

    return {
        "sigma": sigma,
        "d_sigma_dm": d_sigma_dm,
        "d2_sigma_dm2": d2_sigma_dm2,
        "d_sigma_dT": d_sigma_dT,
        "d2_sigma_dmdT": d2_sigma_dmdT,
        "d2_sigma_dT2": d2_sigma_dT2,
    }


# -------------------------
# Engine
# -------------------------

class SurfaceDynamicsEngine:
    """
    Produces the feature tensor:
      [sigma, dσ/dm, d²σ/dm², dσ/dT, dσ/dt_true, d²σ/(dm dt)_true, d²σ/(dT dt)_true,
       sabr_alpha, sabr_rho, sabr_nu, heston_v0, heston_theta, heston_kappa, heston_sigma, heston_rho]

    with shape [nT, nM, 15].
    """

    def __init__(
        self,
        symbol: str,
        state_store: Optional[SurfaceStateStore] = None,
        cfg: DynamicsConfig = DynamicsConfig(),
    ) -> None:
        self.symbol = symbol
        self.cfg = cfg
        self.state_store = state_store or SurfaceStateStore()

    def _select_sigma_fn(
        self,
        F: float,
        primary_model: str,
        sabr_fits: Optional[Dict[float, Dict[str, Any]]],
        heston_fits: Optional[Dict[float, Dict[str, Any]]],
    ) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        pm = primary_model.lower()

        if pm == "sabr":
            if sabr_fits:
                return _build_sabr_sigma_fn(F, sabr_fits, self.cfg)
            if heston_fits:
                return _build_heston_sigma_fn(F, heston_fits, self.cfg)
            raise ValueError("Need sabr_fits or heston_fits.")
        elif pm == "heston":
            if heston_fits:
                return _build_heston_sigma_fn(F, heston_fits, self.cfg)
            if sabr_fits:
                return _build_sabr_sigma_fn(F, sabr_fits, self.cfg)
            raise ValueError("Need heston_fits or sabr_fits.")
        else:
            raise ValueError(f"Unknown primary_model={primary_model}")

    def compute_features(
        self,
        ts_unix_ns: int,
        F: float,
        m_grid: jnp.ndarray,    # 1D (log-moneyness grid)
        T_grid: jnp.ndarray,    # 1D (years)
        sabr_fits: Optional[Dict[float, Dict[str, Any]]] = None,
        heston_fits: Optional[Dict[float, Dict[str, Any]]] = None,
        primary_model: str = "sabr",
    ) -> Dict[str, Any]:
        m_grid = jnp.asarray(m_grid, dtype=jnp.float64)
        T_grid = jnp.asarray(T_grid, dtype=jnp.float64)

        if m_grid.ndim != 1 or T_grid.ndim != 1:
            raise ValueError("m_grid and T_grid must be 1D arrays.")

        # grid: [nT, nM]
        T2d, m2d = jnp.meshgrid(T_grid, m_grid, indexing="ij")

        sigma_fn = self._select_sigma_fn(
            F=float(F),
            primary_model=primary_model,
            sabr_fits=sabr_fits,
            heston_fits=heston_fits,
        )

        cur = _eval_surface_partials(sigma_fn, m2d, T2d)

        prev: Optional[SurfaceSnapshot] = self.state_store.get(self.symbol)

        # -------------------------
        # Transport-corrected time derivatives
        # -------------------------
        if prev is None or prev.sigma.shape != cur["sigma"].shape:
            dt = self.cfg.dt_floor_seconds
            d_sigma_dt_true = jnp.zeros_like(cur["sigma"])
            d2_sigma_dmdt_true = jnp.zeros_like(cur["sigma"])
            d2_sigma_dTdt_true = jnp.zeros_like(cur["sigma"])
        else:
            dt = dt_seconds(prev.ts_unix_ns, ts_unix_ns, floor=self.cfg.dt_floor_seconds)

            # finite difference in observed coordinates
            raw_dt = (cur["sigma"] - prev.sigma) / dt
            dm_dt = (m2d - prev.m) / dt
            dT_dt = (T2d - prev.T) / dt

            # true local time derivative
            # ∂σ/∂t|local = raw_dt - (∂σ/∂m * dm/dt + ∂σ/∂T * dT/dt)
            d_sigma_dt_true = raw_dt - (
                cur["d_sigma_dm"] * dm_dt + cur["d_sigma_dT"] * dT_dt
            )

            # mixed time derivatives
            raw_dmdt = (cur["d_sigma_dm"] - prev.d_sigma_dm) / dt
            d2_sigma_dmdt_true = raw_dmdt - (
                cur["d2_sigma_dm2"] * dm_dt + cur["d2_sigma_dmdT"] * dT_dt
            )

            raw_dTdt = (cur["d_sigma_dT"] - prev.d_sigma_dT) / dt
            d2_sigma_dTdt_true = raw_dTdt - (
                cur["d2_sigma_dmdT"] * dm_dt + cur["d2_sigma_dT2"] * dT_dt
            )

        d_sigma_dt_true = jnp.clip(
            d_sigma_dt_true, -self.cfg.velocity_clip, self.cfg.velocity_clip
        )
        d2_sigma_dmdt_true = jnp.clip(
            d2_sigma_dmdt_true, -self.cfg.velocity_clip, self.cfg.velocity_clip
        )
        d2_sigma_dTdt_true = jnp.clip(
            d2_sigma_dTdt_true, -self.cfg.velocity_clip, self.cfg.velocity_clip
        )

        # -------------------------
        # Param curves as features
        # -------------------------
        sabr_interp = _build_param_interpolator(
            sabr_fits or {}, ["alpha", "rho", "nu"], self.cfg
        )
        heston_interp = _build_param_interpolator(
            heston_fits or {}, ["v0", "theta", "kappa", "sigma", "rho"], self.cfg
        )

        sabr_T = sabr_interp(T_grid)      # each [nT]
        heston_T = heston_interp(T_grid)  # each [nT]

        n_m = m_grid.shape[0]
        sabr_alpha_2d = _broadcast_T_curve_to_surface(sabr_T["alpha"], n_m)
        sabr_rho_2d = _broadcast_T_curve_to_surface(sabr_T["rho"], n_m)
        sabr_nu_2d = _broadcast_T_curve_to_surface(sabr_T["nu"], n_m)

        h_v0_2d = _broadcast_T_curve_to_surface(heston_T["v0"], n_m)
        h_theta_2d = _broadcast_T_curve_to_surface(heston_T["theta"], n_m)
        h_kappa_2d = _broadcast_T_curve_to_surface(heston_T["kappa"], n_m)
        h_sigma_2d = _broadcast_T_curve_to_surface(heston_T["sigma"], n_m)
        h_rho_2d = _broadcast_T_curve_to_surface(heston_T["rho"], n_m)

        feature_names = [
            "sigma",
            "d_sigma_dm",
            "d2_sigma_dm2",
            "d_sigma_dT",
            "d_sigma_dt_true",
            "d2_sigma_dmdt_true",
            "d2_sigma_dTdt_true",
            "sabr_alpha",
            "sabr_rho",
            "sabr_nu",
            "heston_v0",
            "heston_theta",
            "heston_kappa",
            "heston_sigma",
            "heston_rho",
        ]

        feature_tensor = jnp.stack(
            [
                cur["sigma"],
                cur["d_sigma_dm"],
                cur["d2_sigma_dm2"],
                cur["d_sigma_dT"],
                d_sigma_dt_true,
                d2_sigma_dmdt_true,
                d2_sigma_dTdt_true,
                sabr_alpha_2d,
                sabr_rho_2d,
                sabr_nu_2d,
                h_v0_2d,
                h_theta_2d,
                h_kappa_2d,
                h_sigma_2d,
                h_rho_2d,
            ],
            axis=-1,
        )  # [nT, nM, 15]

        atm_idx = int(jnp.argmin(jnp.abs(m_grid)))
        atm_iv_term = cur["sigma"][:, atm_idx]
        atm_vel_term = d_sigma_dt_true[:, atm_idx]

        # update state for next tick
        self.state_store.update(
            self.symbol,
            SurfaceSnapshot(
                ts_unix_ns=int(ts_unix_ns),
                m=m2d,
                T=T2d,
                sigma=cur["sigma"],
                d_sigma_dm=cur["d_sigma_dm"],
                d2_sigma_dm2=cur["d2_sigma_dm2"],
                d_sigma_dT=cur["d_sigma_dT"],
            ),
        )

        return {
            "symbol": self.symbol,
            "ts_unix_ns": int(ts_unix_ns),
            "F": float(F),
            "primary_model": primary_model,
            "grid": {
                "m_grid": m_grid,         # [nM]
                "T_grid": T_grid,         # [nT]
            },
            "feature_names": feature_names,
            "feature_tensor": feature_tensor,  # [nT, nM, 15]
            "surface": {
                "sigma": cur["sigma"],
                "d_sigma_dm": cur["d_sigma_dm"],
                "d2_sigma_dm2": cur["d2_sigma_dm2"],
                "d_sigma_dT": cur["d_sigma_dT"],
                "d_sigma_dt_true": d_sigma_dt_true,
                "d2_sigma_dmdt_true": d2_sigma_dmdt_true,
                "d2_sigma_dTdt_true": d2_sigma_dTdt_true,
            },
            "term_view": {
                "atm_iv": atm_iv_term,     # [nT]
                "atm_vel": atm_vel_term,   # [nT]
            },
            "params": {
                "sabr": sabr_T,            # dict of [nT] curves
                "heston": heston_T,        # dict of [nT] curves
            },
            "diagnostics": {
                "dt_seconds": float(dt),
                "used_prev_state": prev is not None and prev.sigma.shape == cur["sigma"].shape,
            },
        }

