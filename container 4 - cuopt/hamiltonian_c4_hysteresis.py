# c4/hamiltonian_c4_hysteresis.py
# =============================================================================
# C4 HAMILTONIAN with:
#   - regime-normalized instability terms (rho, E_rot, anis)
#   - Δ penalties
#   - Δ² (jerk) penalties
#   - STATEFUL HYSTERESIS GATE (enter/exit thresholds + hold timer)
#
# Goal:
#   stop allocator ping-pong near boundaries in choppy regimes by:
#     (1) entering DEFENSIVE only when instability is clearly elevated
#     (2) exiting only after instability falls materially below exit threshold
#     (3) enforcing a minimum hold time once DEFENSIVE is entered
#     (4) optionally increasing turnover penalty while DEFENSIVE
#
# XLA-static:
#   all shapes fixed; gate state is int32 scalars stored in stats.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
F64 = jnp.float64
I32 = jnp.int32


# ------------------------------- configs ------------------------------------

@dataclass(frozen=True)
class EWMAConfig:
    alpha: float = 0.02
    min_var: float = 1e-10
    eps: float = 1e-12
    z_clip: float = 12.0


@dataclass(frozen=True)
class HysteresisConfig:
    # Enter DEFENSIVE when score_z >= enter_z
    enter_z: float = 2.25
    # Exit DEFENSIVE only when score_z <= exit_z  (must be < enter_z)
    exit_z: float = 1.25
    # Minimum number of ticks to stay in DEFENSIVE once entered
    min_hold: int = 8

    # What the gate actually does
    inst_scale_on: float = 1.75     # multiply H_inst
    inst_scale_off: float = 1.00
    turn_mult_on: float = 2.50      # multiply turnover penalty during DEFENSIVE
    turn_mult_off: float = 1.00


@dataclass(frozen=True)
class InstabilityConfig:
    # level weights
    lam_rho: float = 1.0
    lam_rot: float = 0.75
    lam_anis: float = 0.50
    lam_cross: float = 0.50        # level cross: z_rot * z_anis

    # Δ weights
    lam_drho: float = 0.35
    lam_drot: float = 0.35
    lam_danis: float = 0.35
    lam_dcross: float = 0.35       # Δ cross: z_drot * z_danis

    # Δ² weights
    lam_ddrho: float = 0.20
    lam_ddrot: float = 0.20
    lam_ddanis: float = 0.20
    lam_ddcross: float = 0.20      # Δ² cross: z_ddrot * z_ddanis

    # softplus temperatures
    tau: float = 1.0

    ewma: EWMAConfig = EWMAConfig()
    hyst: HysteresisConfig = HysteresisConfig()

    # rho(J) mode:
    #   "eig"   exact spectral radius (small latent dims)
    #   "snorm" stable upper bound via ||J||_2
    rho_mode: str = "eig"
    snorm_power_iters: int = 8


@dataclass(frozen=True)
class C4Config:
    n_assets: int
    risk_aversion: float = 1.0
    lambda_turn: float = 0.10
    instability: InstabilityConfig = InstabilityConfig()


# --------------------------- numerics / helpers -----------------------------

@jax.jit
def softplus_stable(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0.0)


@jax.jit
def clip_z(z: jnp.ndarray, z_clip: float) -> jnp.ndarray:
    return jnp.clip(z, -z_clip, z_clip)


@jax.jit
def quad_form(u: jnp.ndarray, Sigma: jnp.ndarray) -> jnp.ndarray:
    return (u @ (Sigma @ u)).astype(F64)


@jax.jit
def _ewma_update(mu: jnp.ndarray, s2: jnp.ndarray, x: jnp.ndarray, alpha: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    mu2 = (1.0 - alpha) * mu + alpha * x
    s22 = (1.0 - alpha) * s2 + alpha * (x * x)
    return mu2.astype(F64), s22.astype(F64)


@jax.jit
def _ewma_sigma(mu: jnp.ndarray, s2: jnp.ndarray, min_var: float, eps: float) -> jnp.ndarray:
    var = jnp.maximum(s2 - mu * mu, jnp.array(min_var, dtype=F64))
    return jnp.sqrt(var + eps).astype(F64)


@partial(jax.jit, static_argnames=("cfg",))
def zscore_from_stats(x: jnp.ndarray, mu: jnp.ndarray, s2: jnp.ndarray, cfg: EWMAConfig) -> jnp.ndarray:
    sig = _ewma_sigma(mu, s2, cfg.min_var, cfg.eps)
    z = (x - mu) / (sig + cfg.eps)
    return clip_z(z.astype(F64), cfg.z_clip)


# -------------------------- rho(J) computations -----------------------------

@jax.jit
def rho_from_eig(J: jnp.ndarray) -> jnp.ndarray:
    ev = jnp.linalg.eigvals(J.astype(F64))
    return jnp.max(jnp.abs(ev)).astype(F64)


@partial(jax.jit, static_argnames=("power_iters",))
def spectral_norm_upper_bound(J: jnp.ndarray, power_iters: int = 8) -> jnp.ndarray:
    A = (J.T @ J).astype(F64)
    n = A.shape[0]
    v = jnp.ones((n,), dtype=F64) / jnp.sqrt(jnp.array(n, dtype=F64))

    def body(_, vv):
        vv = A @ vv
        vv = vv / (jnp.linalg.norm(vv) + jnp.array(1e-12, dtype=F64))
        return vv

    vT = jax.lax.fori_loop(0, power_iters, body, v)
    lam = (vT @ (A @ vT)).astype(F64)
    return jnp.sqrt(jnp.maximum(lam, jnp.array(0.0, dtype=F64))).astype(F64)


@partial(jax.jit, static_argnames=("cfg",))
def compute_rho(J_latent: jnp.ndarray, cfg: InstabilityConfig) -> jnp.ndarray:
    if cfg.rho_mode == "eig":
        return rho_from_eig(J_latent)
    else:
        return spectral_norm_upper_bound(J_latent, power_iters=cfg.snorm_power_iters)


# ---------------------- stats: levels + Δ + Δ² + gate -----------------------

def init_instability_stats_hysteresis() -> Dict[str, jnp.ndarray]:
    z = jnp.array(0.0, dtype=F64)
    one = jnp.array(1.0, dtype=F64)

    return {
        # EWMA for levels
        "mu_rho": z,   "s2_rho": one,
        "mu_rot": z,   "s2_rot": one,
        "mu_anis": z,  "s2_anis": one,

        # EWMA for abs deltas
        "mu_drho": z,  "s2_drho": one,
        "mu_drot": z,  "s2_drot": one,
        "mu_danis": z, "s2_danis": one,

        # EWMA for abs second differences (Δ²)
        "mu_ddrho": z,  "s2_ddrho": one,
        "mu_ddrot": z,  "s2_ddrot": one,
        "mu_ddanis": z, "s2_ddanis": one,

        # previous raw values
        "prev_rho": z,
        "prev_rot": z,
        "prev_anis": z,

        # previous deltas
        "prev_drho": z,
        "prev_drot": z,
        "prev_danis": z,

        # hysteresis gate state
        "gate_on": jnp.array(0, dtype=I32),         # 0/1
        "gate_hold": jnp.array(0, dtype=I32),       # ticks remaining (>=0)
        "gate_score_prev": z,                       # optional logging / debugging
    }


@jax.jit
def _hysteresis_update(
    gate_on: jnp.ndarray,     # int32 scalar
    gate_hold: jnp.ndarray,   # int32 scalar
    score_z: jnp.ndarray,     # float scalar (already clipped)
    enter_z: float,
    exit_z: float,
    min_hold: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Stateful two-threshold gate with minimum hold timer.

    - If OFF and score >= enter: turn ON, set hold=min_hold
    - If ON:
        - If hold>0: decrement hold, stay ON
        - Else (hold==0): if score <= exit: turn OFF else stay ON
    """
    enter = jnp.array(enter_z, dtype=F64)
    exit_ = jnp.array(exit_z, dtype=F64)
    min_h = jnp.array(min_hold, dtype=I32)

    def when_off(_):
        turn_on = (score_z >= enter).astype(I32)
        new_on = turn_on
        new_hold = jnp.where(turn_on == 1, min_h, jnp.array(0, dtype=I32))
        return new_on, new_hold

    def when_on(_):
        # hold countdown has priority
        still_holding = (gate_hold > 0).astype(I32)
        hold2 = jnp.maximum(gate_hold - jnp.array(1, dtype=I32), jnp.array(0, dtype=I32))
        # if not holding, allow exit only below exit threshold
        exit_now = ((score_z <= exit_) & (hold2 == 0)).astype(I32)
        on2 = jnp.where(exit_now == 1, jnp.array(0, dtype=I32), jnp.array(1, dtype=I32))
        hold3 = jnp.where(on2 == 0, jnp.array(0, dtype=I32), hold2)
        return on2, hold3

    return jax.lax.cond(gate_on == 0, when_off, when_on, operand=None)


# ----------------- instability: compute z-components + score_z ---------------

@partial(jax.jit, static_argnames=("cfg",))
def compute_instability_z_components(
    rho: jnp.ndarray,
    E_rot: jnp.ndarray,
    anis_energy: jnp.ndarray,
    stats_prev: Dict[str, jnp.ndarray],
    cfg: InstabilityConfig,
) -> Dict[str, jnp.ndarray]:
    ew = cfg.ewma

    rho = rho.astype(F64)
    E_rot = E_rot.astype(F64)
    anis_energy = anis_energy.astype(F64)

    # levels
    z_rho = zscore_from_stats(rho,         stats_prev["mu_rho"],  stats_prev["s2_rho"],  ew)
    z_rot = zscore_from_stats(E_rot,       stats_prev["mu_rot"],  stats_prev["s2_rot"],  ew)
    z_an  = zscore_from_stats(anis_energy, stats_prev["mu_anis"], stats_prev["s2_anis"], ew)

    # deltas
    d_rho  = jnp.abs(rho - stats_prev["prev_rho"]).astype(F64)
    d_rot  = jnp.abs(E_rot - stats_prev["prev_rot"]).astype(F64)
    d_anis = jnp.abs(anis_energy - stats_prev["prev_anis"]).astype(F64)

    z_drho  = zscore_from_stats(d_rho,  stats_prev["mu_drho"],  stats_prev["s2_drho"],  ew)
    z_drot  = zscore_from_stats(d_rot,  stats_prev["mu_drot"],  stats_prev["s2_drot"],  ew)
    z_danis = zscore_from_stats(d_anis, stats_prev["mu_danis"], stats_prev["s2_danis"], ew)

    # second diffs
    dd_rho  = jnp.abs(d_rho  - stats_prev["prev_drho"]).astype(F64)
    dd_rot  = jnp.abs(d_rot  - stats_prev["prev_drot"]).astype(F64)
    dd_anis = jnp.abs(d_anis - stats_prev["prev_danis"]).astype(F64)

    z_ddrho  = zscore_from_stats(dd_rho,  stats_prev["mu_ddrho"],  stats_prev["s2_ddrho"],  ew)
    z_ddrot  = zscore_from_stats(dd_rot,  stats_prev["mu_ddrot"],  stats_prev["s2_ddrot"],  ew)
    z_ddanis = zscore_from_stats(dd_anis, stats_prev["mu_ddanis"], stats_prev["s2_ddanis"], ew)

    # cross terms (these are the ones that cause boundary thrash in rotation regimes)
    z_cross   = clip_z((z_rot * z_an).astype(F64), ew.z_clip)
    z_dcross  = clip_z((z_drot * z_danis).astype(F64), ew.z_clip)
    z_ddcross = clip_z((z_ddrot * z_ddanis).astype(F64), ew.z_clip)

    return {
        "z_rho": z_rho, "z_rot": z_rot, "z_an": z_an,
        "z_drho": z_drho, "z_drot": z_drot, "z_danis": z_danis,
        "z_ddrho": z_ddrho, "z_ddrot": z_ddrot, "z_ddanis": z_ddanis,
        "z_cross": z_cross, "z_dcross": z_dcross, "z_ddcross": z_ddcross,
    }


@partial(jax.jit, static_argnames=("cfg",))
def compute_gate_score_z(
    rho: jnp.ndarray,
    E_rot: jnp.ndarray,
    anis_energy: jnp.ndarray,
    stats_prev: Dict[str, jnp.ndarray],
    cfg: InstabilityConfig,
) -> jnp.ndarray:
    """
    Gate score should detect “boundary conditions” that cause ping-pong:
      - elevated rho (local expansion)
      - elevated rotation×anis (rotation becoming shear)
      - elevated Δ and Δ² cross terms (transition snap)
    We use max() to keep it conservative.
    """
    zc = compute_instability_z_components(rho, E_rot, anis_energy, stats_prev, cfg)
    score = jnp.max(jnp.stack([
        zc["z_rho"],
        zc["z_cross"],
        zc["z_dcross"],
        zc["z_ddcross"],
    ], axis=0))
    return score.astype(F64)


# -------------------- H_inst with levels + Δ + Δ² penalties ------------------

@partial(jax.jit, static_argnames=("cfg",))
def instability_hamiltonian_with_gate(
    rho: jnp.ndarray,
    E_rot: jnp.ndarray,
    anis_energy: jnp.ndarray,
    stats_prev: Dict[str, jnp.ndarray],
    cfg: InstabilityConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns:
      H_inst_scaled (scalar)
      gate_on_eff   (int32) gate used for scaling in THIS tick

    Desk rule:
      - We compute score_z from (rho, E_rot, anis) using stats_prev.
      - We update an "effective gate" (pure) and STOP_GRAD through it.
      - The optimizer sees a stable discrete regime flag for the tick.
    """
    ew = cfg.ewma
    hy = cfg.hyst
    tau = jnp.array(cfg.tau, dtype=F64)

    zc = compute_instability_z_components(rho, E_rot, anis_energy, stats_prev, cfg)

    # Base instability energy
    lvl = (
        cfg.lam_rho  * softplus_stable(zc["z_rho"] / tau) +
        cfg.lam_rot  * softplus_stable(zc["z_rot"] / tau) +
        cfg.lam_anis * softplus_stable(zc["z_an"]  / tau) +
        cfg.lam_cross * softplus_stable(zc["z_cross"] / tau)
    )

    dlt = (
        cfg.lam_drho  * softplus_stable(zc["z_drho"] / tau) +
        cfg.lam_drot  * softplus_stable(zc["z_drot"] / tau) +
        cfg.lam_danis * softplus_stable(zc["z_danis"] / tau) +
        cfg.lam_dcross * softplus_stable(zc["z_dcross"] / tau)
    )

    jerk = (
        cfg.lam_ddrho  * softplus_stable(zc["z_ddrho"] / tau) +
        cfg.lam_ddrot  * softplus_stable(zc["z_ddrot"] / tau) +
        cfg.lam_ddanis * softplus_stable(zc["z_ddanis"] / tau) +
        cfg.lam_ddcross * softplus_stable(zc["z_ddcross"] / tau)
    )

    H_inst = (lvl + dlt + jerk).astype(F64)

    # Gate score + stateful hysteresis update (pure, uses previous gate state)
    score_z = compute_gate_score_z(rho, E_rot, anis_energy, stats_prev, cfg)
    gate_on_prev = stats_prev["gate_on"].astype(I32)
    gate_hold_prev = stats_prev["gate_hold"].astype(I32)

    gate_on_eff, gate_hold_eff = _hysteresis_update(
        gate_on_prev, gate_hold_prev, score_z,
        enter_z=hy.enter_z, exit_z=hy.exit_z, min_hold=hy.min_hold
    )

    # Gate multiplier (stop-gradient: discrete regime flag should not create garbage grads)
    gate_on_eff = jax.lax.stop_gradient(gate_on_eff)
    scale = jnp.where(gate_on_eff == 1,
                      jnp.array(hy.inst_scale_on, dtype=F64),
                      jnp.array(hy.inst_scale_off, dtype=F64))
    H_inst_scaled = (scale * H_inst).astype(F64)

    return H_inst_scaled, gate_on_eff


# ----------------------------- full C4 H(u) --------------------------------

@partial(jax.jit, static_argnames=("cfg",))
def c4_hamiltonian_hysteresis(
    u: jnp.ndarray,                    # [N]
    u_prev: jnp.ndarray,               # [N]
    mu: jnp.ndarray,                   # [N]
    Sigma: jnp.ndarray,                # [N,N]
    J_latent: jnp.ndarray,             # [d,d]
    E_rot: jnp.ndarray,                # scalar
    anis_energy: jnp.ndarray,          # scalar
    stats_prev: Dict[str, jnp.ndarray],# stats up to t-1
    cfg: C4Config,
) -> jnp.ndarray:
    u = u.astype(F64)
    u_prev = u_prev.astype(F64)
    mu = mu.astype(F64)
    Sigma = Sigma.astype(F64)

    # PnL (maximize mu·u)
    H_pnl = (-(mu @ u)).astype(F64)

    # risk
    H_risk = (jnp.array(cfg.risk_aversion, dtype=F64) * quad_form(u, Sigma)).astype(F64)

    # instability scalars
    rho = compute_rho(J_latent.astype(F64), cfg.instability)
    H_inst_scaled, gate_on_eff = instability_hamiltonian_with_gate(
        rho, E_rot, anis_energy, stats_prev, cfg.instability
    )

    # turnover (optionally tighten during DEFENSIVE to stop flip-flopping)
    hy = cfg.instability.hyst
    turn_mult = jnp.where(gate_on_eff == 1,
                          jnp.array(hy.turn_mult_on, dtype=F64),
                          jnp.array(hy.turn_mult_off, dtype=F64))
    du = (u - u_prev).astype(F64)
    H_turn = (jnp.array(cfg.lambda_turn, dtype=F64) * turn_mult * (du @ du)).astype(F64)

    return (H_pnl + H_risk + H_turn + H_inst_scaled).astype(F64)


@partial(jax.jit, static_argnames=("cfg",))
def c4_objective_and_grad_hysteresis(
    u: jnp.ndarray,
    u_prev: jnp.ndarray,
    mu: jnp.ndarray,
    Sigma: jnp.ndarray,
    J_latent: jnp.ndarray,
    E_rot: jnp.ndarray,
    anis_energy: jnp.ndarray,
    stats_prev: Dict[str, jnp.ndarray],
    cfg: C4Config,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    f = lambda uu: c4_hamiltonian_hysteresis(
        uu, u_prev, mu, Sigma, J_latent, E_rot, anis_energy, stats_prev, cfg
    )
    H = f(u)
    g = jax.grad(f)(u).astype(F64)
    return H, g


# -------------------------- end-of-tick stats update -------------------------

@partial(jax.jit, static_argnames=("cfg",))
def update_instability_stats_with_gate(
    stats: Dict[str, jnp.ndarray],
    rho: jnp.ndarray,
    E_rot: jnp.ndarray,
    anis_energy: jnp.ndarray,
    cfg: InstabilityConfig,
) -> Dict[str, jnp.ndarray]:
    """
    Call AFTER you commit u_t and observe rho_t/E_rot_t/anis_t for tick t.

    Updates:
      - level EWMAs
      - Δ EWMAs
      - Δ² EWMAs
      - hysteresis gate state + hold timer (using stats from t-1, score at t)
    """
    ew = cfg.ewma
    alpha = ew.alpha

    rho = rho.astype(F64)
    E_rot = E_rot.astype(F64)
    anis_energy = anis_energy.astype(F64)

    # deltas
    d_rho  = jnp.abs(rho - stats["prev_rho"]).astype(F64)
    d_rot  = jnp.abs(E_rot - stats["prev_rot"]).astype(F64)
    d_anis = jnp.abs(anis_energy - stats["prev_anis"]).astype(F64)

    # second diffs
    dd_rho  = jnp.abs(d_rho  - stats["prev_drho"]).astype(F64)
    dd_rot  = jnp.abs(d_rot  - stats["prev_drot"]).astype(F64)
    dd_anis = jnp.abs(d_anis - stats["prev_danis"]).astype(F64)

    # update level ewmas
    mu_rho, s2_rho = _ewma_update(stats["mu_rho"],  stats["s2_rho"],  rho,         alpha)
    mu_rot, s2_rot = _ewma_update(stats["mu_rot"],  stats["s2_rot"],  E_rot,       alpha)
    mu_an,  s2_an  = _ewma_update(stats["mu_anis"], stats["s2_anis"], anis_energy, alpha)

    # update delta ewmas
    mu_dr,   s2_dr   = _ewma_update(stats["mu_drho"],  stats["s2_drho"],  d_rho,  alpha)
    mu_drot, s2_drot = _ewma_update(stats["mu_drot"],  stats["s2_drot"],  d_rot,  alpha)
    mu_dan,  s2_dan  = _ewma_update(stats["mu_danis"], stats["s2_danis"], d_anis, alpha)

    # update second-diff ewmas
    mu_ddr,   s2_ddr   = _ewma_update(stats["mu_ddrho"],  stats["s2_ddrho"],  dd_rho,  alpha)
    mu_ddrot, s2_ddrot = _ewma_update(stats["mu_ddrot"],  stats["s2_ddrot"],  dd_rot,  alpha)
    mu_ddan,  s2_ddan  = _ewma_update(stats["mu_ddanis"], stats["s2_ddanis"], dd_anis, alpha)

    # gate update (use score_z based on pre-update stats to keep it consistent)
    score_z = compute_gate_score_z(rho, E_rot, anis_energy, stats, cfg)
    gate_on2, gate_hold2 = _hysteresis_update(
        stats["gate_on"].astype(I32),
        stats["gate_hold"].astype(I32),
        score_z,
        enter_z=cfg.hyst.enter_z,
        exit_z=cfg.hyst.exit_z,
        min_hold=cfg.hyst.min_hold,
    )

    return {
        "mu_rho": mu_rho,   "s2_rho": s2_rho,
        "mu_rot": mu_rot,   "s2_rot": s2_rot,
        "mu_anis": mu_an,   "s2_anis": s2_an,

        "mu_drho": mu_dr,   "s2_drho": s2_dr,
        "mu_drot": mu_drot, "s2_drot": s2_drot,
        "mu_danis": mu_dan, "s2_danis": s2_dan,

        "mu_ddrho": mu_ddr,   "s2_ddrho": s2_ddr,
        "mu_ddrot": mu_ddrot, "s2_ddrot": s2_ddrot,
        "mu_ddanis": mu_ddan, "s2_ddanis": s2_ddan,

        "prev_rho": rho,
        "prev_rot": E_rot,
        "prev_anis": anis_energy,

        "prev_drho": d_rho,
        "prev_drot": d_rot,
        "prev_danis": d_anis,

        "gate_on": gate_on2.astype(I32),
        "gate_hold": gate_hold2.astype(I32),
        "gate_score_prev": score_z.astype(F64),
    }


# ------------------------------- wiring -------------------------------------
# Tick t:
#   inputs from C3: J_latent_t, E_rot_t, anis_t, plus mu_t, Sigma_t, u_{t-1}
#
#   rho_t = compute_rho(J_latent_t, cfg.instability)
#
#   Solve:
#     u_t = argmin_u c4_hamiltonian_hysteresis(..., stats_prev)
#
#   Then update stats:
#     stats_next = update_instability_stats_with_gate(stats_prev, rho_t, E_rot_t, anis_t, cfg.instability)
#
# Carry stats_next into tick t+1.
# =============================================================================

if __name__ == "__main__":
    # Static-shape smoke test
    N = 200
    d = 32
    cfg = C4Config(n_assets=N)

    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)

    u = jnp.zeros((N,), dtype=F64)
    u_prev = jnp.zeros((N,), dtype=F64)
    mu = jax.random.normal(k1, (N,), dtype=F64) * 0.001

    A = jax.random.normal(k2, (N, N), dtype=F64)
    Sigma = (A @ A.T) / jnp.array(N, dtype=F64) + 1e-6 * jnp.eye(N, dtype=F64)

    J_latent = jax.random.normal(k3, (d, d), dtype=F64) * 0.05
    E_rot = jnp.array(0.10, dtype=F64)
    anis_energy = jnp.array(0.05, dtype=F64)

    stats = init_instability_stats_hysteresis()

    H, g = c4_objective_and_grad_hysteresis(u, u_prev, mu, Sigma, J_latent, E_rot, anis_energy, stats, cfg)
    print("H:", float(H), "||grad||:", float(jnp.linalg.norm(g)), "gate_on(prev):", int(stats["gate_on"]))

    rho = compute_rho(J_latent, cfg.instability)
    stats2 = update_instability_stats_with_gate(stats, rho, E_rot, anis_energy, cfg.instability)
    print("gate_on(next):", int(stats2["gate_on"]), "hold(next):", int(stats2["gate_hold"]), "score_z:", float(stats2["gate_score_prev"]))

