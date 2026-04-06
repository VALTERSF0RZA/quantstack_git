# c4/hamiltonian_c4_gate3.py
# =============================================================================
# C4 HAMILTONIAN with:
#   - regime-normalized instability terms (rho, E_rot, anis)
#   - Δ penalties
#   - Δ² penalties
#   - MULTI-STATE GATE FSM: NORMAL / CAUTION / DEFENSIVE
#       * different enter/exit bands (hysteresis)
#       * per-state instability scaling + turnover multipliers
#       * per-state minimum hold timers (prevents snap flips)
#
# Gate states:
#   0 = NORMAL
#   1 = CAUTION
#   2 = DEFENSIVE
#
# XLA-static:
#   fixed shapes only; gate state is int32 scalar in stats.
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
class Gate3Config:
    """
    Two boundaries, each with hysteresis bands:

      NORMAL <-> CAUTION boundary:
        enter CAUTION if score_z >= enter_caution_z
        exit  CAUTION to NORMAL if score_z <= exit_caution_z

      CAUTION <-> DEFENSIVE boundary:
        enter DEFENSIVE if score_z >= enter_defensive_z
        exit  DEFENSIVE to CAUTION if score_z <= exit_defensive_z

    Requirements:
      exit_caution_z  < enter_caution_z
      exit_defensive_z < enter_defensive_z
      enter_defensive_z >= enter_caution_z (usually higher)
    """
    enter_caution_z: float = 1.75
    exit_caution_z: float = 1.10

    enter_defensive_z: float = 2.75
    exit_defensive_z: float = 2.00

    min_hold_caution: int = 6
    min_hold_defensive: int = 10

    # Multipliers indexed by state: [NORMAL, CAUTION, DEFENSIVE]
    inst_scale: Tuple[float, float, float] = (1.00, 1.25, 1.80)
    turn_mult: Tuple[float, float, float] = (1.00, 1.50, 2.75)


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

    tau: float = 1.0
    ewma: EWMAConfig = EWMAConfig()
    gate: Gate3Config = Gate3Config()

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
    # cfg is static, so this branch is compile-time.
    if cfg.rho_mode == "eig":
        return rho_from_eig(J_latent)
    else:
        return spectral_norm_upper_bound(J_latent, power_iters=cfg.snorm_power_iters)


# ---------------------- stats: levels + Δ + Δ² + gate3 ----------------------

def init_instability_stats_gate3() -> Dict[str, jnp.ndarray]:
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

        # EWMA for abs second differences
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

        # gate state (0/1/2) + hold timer
        "gate_state": jnp.array(0, dtype=I32),   # 0=NORMAL
        "gate_hold": jnp.array(0, dtype=I32),

        "gate_score_prev": z,
    }


# ------------------------- gate3 FSM (hysteresis) ---------------------------

@jax.jit
def _gate3_update(
    state: jnp.ndarray,      # int32: 0/1/2
    hold: jnp.ndarray,       # int32 >=0
    score_z: jnp.ndarray,    # float
    enter_caution_z: float,
    exit_caution_z: float,
    enter_defensive_z: float,
    exit_defensive_z: float,
    min_hold_caution: int,
    min_hold_defensive: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Three-state hysteresis FSM with per-state hold.

    Priority:
      - hold > 0 => decrement, no transitions
      - otherwise transitions by bands

    Transitions:
      NORMAL -> CAUTION      if score >= enter_caution
      CAUTION -> DEFENSIVE   if score >= enter_defensive
      CAUTION -> NORMAL      if score <= exit_caution
      DEFENSIVE -> CAUTION   if score <= exit_defensive

    (No direct DEFENSIVE -> NORMAL jump; it must pass through CAUTION.)
    """
    ec = jnp.array(enter_caution_z, dtype=F64)
    xc = jnp.array(exit_caution_z, dtype=F64)
    ed = jnp.array(enter_defensive_z, dtype=F64)
    xd = jnp.array(exit_defensive_z, dtype=F64)

    hC = jnp.array(min_hold_caution, dtype=I32)
    hD = jnp.array(min_hold_defensive, dtype=I32)

    # if holding: just count down
    def holding(_):
        hold2 = jnp.maximum(hold - jnp.array(1, dtype=I32), jnp.array(0, dtype=I32))
        return state, hold2

    def not_holding(_):
        def from_normal(_):
            go_caution = (score_z >= ec)
            s2 = jnp.where(go_caution, jnp.array(1, dtype=I32), jnp.array(0, dtype=I32))
            h2 = jnp.where(go_caution, hC, jnp.array(0, dtype=I32))
            return s2, h2

        def from_caution(_):
            go_def = (score_z >= ed)
            go_norm = (score_z <= xc)

            # DEFENSIVE has priority over NORMAL if both somehow true (rare)
            s2 = jnp.where(go_def, jnp.array(2, dtype=I32),
                           jnp.where(go_norm, jnp.array(0, dtype=I32), jnp.array(1, dtype=I32)))

            h2 = jnp.where(go_def, hD,
                           jnp.where(go_norm, jnp.array(0, dtype=I32), jnp.array(0, dtype=I32)))
            # if we *enter* CAUTION from CAUTION (no), no hold; hold triggers only on upward transitions
            return s2, h2

        def from_defensive(_):
            go_caution = (score_z <= xd)
            s2 = jnp.where(go_caution, jnp.array(1, dtype=I32), jnp.array(2, dtype=I32))
            # if we drop from DEF->CAUTION, impose CAUTION hold (prevents immediate bounce back)
            h2 = jnp.where(go_caution, hC, jnp.array(0, dtype=I32))
            return s2, h2

        return jax.lax.switch(state, [from_normal, from_caution, from_defensive], operand=None)

    return jax.lax.cond(hold > 0, holding, not_holding, operand=None)


@jax.jit
def _state_multiplier3(values: Tuple[float, float, float], state: jnp.ndarray) -> jnp.ndarray:
    arr = jnp.array(values, dtype=F64)  # [3]
    return arr[jnp.clip(state, 0, 2)].astype(F64)


# ----------------- instability z-components + gate score --------------------

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

    # cross terms
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
    zc = compute_instability_z_components(rho, E_rot, anis_energy, stats_prev, cfg)
    # conservative: score is the max of “things that cause snap + ping-pong”
    score = jnp.max(jnp.stack([zc["z_rho"], zc["z_cross"], zc["z_dcross"], zc["z_ddcross"]], axis=0))
    return score.astype(F64)


# ---------------------- H_inst + gate3 (stateful) ---------------------------

@partial(jax.jit, static_argnames=("cfg",))
def instability_hamiltonian_with_gate3(
    rho: jnp.ndarray,
    E_rot: jnp.ndarray,
    anis_energy: jnp.ndarray,
    stats_prev: Dict[str, jnp.ndarray],
    cfg: InstabilityConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns:
      H_inst_scaled: scalar
      gate_state_eff: int32 in {0,1,2} used for THIS tick (stop-grad)
    """
    tau = jnp.array(cfg.tau, dtype=F64)
    zc = compute_instability_z_components(rho, E_rot, anis_energy, stats_prev, cfg)

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

    # Gate3 update (uses previous state/hold; computed from score_z)
    g = cfg.gate
    score_z = compute_gate_score_z(rho, E_rot, anis_energy, stats_prev, cfg)
    s_prev = stats_prev["gate_state"].astype(I32)
    h_prev = stats_prev["gate_hold"].astype(I32)

    s_eff, h_eff = _gate3_update(
        s_prev, h_prev, score_z,
        enter_caution_z=g.enter_caution_z,
        exit_caution_z=g.exit_caution_z,
        enter_defensive_z=g.enter_defensive_z,
        exit_defensive_z=g.exit_defensive_z,
        min_hold_caution=g.min_hold_caution,
        min_hold_defensive=g.min_hold_defensive,
    )

    s_eff = jax.lax.stop_gradient(s_eff)
    inst_scale = _state_multiplier3(g.inst_scale, s_eff)
    H_inst_scaled = (inst_scale * H_inst).astype(F64)
    return H_inst_scaled, s_eff


# ----------------------------- full C4 H(u) --------------------------------

@partial(jax.jit, static_argnames=("cfg",))
def c4_hamiltonian_gate3(
    u: jnp.ndarray,                     # [N]
    u_prev: jnp.ndarray,                # [N]
    mu: jnp.ndarray,                    # [N]
    Sigma: jnp.ndarray,                 # [N,N]
    J_latent: jnp.ndarray,              # [d,d]
    E_rot: jnp.ndarray,                 # scalar
    anis_energy: jnp.ndarray,           # scalar
    stats_prev: Dict[str, jnp.ndarray], # stats up to t-1
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
    H_inst_scaled, gate_state_eff = instability_hamiltonian_with_gate3(
        rho, E_rot, anis_energy, stats_prev, cfg.instability
    )

    # turnover (state-dependent)
    g = cfg.instability.gate
    turn_mult = _state_multiplier3(g.turn_mult, gate_state_eff)
    du = (u - u_prev).astype(F64)
    H_turn = (jnp.array(cfg.lambda_turn, dtype=F64) * turn_mult * (du @ du)).astype(F64)

    return (H_pnl + H_risk + H_turn + H_inst_scaled).astype(F64)


@partial(jax.jit, static_argnames=("cfg",))
def c4_objective_and_grad_gate3(
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
    f = lambda uu: c4_hamiltonian_gate3(
        uu, u_prev, mu, Sigma, J_latent, E_rot, anis_energy, stats_prev, cfg
    )
    H = f(u)
    g = jax.grad(f)(u).astype(F64)
    return H, g


# ---------------------- end-of-tick stats update (gate3) --------------------

@partial(jax.jit, static_argnames=("cfg",))
def update_instability_stats_gate3(
    stats: Dict[str, jnp.ndarray],
    rho: jnp.ndarray,
    E_rot: jnp.ndarray,
    anis_energy: jnp.ndarray,
    cfg: InstabilityConfig,
) -> Dict[str, jnp.ndarray]:
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

    # gate update (use score_z based on pre-update stats for consistency)
    g = cfg.gate
    score_z = compute_gate_score_z(rho, E_rot, anis_energy, stats, cfg)
    s2, h2 = _gate3_update(
        stats["gate_state"].astype(I32),
        stats["gate_hold"].astype(I32),
        score_z,
        enter_caution_z=g.enter_caution_z,
        exit_caution_z=g.exit_caution_z,
        enter_defensive_z=g.enter_defensive_z,
        exit_defensive_z=g.exit_defensive_z,
        min_hold_caution=g.min_hold_caution,
        min_hold_defensive=g.min_hold_defensive,
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

        "gate_state": s2.astype(I32),
        "gate_hold": h2.astype(I32),
        "gate_score_prev": score_z.astype(F64),
    }


# ------------------------------- wiring -------------------------------------
# Tick t:
#   - from C3: J_latent_t, E_rot_t, anis_t
#   - compute rho_t
#   - solve u_t = argmin H(u) using stats_prev (includes gate_state/hold)
#   - update stats_next = update_instability_stats_gate3(stats_prev, rho_t, E_rot_t, anis_t, cfg.instability)
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

    stats = init_instability_stats_gate3()

    H, g = c4_objective_and_grad_gate3(u, u_prev, mu, Sigma, J_latent, E_rot, anis_energy, stats, cfg)
    rho = compute_rho(J_latent, cfg.instability)

    print("H:", float(H),
          "||grad||:", float(jnp.linalg.norm(g)),
          "gate_state(prev):", int(stats["gate_state"]))

    stats2 = update_instability_stats_gate3(stats, rho, E_rot, anis_energy, cfg.instability)
    print("gate_state(next):", int(stats2["gate_state"]),
          "hold(next):", int(stats2["gate_hold"]),
          "score_z:", float(stats2["gate_score_prev"]))

