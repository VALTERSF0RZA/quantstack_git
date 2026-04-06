# c4/hamiltonian_c4.py
# =============================================================================
# DESK-GRADE C4 HAMILTONIAN (FP64, JAX/JIT/XLA)
#
# هدف:
#   H(u) = -E[PnL] + u^T Σ u + λ_turn ||u-u_prev||^2 + H_inst(ρ, E_rot, anis_energy)
#
# Where (regime-aware):
#   z_x = (x - μ_x) / (σ_x + eps)   with (μ, σ) tracked via EWMA online
#   φ(z) = softplus(z / τ)         (temperature τ prevents overreaction)
#   H_inst = λρ φ(zρ) + λr φ(zr) + λa φ(za) + λra φ(zr*za)
#
# Notes (XLA-safe):
#   - fixed shapes
#   - FP64 throughout
#   - optional ρ approximation (spectral norm upper bound) for safer compilation
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
F64 = jnp.float64


# ------------------------------- configs ------------------------------------

@dataclass(frozen=True)
class EWMAConfig:
    alpha: float = 0.02            # slower = more regime-stable
    min_var: float = 1e-10         # floor to avoid sigma collapse in quiet regimes
    eps: float = 1e-12
    z_clip: float = 12.0           # avoid pathological tails


@dataclass(frozen=True)
class InstabilityConfig:
    lam_rho: float = 1.0
    lam_rot: float = 0.75
    lam_anis: float = 0.50
    lam_cross: float = 0.50

    # temperatures (smaller => harsher reaction to z)
    tau_rho: float = 1.0
    tau_rot: float = 1.0
    tau_anis: float = 1.0
    tau_cross: float = 1.0

    ewma: EWMAConfig = EWMAConfig()

    # ρ mode:
    #   "eig"   => exact spectral radius via eigvals (best when latent dim small, e.g. 16..64)
    #   "snorm" => spectral-norm upper bound via power-iter on J^T J (very XLA-stable)
    rho_mode: str = "eig"
    snorm_power_iters: int = 8


@dataclass(frozen=True)
class C4Config:
    n_assets: int
    risk_aversion: float = 1.0      # scales u^T Σ u
    lambda_turn: float = 0.10       # turnover penalty
    instability: InstabilityConfig = InstabilityConfig()


# --------------------------- numerics / helpers -----------------------------

@jax.jit
def softplus_stable(x: jnp.ndarray) -> jnp.ndarray:
    # stable softplus: log(1 + exp(x))
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0.0)


@jax.jit
def clip_z(z: jnp.ndarray, z_clip: float) -> jnp.ndarray:
    return jnp.clip(z, -z_clip, z_clip)


@jax.jit
def quad_form(u: jnp.ndarray, Sigma: jnp.ndarray) -> jnp.ndarray:
    # u: [N], Sigma: [N,N]
    return (u @ (Sigma @ u)).astype(F64)


# --------------------------- EWMA stats (online) ----------------------------

# We track EWMA mean and EWMA second moment; var = s2 - mu^2 (floored).
# State is a PyTree-friendly dict of scalars.

def init_instability_stats() -> Dict[str, jnp.ndarray]:
    z = jnp.array(0.0, dtype=F64)
    one = jnp.array(1.0, dtype=F64)  # start with nonzero variance baseline if you want
    return {
        "mu_rho": z,  "s2_rho": one,
        "mu_rot": z,  "s2_rot": one,
        "mu_anis": z, "s2_anis": one,
    }


@jax.jit
def _ewma_update(mu: jnp.ndarray, s2: jnp.ndarray, x: jnp.ndarray, alpha: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # EWMA on mean and second moment
    mu2 = (1.0 - alpha) * mu + alpha * x
    s22 = (1.0 - alpha) * s2 + alpha * (x * x)
    return mu2.astype(F64), s22.astype(F64)


@jax.jit
def _ewma_sigma(mu: jnp.ndarray, s2: jnp.ndarray, min_var: float, eps: float) -> jnp.ndarray:
    var = jnp.maximum(s2 - mu * mu, jnp.array(min_var, dtype=F64))
    return jnp.sqrt(var + eps).astype(F64)


@partial(jax.jit, static_argnames=("cfg",))
def update_instability_stats(
    stats: Dict[str, jnp.ndarray],
    rho: jnp.ndarray,
    E_rot: jnp.ndarray,
    anis_energy: jnp.ndarray,
    cfg: EWMAConfig,
) -> Dict[str, jnp.ndarray]:
    mu_rho, s2_rho = _ewma_update(stats["mu_rho"], stats["s2_rho"], rho, cfg.alpha)
    mu_rot, s2_rot = _ewma_update(stats["mu_rot"], stats["s2_rot"], E_rot, cfg.alpha)
    mu_an,  s2_an  = _ewma_update(stats["mu_anis"], stats["s2_anis"], anis_energy, cfg.alpha)
    return {
        "mu_rho": mu_rho, "s2_rho": s2_rho,
        "mu_rot": mu_rot, "s2_rot": s2_rot,
        "mu_anis": mu_an, "s2_anis": s2_an,
    }


@partial(jax.jit, static_argnames=("cfg",))
def zscore_from_stats(
    x: jnp.ndarray,
    mu: jnp.ndarray,
    s2: jnp.ndarray,
    cfg: EWMAConfig,
) -> jnp.ndarray:
    sig = _ewma_sigma(mu, s2, cfg.min_var, cfg.eps)
    z = (x - mu) / (sig + cfg.eps)
    return clip_z(z.astype(F64), cfg.z_clip)


# -------------------------- rho(J) computations -----------------------------

@jax.jit
def rho_from_eig(J: jnp.ndarray) -> jnp.ndarray:
    # exact spectral radius for small latent dimension
    ev = jnp.linalg.eigvals(J.astype(F64))
    return jnp.max(jnp.abs(ev)).astype(F64)


@partial(jax.jit, static_argnames=("power_iters",))
def spectral_norm_upper_bound(J: jnp.ndarray, power_iters: int = 8) -> jnp.ndarray:
    # ||J||_2 = sqrt( lambda_max(J^T J) ); upper-bounds spectral radius, very stable.
    A = (J.T @ J).astype(F64)
    n = A.shape[0]
    v = jnp.ones((n,), dtype=F64) / jnp.sqrt(jnp.array(n, dtype=F64))

    def body(_, vv):
        vv = A @ vv
        vv = vv / (jnp.linalg.norm(vv) + jnp.array(1e-12, dtype=F64))
        return vv

    vT = jax.lax.fori_loop(0, power_iters, body, v)
    Av = A @ vT
    lam = (vT @ Av).astype(F64)  # Rayleigh quotient
    return jnp.sqrt(jnp.maximum(lam, jnp.array(0.0, dtype=F64))).astype(F64)


@partial(jax.jit, static_argnames=("cfg",))
def compute_rho(J_latent: jnp.ndarray, cfg: InstabilityConfig) -> jnp.ndarray:
    # keep branching static via cfg (static_argnames)
    if cfg.rho_mode == "eig":
        return rho_from_eig(J_latent)
    else:
        return spectral_norm_upper_bound(J_latent, power_iters=cfg.snorm_power_iters)


# ------------------------ H_inst(ρ, E_rot, anis) ----------------------------

@partial(jax.jit, static_argnames=("cfg",))
def instability_hamiltonian(
    rho: jnp.ndarray,
    E_rot: jnp.ndarray,
    anis_energy: jnp.ndarray,
    stats: Dict[str, jnp.ndarray],
    cfg: InstabilityConfig,
) -> jnp.ndarray:
    ew = cfg.ewma

    z_rho = zscore_from_stats(rho, stats["mu_rho"], stats["s2_rho"], ew)
    z_rot = zscore_from_stats(E_rot, stats["mu_rot"], stats["s2_rot"], ew)
    z_an  = zscore_from_stats(anis_energy, stats["mu_anis"], stats["s2_anis"], ew)

    # temperature-scaled softplus
    t_rho = cfg.lam_rho * softplus_stable(z_rho / jnp.array(cfg.tau_rho, dtype=F64))
    t_rot = cfg.lam_rot * softplus_stable(z_rot / jnp.array(cfg.tau_rot, dtype=F64))
    t_an  = cfg.lam_anis * softplus_stable(z_an  / jnp.array(cfg.tau_anis, dtype=F64))

    # crucial interaction: rotation × anisotropy
    z_cross = clip_z((z_rot * z_an).astype(F64), ew.z_clip)
    t_cross = cfg.lam_cross * softplus_stable(z_cross / jnp.array(cfg.tau_cross, dtype=F64))

    return (t_rho + t_rot + t_an + t_cross).astype(F64)


# ----------------------------- full C4 H(u) --------------------------------

@partial(jax.jit, static_argnames=("cfg",))
def c4_hamiltonian(
    u: jnp.ndarray,                 # [N] positions/weights
    u_prev: jnp.ndarray,            # [N]
    mu: jnp.ndarray,                # [N] expected return / alpha
    Sigma: jnp.ndarray,             # [N,N] risk covariance in position space
    J_latent: jnp.ndarray,          # [d,d] from C3 (∂f/∂x)
    E_rot: jnp.ndarray,             # scalar from risk_manifold feats["E_rot"]
    anis_energy: jnp.ndarray,       # scalar from encoder anisotropy module
    stats: Dict[str, jnp.ndarray],  # EWMA stats for normalization
    cfg: C4Config,
) -> jnp.ndarray:
    # enforce FP64
    u = u.astype(F64)
    u_prev = u_prev.astype(F64)
    mu = mu.astype(F64)
    Sigma = Sigma.astype(F64)

    # --- PnL term: -E[PnL] (maximize mu·u)
    pnl = (mu @ u).astype(F64)
    H_pnl = (-pnl).astype(F64)

    # --- Risk term: risk_aversion * u^T Σ u
    H_risk = (jnp.array(cfg.risk_aversion, dtype=F64) * quad_form(u, Sigma)).astype(F64)

    # --- Turnover term: λ_turn ||u-u_prev||^2
    du = (u - u_prev).astype(F64)
    H_turn = (jnp.array(cfg.lambda_turn, dtype=F64) * (du @ du)).astype(F64)

    # --- Instability term (regime-aware)
    rho = compute_rho(J_latent.astype(F64), cfg.instability)
    H_inst = instability_hamiltonian(rho, E_rot.astype(F64), anis_energy.astype(F64), stats, cfg.instability)

    return (H_pnl + H_risk + H_turn + H_inst).astype(F64)


@partial(jax.jit, static_argnames=("cfg",))
def c4_objective_and_grad(
    u: jnp.ndarray,
    u_prev: jnp.ndarray,
    mu: jnp.ndarray,
    Sigma: jnp.ndarray,
    J_latent: jnp.ndarray,
    E_rot: jnp.ndarray,
    anis_energy: jnp.ndarray,
    stats: Dict[str, jnp.ndarray],
    cfg: C4Config,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    f = lambda uu: c4_hamiltonian(
        uu, u_prev, mu, Sigma, J_latent, E_rot, anis_energy, stats, cfg
    )
    H = f(u)
    g = jax.grad(f)(u).astype(F64)
    return H, g


# -------------------------- optional: one PGD step --------------------------

@partial(jax.jit, static_argnames=("cfg",))
def c4_pgd_step(
    u: jnp.ndarray,
    u_prev: jnp.ndarray,
    mu: jnp.ndarray,
    Sigma: jnp.ndarray,
    J_latent: jnp.ndarray,
    E_rot: jnp.ndarray,
    anis_energy: jnp.ndarray,
    stats: Dict[str, jnp.ndarray],
    step_size: float,
    cfg: C4Config,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    H, g = c4_objective_and_grad(u, u_prev, mu, Sigma, J_latent, E_rot, anis_energy, stats, cfg)
    u2 = (u - jnp.array(step_size, dtype=F64) * g).astype(F64)
    return u2, H


# ------------------------------- usage note ---------------------------------
# Runtime wiring (one decision tick):
#
# 1) C2 -> risk_manifold gives feats (incl. feats["E_rot"])
# 2) Encoder anisotropy module gives anis_energy
# 3) C3 latent_dynamics gives J_latent (∂f/∂x) for the *latent* state
# 4) Update EWMA stats ONCE per tick (outside optimization):
#       rho = compute_rho(J_latent, cfg.instability)
#       stats = update_instability_stats(stats, rho, feats["E_rot"], anis_energy, cfg.instability.ewma)
#    Then pass stats into optimization for consistent regime scaling.
#
# 5) Optimize u by your solver (QP/MPC/PGD). This file gives you H(u) + ∇H(u).
# =============================================================================

if __name__ == "__main__":
    # Minimal smoke-test (static shapes)
    N = 200
    d = 32
    cfg = C4Config(n_assets=N)

    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    u = jnp.zeros((N,), dtype=F64)
    u_prev = jnp.zeros((N,), dtype=F64)
    mu = jax.random.normal(k1, (N,), dtype=F64) * 0.001

    A = jax.random.normal(k2, (N, N), dtype=F64)
    Sigma = (A @ A.T) / jnp.array(N, dtype=F64) + 1e-6 * jnp.eye(N, dtype=F64)

    J_latent = jax.random.normal(k3, (d, d), dtype=F64) * 0.05
    E_rot = jnp.array(0.10, dtype=F64)
    anis_energy = jnp.array(0.05, dtype=F64)

    stats = init_instability_stats()

    # update stats once (like live monitor tick)
    rho = compute_rho(J_latent, cfg.instability)
    stats = update_instability_stats(stats, rho, E_rot, anis_energy, cfg.instability.ewma)

    H, g = c4_objective_and_grad(u, u_prev, mu, Sigma, J_latent, E_rot, anis_energy, stats, cfg)
    print("H:", float(H), "||grad||:", float(jnp.linalg.norm(g)))

