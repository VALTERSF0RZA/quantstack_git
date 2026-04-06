# c3/risk_manifold.py
# =============================================================================
# ELITE VOL ENGINE: RISK-MANIFOLD STATE FROM C2 FACTORS (FP64, JAX)
#
# Purpose
#   C2 emits per-surface geometry factors (skew/convexity/fragility/entropy/alignment/...).
#   This module turns those into a C3 *state vector* by building an NxN risk covariance,
#   extracting the dominant eigen-subspace, and tracking:
#     - eigenvalues (Λ): risk energy in each systemic mode
#     - eigenvectors (Q): systemic mode portfolios across surfaces
#     - subspace drift magnitude + direction (Grassmann)
#     - curvature (second-order change in drift)
#     - spectral-gap normalized drift (noise-robust)
#     - EWMA smoothing (so C4 doesn't penalize noise)
#     - a C4-ready rotational energy penalty term
#
# Shape contracts (static):
#   N      = number of surfaces/assets (e.g. 200)
#   P      = number of C2 factors per surface (e.g. 5..64)
#   K_MAX  = max systemic modes you track (compile-time, e.g. 16)
#
# Output:
#   A ManifoldFeatures dict with *fixed shapes* suitable for XLA fusion.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
F64 = jnp.float64


# ------------------------------- config/state -------------------------------

@dataclass(frozen=True)
class ManifoldConfig:
    N: int
    P: int
    K_max: int = 16                 # compile-time cap for systemic modes
    ewma_alpha: float = 0.15        # smoothing for drift/curvature signals
    cov_ewma_alpha: float = 0.05    # optional smoothing of Sigma itself
    eps: float = 1e-12

    # C4 penalty weights (tune on desk PnL / risk objectives)
    w_drift: float = 1.0
    w_curv: float = 0.25
    w_rot_dir: float = 0.05         # small regularizer on direction changes


@dataclass
class ManifoldState:
    # Previous top-K subspace (orthonormal columns)
    Q_prev: jnp.ndarray            # [N, K_max]
    # Previous top-(K_max+1) eigenvalues (descending)
    lam_prev: jnp.ndarray          # [K_max+1]
    # Previous "rotation generator" in subspace coordinates
    A_prev: jnp.ndarray            # [K_max, K_max]

    # Smoothed metrics
    ema_drift: jnp.ndarray         # scalar
    ema_curv: jnp.ndarray          # scalar
    ema_drift_norm: jnp.ndarray    # scalar

    # Optional smoothed covariance (for stability)
    Sigma_ema: jnp.ndarray         # [N, N]

    t: jnp.int32                   # step counter


def init_manifold_state(cfg: ManifoldConfig) -> ManifoldState:
    N, K = cfg.N, cfg.K_max
    Q0 = jnp.zeros((N, K), dtype=F64).at[0, 0].set(1.0)  # harmless placeholder
    lam0 = jnp.ones((K + 1,), dtype=F64)
    A0 = jnp.zeros((K, K), dtype=F64)
    Sigma0 = jnp.eye(N, dtype=F64)
    z = jnp.array(0.0, dtype=F64)
    return ManifoldState(
        Q_prev=Q0,
        lam_prev=lam0,
        A_prev=A0,
        ema_drift=z,
        ema_curv=z,
        ema_drift_norm=z,
        Sigma_ema=Sigma0,
        t=jnp.int32(0),
    )


# ----------------------------- core math utils ------------------------------

@jax.jit
def participation_ratio(lam: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """
    Participation ratio (effective dimension):
      PR = (sum lam)^2 / sum(lam^2)
    """
    lam = jnp.maximum(lam, 0.0)
    s1 = jnp.sum(lam)
    s2 = jnp.sum(lam * lam)
    return (s1 * s1) / jnp.maximum(s2, eps)


@jax.jit
def k_from_participation_ratio(lam: jnp.ndarray, K_max: int) -> jnp.ndarray:
    """
    Returns K_eff in [1, K_max] as int32.
    Note: K_eff is dynamic, but we never slice arrays by it (static shapes preserved).
    """
    pr = participation_ratio(lam)
    k = jnp.ceil(pr).astype(jnp.int32)
    k = jnp.clip(k, 1, jnp.int32(K_max))
    return k


@jax.jit
def _center_columns(F: jnp.ndarray) -> jnp.ndarray:
    # F: [N, P]
    mu = jnp.mean(F, axis=1, keepdims=True)  # center per-surface across factors
    return F - mu


@jax.jit
def sigma_from_factors(F: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """
    Build an NxN risk matrix from C2 factor vectors.
    We use a Gram-style covariance across surfaces:
        Sigma = (Fc @ Fc^T) / (P - 1)
    where Fc is centered across factor-dimension.
    """
    Fc = _center_columns(F).astype(F64)  # [N,P]
    P = Fc.shape[1]
    denom = jnp.maximum(P - 1, 1)
    Sigma = (Fc @ Fc.T) / denom
    # small ridge for numerical stability
    Sigma = Sigma + (eps * jnp.eye(Sigma.shape[0], dtype=F64))
    return Sigma


@jax.jit
def top_eigh_desc(Sigma: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Symmetric eigendecomposition, descending order.
    Returns:
      lam: [N] descending
      Q:   [N,N] columns are eigenvectors aligned with lam
    """
    lam, Q = jnp.linalg.eigh(Sigma)          # ascending
    lam = lam[::-1]
    Q = Q[:, ::-1]
    lam = jnp.maximum(lam, 0.0)
    return lam, Q


@jax.jit
def align_eigenvector_signs(Q_prev: jnp.ndarray, Q_t: jnp.ndarray) -> jnp.ndarray:
    """
    Eigenvectors are sign-ambiguous. Align column signs to previous subspace.
    This prevents fake 'rotation' from random sign flips.
    """
    dots = jnp.sum(Q_prev * Q_t, axis=0)  # [K]
    sgn = jnp.where(dots >= 0.0, 1.0, -1.0).astype(F64)
    return Q_t * sgn


# ----------------------- subspace drift + direction -------------------------

@jax.jit
def subspace_drift_and_direction(
    Q_prev: jnp.ndarray,  # [N,K]
    Q_t: jnp.ndarray,     # [N,K]
    eps: float = 1e-12,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Returns:
      drift  : scalar ||theta||_2  (principal angles)
      theta  : [K] principal angles
      A      : [K,K] "rotation generator" in subspace coordinates (direction)
               A ~ U * diag(theta) * V^T from SVD(Q_prev^T Q_t) = U cosθ V^T
               This is the *directional* object you were missing.
    """
    # Orthonormal columns assumed from eigh; we still align sign outside.
    M = Q_prev.T @ Q_t                          # [K,K]
    s = jnp.linalg.svd(M, compute_uv=False)     # singular values
    s = jnp.clip(s, 0.0, 1.0)
    theta = jnp.arccos(jnp.maximum(s, eps))     # [K]
    drift = jnp.sqrt(jnp.sum(theta * theta))

    # Direction: U, V^T from full SVD of M
    U, _, Vt = jnp.linalg.svd(M, full_matrices=False)
    A = (U * theta) @ Vt                        # U diag(theta) V^T, [K,K]
    return drift, theta, A


@jax.jit
def drift_direction_change(A_prev: jnp.ndarray, A_t: jnp.ndarray) -> jnp.ndarray:
    """
    Measures how the *direction* of rotation is changing (Frobenius norm).
    This is distinct from "how much it moved".
    """
    dA = A_t - A_prev
    return jnp.sqrt(jnp.sum(dA * dA))


@jax.jit
def spectral_gap_at_k(lam_top: jnp.ndarray, k_eff: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """
    lam_top: [K_max+1] descending
    k_eff in [1,K_max]
    gap = |λ_k - λ_{k+1}| (1-indexed), robust to boundary
    """
    k0 = jnp.clip(k_eff - 1, 0, lam_top.shape[0] - 2)  # index for λ_k
    lam_k = lam_top[k0]
    lam_k1 = lam_top[k0 + 1]
    return jnp.maximum(jnp.abs(lam_k - lam_k1), eps)


# ----------------------- C4-ready penalty / stability -----------------------

@jax.jit
def rotational_energy(
    drift_norm: jnp.ndarray,
    curvature_norm: jnp.ndarray,
    dir_change: jnp.ndarray,
    w_drift: float,
    w_curv: float,
    w_dir: float,
) -> jnp.ndarray:
    """
    Energy term C4 can penalize directly:
      E_rot = w_d * drift_norm^2 + w_k * curvature_norm^2 + w_dir * dir_change^2
    """
    return (w_drift * drift_norm * drift_norm
            + w_curv * curvature_norm * curvature_norm
            + w_dir * dir_change * dir_change)


@jax.jit
def ewma(prev: jnp.ndarray, x: jnp.ndarray, alpha: float) -> jnp.ndarray:
    return (1.0 - alpha) * prev + alpha * x


# ---------------------------------- update ---------------------------------

@jax.jit
def update_manifold(
    state: ManifoldState,
    F_t: jnp.ndarray,                 # [N,P] C2 factors at time t
    cfg: ManifoldConfig,
) -> Tuple[ManifoldState, Dict[str, jnp.ndarray]]:
    """
    One step update. Fixed-shape outputs, C3-state-vector friendly.
    """
    eps = cfg.eps
    N = cfg.N
    K = cfg.K_max

    # 1) Build Sigma from C2 factors (optionally EWMA-smooth Sigma)
    Sigma_now = sigma_from_factors(F_t, eps=eps)              # [N,N]
    Sigma_ema = ewma(state.Sigma_ema, Sigma_now, cfg.cov_ewma_alpha)

    # 2) Eigen-decompose
    lam_full, Q_full = top_eigh_desc(Sigma_ema)               # lam: [N], Q: [N,N]

    # 3) Choose K_eff automatically via participation ratio (dynamic scalar)
    k_eff = k_from_participation_ratio(lam_full, K_max=K)     # int32

    # 4) Take top-(K_max+1) eigenvalues and top-K eigenvectors (static)
    lam_top = lam_full[: (K + 1)]                             # [K+1]
    Q_top = Q_full[:, :K]                                     # [N,K]

    # 5) Sign-align eigenvectors to avoid fake rotations
    Q_top_aligned = align_eigenvector_signs(state.Q_prev, Q_top)

    # 6) Subspace drift magnitude + direction (direction = A in subspace coords)
    drift, theta, A_t = subspace_drift_and_direction(state.Q_prev, Q_top_aligned, eps=eps)

    # 7) Curvature proxy (second difference on drift, plus direction change)
    #    We store only the scalar drift/curv EMAs in state; curvature here is instantaneous.
    curv = drift - state.ema_drift
    dir_change = drift_direction_change(state.A_prev, A_t)

    # 8) Spectral-gap normalization (noise-robust)
    gap = spectral_gap_at_k(lam_top, k_eff, eps=eps)
    drift_norm = drift / gap
    curv_norm = curv / gap

    # 9) Smooth the scalars (so C4 isn't reacting to micro-noise)
    ema_drift = ewma(state.ema_drift, drift, cfg.ewma_alpha)
    ema_curv = ewma(state.ema_curv, curv, cfg.ewma_alpha)
    ema_drift_norm = ewma(state.ema_drift_norm, drift_norm, cfg.ewma_alpha)

    # 10) C4 penalty term
    E_rot = rotational_energy(
        drift_norm=drift_norm,
        curvature_norm=curv_norm,
        dir_change=dir_change,
        w_drift=cfg.w_drift,
        w_curv=cfg.w_curv,
        w_dir=cfg.w_rot_dir,
    )

    # 11) Update state
    new_state = ManifoldState(
        Q_prev=Q_top_aligned,
        lam_prev=lam_top,
        A_prev=A_t,
        ema_drift=ema_drift,
        ema_curv=ema_curv,
        ema_drift_norm=ema_drift_norm,
        Sigma_ema=Sigma_ema,
        t=state.t + jnp.int32(1),
    )

    # Fixed-shape feature packet for C3 (state vector inputs)
    feats = {
        # Core state components
        "k_eff": k_eff.astype(jnp.int32),              # scalar
        "lam_top": lam_top.astype(F64),                # [K+1]
        "theta": theta.astype(F64),                    # [K]
        "A_rot": A_t.astype(F64),                      # [K,K] direction object (compact)

        # Stability-aware metrics
        "gap": gap.astype(F64),                        # scalar
        "drift": drift.astype(F64),                    # scalar
        "drift_norm": drift_norm.astype(F64),          # scalar
        "curvature": curv.astype(F64),                 # scalar
        "curvature_norm": curv_norm.astype(F64),       # scalar
        "dir_change": dir_change.astype(F64),          # scalar

        # Smoothed (allocator-friendly)
        "ema_drift": ema_drift.astype(F64),            # scalar
        "ema_curv": ema_curv.astype(F64),              # scalar
        "ema_drift_norm": ema_drift_norm.astype(F64),  # scalar

        # C4 penalty term
        "E_rot": E_rot.astype(F64),                    # scalar
    }
    return new_state, feats


# ----------------------- optional: Jacobian coupling hook --------------------

@jax.jit
def jacobian_stability_penalty(
    J_state: jnp.ndarray,          # [d,d] Jacobian of learned dynamics f(x)
    drift_norm: jnp.ndarray,       # scalar
    curvature_norm: jnp.ndarray,   # scalar
    beta: float = 0.25,
) -> jnp.ndarray:
    """
    Minimal institutional hook:
      - compute spectral radius proxy of J_state (via eigenvalues, d small)
      - inflate by drift/curvature (risk-manifold instability)
    This returns a scalar penalty C4 can add to its Hamiltonian.
    """
    # For small d (your C3 state dim), eig is fine.
    ev = jnp.linalg.eigvals(J_state)
    rho = jnp.max(jnp.abs(ev))
    return rho + beta * (drift_norm + jnp.abs(curvature_norm))


# ---------------------------------- usage -----------------------------------

def build_c3_state_vector(feats: Dict[str, jnp.ndarray], cfg: ManifoldConfig) -> jnp.ndarray:
    """
    Example: compress features into a C3 state vector x_t with fixed length.
    You can expand this, but keep it deterministic + static.
    """
    K = cfg.K_max
    # Take top-K eigenvalues (drop the +1 used only for gap)
    lam_k = feats["lam_top"][:K]
    # Flatten the KxK rotation generator (direction) — compact but informative
    A_flat = feats["A_rot"].reshape((K * K,))

    # Scalars
    scalars = jnp.array([
        feats["drift_norm"],
        feats["curvature_norm"],
        feats["dir_change"],
        feats["ema_drift_norm"],
        feats["E_rot"],
    ], dtype=F64)

    # Final x_t (static)
    x_t = jnp.concatenate([lam_k, feats["theta"], scalars, A_flat], axis=0)
    return x_t


# ---------------------------------- demo ------------------------------------

if __name__ == "__main__":
    # Example: N=200 surfaces, P=5 C2 factors (skew/convexity/fragility/entropy/alignment)
    cfg = ManifoldConfig(N=200, P=5, K_max=16)

    st = init_manifold_state(cfg)

    key = jax.random.PRNGKey(0)

    # Simulate 10 steps of C2 factor snapshots
    for t in range(10):
        key, k1 = jax.random.split(key)
        # pretend factors shift slowly (add a drift component)
        drift = (t / 10.0) * 0.05
        F_t = jax.random.normal(k1, (cfg.N, cfg.P), dtype=F64) * 0.5 + drift

        st, feats = update_manifold(st, F_t, cfg)
        x_t = build_c3_state_vector(feats, cfg)

        print(
            f"t={int(st.t)}  K_eff={int(feats['k_eff'])}  "
            f"drift_norm={float(feats['drift_norm']):.6f}  "
            f"E_rot={float(feats['E_rot']):.6f}  "
            f"|x_t|={x_t.shape[0]}"
        )

