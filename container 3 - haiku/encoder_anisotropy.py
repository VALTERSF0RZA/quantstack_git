# c3/encoder_anisotropy.py
from __future__ import annotations
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
F64 = jnp.float64


# ---------------------------------------------------------------------
# Core: anisotropy from encoder Jacobian via Gram eigenvalues (fast)
# ---------------------------------------------------------------------

@jax.jit
def _anisotropy_from_gram_eigs(G: jnp.ndarray, eps: float = 1e-12) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    G = J J^T : [d, d] symmetric PSD
    Returns:
      anis_energy: scalar  Var(log sigma_i)
      diag: dict with useful extras (log_cond, sigma, iso_score)
    """
    # eigenvalues of G are sigma^2
    lam = jnp.linalg.eigvalsh(G)                  # [d] ascending
    lam = jnp.maximum(lam, eps)

    # log sigma = 0.5 log lam
    log_sigma = 0.5 * jnp.log(lam)                # [d]

    anis_energy = jnp.var(log_sigma).astype(F64)  # scale-invariant anisotropy energy

    # practical monitoring extras
    log_cond = (0.5 * (jnp.max(jnp.log(lam)) - jnp.min(jnp.log(lam)))).astype(F64)  # log(kappa)
    sigma = jnp.sqrt(lam).astype(F64)
    iso_score = jnp.exp(-anis_energy).astype(F64)  # 1 ~ isotropic, 0 ~ highly anisotropic

    diag = {
        "sigma": sigma,               # [d]
        "log_cond": log_cond,         # scalar
        "iso_score": iso_score,       # scalar
    }
    return anis_energy, diag


def encoder_jacobian(
    encoder_apply: Callable[[dict, jnp.ndarray], jnp.ndarray],
    params: dict,
    obs: jnp.ndarray,
) -> jnp.ndarray:
    """
    encoder_apply(params, obs) -> latent x  (shape [d])
    Returns J: [d, D]
    """
    obs = obs.astype(F64)

    def f(o):
        return encoder_apply(params, o).astype(F64)

    J = jax.jacfwd(f)(obs)  # [d, D]
    return J.astype(F64)


@jax.jit
def anisotropy_from_jacobian(J: jnp.ndarray, eps: float = 1e-12) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    J: [d, D]
    Returns anis_energy scalar + diagnostics.
    Uses G = J J^T (dxd) then eigvalsh (cheap when d=32..64).
    """
    J = J.astype(F64)
    G = J @ J.T  # [d, d]
    # small ridge for numerical stability
    d = G.shape[0]
    G = G + eps * jnp.eye(d, dtype=F64)
    return _anisotropy_from_gram_eigs(G, eps=eps)


def encoder_anisotropy_scalar(
    encoder_apply: Callable[[dict, jnp.ndarray], jnp.ndarray],
    params: dict,
    obs: jnp.ndarray,
    eps: float = 1e-12,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    End-to-end:
      obs -> J_E -> anis_energy (single scalar)
    """
    J = encoder_jacobian(encoder_apply, params, obs)
    return anisotropy_from_jacobian(J, eps=eps)


# ---------------------------------------------------------------------
# Batch version (vmap)
# ---------------------------------------------------------------------

def batch_encoder_anisotropy_scalar(
    encoder_apply: Callable[[dict, jnp.ndarray], jnp.ndarray],
    params: dict,
    obs_batch: jnp.ndarray,   # [B, D]
    eps: float = 1e-12,
):
    """
    Returns:
      anis_energy: [B]
      log_cond:    [B]
      iso_score:   [B]
    """
    def one(obs):
        a, diag = encoder_anisotropy_scalar(encoder_apply, params, obs, eps=eps)
        return a, diag["log_cond"], diag["iso_score"]

    a, logc, iso = jax.vmap(one)(obs_batch.astype(F64))
    return a.astype(F64), logc.astype(F64), iso.astype(F64)

