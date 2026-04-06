# c3/haiku_dynamics_fp64.py
# =============================================================================
# C3 DYNAMICS (Haiku, JAX, FP64)
#
# Contract:
#   input : x_t  [D]   (C3 state vector; engineered or learned — ML not required)
#   output: x_{t+1} [D]
#           J_t = ∂f/∂x |_{x_t}  [D,D]   (Jacobian of the one-step dynamics)
#
# This is the full bridge you asked for:
#   (x_next, J) feeds directly into jacobian_stability_penalty(J, drift_norm, curvature_norm)
#   so C4 can penalize instability (spectral radius + risk-manifold rotation terms).
#
# FP64 notes:
#   - jax_enable_x64=True
#   - custom Dense64/LayerNorm64 force float64 params
#   - deterministic (no dropout) by default
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Tuple, Dict, Any, Optional

import jax
import jax.numpy as jnp
import haiku as hk

jax.config.update("jax_enable_x64", True)
F64 = jnp.float64


# ----------------------------- small math utils -----------------------------

@jax.jit
def gelu64(x: jnp.ndarray) -> jnp.ndarray:
    # Stable GELU, FP64
    return 0.5 * x * (1.0 + jax.lax.erf(x / jnp.sqrt(F64(2.0))))


class Dense64(hk.Module):
    """FP64 dense layer with FP64 parameters."""
    def __init__(self, out_dim: int, with_bias: bool = True, name: Optional[str] = None):
        super().__init__(name=name)
        self.out_dim = int(out_dim)
        self.with_bias = bool(with_bias)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(F64)
        in_dim = int(x.shape[-1])
        w = hk.get_parameter(
            "w",
            shape=(in_dim, self.out_dim),
            dtype=F64,
            init=hk.initializers.Orthogonal(scale=1.0),
        )
        y = x @ w
        if self.with_bias:
            b = hk.get_parameter(
                "b",
                shape=(self.out_dim,),
                dtype=F64,
                init=hk.initializers.Constant(0.0),
            )
            y = y + b
        return y


class LayerNorm64(hk.Module):
    """FP64 layer norm with FP64 parameters."""
    def __init__(self, eps: float = 1e-5, name: Optional[str] = None):
        super().__init__(name=name)
        self.eps = float(eps)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(F64)
        d = int(x.shape[-1])

        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean((x - mean) * (x - mean), axis=-1, keepdims=True)
        xhat = (x - mean) / jnp.sqrt(var + F64(self.eps))

        gamma = hk.get_parameter("gamma", shape=(d,), dtype=F64, init=hk.initializers.Constant(1.0))
        beta = hk.get_parameter("beta", shape=(d,), dtype=F64, init=hk.initializers.Constant(0.0))
        return xhat * gamma + beta


# ------------------------------ C3 dynamics core ----------------------------

@dataclass(frozen=True)
class C3DynConfig:
    state_dim: int                  # D
    hidden_dim: int = 256
    depth: int = 3
    dt: float = 1.0                 # discrete step size
    ln_eps: float = 1e-5
    clip_dx: float = 0.0            # 0 disables; otherwise clip delta norm


class C3Dynamics(hk.Module):
    """
    Deterministic one-step dynamics:
      x_{t+1} = x_t + dt * g(x_t)
    Residual form is the institutional default for stability.
    """
    def __init__(self, cfg: C3DynConfig, name: Optional[str] = None):
        super().__init__(name=name)
        self.cfg = cfg

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        cfg = self.cfg
        x = x.astype(F64)

        h = x
        for i in range(cfg.depth):
            h = Dense64(cfg.hidden_dim, name=f"fc{i}")(h)
            h = gelu64(h)
            h = LayerNorm64(eps=cfg.ln_eps, name=f"ln{i}")(h)

        dx = Dense64(cfg.state_dim, name="out")(h)

        if cfg.clip_dx and cfg.clip_dx > 0.0:
            nrm = jnp.sqrt(jnp.sum(dx * dx) + F64(1e-12))
            scale = jnp.minimum(F64(1.0), F64(cfg.clip_dx) / nrm)
            dx = dx * scale

        x_next = x + F64(cfg.dt) * dx
        return x_next


# -------------------------- transform + jacobian API ------------------------

def build_c3_dynamics(cfg: C3DynConfig):
    """
    Returns:
      init(rng, x0) -> params
      step(params, x_t) -> x_next
      step_with_jacobian(params, x_t) -> (x_next, J)
      step_with_jvp(params, x_t, v) -> (x_next, Jv)   (faster alternative)
    """
    def forward(x: jnp.ndarray) -> jnp.ndarray:
        model = C3Dynamics(cfg)
        return model(x)

    tx = hk.without_apply_rng(hk.transform(forward))

    @jax.jit
    def init(rng: jax.Array, x0: jnp.ndarray) -> hk.Params:
        x0 = x0.astype(F64)
        return tx.init(rng, x0)

    @jax.jit
    def step(params: hk.Params, x_t: jnp.ndarray) -> jnp.ndarray:
        x_t = x_t.astype(F64)
        return tx.apply(params, x_t)

    # Full Jacobian: J = ∂f/∂x at x_t
    @jax.jit
    def step_with_jacobian(params: hk.Params, x_t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x_t = x_t.astype(F64)

        def f(x):
            return tx.apply(params, x)

        x_next = f(x_t)
        # jacfwd is typically better for medium D; jacrev for large D. Pick one.
        J = jax.jacfwd(f)(x_t).astype(F64)   # [D,D]
        return x_next, J

    # Faster: Jacobian-vector product (useful if C4 only needs stability proxy / gradients)
    @jax.jit
    def step_with_jvp(params: hk.Params, x_t: jnp.ndarray, v: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x_t = x_t.astype(F64)
        v = v.astype(F64)

        def f(x):
            return tx.apply(params, x)

        x_next, Jv = jax.jvp(f, (x_t,), (v,))
        return x_next.astype(F64), Jv.astype(F64)

    return init, step, step_with_jacobian, step_with_jvp


# ----------------------------- C4 stability hook ----------------------------

@jax.jit
def jacobian_stability_penalty(
    J_state: jnp.ndarray,          # [D,D]
    drift_norm: jnp.ndarray,       # scalar (from risk_manifold feats)
    curvature_norm: jnp.ndarray,   # scalar (from risk_manifold feats)
    beta: float = 0.25,
) -> jnp.ndarray:
    """
    Minimal, desk-grade stability penalty:
      penalty = spectral_radius(J) + beta*(drift_norm + |curvature_norm|)
    """
    ev = jnp.linalg.eigvals(J_state)
    rho = jnp.max(jnp.abs(ev))
    return rho + F64(beta) * (drift_norm + jnp.abs(curvature_norm))


# ------------------------------ example wiring ------------------------------

def example_bridge():
    """
    End-to-end illustration:
      - you already built x_t from manifold features (or a smaller latent state)
      - run Haiku dynamics to get x_next and J
      - compute a C4 penalty term using drift_norm/curvature_norm from manifold
    """
    D = 32  # IMPORTANT: if you want full J, keep D small (latent state). 293x293 is possible but heavy.
    cfg = C3DynConfig(state_dim=D, hidden_dim=256, depth=3, dt=1.0, clip_dx=5.0)

    init, step, stepJ, stepJVP = build_c3_dynamics(cfg)

    rng = jax.random.PRNGKey(0)
    x0 = jnp.zeros((D,), dtype=F64)
    params = init(rng, x0)

    # Suppose these come from your risk_manifold.py feats packet:
    drift_norm = jnp.array(0.12, dtype=F64)
    curvature_norm = jnp.array(-0.03, dtype=F64)

    # One step:
    x1, J0 = stepJ(params, x0)

    pen = jacobian_stability_penalty(J0, drift_norm, curvature_norm, beta=0.25)

    return x1, J0, pen


if __name__ == "__main__":
    x1, J0, pen = example_bridge()
    print("x1 shape:", x1.shape)
    print("J0 shape:", J0.shape)
    print("stability penalty:", float(pen))

