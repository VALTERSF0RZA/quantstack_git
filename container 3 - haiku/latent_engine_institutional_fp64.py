# c3/latent_engine_institutional_fp64.py
# =============================================================================
# HARD INSTITUTIONAL C3 (FP64):
#   obs_t (293-d)  --->  Encoder  --->  latent x_t (32-d, EMA-filtered)
#                              |
#                              v
#                    Latent Dynamics f(x_t)  --->  x_{t+1}, J_latent (32x32)
#                              |
#                              v
#             Raw instability (power-iter spectral radius + drift/curvature terms)
#                              |
#                              v
#     Calibrated instability scalar in [0,1] with online EWMA z-score + EMA smoothing
#
# "Run all day" features:
#   - FP64 params + activations
#   - fixed shapes, jit-friendly, no data-dependent control flow
#   - power iteration uses warm-start vector in state
#   - EMA filter on latent and on instability scalar (noise robustness)
#   - online calibration (EWMA mean/var) so thresholds are stable across regimes
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import jax
import jax.numpy as jnp
import haiku as hk

jax.config.update("jax_enable_x64", True)
F64 = jnp.float64


# ----------------------------- FP64 primitives ------------------------------

@jax.jit
def gelu64(x: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * x * (1.0 + jax.lax.erf(x / jnp.sqrt(F64(2.0))))


class Dense64(hk.Module):
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
    def __init__(self, eps: float = 1e-5, name: Optional[str] = None):
        super().__init__(name=name)
        self.eps = float(eps)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(F64)
        d = int(x.shape[-1])
        mu = jnp.mean(x, axis=-1, keepdims=True)
        v = jnp.mean((x - mu) * (x - mu), axis=-1, keepdims=True)
        xhat = (x - mu) / jnp.sqrt(v + F64(self.eps))
        g = hk.get_parameter("gamma", shape=(d,), dtype=F64, init=hk.initializers.Constant(1.0))
        b = hk.get_parameter("beta", shape=(d,), dtype=F64, init=hk.initializers.Constant(0.0))
        return xhat * g + b


# ----------------------------- Config + State --------------------------------

@dataclass(frozen=True)
class LatentEngineCfg:
    obs_dim: int = 293
    latent_dim: int = 32

    enc_hidden: int = 256
    enc_depth: int = 3

    dyn_hidden: int = 256
    dyn_depth: int = 3
    dt: float = 1.0
    clip_dx: float = 5.0

    # Filtering (noise robustness)
    ema_x: float = 0.98           # latent smoothing; higher = smoother
    ema_inst: float = 0.995       # scalar smoothing; higher = smoother

    # Online calibration of raw instability -> z-score
    calib_ema: float = 0.995      # EWMA mean/var speed

    # Instability construction
    power_iter_steps: int = 12
    w_drift: float = 0.35
    w_curv: float = 0.20
    beta_rho: float = 1.0         # how hard you care about rho(J)
    eps: float = 1e-12


@jax.tree_util.register_pytree_node_class
@dataclass
class FilterState:
    # EMA latent state (what you actually run on)
    x_ema: jnp.ndarray          # [D]
    # Power-iteration warm start vector (stabilizes rho estimate)
    v_pi: jnp.ndarray           # [D]
    # Online calibration stats for raw instability
    mu: jnp.ndarray             # scalar
    var: jnp.ndarray            # scalar
    # Smoothed final instability scalar
    inst_ema: jnp.ndarray       # scalar

    def tree_flatten(self):
        return (self.x_ema, self.v_pi, self.mu, self.var, self.inst_ema), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


# ----------------------------- Model blocks ----------------------------------

class Encoder(hk.Module):
    def __init__(self, cfg: LatentEngineCfg, name: Optional[str] = None):
        super().__init__(name=name)
        self.cfg = cfg

    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        cfg = self.cfg
        h = obs.astype(F64)
        for i in range(cfg.enc_depth):
            h = Dense64(cfg.enc_hidden, name=f"enc_fc{i}")(h)
            h = gelu64(h)
            h = LayerNorm64(name=f"enc_ln{i}")(h)
        x = Dense64(cfg.latent_dim, name="enc_out")(h)
        # light norm to keep scale stable (important for all-day streaming)
        x = LayerNorm64(name="enc_out_ln")(x)
        return x.astype(F64)


class LatentDynamics(hk.Module):
    """
    Residual dynamics in latent:
      x_{t+1} = x_t + dt * g(x_t)
    """
    def __init__(self, cfg: LatentEngineCfg, name: Optional[str] = None):
        super().__init__(name=name)
        self.cfg = cfg

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        cfg = self.cfg
        h = x.astype(F64)
        for i in range(cfg.dyn_depth):
            h = Dense64(cfg.dyn_hidden, name=f"dyn_fc{i}")(h)
            h = gelu64(h)
            h = LayerNorm64(name=f"dyn_ln{i}")(h)
        dx = Dense64(cfg.latent_dim, name="dyn_out")(h)

        if cfg.clip_dx and cfg.clip_dx > 0.0:
            nrm = jnp.sqrt(jnp.sum(dx * dx) + F64(cfg.eps))
            scale = jnp.minimum(F64(1.0), F64(cfg.clip_dx) / nrm)
            dx = dx * scale

        return (x + F64(cfg.dt) * dx).astype(F64)


class InstabilityHead(hk.Module):
    """
    Optional learnable monotone calibration on z-score:
      inst = sigmoid(a * z + b)
    This is *not* the online calibration (mu/var); it's a stable mapping layer.
    """
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        a = hk.get_parameter("a", shape=(), dtype=F64, init=hk.initializers.Constant(1.25))
        b = hk.get_parameter("b", shape=(), dtype=F64, init=hk.initializers.Constant(0.0))
        y = a * z + b
        return jax.nn.sigmoid(y.astype(F64)).astype(F64)


# ----------------------------- Core math -------------------------------------

@jax.jit
def ema_update(prev: jnp.ndarray, new: jnp.ndarray, alpha: float) -> jnp.ndarray:
    a = F64(alpha)
    return (a * prev + (F64(1.0) - a) * new).astype(F64)


@jax.jit
def power_iter_spectral_radius(
    J: jnp.ndarray, v0: jnp.ndarray, steps: int, eps: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns (rho_hat, v_last) where rho_hat ~ spectral radius of J.
    Uses ||J v|| / ||v|| with fixed-step power iteration; warm-start makes it stable.
    """
    v = v0.astype(F64)
    v = v / (jnp.sqrt(jnp.sum(v * v)) + F64(eps))

    def body(_, v):
        w = J @ v
        n = jnp.sqrt(jnp.sum(w * w)) + F64(eps)
        return w / n

    v = jax.lax.fori_loop(0, int(steps), body, v)
    Jv = J @ v
    rho = jnp.sqrt(jnp.sum(Jv * Jv)) / (jnp.sqrt(jnp.sum(v * v)) + F64(eps))
    return rho.astype(F64), v.astype(F64)


@jax.jit
def softplus64(x: jnp.ndarray) -> jnp.ndarray:
    # stable softplus
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0)


@jax.jit
def online_calibrate_z(
    raw: jnp.ndarray, mu: jnp.ndarray, var: jnp.ndarray, alpha: float, eps: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    EWMA mean/var for raw instability -> z-score.
    """
    mu_new = ema_update(mu, raw, alpha)
    # EWMA variance on centered residual
    resid = (raw - mu_new)
    var_new = ema_update(var, resid * resid, alpha)
    z = (raw - mu_new) / jnp.sqrt(var_new + F64(eps))
    return z.astype(F64), mu_new.astype(F64), var_new.astype(F64)


# ----------------------------- Build API -------------------------------------

def build_latent_engine(cfg: LatentEngineCfg):
    """
    init(rng, obs0) -> params, filter_state
    step(params, filter_state, obs_t, drift_norm, curvature_norm)
        -> (x_latent_next, J_latent, instability_scalar, filter_state_next, aux)
    """

    def forward(obs: jnp.ndarray, x_prev: jnp.ndarray, drift_norm: jnp.ndarray, curvature_norm: jnp.ndarray):
        enc = Encoder(cfg)
        dyn = LatentDynamics(cfg)
        head = InstabilityHead()

        x_enc = enc(obs)                 # [D]
        x_next = dyn(x_prev)             # [D]

        # Jacobian in latent space only (32x32): this is the point
        def f_lat(x):
            return dyn(x)

        J = jax.jacfwd(f_lat)(x_prev).astype(F64)  # [D,D] (D=32)

        # raw instability (rho around 1 is the boundary)
        # NOTE: rho computed outside Haiku here would be fine too; we keep the head learnable only.
        rho = jnp.array(0.0, dtype=F64)  # placeholder; overwritten in step() where v_pi exists

        # head expects z-score; we pass placeholder (computed in step)
        z = jnp.array(0.0, dtype=F64)
        inst = head(z)

        return x_enc, x_next, J, inst

    tx = hk.without_apply_rng(hk.transform(forward))

    @jax.jit
    def init(rng: jax.Array, obs0: jnp.ndarray) -> Tuple[hk.Params, FilterState]:
        obs0 = obs0.astype(F64)
        D = cfg.latent_dim
        x0 = jnp.zeros((D,), dtype=F64)
        drift0 = jnp.array(0.0, dtype=F64)
        curv0 = jnp.array(0.0, dtype=F64)

        params = tx.init(rng, obs0, x0, drift0, curv0)

        # filter state starts neutral; var non-zero to avoid divide-by-zero
        fs = FilterState(
            x_ema=x0,
            v_pi=jnp.ones((D,), dtype=F64) / jnp.sqrt(F64(D)),
            mu=jnp.array(0.0, dtype=F64),
            var=jnp.array(1.0, dtype=F64),
            inst_ema=jnp.array(0.0, dtype=F64),
        )
        return params, fs

    @jax.jit
    def step(
        params: hk.Params,
        fs: FilterState,
        obs_t: jnp.ndarray,
        drift_norm: jnp.ndarray,
        curvature_norm: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, FilterState, Dict[str, jnp.ndarray]]:
        """
        obs_t: [obs_dim] FP64
        drift_norm, curvature_norm: scalars (from your risk-manifold / eigen drift module)
        """
        obs_t = obs_t.astype(F64)
        drift_norm = drift_norm.astype(F64)
        curvature_norm = curvature_norm.astype(F64)

        # 1) Encode observation -> latent (then filter it)
        #    We run forward once to get x_enc and J skeleton; dynamics uses filtered latent
        x_enc, _, J_lat, _ = tx.apply(params, obs_t, fs.x_ema, drift_norm, curvature_norm)
        x_filt = ema_update(fs.x_ema, x_enc, cfg.ema_x)  # stable latent for "all day" running

        # 2) Predict next latent using dynamics on filtered state
        #    (apply again so x_prev is the filtered latent)
        _, x_next, J_lat, _ = tx.apply(params, obs_t, x_filt, drift_norm, curvature_norm)

        # 3) Instability: rho(J) via power iteration w/ warm start
        rho, v_pi_next = power_iter_spectral_radius(J_lat, fs.v_pi, cfg.power_iter_steps, cfg.eps)

        # 4) Compose raw instability (local expansion + manifold rotation terms)
        #    softplus(rho - 1) is the "crossing contraction boundary" indicator.
        raw = (
            F64(cfg.beta_rho) * softplus64(rho - F64(1.0))
            + F64(cfg.w_drift) * drift_norm
            + F64(cfg.w_curv) * jnp.abs(curvature_norm)
        ).astype(F64)

        # 5) Online calibration (EWMA mean/var -> z-score), then learnable sigmoid map
        z, mu_new, var_new = online_calibrate_z(raw, fs.mu, fs.var, cfg.calib_ema, cfg.eps)

        # apply just the instability head (we reuse tx.apply but ignore other outputs)
        def head_only(z_in: jnp.ndarray) -> jnp.ndarray:
            h = InstabilityHead()
            return h(z_in)

        head_tx = hk.without_apply_rng(hk.transform(head_only))
        # NOTE: head parameters are inside `params` already; reuse by applying with same params subtree name.
        # Simplest: call full forward and read inst; we pass placeholders for unused.
        # (This keeps names consistent and avoids maintaining a second params object.)
        _, _, _, inst = tx.apply(params, obs_t, x_filt, drift_norm, curvature_norm)
        # overwrite with proper z by directly evaluating head module inside a named scope
        # (robust option: put head inside forward, but we already did; so we compute inst deterministically below)
        # We'll compute the calibrated inst explicitly to avoid any ambiguity in param scoping:
        #   inst = sigmoid(a*z + b) with a,b from params. We retrieve them safely.
        # However, Haiku param dict structure depends on module naming; easiest: just re-run a small head transform
        # with the SAME naming by using hk.experimental.name_scope in forward. To keep this file drop-in,
        # we provide a param-free calibrated inst and keep the head learnable optional.
        inst_cal = jax.nn.sigmoid(z).astype(F64)

        # 6) Smooth the scalar (this is what you trade on / gate on)
        inst_ema = ema_update(fs.inst_ema, inst_cal, cfg.ema_inst)

        fs_next = FilterState(
            x_ema=x_filt,
            v_pi=v_pi_next,
            mu=mu_new,
            var=var_new,
            inst_ema=inst_ema,
        )

        aux = {
            "x_enc": x_enc,
            "x_filt": x_filt,
            "rho": rho,
            "raw_instability": raw,
            "z_instability": z,
            "inst_cal": inst_cal,
        }

        return x_next, J_lat, inst_ema, fs_next, aux

    return init, step


# ----------------------------- Example run -----------------------------------

def _demo():
    cfg = LatentEngineCfg(obs_dim=293, latent_dim=32)
    init, step = build_latent_engine(cfg)

    rng = jax.random.PRNGKey(0)
    obs0 = jnp.zeros((cfg.obs_dim,), dtype=F64)
    params, fs = init(rng, obs0)

    # pretend these come from your eigen-subspace module
    drift = jnp.array(0.10, dtype=F64)
    curv = jnp.array(-0.02, dtype=F64)

    x1, J0, inst, fs, aux = step(params, fs, obs0, drift, curv)
    print("x1:", x1.shape, "J0:", J0.shape, "inst:", float(inst), "rho:", float(aux["rho"]))


if __name__ == "__main__":
    _demo()

