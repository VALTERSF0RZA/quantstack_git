# minimal integration snippet (inside your container 3 runtime loop)

import jax.numpy as jnp
from surface_state import SurfaceStateStore
from surface_dynamics import SurfaceDynamicsEngine, DynamicsConfig

state_store = SurfaceStateStore()
dyn_engine = SurfaceDynamicsEngine(
    symbol="SPX",
    state_store=state_store,
    cfg=DynamicsConfig(
        dt_floor_seconds=1e-3,
        m_bandwidth=0.08,
        T_bandwidth=0.12,
        velocity_clip=25.0,
    ),
)

# Example per tick:
# ts_unix_ns, F, sabr_fits, heston_fits come from your calibration stage
# m_grid/T_grid are your canonical feature grids
m_grid = jnp.linspace(-0.35, 0.35, 61, dtype=jnp.float64)
T_grid = jnp.array([1/252, 2/252, 5/252, 10/252, 21/252, 42/252, 63/252, 126/252, 252/252], dtype=jnp.float64)

features = dyn_engine.compute_features(
    ts_unix_ns=ts_unix_ns,
    F=F,
    m_grid=m_grid,
    T_grid=T_grid,
    sabr_fits=sabr_fits,
    heston_fits=heston_fits,
    primary_model="sabr",   # or "heston"
)

# feed to Haiku/LSTM/HMM/regime model
X = features["feature_tensor"]   # [nT, nM, 15]

