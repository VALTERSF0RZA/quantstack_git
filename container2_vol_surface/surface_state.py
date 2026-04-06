# surface_state.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import jax.numpy as jnp


@dataclass
class SurfaceSnapshot:
    """
    Stored state at previous tick for transport-corrected time derivatives.
    Shapes are expected [nT, nM].
    """
    ts_unix_ns: int
    m: jnp.ndarray
    T: jnp.ndarray
    sigma: jnp.ndarray
    d_sigma_dm: jnp.ndarray
    d2_sigma_dm2: jnp.ndarray
    d_sigma_dT: jnp.ndarray


class SurfaceStateStore:
    """
    In-memory state store keyed by symbol (or any routing key).
    """
    def __init__(self) -> None:
        self._state: Dict[str, SurfaceSnapshot] = {}

    def get(self, key: str) -> Optional[SurfaceSnapshot]:
        return self._state.get(key)

    def update(self, key: str, snapshot: SurfaceSnapshot) -> None:
        self._state[key] = snapshot

    def clear(self, key: Optional[str] = None) -> None:
        if key is None:
            self._state.clear()
        else:
            self._state.pop(key, None)


def dt_seconds(prev_ts_unix_ns: int, now_ts_unix_ns: int, floor: float = 1e-6) -> float:
    """
    Nanoseconds -> seconds with floor guard.
    """
    dt = (float(now_ts_unix_ns) - float(prev_ts_unix_ns)) * 1e-9
    return max(dt, floor)

