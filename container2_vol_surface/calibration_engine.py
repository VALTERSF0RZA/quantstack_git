# calibration_engine.py
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional

import jax

from sabr_calibration import calibrate_sabr_surface_from_points
from heston_calibration import (
    HestonCalibConfig,
    calibrate_heston_surface_from_points,
)

def _run_on_device(device, fn, *args, **kwargs):
    with jax.default_device(device):
        return fn(*args, **kwargs)

def calibrate_sabr_heston_parallel_from_payload(
    payload: Dict[str, Any],
    r: float = 0.0,
    q: float = 0.0,
    sabr_beta: float = 0.7,
    min_points_per_expiry: int = 8,
    heston_cfg: Optional[HestonCalibConfig] = None,
) -> Dict[str, Any]:
    """
    Multi-GPU: true model-parallel dispatch (SABR on GPU0, Heston on GPU1)
    Single-GPU: sequential fallback (same API).
    """
    if heston_cfg is None:
        heston_cfg = HestonCalibConfig()

    points = payload["points"]
    F = float(payload["S_mid"])
    gpus = jax.devices("gpu")

    if len(gpus) >= 2:
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_sabr = ex.submit(
                _run_on_device,
                gpus[0],
                calibrate_sabr_surface_from_points,
                points,
                F,
                sabr_beta,
                min_points_per_expiry,
            )
            f_heston = ex.submit(
                _run_on_device,
                gpus[1],
                calibrate_heston_surface_from_points,
                points,
                F,
                float(r),
                float(q),
                min_points_per_expiry,
                heston_cfg,
            )
            sabr = f_sabr.result()
            heston = f_heston.result()
    else:
        # single GPU / CPU fallback
        sabr = calibrate_sabr_surface_from_points(
            points=points,
            F=F,
            beta=sabr_beta,
            min_points_per_expiry=min_points_per_expiry,
        )
        heston = calibrate_heston_surface_from_points(
            points=points,
            F=F,
            r=float(r),
            q=float(q),
            min_points_per_expiry=min_points_per_expiry,
            cfg=heston_cfg,
        )

    return {
        "symbol": payload.get("symbol"),
        "ts_unix_ns": payload.get("ts_unix_ns"),
        "sabr": sabr,
        "heston": heston,
    }


