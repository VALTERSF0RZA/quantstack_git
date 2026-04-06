# run_c2_pipeline.py
# =============================================================================
# C2 runner: JIT path + XLA AOT compile path (FP64)
# - Builds synthetic [N,K,T] surface inputs
# - Executes canonical run_c2_state_packet()
# - Executes lowered/compiled aot_compile_c2_state_core()
# =============================================================================

from __future__ import annotations

import time
from typing import Dict, Any

import jax
import jax.numpy as jnp

# Import from your c2 package
from container2_vol_surface import C2Config, F64
from container2_vol_surface import (
    run_c2_state_packet,
    aot_compile_c2_state_core,
)

# Enforce FP64 globally
jax.config.update("jax_enable_x64", True)


def _make_synthetic_inputs(cfg: C2Config):
    """
    Deterministic synthetic test tensors with static shapes:
      sigma_raw: [N,K,T]
      dsdt_raw : [N,K,T]
      log_m    : [K]
      tau      : [T]
    """
    N, K, T = cfg.n_assets, cfg.n_strikes, cfg.n_tenors

    # Log-moneyness and tenor grids
    log_m = jnp.linspace(-0.5, 0.5, K, dtype=F64)
    tau = jnp.linspace(1.0 / 365.0, 1.5, T, dtype=F64)

    # Synthetic volatility surface (shape-stable, smooth, cross-asset shifted)
    # base: [1,K,T]
    base = (
        0.18
        + 0.04 * jnp.exp(-3.0 * tau)[None, None, :]
        + 0.06 * (log_m[None, :, None] ** 2)
    )

    # asset_shift: [N,1,1]
    asset_shift = jnp.linspace(-0.03, 0.03, N, dtype=F64)[:, None, None]

    # sigma_raw: [N,K,T]
    sigma_raw = jnp.maximum(base + asset_shift, F64(0.04))

    # If you do not yet have ds/dt from upstream, use zeros
    dsdt_raw = jnp.zeros_like(sigma_raw, dtype=F64)

    return sigma_raw, log_m, tau, dsdt_raw


def _print_packet_summary(packet: Dict[str, Any], title: str):
    print(f"\n=== {title} ===")
    print("features_raw:", packet["features_raw"].shape, packet["features_raw"].dtype)
    print("features_z:", packet["features_z"].shape, packet["features_z"].dtype)
    print("pca_scores:", packet["pca_scores"].shape, packet["pca_scores"].dtype)
    print("asset_corr:", packet["asset_corr"].shape, packet["asset_corr"].dtype)
    print("coupling_NN3:", packet["coupling_NN3"].shape, packet["coupling_NN3"].dtype)
    print("regime_id:", packet["regime_id"].shape, packet["regime_id"].dtype)
    print("cluster_id:", packet["cluster_id"].shape, packet["cluster_id"].dtype)
    print("arb_metrics [cal_pre, cal_post, bf_pre, bf_post]:", packet["arb_metrics"])


def main():
    # -------------------------------------------------------------------------
    # Config (static-shape contract for XLA)
    # -------------------------------------------------------------------------
    cfg = C2Config(
        n_assets=200,
        n_strikes=64,
        n_tenors=32,
        pca_components=5,
    )

    # -------------------------------------------------------------------------
    # Build deterministic synthetic test input
    # -------------------------------------------------------------------------
    sigma_raw, log_m, tau, dsdt_raw = _make_synthetic_inputs(cfg)

    # -------------------------------------------------------------------------
    # 1) JIT runtime path (through your canonical runtime wrapper)
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    packet = run_c2_state_packet(
        sigma_raw=sigma_raw,
        log_m=log_m,
        tau=tau,
        dsdt_raw=dsdt_raw,
        state_id_ns=1730000000000000000,
        cfg=cfg,
    )
    # Explicit sync for accurate timing
    _ = packet["features_z"].block_until_ready()
    t1 = time.perf_counter()

    _print_packet_summary(packet, "JIT path")
    print(f"JIT elapsed: {(t1 - t0) * 1000.0:.2f} ms")

    # -------------------------------------------------------------------------
    # 2) XLA AOT path (lower + compile + execute compiled callable)
    # -------------------------------------------------------------------------
    c0 = time.perf_counter()
    compiled = aot_compile_c2_state_core(cfg)  # one-time compile cost
    c1 = time.perf_counter()

    e0 = time.perf_counter()
    out2 = compiled(sigma_raw, log_m, tau, dsdt_raw)
    _ = out2["features_z"].block_until_ready()
    e1 = time.perf_counter()

    _print_packet_summary(out2, "AOT compiled path")
    print(f"AOT compile time: {(c1 - c0) * 1000.0:.2f} ms")
    print(f"AOT execute time: {(e1 - e0) * 1000.0:.2f} ms")

    # Optional quick consistency check (shape + finite values)
    assert packet["features_z"].shape == out2["features_z"].shape, "Shape mismatch JIT vs AOT"
    assert jnp.all(jnp.isfinite(packet["features_z"])), "Non-finite values in JIT output"
    assert jnp.all(jnp.isfinite(out2["features_z"])), "Non-finite values in AOT output"

    print("\nAOT compiled run OK:", out2["features_z"].shape)


if __name__ == "__main__":
    main()

