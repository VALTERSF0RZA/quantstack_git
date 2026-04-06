# surface_engine.py
# =============================================================================
# DESK-GRADE VOLATILITY SURFACE ENGINE (FP64, Production Hardened)
#
# PURPOSE
# -------
# This module constructs an institutional-grade implied volatility surface
# from a live option book.
#
# ARCHITECTURE SPLIT
# ------------------
#   - CPU (Python):
#       • market hygiene
#       • liquidity + no-arbitrage filtering
#       • bucket selection
#       • data shaping
#       • orchestration & publishing
#
#   - GPU (XLA / JAX):
#       • implied volatility inversion
#       • Black–Scholes prices & Greeks
#
# This separation is deliberate:
#   • Market logic evolves; math must remain deterministic
#   • GPU kernels must be auditable and side-effect free
#
# RULES
# -----
#   • No market logic inside GPU kernels
#   • No numerical loops on the CPU
#   • FP64 everywhere for numerical stability
#
# =============================================================================

import time
import math
from dataclasses import dataclass
from typing import Dict, Optional, Any, List, Tuple
from collections import defaultdict

import jax
import jax.numpy as jnp

# -----------------------------------------------------------------------------
# FP64 CONFIGURATION
# -----------------------------------------------------------------------------
# JAX defaults to FP32.
# FP32 is insufficient for:
#   • near-expiry options
#   • deep OTM gamma
#   • IV inversion stability
#
# Enabling FP64 ensures IEEE-754 double precision end-to-end.
#
jax.config.update("jax_enable_x64", True)

from bs_jax import implied_vol, bs_price_greeks


# -----------------------------------------------------------------------------
# DEVICE DISCOVERY
# -----------------------------------------------------------------------------
# Explicit device binding prevents accidental placement and
# makes latency + numerical debugging tractable.
#
CPU = jax.devices("cpu")[0]
GPUS = jax.devices("gpu")
GPU = GPUS[0] if GPUS else None


# =============================================================================
# DATA CONTAINERS (CPU CONTROL PLANE)
# =============================================================================
# These objects:
#   • live only on the CPU
#   • never enter JAX / XLA
#   • represent market state, not math
#
@dataclass(frozen=True)
class OptionKey:
    T: float      # time to expiry (years)
    K: float      # strike
    cp: int       # +1 call, -1 put


@dataclass
class OptionQuote:
    bid: float
    ask: float
    mid: float
    ts_recv_ns: int


# =============================================================================
# SURFACE ENGINE
# =============================================================================
class SurfaceEngine:
    """
    Desk-grade volatility surface engine.

    Responsibilities:
      • maintain an in-memory option book
      • apply desk-grade filtering (CPU)
      • construct fixed-shape tensors
      • execute FP64 math on GPU
      • aggregate exposures
    """

    # -------------------------------------------------------------------------
    # INITIALIZATION (CPU ONLY)
    # -------------------------------------------------------------------------
    def __init__(
        self,
        symbol: str,
        r: float = 0.0,
        q: float = 0.0,
        max_points: int = 512,
        compute_on: str = "gpu",
        maturity_bucket_hours: int = 2,
        min_per_bucket: int = 10,
        max_per_bucket: int = 80,
        # ---------------------------------------------------------------------
        # DESK PARAMETERS (NOT MATH CONSTANTS)
        #
        # These are intentionally configurable:
        #   • SPX vs NQ vs single names
        #   • 0DTE vs longer dated
        #
        # Institutional distinction:
        #   tuning = config change, not code change
        #
        # ---------------------------------------------------------------------
        max_relative_spread: float = 0.35,
        max_absolute_spread: float = 5.0,
        arb_bound_leeway: float = 1.2,
        max_moneyness: float = 1.5,
        min_option_price: float = 0.05,
    ):
        # --- static configuration ---
        self.symbol = symbol
        self.r = float(r)
        self.q = float(q)
        self.max_points = int(max_points)

        self.maturity_bucket_hours = int(maturity_bucket_hours)
        self.min_per_bucket = int(min_per_bucket)
        self.max_per_bucket = int(max_per_bucket)

        # --- tunable desk parameters ---
        self.max_relative_spread = float(max_relative_spread)
        self.max_absolute_spread = float(max_absolute_spread)
        self.arb_bound_leeway = float(arb_bound_leeway)
        self.max_moneyness = float(max_moneyness)
        self.min_option_price = float(min_option_price)

        # --- mutable market state ---
        self._underlying_mid: Optional[float] = None
        self._book: Dict[OptionKey, OptionQuote] = {}

        # --- execution target ---
        self.compute_on = compute_on.lower()
        if self.compute_on == "gpu" and GPU is None:
            self.compute_on = "cpu"

        # ---------------------------------------------------------------------
        # GPU / XLA NUMERICAL KERNEL
        # ---------------------------------------------------------------------
        # This is the ONLY code that runs on GPU.
        #
        def kernel(S, K, T, r, q, mid, cp):
            # Runtime guard: FP64 must already be enforced upstream
            assert S.dtype == jnp.float64, f"Expected float64, got {S.dtype}"

            iv, ok, proj, ident = implied_vol(S, K, T, r, q, mid, cp)
            price, delta, gamma, vega = bs_price_greeks(S, K, T, r, q, iv, cp)
            return iv, ok, proj, ident, price, delta, gamma, vega

        self._kernel = jax.jit(kernel)

        # instrumentation
        self._warm = False
        self._compiled = False
        self._last_exec_ms = 0.0
        self._last_compile_ms = 0.0

    # -------------------------------------------------------------------------
    # DEVICE RESOLUTION
    # -------------------------------------------------------------------------
    def _dev(self):
        return GPU if self.compute_on == "gpu" else CPU

    # -------------------------------------------------------------------------
    # XLA WARMUP
    # -------------------------------------------------------------------------
    def _warmup(self):
        """
        Forces XLA compilation ahead of live trading.
        Prevents first-tick latency spikes.
        """
        n = self.max_points
        dev = self._dev()

        with jax.default_device(CPU):
            S   = jnp.ones((n,), jnp.float64) * 5000.0
            K   = jnp.ones((n,), jnp.float64) * 5000.0
            T   = jnp.ones((n,), jnp.float64) * 0.01
            r   = jnp.zeros((n,), jnp.float64)
            q   = jnp.zeros((n,), jnp.float64)
            mid = jnp.ones((n,), jnp.float64) * 10.0
            cp  = jnp.ones((n,), jnp.int32)

        payload = jax.device_put(
            {"S": S, "K": K, "T": T, "r": r, "q": q, "mid": mid, "cp": cp},
            dev,
        )

        out = self._kernel(**payload)
        _ = jnp.nansum(out[0]).block_until_ready()

        self._warm = True

    # -------------------------------------------------------------------------
    # MARKET INGESTION (CPU)
    # -------------------------------------------------------------------------
    def update_underlying(self, S_mid: float):
        self._underlying_mid = float(S_mid)

    def upsert_option(self, K: float, T: float, cp: int, bid: float, ask: float):
        if bid <= 0 and ask <= 0:
            return

        mid = 0.5 * (bid + ask) if (bid > 0 and ask > 0) else max(bid, ask)
        if mid <= 0:
            return

        self._book[OptionKey(T, K, cp)] = OptionQuote(
            bid, ask, mid, int(time.time() * 1e9)
        )

    # -------------------------------------------------------------------------
    # CONTRACT SELECTION (CPU DESK LOGIC)
    # -------------------------------------------------------------------------
    def _select_contracts(
        self,
        keys: List[OptionKey],
        quotes: List[OptionQuote],
        S0: float,
    ) -> Tuple[List[OptionKey], List[OptionQuote]]:
        eps = 1e-12
        S0 = max(float(S0), eps)

        rows = []

        for k, q in zip(keys, quotes):
            spread = max(q.ask - q.bid, 0.0)

            # liquidity filters
            if spread / max(q.mid, 0.01) > self.max_relative_spread:
                continue
            if spread > self.max_absolute_spread:
                continue

            # no-arbitrage bounds
            disc_S = S0 * math.exp(-self.q * k.T)
            disc_K = k.K * math.exp(-self.r * k.T)

            if k.cp > 0:
                lower = max(0.0, disc_S - disc_K)
                upper = disc_S
            else:
                lower = max(0.0, disc_K - disc_S)
                upper = disc_K

            if not (lower <= q.mid <= upper * self.arb_bound_leeway):
                continue

            # moneyness & noise filters
            moneyness = abs(math.log(k.K / S0))
            if moneyness > self.max_moneyness:
                continue
            if q.mid < self.min_option_price:
                continue

            rel = spread / max(q.mid, 0.01)
            hours = k.T * 365.0 * 24.0
            b = int(hours // self.maturity_bucket_hours)

            score = 5.0 * moneyness + rel + 0.001 * spread
            rows.append((b, score, k, q))

        if not rows:
            return [], []

        buckets = defaultdict(list)
        for b, score, k, q in rows:
            buckets[b].append((score, k, q))

        for b in buckets:
            buckets[b].sort(key=lambda x: x[0])

        selected = []
        remaining = self.max_points

        for b in sorted(buckets):
            take = min(self.min_per_bucket, remaining, len(buckets[b]))
            selected.extend(buckets[b][:take])
            buckets[b] = buckets[b][take:]
            remaining -= take

        cap_extra = max(0, self.max_per_bucket - self.min_per_bucket)
        for b in sorted(buckets):
            take = min(cap_extra, remaining, len(buckets[b]))
            selected.extend(buckets[b][:take])
            remaining -= take

        if remaining > 0:
            leftovers = []
            for b in buckets:
                leftovers.extend(buckets[b])
            leftovers.sort(key=lambda x: x[0])
            selected.extend(leftovers[:remaining])

        return [x[1] for x in selected], [x[2] for x in selected]

    # -------------------------------------------------------------------------
    # ARRAY BUILD (CPU → GPU STAGING)
    # -------------------------------------------------------------------------
    def _build_arrays(self, keys, quotes, S0):
        """
        Builds fixed-shape arrays required by XLA.
        Padding + masking is mandatory for GPU determinism.
        """
        n = self.max_points
        n_real = min(len(keys), n)

        with jax.default_device(CPU):
            K = jnp.pad(jnp.array([k.K for k in keys[:n_real]], jnp.float64), (0, n - n_real))
            T = jnp.pad(jnp.array([k.T for k in keys[:n_real]], jnp.float64), (0, n - n_real))
            cp = jnp.pad(jnp.array([k.cp for k in keys[:n_real]], jnp.int32), (0, n - n_real))
            mid = jnp.pad(jnp.array([q.mid for q in quotes[:n_real]], jnp.float64), (0, n - n_real))

            # mask marks which entries are real vs padded
            mask = jnp.arange(n) < n_real

            S = jnp.full((n,), S0, jnp.float64)
            r = jnp.full((n,), self.r, jnp.float64)
            q = jnp.full((n,), self.q, jnp.float64)

        return S, K, T, r, q, mid, cp, mask, n_real

    # -------------------------------------------------------------------------
    # SURFACE COMPUTATION (CPU → GPU → CPU)
    # -------------------------------------------------------------------------
    def compute_surface_points(self) -> Optional[Dict[str, Any]]:
        if self._underlying_mid is None or not self._book:
            return None

        if not self._warm:
            self._warmup()

        keys = list(self._book.keys())
        quotes = [self._book[k] for k in keys]
        S0 = self._underlying_mid

        keys, quotes = self._select_contracts(keys, quotes, S0)
        if not keys:
            return None

        S, K, T, r, q, mid, cp, mask, n_real = self._build_arrays(keys, quotes, S0)
        dev = self._dev()

        payload = jax.device_put(
            {"S": S, "K": K, "T": T, "r": r, "q": q, "mid": mid, "cp": cp},
            dev,
        )

        t0 = time.perf_counter()
        iv, ok, proj, ident, price, delta, gamma, vega = self._kernel(**payload)
        _ = jnp.nansum(iv).block_until_ready()
        elapsed = (time.perf_counter() - t0) * 1000

        if not self._compiled:
            self._last_compile_ms = elapsed
            self._compiled = True
        else:
            self._last_exec_ms = elapsed

        # boolean-pure masking
        good = ident & mask

        delta_exp = float(jnp.sum(jnp.where(good, delta * S, 0.0)))
        gamma_exp = float(jnp.sum(jnp.where(good, gamma * S * S, 0.0)))
        vega_exp = float(jnp.sum(jnp.where(good, vega, 0.0)))

        # ATM selection:
        # jnp.inf guarantees padded entries can NEVER be chosen
        atm_idx = int(jnp.argmin(jnp.where(mask, jnp.abs(K - S0), jnp.inf)))
        atm_iv = float(iv[atm_idx])

        iv_h = jax.device_get(iv[:n_real])
        K_h = jax.device_get(K[:n_real])
        T_h = jax.device_get(T[:n_real])
        cp_h = jax.device_get(cp[:n_real])
        mid_h = jax.device_get(mid[:n_real])
        id_h = jax.device_get(ident[:n_real])

        points = [
            {
                "K": float(K_h[i]),
                "T": float(T_h[i]),
                "cp": int(cp_h[i]),
                "bid": float(quotes[i].bid),
                "ask": float(quotes[i].ask),
                "mid": float(mid_h[i]),
                "iv": float(iv_h[i]),
            }
            for i in range(n_real)
            if id_h[i]
        ]

        return {
            "schema_version": 5,
            "symbol": self.symbol,
            "ts_unix_ns": int(time.time() * 1e9),
            "S_mid": S0,
            "count_real": int(n_real),
            "count_max": int(self.max_points),
            "points_emitted": len(points),
            "compute_device": "gpu" if (dev is GPU) else "cpu",
            "iv_compute_ms": round(elapsed, 3),
            "metrics": {
                "atm_iv": atm_iv,
                "delta_exposure": delta_exp,
                "gamma_exposure": gamma_exp,
                "vega_exposure": vega_exp,
            },
            "xla": {
                "compile_ms": round(self._last_compile_ms, 3),
                "exec_ms": round(self._last_exec_ms, 3),
            },
            "points": points,
        }

