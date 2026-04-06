     1 import jax
     2 import jax.numpy as jnp
     3 import jax.scipy as jsp
     4
     5 # --- FP64 Requirement 1: Enforce FP64 and define constants ---
     6 jax.config.update("jax_enable_x64", True)
     7 F64 = jnp.float64
     8
     9 EPS_S = F64(1e-12)
    10 EPS_T = F64(1e-8)
    11 EPS_V = F64(1e-12)
    12
    13
    14 # -----------------------------
    15 # Normal CDF / PDF
    16 # -----------------------------
    17 @jax.jit
    18 def _norm_cdf(x: jnp.ndarray) -> jnp.ndarray:
    19     # --- FP64 Requirement 3: Cast input and use FP64 constants ---
    20     x = jnp.asarray(x, F64)
    21     return F64(0.5) * (F64(1.0) + jsp.special.erf(x / jnp.sqrt(F64(2.0))))
    22
    23
    24 @jax.jit
    25 def _norm_pdf(x: jnp.ndarray) -> jnp.ndarray:
    26     # --- FP64 Requirement 3: Cast input and use FP64 constants ---
    27     x = jnp.asarray(x, F64)
    28     return jnp.exp(F64(-0.5) * x * x) / jnp.sqrt(F64(2.0) * F64(jnp.pi))
    29
    30
    31 # -----------------------------
    32 # Black–Scholes price + greeks
    33 # -----------------------------
    34 @jax.jit
    35 def bs_price_greeks(S, K, T, r, q, vol, cp):
    36     # --- FP64 Requirement 4: Cast all inputs to F64 ---
    37     S = jnp.asarray(S, F64)
    38     K = jnp.asarray(K, F64)
    39     T = jnp.asarray(T, F64)
    40     r = jnp.asarray(r, F64)
    41     q = jnp.asarray(q, F64)
    42     vol = jnp.asarray(vol, F64)
    43
    44     S = jnp.maximum(S, EPS_S)
    45     K = jnp.maximum(K, EPS_S)
    46     T = jnp.maximum(T, EPS_T)
    47     vol = jnp.maximum(vol, F64(1e-8))
    48
    49     disc_r = jnp.exp(-r * T)
    50     disc_q = jnp.exp(-q * T)
    51
    52     sqrtT = jnp.sqrt(T)
    53     inv = F64(1.0) / (vol * sqrtT + EPS_V)
    54
    55     d1 = (jnp.log(S / K) + (r - q + F64(0.5) * vol * vol) * T) * inv
    56     d2 = d1 - vol * sqrtT
    57
    58     Nd1 = _norm_cdf(cp * d1)
    59     Nd2 = _norm_cdf(cp * d2)
    60
    61     price = cp * (S * disc_q * Nd1 - K * disc_r * Nd2)
    62
    63     pdf_d1 = _norm_pdf(d1)
    64     delta = cp * disc_q * Nd1
    65     gamma = disc_q * pdf_d1 / (S * vol * sqrtT + EPS_V)
    66     vega = S * disc_q * pdf_d1 * sqrtT
    67
    68     return price, delta, gamma, vega
    69
    70
    71 # -----------------------------
    72 # No-arbitrage bounds
    73 # -----------------------------
    74 @jax.jit
    75 def _arb_bounds(S, K, T, r, q, cp):
    76     # Inputs are already F64 from calling functions
    77     disc_r = jnp.exp(-r * T)
    78     disc_q = jnp.exp(-q * T)
    79
    80     fwd_pv = S * disc_q
    81     strike_pv = K * disc_r
    82
    83     call_lb = jnp.maximum(F64(0.0), fwd_pv - strike_pv)
    84     put_lb  = jnp.maximum(F64(0.0), strike_pv - fwd_pv)
    85     lb = jnp.where(cp > 0, call_lb, put_lb)
    86
    87     call_ub = fwd_pv
    88     put_ub  = strike_pv
    89     ub = jnp.where(cp > 0, call_ub, put_ub)
    90
    91     return lb, ub
    92
    93
    94 # -----------------------------
    95 # Desk-grade implied vol
    96 # -----------------------------
    97 @jax.jit
    98 def implied_vol(
    99     S,
   100     K,
   101     T,
   102     r,
   103     q,
   104     mid,
   105     cp,
   106     # --- FP64 Requirement 5: Use F64 for default arguments ---
   107     vol_init=F64(0.25),
   108     vol_lo=F64(1e-4),
   109     vol_hi=F64(5.0),
   110     newton_steps=8,
   111     bisect_steps=24,
   112     tol=F64(1e-3),
   113     vega_min=F64(1e-6),
   114     extrinsic_eps=F64(0.05),
   115 ):
   116     # --- FP64 Requirement 5: Cast all inputs to F64 ---
   117     S = jnp.asarray(S, F64)
   118     K = jnp.asarray(K, F64)
   119     T = jnp.asarray(T, F64)
   120     r = jnp.asarray(r, F64)
   121     q = jnp.asarray(q, F64)
   122     mid = jnp.asarray(mid, F64)
   123
   124     S = jnp.maximum(S, EPS_S)
   125     K = jnp.maximum(K, EPS_S)
   126     T = jnp.maximum(T, EPS_T)
   127     mid = jnp.maximum(mid, F64(0.0))
   128
   129     # --- 1) No-arb projection ---
   130     lb, ub = _arb_bounds(S, K, T, r, q, cp)
   131     is_price_valid = (mid >= lb) & (mid <= ub)
   132     mid_clip = jnp.clip(mid, lb, ub)
   133     is_price_projected = ~is_price_valid
   134
   135     # --- 2) Extrinsic test (observability) ---
   136     extrinsic = mid_clip - lb
   137     near_intrinsic = extrinsic <= extrinsic_eps
   138
   139     # --- 3) Newton ---
   140     vol = jnp.full_like(mid_clip, vol_init)
   141
   142     def newton_body(v, _):
   143         price, _, _, vega = bs_price_greeks(S, K, T, r, q, v, cp)
   144         err = price - mid_clip
   145         step = err / (vega + EPS_V)
   146         return jnp.clip(v - step, vol_lo, vol_hi), None
   147
   148     volN, _ = jax.lax.scan(newton_body, vol, xs=None, length=newton_steps)
   149     pN, _, _, vN = bs_price_greeks(S, K, T, r, q, volN, cp)
   150
   151     # --- 4) Vega conditioning (critical fix) ---
   152     vega_cut = vega_min * S * jnp.sqrt(T)
   153     low_vega = vN <= vega_cut
   154     is_vega_identifiable = (~near_intrinsic) & (~low_vega)
   155
   156     need = is_vega_identifiable & (jnp.abs(pN - mid_clip) > tol)
   157
   158     # --- 5) Masked bisection ---
   159     def do_bisect(_):
   160         lo = jnp.full_like(volN, vol_lo)
   161         hi = jnp.full_like(volN, vol_hi)
   162
   163         def bis_body(state, _):
   164             lo, hi = state
   165             vm = F64(0.5) * (lo + hi)
   166             pm, _, _, _ = bs_price_greeks(S, K, T, r, q, vm, cp)
   167
   168             go_hi = pm > mid_clip
   169             hi2 = jnp.where(need & go_hi, vm, hi)
   170             lo2 = jnp.where(need & (~go_hi), vm, lo)
   171             return (lo2, hi2), None
   172
   173         (loF, hiF), _ = jax.lax.scan(bis_body, (lo, hi), xs=None, length=bisect_steps)
   174         return jnp.where(need, F64(0.5) * (loF + hiF), volN)
   175
   176     volB = jax.lax.cond(jnp.any(need), do_bisect, lambda _: volN, operand=None)
   177
   178     # --- 6) Final institutional rule ---
   179     iv = jnp.where(is_vega_identifiable, volB, jnp.nan)
   180
   181     return iv, is_price_valid, is_price_projected, is_vega_identifiable

