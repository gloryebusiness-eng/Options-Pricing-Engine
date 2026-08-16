"""
Analytical Greeks for European options under Black-Scholes-Merton.

Each function is a closed-form partial derivative of the pricing formula,
verified in tests/test_greeks.py against a central-difference approximation
of the same derivative computed from the pricer.

A recurring identity underlies most of the simplifications here:

    S*phi(d1) = K*exp(-rT)*phi(d2)

Substituting it collapses the cross terms produced by the product rule, which
is why delta reduces to N(d1) alone and why vega and gamma are single-term
expressions. It also explains why gamma and vega are identical for calls and
puts: put-call parity differs by S - K*exp(-rT), which is linear in S and
independent of sigma, so both derivatives difference away.

Sign and scaling conventions:
  - theta is returned as -dC/dT per calendar day, so long options report
    negative theta (decay is a cost to the holder).
  - vega and rho are returned per unit change (1.00), not per point (0.01).
    Desk quoting conventions divide by 100; scaling is left to the caller so
    the raw derivative is what the finite-difference tests compare against.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ope.models.black_scholes import d1_d2

__all__ = ["delta", "gamma", "vega", "theta", "rho", "all_greeks"]

DAYS_PER_YEAR = 365.0


def _validate(option_type: str) -> str:
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    return option_type


def delta(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    option_type: str = "call",
) -> float | np.ndarray:
    """dC/dS -- the hedge ratio, in shares per option.

    Call delta is N(d1), bounded in [0, 1]; put delta is N(d1) - 1, bounded
    in [-1, 0]. The unit gap between them is the derivative of the put-call
    parity relation with respect to S.

    The three cross terms produced by the product rule cancel via the
    S*phi(d1) = K*exp(-rT)*phi(d2) identity, leaving N(d1) alone.
    """
    option_type = _validate(option_type)
    d1, _ = d1_d2(S, K, T, r, sigma)

    result = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1.0
    return float(result) if np.ndim(result) == 0 else result


def gamma(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
) -> float | np.ndarray:
    """d2C/dS2 -- the rate of change of delta.

    Identical for calls and puts. Peaks near the forward and decays in both
    tails, since deep ITM and deep OTM options have near-constant delta
    (1 and 0) and therefore no convexity. Gamma is what forces a delta hedge
    to be rebalanced as spot moves.
    """
    d1, _ = d1_d2(S, K, T, r, sigma)
    result = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return float(result) if np.ndim(result) == 0 else result


def vega(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
) -> float | np.ndarray:
    """dC/dsigma per unit volatility -- identical for calls and puts.

    Strictly positive: optionality caps the downside at the premium while
    leaving the upside open, so a wider terminal distribution can only add
    value.

    At the money phi(d1) is close to 0.4, giving the desk approximation
    vega ~ 0.4*S*sqrt(T), which is a useful sanity check on any quoted number.
    """
    d1, _ = d1_d2(S, K, T, r, sigma)
    result = S * norm.pdf(d1) * np.sqrt(T)
    return float(result) if np.ndim(result) == 0 else result


def theta(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    option_type: str = "call",
) -> float | np.ndarray:
    """-dC/dT per calendar day.

    Time enters the price through two distinct channels, and unlike the other
    Greeks they do not collapse into one term:

      1. Diffusion decay -- sigma*sqrt(T) shrinks as expiry approaches, so
         the terminal distribution narrows and optionality is lost. This is
         the -S*phi(d1)*sigma / (2*sqrt(T)) term, negative for both types.
      2. Discounting -- the present value of the strike changes with T. This
         term carries opposite signs for calls and puts, since the call
         holder pays K in the future while the put holder receives it.

    Division by 365 converts the annualised derivative to the per-day figure
    desks quote. Long options report negative theta.
    """
    option_type = _validate(option_type)
    d1, d2 = d1_d2(S, K, T, r, sigma)

    diffusion = -(S * norm.pdf(d1) * sigma) / (2.0 * np.sqrt(T))

    if option_type == "call":
        discounting = -r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        discounting = r * K * np.exp(-r * T) * norm.cdf(-d2)

    result = (diffusion + discounting) / DAYS_PER_YEAR
    return float(result) if np.ndim(result) == 0 else result


def rho(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    option_type: str = "call",
) -> float | np.ndarray:
    """dC/dr per unit rate.

    Positive for calls and negative for puts: a higher rate lowers the
    present value of the strike, which benefits the party who pays it and
    penalises the party who receives it. Rho scales with T, so it is
    negligible for short-dated options and material for LEAPS.
    """
    option_type = _validate(option_type)
    _, d2 = d1_d2(S, K, T, r, sigma)
    discounted_strike = K * T * np.exp(-r * T)

    if option_type == "call":
        result = discounted_strike * norm.cdf(d2)
    else:
        result = -discounted_strike * norm.cdf(-d2)

    return float(result) if np.ndim(result) == 0 else result


def all_greeks(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    option_type: str = "call",
) -> dict[str, float | np.ndarray]:
    """Return every Greek in one call.

    Convenience wrapper for risk reporting, where the full sensitivity vector
    is usually wanted at once. Recomputes d1 and d2 per Greek; caching them
    is a possible optimisation if this ever sits on a latency path.
    """
    option_type = _validate(option_type)
    return {
        "delta": delta(S, K, T, r, sigma, option_type=option_type),
        "gamma": gamma(S, K, T, r, sigma),
        "vega": vega(S, K, T, r, sigma),
        "theta": theta(S, K, T, r, sigma, option_type=option_type),
        "rho": rho(S, K, T, r, sigma, option_type=option_type),
    }
