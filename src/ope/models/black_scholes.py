"""
Black-Scholes-Merton pricing for European options on a non-dividend-paying asset.

The closed-form solution is the discounted risk-neutral expectation of the
terminal payoff under geometric Brownian motion. For a call:

    C = S*N(d1) - K*exp(-rT)*N(d2)

where N(d2) is the risk-neutral probability of finishing in the money, and
N(d1) is the option's delta -- the expected terminal stock value conditional
on exercise, expressed under the stock-numeraire measure. The two differ by
exactly sigma*sqrt(T), the total accumulated uncertainty over the option's
life, which is why N(d1) > N(d2) for every non-degenerate input.

Model assumptions and their known failures are documented in
docs/MODEL_ASSUMPTIONS.md. The most consequential in practice is constant
volatility, whose violation is directly observable as the volatility smile.

All functions here are pure: they accept and return numbers or NumPy arrays
and perform no I/O. This keeps them vectorised, trivially testable, and
interchangeable with the binomial and Monte Carlo kernels behind one interface.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["bs_price", "d1_d2"]


def d1_d2(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the standardised moneyness terms d1 and d2.

    d1 = [ln(S/K) + (r + sigma^2/2)T] / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    The numerator of d1 is the log-moneyness of the *forward* price: the
    drift term (r + sigma^2/2)T is the mean of the log-normal terminal
    distribution, so d1 measures how many standard deviations the strike sits
    below the expected terminal log-price.

    When sigma*sqrt(T) is zero the terms are undefined (the payoff is
    deterministic). The denominator is substituted with 1.0 to avoid a
    division warning; callers must branch on the degenerate case separately.
    """
    vol_time = sigma * np.sqrt(T)
    safe_vol_time = np.where(vol_time > 0.0, vol_time, 1.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / safe_vol_time
    d2 = d1 - vol_time

    return d1, d2


def bs_price(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    option_type: str = "call",
) -> float | np.ndarray:
    """Price a European option under Black-Scholes-Merton.

    Parameters
    ----------
    S : Spot price of the underlying. Must be strictly positive.
    K : Strike price. Must be strictly positive.
    T : Time to expiry in years. Zero is permitted and returns intrinsic value.
    r : Continuously compounded risk-free rate, as a decimal. May be negative.
    sigma : Annualised volatility, as a decimal. Zero returns the forward payoff.
    option_type : Either "call" or "put".

    Returns
    -------
    The option premium. Scalar inputs return a float; array inputs broadcast
    and return an ndarray.

    Notes
    -----
    When sigma*sqrt(T) is zero the terminal price is known with certainty to be
    the forward S*exp(rT), so the option is worth its discounted intrinsic
    value. This branch is handled explicitly rather than relying on limiting
    behaviour of the normal CDF, which would divide by zero.
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(
            f"option_type must be 'call' or 'put', got {option_type!r}"
        )

    S, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma))

    if np.any(S <= 0.0) or np.any(K <= 0.0):
        raise ValueError("S and K must be strictly positive")
    if np.any(T < 0.0):
        raise ValueError("T must be non-negative")
    if np.any(sigma < 0.0):
        raise ValueError("sigma must be non-negative")

    discount = np.exp(-r * T)
    d1, d2 = d1_d2(S, K, T, r, sigma)
    is_degenerate = (sigma * np.sqrt(T)) <= 0.0

    if option_type == "call":
        stochastic = S * norm.cdf(d1) - K * discount * norm.cdf(d2)
        degenerate = np.maximum(S - K * discount, 0.0)
    else:
        stochastic = K * discount * norm.cdf(-d2) - S * norm.cdf(-d1)
        degenerate = np.maximum(K * discount - S, 0.0)

    price = np.where(is_degenerate, degenerate, stochastic)

    return float(price) if price.ndim == 0 else price