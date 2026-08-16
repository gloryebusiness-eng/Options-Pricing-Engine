"""
Implied volatility: inverting Black-Scholes for the volatility input.

Given an observed market price, find the sigma that reproduces it. No closed
form exists -- sigma appears inside the normal CDF in both d1 and d2 -- so
this is a root-finding problem on

    f(sigma) = BS(sigma) - market_price

Two solvers are layered. Newton-Raphson is tried first: its derivative is
vega, available analytically, and convergence is quadratic. Newton fails when
vega approaches zero, so Brent's method takes over, guaranteeing convergence
given a sign-changing bracket at the cost of more iterations.

CONDITIONING
------------
The sensitivity of the recovered volatility to price error is 1/vega, so vega
is the condition number of this inversion. For deep in- or out-of-the-money
options vega collapses toward zero and the problem becomes ill-posed: every
volatility across a wide range reproduces the observed price to the last
representable bit, and no algorithm can distinguish them.

This is a property of the problem, not a limitation of the solver. Returning
a confident number in that regime is misleading, so results carry a vega
measurement and an identifiable flag. Callers building surfaces across a full
chain should filter on that flag rather than trusting every strike equally.

Convergence is assessed on the change in sigma rather than the price
residual. An absolute price tolerance is not scale-free: for a deep OTM
option priced at 1e-55, any sigma satisfies a 1e-8 price tolerance
immediately, and the solver would return its initial guess unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq

from ope.models.black_scholes import bs_price
from ope.models.greeks import vega as bs_vega

__all__ = ["implied_vol", "ImpliedVolError", "ImpliedVolResult"]

VOL_LOWER_BOUND = 1e-6
VOL_UPPER_BOUND = 5.0

# Below this, the Newton step divides by a quantity small enough that the
# update is dominated by floating-point noise.
MIN_USABLE_VEGA = 1e-8

# Below this, the inversion is ill-conditioned: a price perturbation at the
# level of double precision maps to a volatility error exceeding any useful
# tolerance. Results are returned but flagged as not identifiable.
MIN_IDENTIFIABLE_VEGA = 1e-6

DEFAULT_SIGMA_TOLERANCE = 1e-10
DEFAULT_MAX_ITERATIONS = 50


class ImpliedVolError(ValueError):
    """Raised when no volatility can reproduce the observed price.

    Distinct from ill-conditioning: this signals the price is inconsistent
    with the model's arbitrage bounds, so no solution exists at all.
    """


@dataclass(frozen=True)
class ImpliedVolResult:
    """Solver output with conditioning diagnostics.

    Attributes
    ----------
    volatility : The recovered sigma.
    iterations : Newton steps taken before convergence or handoff.
    converged : Whether a root was found within the search bracket.
    method : "newton" or "brent" -- which solver produced the result. A
        contiguous region of a surface falling back to Brent identifies
        exactly where vega collapsed.
    residual : Price error at the returned sigma.
    vega : Vega at the returned sigma, the condition number of the inversion.
    identifiable : False when vega is too small for the recovered volatility
        to carry meaningful information, regardless of convergence.
    """

    volatility: float
    iterations: int
    converged: bool
    method: str
    residual: float
    vega: float
    identifiable: bool


def _arbitrage_bounds(
    S: float, K: float, T: float, r: float, option_type: str
) -> tuple[float, float]:
    """No-arbitrage price bounds for the given contract."""
    discounted_strike = K * math.exp(-r * T)

    if option_type == "call":
        return max(S - discounted_strike, 0.0), S
    return max(discounted_strike - S, 0.0), discounted_strike


def _initial_guess(
    price: float, S: float, K: float, T: float, r: float, option_type: str
) -> float:
    """Corrado-Miller approximation to the implied volatility.

    Brenner-Subrahmanyam, sigma ~ (price/S)*sqrt(2*pi/T), is derived at the
    money and degrades quickly away from it: for a strike 23% out of the money
    it returns a value near the clamp floor while the true volatility is
    several times larger. Newton started from there overshoots the search
    bracket on its first step and hands off to the bracketed solver, which
    converges correctly but far more slowly.

    Corrado-Miller adds a correction term in the forward moneyness
    (S - K*exp(-rT)), which is the information the at-the-money derivation
    discards. It remains usable across the wings and reduces the fallback rate
    substantially.

    The discriminant can go negative for prices near the arbitrage bounds,
    where the quadratic has no real root. Brenner-Subrahmanyam is used as a
    backstop in that case; it is a poor guess but a finite one, and the
    bracketed solver remains available behind it.

    Puts are converted to the equivalent call price via put-call parity, since
    the approximation is derived for calls.
    """
    discounted_strike = K * math.exp(-r * T)

    if option_type == "put":
        price = price + S - discounted_strike

    moneyness_gap = S - discounted_strike
    average = (S + discounted_strike) / 2.0

    discriminant = (price - moneyness_gap / 2.0) ** 2 - (moneyness_gap**2) / math.pi

    if discriminant < 0.0:
        fallback = (price / S) * math.sqrt(2.0 * math.pi / T)
        return min(max(fallback, 0.01), 3.0)

    guess = (
        math.sqrt(2.0 * math.pi / T)
        / average
        * (price - moneyness_gap / 2.0 + math.sqrt(discriminant))
    )

    return min(max(guess, 0.01), 3.0)


def _build_result(
    sigma: float,
    S: float,
    K: float,
    T: float,
    r: float,
    price: float,
    option_type: str,
    iterations: int,
    method: str,
) -> ImpliedVolResult:
    """Attach conditioning diagnostics to a converged solve."""
    v = float(bs_vega(S, K, T, r, sigma))
    residual = float(bs_price(S, K, T, r, sigma, option_type=option_type)) - price

    return ImpliedVolResult(
        volatility=sigma,
        iterations=iterations,
        converged=True,
        method=method,
        residual=residual,
        vega=v,
        identifiable=v >= MIN_IDENTIFIABLE_VEGA,
    )


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    sigma_tolerance: float = DEFAULT_SIGMA_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    return_diagnostics: bool = False,
) -> float | ImpliedVolResult:
    """Solve for the volatility implied by an observed option price.

    Parameters
    ----------
    price : Observed market premium.
    S, K, T, r : Contract and market parameters, as in bs_price.
    option_type : Either "call" or "put".
    sigma_tolerance : Convergence threshold on the change in sigma between
        iterations. Assessed in volatility units rather than price units so
        the criterion is scale-free across moneyness.
    max_iterations : Iteration cap for the Newton phase.
    return_diagnostics : When True, return the full ImpliedVolResult. Callers
        processing a full chain should use this and filter on identifiable.

    Raises
    ------
    ImpliedVolError
        If the price violates the no-arbitrage bounds, or if both solvers
        fail to converge.
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(
            f"option_type must be 'call' or 'put', got {option_type!r}"
        )

    if not math.isfinite(price):
        raise ImpliedVolError(f"price must be finite, got {price}")

    lower, upper = _arbitrage_bounds(S, K, T, r, option_type)

    if price < lower - 1e-10:
        raise ImpliedVolError(
            f"price {price:.6g} is below the no-arbitrage floor {lower:.6g}; "
            "no volatility reproduces this price"
        )
    if price > upper + 1e-10:
        raise ImpliedVolError(
            f"price {price:.6g} exceeds the no-arbitrage ceiling {upper:.6g}; "
            "no volatility reproduces this price"
        )

    sigma = sigma = _initial_guess(price, S, K, T, r, option_type)
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        model_price = float(bs_price(S, K, T, r, sigma, option_type=option_type))
        residual = model_price - price
        v = float(bs_vega(S, K, T, r, sigma))

        # Vega collapses in the deep tails. Dividing by it there produces a
        # step dominated by floating-point noise, so hand off rather than
        # iterating into a diverged state.
        if v < MIN_USABLE_VEGA:
            break

        step = residual / v
        candidate = sigma - step

        # Newton knows nothing about the search bracket and can overshoot
        # into a non-physical region even when vega is usable.
        if not (VOL_LOWER_BOUND < candidate < VOL_UPPER_BOUND):
            break

        sigma = candidate

        # Convergence is assessed on the step in sigma, not the price
        # residual: the latter is not scale-free and is satisfied trivially
        # wherever the option price is itself smaller than the tolerance.
        if abs(step) < sigma_tolerance:
            result = _build_result(
                sigma, S, K, T, r, price, option_type, iterations, "newton"
            )
            return result if return_diagnostics else result.volatility

    return _solve_by_bracketing(
        price, S, K, T, r, option_type, iterations, return_diagnostics
    )


def _solve_by_bracketing(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    newton_iterations: int,
    return_diagnostics: bool,
) -> float | ImpliedVolResult:
    """Bracketed fallback using Brent's method.

    Black-Scholes is monotonically increasing in sigma, so the prices at the
    bracket endpoints straddle any attainable market price. Brent combines
    the guaranteed convergence of bisection with faster interpolation where
    the function is locally well behaved.

    Where vega is negligible the objective is flat to machine precision and
    Brent returns an arbitrary point on that plateau. The returned result is
    flagged as not identifiable rather than presented as a confident answer.
    """

    def objective(sigma: float) -> float:
        return float(bs_price(S, K, T, r, sigma, option_type=option_type)) - price

    try:
        sigma = brentq(
            objective,
            VOL_LOWER_BOUND,
            VOL_UPPER_BOUND,
            xtol=1e-14,
            rtol=8.9e-16,
            maxiter=200,
        )
    except (ValueError, RuntimeError) as exc:
        raise ImpliedVolError(
            f"no volatility in [{VOL_LOWER_BOUND}, {VOL_UPPER_BOUND}] "
            f"reproduces price {price:.6g}"
        ) from exc

    result = _build_result(
        sigma, S, K, T, r, price, option_type, newton_iterations, "brent"
    )
    return result if return_diagnostics else result.volatility