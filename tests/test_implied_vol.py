"""
Correctness tests for the implied volatility solver.

The primary test is a round trip: price at a known sigma, then recover it.
Recovery is asserted only where the inversion is well conditioned.

Vega is the condition number of this inversion, since d(sigma)/d(price) is
1/vega. Where vega collapses -- deep in- and out-of-the-money strikes -- a
price perturbation at the level of double precision maps to an unbounded
volatility error, and no implied volatility exists to be recovered. Asserting
recovery there would assert something numerically impossible, so those cases
instead assert that the solver reports the result as not identifiable.

Tolerances are conditioning-aware: the achievable error in sigma scales as
the price precision divided by vega, so a fixed tolerance would be either
vacuous near the money or unsatisfiable in the wings.
"""

import math

import pytest
from hypothesis import assume, given, settings, strategies as st

from ope.models.black_scholes import bs_price
from ope.models.greeks import vega as bs_vega
from ope.vol.implied import ImpliedVolError, implied_vol

spot = st.floats(min_value=10.0, max_value=500.0, allow_nan=False)
strike = st.floats(min_value=10.0, max_value=500.0, allow_nan=False)
tenor = st.floats(min_value=0.05, max_value=3.0, allow_nan=False)
rate = st.floats(min_value=0.0, max_value=0.10, allow_nan=False)
vol = st.floats(min_value=0.05, max_value=1.5, allow_nan=False)
kind = st.sampled_from(["call", "put"])

# Below this vega the recovered volatility carries too little information for
# a round-trip assertion to be meaningful.
WELL_CONDITIONED_VEGA = 1e-2


def sigma_error_budget(S, K, T, r, sigma, option_type):
    """Achievable absolute error in sigma for this contract.

    Price is representable to roughly max(|price|, 1) * 1e-13 in double
    precision after accumulation through the normal CDF. Dividing by vega
    converts that price uncertainty into the corresponding uncertainty in
    volatility, which is the tightest tolerance any solver could satisfy.
    """
    price = float(bs_price(S, K, T, r, sigma, option_type=option_type))
    v = float(bs_vega(S, K, T, r, sigma))
    price_precision = max(abs(price), 1.0) * 1e-13

    return max(1e-8, price_precision / max(v, 1e-300))


@given(S=spot, K=strike, T=tenor, r=rate, sigma=vol, option_type=kind)
@settings(deadline=None, max_examples=400)
def test_round_trip_recovers_input_volatility(S, K, T, r, sigma, option_type):
    """Price at a known sigma, then invert the price to recover it.

    Restricted to well-conditioned contracts. Black-Scholes is strictly
    increasing in sigma, so the inverse is unique wherever vega is
    non-negligible; deviation beyond the conditioning budget indicates a
    broken iteration rather than an ill-posed problem.
    """
    assume(float(bs_vega(S, K, T, r, sigma)) > WELL_CONDITIONED_VEGA)

    price = bs_price(S, K, T, r, sigma, option_type=option_type)
    recovered = implied_vol(price, S, K, T, r, option_type=option_type)
    budget = sigma_error_budget(S, K, T, r, sigma, option_type)

    assert recovered == pytest.approx(sigma, abs=budget)


@pytest.mark.parametrize("moneyness", [0.8, 0.9, 1.0, 1.1, 1.25])
@pytest.mark.parametrize("sigma", [0.1, 0.35, 1.2])
def test_round_trip_near_the_money(moneyness, sigma):
    """Recovery across the strikes that carry real volatility information."""
    S, T, r = 100.0, 1.0, 0.03
    K = S / moneyness

    price = bs_price(S, K, T, r, sigma, option_type="call")
    recovered = implied_vol(price, S, K, T, r, option_type="call")
    budget = sigma_error_budget(S, K, T, r, sigma, "call")

    assert recovered == pytest.approx(sigma, abs=budget)


@pytest.mark.parametrize("moneyness", [0.2, 5.0])
def test_deep_wing_volatility_is_not_identifiable(moneyness):
    """Deep wings must be reported as not identifiable, not solved silently.

    At S/K of 5.0 with 10% volatility, vega is on the order of 1e-57: every
    volatility from 1% to 300% produces the same price to the last
    representable bit. The correct behaviour is to flag the result, since a
    confident number here would be indistinguishable from a correct one to
    any downstream consumer.
    """
    S, T, r, sigma = 100.0, 1.0, 0.03, 0.1
    K = S / moneyness

    price = bs_price(S, K, T, r, sigma, option_type="call")
    result = implied_vol(
        price, S, K, T, r, option_type="call", return_diagnostics=True
    )

    assert not result.identifiable
    assert result.vega < 1e-6


def test_identifiable_flag_is_set_near_the_money():
    """The flag must not fire where the inversion is well posed."""
    price = bs_price(100.0, 100.0, 1.0, 0.05, 0.25, option_type="call")
    result = implied_vol(
        price, 100.0, 100.0, 1.0, 0.05, option_type="call", return_diagnostics=True
    )

    assert result.identifiable
    assert result.vega > 1.0


def test_rejects_price_below_intrinsic():
    """A price under the no-arbitrage floor has no corresponding sigma.

    Not a convergence failure: the lower bound holds for every sigma, so no
    volatility reproduces such a price. Stale and crossed quotes produce
    these routinely, so the solver must reject rather than iterate.
    """
    S, K, T, r = 100.0, 80.0, 1.0, 0.05
    floor = max(S - K * math.exp(-r * T), 0.0)

    with pytest.raises(ImpliedVolError):
        implied_vol(floor - 1.0, S, K, T, r, option_type="call")


def test_rejects_price_above_spot():
    """A call cannot be worth more than the underlying."""
    with pytest.raises(ImpliedVolError):
        implied_vol(150.0, 100.0, 90.0, 1.0, 0.05, option_type="call")


def test_rejects_negative_price():
    with pytest.raises(ImpliedVolError):
        implied_vol(-1.0, 100.0, 100.0, 1.0, 0.05, option_type="call")


def test_converges_within_iteration_budget():
    """Newton should reach tolerance in a handful of steps near the money.

    Quadratic convergence roughly doubles the correct digits per iteration,
    so an ATM solve needing many steps signals a poor initial guess or a
    broken update.
    """
    price = bs_price(100.0, 100.0, 1.0, 0.05, 0.25, option_type="call")
    result = implied_vol(
        price, 100.0, 100.0, 1.0, 0.05, option_type="call", return_diagnostics=True
    )

    assert result.iterations <= 8
    assert result.converged
    assert result.volatility == pytest.approx(0.25, abs=1e-8)


def test_reports_which_method_succeeded():
    """Diagnostics must distinguish the Newton path from the fallback.

    Knowing which solver produced a number matters when auditing a surface
    built from thousands of strikes: a shift to the bracketed method across
    a contiguous region flags where vega collapsed.
    """
    price = bs_price(100.0, 100.0, 1.0, 0.05, 0.25, option_type="call")
    result = implied_vol(
        price, 100.0, 100.0, 1.0, 0.05, option_type="call", return_diagnostics=True
    )

    assert result.method in ("newton", "brent")


def test_put_round_trip():
    price = bs_price(100.0, 110.0, 0.5, 0.04, 0.3, option_type="put")
    recovered = implied_vol(price, 100.0, 110.0, 0.5, 0.04, option_type="put")

    assert recovered == pytest.approx(0.3, abs=1e-8)


def test_rejects_unknown_option_type():
    with pytest.raises(ValueError):
        implied_vol(10.0, 100.0, 100.0, 1.0, 0.05, option_type="strangle")