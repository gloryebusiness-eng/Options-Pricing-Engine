"""
Correctness tests for the Black-Scholes European option pricer.

These tests assert no-arbitrage properties rather than hardcoded reference
values. An arbitrage relation must hold for every valid input regardless of
which model produced the price, so violating one is unambiguous evidence of
an implementation error -- not a tolerance or calibration issue.

Property-based tests use hypothesis to g       enerate adversarial inputs across the
full parameter space, including the extreme moneyness and near-expiry regions
where naive implementations fail numerically.
"""

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ope.models.black_scholes import bs_price

# Parameter ranges span realistic markets: penny stocks to index levels,
# one week to five years, negative-to-elevated rates, and vol from a quiet
# mega-cap (10%) to a distressed single name (200%).
spot = st.floats(min_value=1.0, max_value=1000.0, allow_nan=False)
strike = st.floats(min_value=1.0, max_value=1000.0, allow_nan=False)
tenor = st.floats(min_value=0.02, max_value=5.0, allow_nan=False)
rate = st.floats(min_value=-0.01, max_value=0.15, allow_nan=False)
vol = st.floats(min_value=0.05, max_value=2.0, allow_nan=False)


@given(S=spot, K=strike, T=tenor, r=rate, sigma=vol)
@settings(deadline=None, max_examples=500)
def test_put_call_parity(S, K, T, r, sigma):
    """C - P = S - K*exp(-rT) for European options on a non-dividend payer.

    This is a static replication argument, not a modelling assumption: a long
    call plus a short put pays S_T - K at expiry in every state of the world,
    which is the payoff of holding the stock financed by a zero-coupon bond.
    Any deviation is a riskless arbitrage.
    """
    call = bs_price(S, K, T, r, sigma, option_type="call")
    put = bs_price(S, K, T, r, sigma, option_type="put")

    assert call - put == pytest.approx(S - K * math.exp(-r * T), rel=1e-9, abs=1e-9)


@given(S=spot, K=strike, T=tenor, r=rate, sigma=vol)
@settings(deadline=None, max_examples=500)
def test_call_within_arbitrage_bounds(S, K, T, r, sigma):
    """max(S - K*exp(-rT), 0) <= C <= S.

    Lower bound: the call dominates the discounted intrinsic value, and is
    never negative since exercise is optional. Upper bound: the right to buy
    the asset cannot exceed the value of the asset itself.
    """
    call = bs_price(S, K, T, r, sigma, option_type="call")
    lower = max(S - K * math.exp(-r * T), 0.0)

    assert call >= lower - 1e-9
    assert call <= S + 1e-9


@given(S=spot, K=strike, T=tenor, r=rate, sigma=vol)
@settings(deadline=None, max_examples=500)
def test_put_within_arbitrage_bounds(S, K, T, r, sigma):
    """max(K*exp(-rT) - S, 0) <= P <= K*exp(-rT).

    The put is capped by the present value of the strike, which is the most
    it can ever pay (realised when the underlying goes to zero).
    """
    put = bs_price(S, K, T, r, sigma, option_type="put")
    discounted_strike = K * math.exp(-r * T)
    lower = max(discounted_strike - S, 0.0)

    assert put >= lower - 1e-9
    assert put <= discounted_strike + 1e-9


@given(S=spot, K=strike, T=tenor, r=rate, sigma=vol)
@settings(deadline=None, max_examples=300)
def test_call_increasing_in_spot(S, K, T, r, sigma):
    """Delta > 0: a call is worth more when the underlying is worth more."""
    bumped = bs_price(S * 1.01, K, T, r, sigma, option_type="call")
    base = bs_price(S, K, T, r, sigma, option_type="call")

    assert bumped >= base - 1e-9


@given(S=spot, K=strike, T=tenor, r=rate, sigma=vol)
@settings(deadline=None, max_examples=300)
def test_call_increasing_in_volatility(S, K, T, r, sigma):
    """Vega > 0: the holder keeps unlimited upside but caps downside at the
    premium paid, so a wider terminal distribution is strictly more valuable.
    """
    higher_vol = bs_price(S, K, T, r, sigma * 1.1, option_type="call")
    base = bs_price(S, K, T, r, sigma, option_type="call")

    assert higher_vol >= base - 1e-9


def test_deep_itm_call_approaches_discounted_intrinsic():
    """As S/K -> infinity, exercise becomes certain and the call converges to
    a forward: S - K*exp(-rT). Optionality is worthless when there is no
    realistic path to finishing out of the money.
    """
    price = bs_price(S=1000.0, K=1.0, T=1.0, r=0.05, sigma=0.2, option_type="call")
    forward = 1000.0 - 1.0 * math.exp(-0.05)

    assert price == pytest.approx(forward, rel=1e-6)


def test_deep_otm_call_approaches_zero():
    """As S/K -> 0, the probability of finishing in the money vanishes."""
    price = bs_price(S=1.0, K=1000.0, T=0.25, r=0.05, sigma=0.2, option_type="call")

    assert price == pytest.approx(0.0, abs=1e-8)


def test_matches_reference_value():
    """Anchor against a published benchmark to catch a globally consistent
    error that still satisfies every arbitrage relation.

    S=100, K=100, T=1, r=5%, sigma=20% -> C = 10.4506, P = 5.5735.
    """
    call = bs_price(100.0, 100.0, 1.0, 0.05, 0.20, option_type="call")
    put = bs_price(100.0, 100.0, 1.0, 0.05, 0.20, option_type="put")

    assert call == pytest.approx(10.4506, rel=1e-4)
    assert put == pytest.approx(5.5735, rel=1e-4)


def test_rejects_unknown_option_type():
    """Fail loudly on invalid input rather than silently defaulting."""
    with pytest.raises(ValueError):
        bs_price(100.0, 100.0, 1.0, 0.05, 0.20, option_type="straddle")
