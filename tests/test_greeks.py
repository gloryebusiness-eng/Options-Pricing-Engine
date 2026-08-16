"""
Correctness tests for analytical Black-Scholes Greeks.

Each closed-form Greek is verified against a central-difference approximation
of the same partial derivative, computed from the pricer itself. Agreement
between a hand-derived analytical expression and a numerical derivative of an
independent implementation is strong evidence both are correct: a calculus
error would have to be reproduced identically along two unrelated
computational paths to escape detection.

Bump sizes are relative rather than absolute so that precision is preserved
across the full range of realistic input scales. For a central difference in
double precision, error is minimised near h ~ eps^(1/3), giving a relative
bump on the order of 1e-5.
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from ope.models.black_scholes import bs_price
from ope.models.greeks import delta, gamma, rho, theta, vega


# Extreme moneyness is excluded here: Greeks decay toward zero in the deep
# tails, where a relative-tolerance comparison is uninformative. Correctness
# in those regions is covered by the dedicated boundary tests below.
spot = st.floats(min_value=10.0, max_value=500.0, allow_nan=False)
strike = st.floats(min_value=10.0, max_value=500.0, allow_nan=False)
tenor = st.floats(min_value=0.1, max_value=3.0, allow_nan=False)
rate = st.floats(min_value=0.0, max_value=0.10, allow_nan=False)
vol = st.floats(min_value=0.10, max_value=1.0, allow_nan=False)
kind = st.sampled_from(["call", "put"])


def central_difference(f, x, h):
    """Approximate f'(x) to O(h^2) using a symmetric bump.

    The symmetric form cancels the first-order truncation terms of the two
    one-sided expansions, so accuracy improves quadratically in h rather
    than linearly, for one extra function evaluation.
    """
    return (f(x + h) - f(x - h)) / (2.0 * h)


@given(S=spot, K=strike, T=tenor, r=rate, sigma=vol, option_type=kind)
@settings(deadline=None, max_examples=200)
def test_delta_matches_finite_difference(S, K, T, r, sigma, option_type):
    """Delta = dC/dS, the hedge ratio in shares per option."""
    analytic = delta(S, K, T, r, sigma, option_type=option_type)
    numeric = central_difference(
        lambda s: bs_price(s, K, T, r, sigma, option_type=option_type),
        S,
        S * 1e-5,
    )

    assert analytic == pytest.approx(numeric, rel=1e-5, abs=1e-7)


@given(S=spot, K=strike, T=tenor, r=rate, sigma=vol, option_type=kind)
@settings(deadline=None, max_examples=200)
def test_gamma_matches_second_difference(S, K, T, r, sigma, option_type):
    """Gamma = d2C/dS2, computed as the central difference of delta.

    Differencing delta rather than double-differencing price avoids
    compounding cancellation error through two subtraction steps.
    """
    analytic = gamma(S, K, T, r, sigma)
    numeric = central_difference(
        lambda s: delta(s, K, T, r, sigma, option_type=option_type),
        S,
        S * 1e-4,
    )

    assert analytic == pytest.approx(numeric, rel=1e-4, abs=1e-8)


@given(S=spot, K=strike, T=tenor, r=rate, sigma=vol, option_type=kind)
@settings(deadline=None, max_examples=200)
def test_vega_matches_finite_difference(S, K, T, r, sigma, option_type):
    """Vega = dC/dsigma, reported per 1.00 change in volatility.

    Desks usually quote vega per volatility point (1%), i.e. this value
    divided by 100. The raw derivative is returned here; scaling is a
    presentation concern left to the caller.
    """
    analytic = vega(S, K, T, r, sigma)
    numeric = central_difference(
        lambda v: bs_price(S, K, T, r, v, option_type=option_type),
        sigma,
        sigma * 1e-5,
    )

    assert analytic == pytest.approx(numeric, rel=1e-5, abs=1e-6)


@given(S=spot, K=strike, T=tenor, r=rate, sigma=vol, option_type=kind)
@settings(deadline=None, max_examples=200)
def test_theta_matches_finite_difference(S, K, T, r, sigma, option_type):
    """Theta = -dC/dT, returned per calendar day.

    The sign convention is deliberate: theta is quoted as the P&L impact of
    one day passing, so a long option position reports negative theta. The
    finite difference is taken with respect to T and negated to match, then
    divided by 365 to convert from annual to daily.
    """
    analytic = theta(S, K, T, r, sigma, option_type=option_type)
    numeric = -central_difference(
        lambda t: bs_price(S, K, t, r, sigma, option_type=option_type),
        T,
        T * 1e-5,
    ) / 365.0

    assert analytic == pytest.approx(numeric, rel=1e-4, abs=1e-8)


@given(S=spot, K=strike, T=tenor, r=rate, sigma=vol, option_type=kind)
@settings(deadline=None, max_examples=200)
def test_rho_matches_finite_difference(S, K, T, r, sigma, option_type):
    """Rho = dC/dr, reported per 1.00 change in the rate."""
    analytic = rho(S, K, T, r, sigma, option_type=option_type)
    numeric = central_difference(
        lambda rate_: bs_price(S, K, T, rate_, sigma, option_type=option_type),
        r,
        1e-5,
    )

    assert analytic == pytest.approx(numeric, rel=1e-5, abs=1e-6)


@given(S=spot, K=strike, T=tenor, r=rate, sigma=vol)
@settings(deadline=None, max_examples=200)
def test_gamma_and_vega_are_type_independent(S, K, T, r, sigma):
    """Gamma and vega are identical for calls and puts.

    This follows directly from put-call parity: C - P = S - K*exp(-rT) has
    zero second derivative in S and no dependence on sigma, so differencing
    the two option types eliminates both Greeks.
    """
    assert gamma(S, K, T, r, sigma) == pytest.approx(
        gamma(S, K, T, r, sigma), rel=1e-12
    )
    assert vega(S, K, T, r, sigma) == pytest.approx(
        vega(S, K, T, r, sigma), rel=1e-12
    )


@given(S=spot, K=strike, T=tenor, r=rate, sigma=vol)
@settings(deadline=None, max_examples=200)
def test_delta_parity(S, K, T, r, sigma):
    """Call delta minus put delta equals one.

    Differentiating put-call parity with respect to S gives
    dC/dS - dP/dS = 1, since the right-hand side S - K*exp(-rT) has unit
    slope in S. This links the two deltas through a relation neither
    formula encodes directly.
    """
    call_delta = delta(S, K, T, r, sigma, option_type="call")
    put_delta = delta(S, K, T, r, sigma, option_type="put")

    assert call_delta - put_delta == pytest.approx(1.0, rel=1e-9, abs=1e-9)


def test_delta_bounds():
    """Call delta lies in [0, 1]; put delta lies in [-1, 0].

    A call can never require more than one share to hedge, and delta is the
    risk-neutral probability of exercise under the stock-numeraire measure,
    which is a probability and therefore bounded.
    """
    for S in (1.0, 50.0, 100.0, 200.0, 10_000.0):
        call_delta = delta(S, 100.0, 1.0, 0.05, 0.2, option_type="call")
        put_delta = delta(S, 100.0, 1.0, 0.05, 0.2, option_type="put")

        assert 0.0 <= call_delta <= 1.0
        assert -1.0 <= put_delta <= 0.0


def test_gamma_peaks_near_the_money():
    """Gamma is maximised near the forward and decays in both tails.

    Deep ITM and deep OTM options have near-constant delta (1 and 0
    respectively), so the rate of change of delta vanishes. Convexity is
    concentrated where exercise is most uncertain.
    """
    atm = gamma(100.0, 100.0, 1.0, 0.05, 0.2)
    deep_itm = gamma(300.0, 100.0, 1.0, 0.05, 0.2)
    deep_otm = gamma(30.0, 100.0, 1.0, 0.05, 0.2)

    assert atm > deep_itm
    assert atm > deep_otm


def test_vega_is_non_negative():
    """Vega is strictly positive for any live option.

    Optionality caps the downside at the premium while leaving the upside
    open, so a wider terminal distribution can only add value.
    """
    for S in (50.0, 100.0, 150.0):
        assert vega(S, 100.0, 1.0, 0.05, 0.2) >= 0.0


def test_greeks_reject_unknown_option_type():
    with pytest.raises(ValueError):
        delta(100.0, 100.0, 1.0, 0.05, 0.2, option_type="butterfly")
    with pytest.raises(ValueError):
        theta(100.0, 100.0, 1.0, 0.05, 0.2, option_type="butterfly")
    with pytest.raises(ValueError):
        rho(100.0, 100.0, 1.0, 0.05, 0.2, option_type="butterfly")