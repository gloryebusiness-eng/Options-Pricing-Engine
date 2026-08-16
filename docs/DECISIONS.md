# Design decisions

Format: what was decided, what alternatives were considered, why this one.

## D-001: src/ layout with an installable package
Alternative: flat scripts at repo root.
Chose src/ so tests import the installed package, not stray local files.
Prevents "works on my machine" import-shadowing bugs.

## D-002: Pricing kernels are pure functions with no I/O
Alternative: classes holding market data as state.
Pure kernels vectorize over NumPy arrays, test in microseconds, and let
numerical methods be swapped behind one interface.

## D-003: Validate against no-arbitrage properties, not reference tables
Alternative: assert against hardcoded values from a textbook.
Property-based tests (hypothesis) generate hundreds of adversarial inputs and
assert relations that must hold for any correct pricer -- put-call parity,
arbitrage bounds, monotonicity in spot and vol. Reference values are retained
as a single anchor to catch globally consistent errors that still satisfy
every arbitrage relation.


## D-005: Compute the put with N(-d) rather than 1 - N(d)
Alternative: derive the put from the call via put-call parity, or use 1-N(d).
For large d, N(d) rounds to 1.0 and 1-N(d) loses all significant digits to
catastrophic cancellation. N(-d) evaluates the tail directly and stays
accurate into the far tails, which matters for deep OTM strikes.

## D-006: Handle sigma*sqrt(T)=0 as an explicit branch
Alternative: rely on limiting behaviour of the normal CDF.
When total uncertainty is zero the terminal price is deterministic and the
option is worth discounted intrinsic value. Branching explicitly avoids a
0/0 in d1 and makes the degenerate case a documented behaviour rather than
an accident of floating-point handling.

## D-007: Verify analytical Greeks against finite differences
Alternative: assert against published reference values per Greek.
Each closed-form Greek is compared to a central-difference derivative of the
pricer. Agreement between a hand-derived expression and a numerical
derivative of an independent implementation is strong correctness evidence:
a calculus error would have to reproduce identically along two unrelated
paths. Relative bump sizes near 1e-5 balance truncation against cancellation.

## D-008: Theta returned per calendar day, vega and rho per unit
Alternative: return all Greeks as raw annualised derivatives.
Theta is divided by 365 to match desk quoting, where it represents one day's
decay and is negative for long options. Vega and rho are left per unit change
so they compare directly against finite differences; per-point scaling is a
presentation concern.

## D-011: Convergence assessed on the sigma step, not the price residual
Alternative: absolute tolerance on |BS(sigma) - market_price|.
A price tolerance is not scale-free. For a deep OTM option priced at 1e-55,
any sigma satisfies a 1e-8 price tolerance on the first evaluation, and the
solver returns its initial guess unchanged. The change in sigma between
iterations asks the right question -- has the answer stopped moving -- and
behaves identically across moneyness.

## D-012: Report identifiability rather than returning a confident number
Alternative: return the root and let callers assume it is meaningful.
Vega is the condition number of the inversion, since d(sigma)/d(price) is
1/vega. For deep wings vega falls to ~1e-57, so every volatility in a wide
range reproduces the observed price to machine precision and no implied
volatility exists to recover. The solver returns the root but flags it, since
a wrong number is indistinguishable from a right one downstream. Surface
construction filters on this flag.

Discovered when a round-trip test failed in the deep wings. The test was
asserting a property the problem does not have; the fix was to correct the
test's premise and add the missing diagnostic, not to loosen the tolerance.

## D-013: Build the smile from out-of-the-money contracts only
Alternative: use calls across all strikes.
In-the-money options hold most of their premium as intrinsic value, so the
time value carrying the volatility signal is a small residual that quote
error dominates. Observed on live SPY data: ITM call quotes violated the
vertical spread bound -1 <= dC/dK <= 0 by a factor of five, and the recovered
IV curve sloped the wrong way across the ITM wing. Puts below spot and calls
above it keep every contract's premium entirely time value.

## D-014: Filter on vertical spread arbitrage across adjacent strikes
Alternative: rely on per-quote spread and open-interest filters.
Individually plausible quotes can be jointly inconsistent. Only the relation
between neighbouring strikes reveals staleness, and violations of the
monotonicity bounds are direct arbitrage rather than marginal mispricing.

## D-015: Dividends are not modelled (known bias)
The pricer assumes a non-dividend-paying underlying. SPY yields roughly 1.1%,
which depresses calls and lifts puts, tilting recovered implied volatilities
systematically. Merton's q extension is the correction; recording the bias
explicitly rather than presenting the surface as unbiased.