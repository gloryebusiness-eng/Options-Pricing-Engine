"""
Implied volatility curve construction from a market option chain.

Black-Scholes assumes a single constant volatility for the underlying. If
that assumption held, solving for implied volatility across every strike of
one expiry would return the same number each time, and the resulting curve
would be flat.

It is not flat. Equity index options exhibit a pronounced negative skew:
implied volatility rises as strikes fall below spot and flattens above. This
is the market's correction to a model it knows to be wrong. Real returns have
fatter tails and stronger negative skew than the log-normal distribution
Black-Scholes assumes, so downside protection is worth more than the model
says -- and volatility is the only free parameter available to express that.

The skew is not a permanent feature of markets. It was largely absent from
equity index options before October 1987 and has persisted since, which makes
it a fingerprint of a specific historical repricing of tail risk rather than
a mathematical necessity.

Strikes whose implied volatility is not identifiable are excluded rather than
plotted. Including them would draw a curve through numbers that carry no
volatility information, which is worse than showing a gap.
"""

from __future__ import annotations

import pandas as pd

from ope.data.chains import ChainSnapshot
from ope.vol.implied import ImpliedVolError, implied_vol

__all__ = ["build_smile"]


def build_smile(
    snapshot: ChainSnapshot,
    risk_free_rate: float = 0.04,
    option_type: str = "call",
) -> pd.DataFrame:
    """Solve implied volatility for every strike in a cleaned chain.

    Parameters
    ----------
    snapshot : A cleaned chain from fetch_chain.
    risk_free_rate : Continuously compounded rate matching the option tenor.
        A single flat rate is used here; a term structure from Treasury
        yields would be the production choice and is a documented
        simplification rather than an oversight.
    option_type : Must match the type retrieved in the snapshot.

    Returns
    -------
    One row per strike with the recovered volatility and its diagnostics.
    Strikes that failed to solve or produced non-identifiable results are
    retained with a status column, so the discard is visible rather than
    silent.
    """
    rows = []

    for _, row in snapshot.options.iterrows():
        strike = float(row["strike"])
        market_price = float(row["mid"])

        record = {
            "strike": strike,
            "log_moneyness": float(row["log_moneyness"]),
            "moneyness": snapshot.spot / strike,
            "option_type": row.get("option_type", option_type),
            "mid": market_price,
            "relative_spread": float(row["relative_spread"]),
        }

        try:
           result = implied_vol(
                market_price,
                snapshot.spot,
                strike,
                snapshot.tenor,
                risk_free_rate,
                option_type=row.get("option_type", option_type),
                return_diagnostics=True,
            )
        except ImpliedVolError as exc:
            record |= {
                "implied_vol": None,
                "vega": None,
                "method": None,
                "identifiable": False,
                "status": f"rejected: {exc.__class__.__name__}",
            }
        else:
            record |= {
                "implied_vol": result.volatility,
                "vega": result.vega,
                "method": result.method,
                "identifiable": result.identifiable,
                "status": "ok" if result.identifiable else "not identifiable",
            }

        rows.append(record)

    return pd.DataFrame(rows)