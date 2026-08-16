"""
Market option chain retrieval and cleaning.

This is the only module in the package that performs I/O. Pricing kernels and
the volatility solver operate on numbers alone, which keeps them vectorised,
fast to test, and independent of any particular data vendor. Swapping yfinance
for a paid feed touches this file and nothing else.

Quote quality is the dominant practical problem. Retail option data contains
stale prints, zero-bid strikes, crossed markets, and contracts that have not
traded in weeks. Feeding these into a solver produces implied volatilities
that are numerically valid and economically meaningless. Filtering is
therefore part of the data contract, not an optional refinement, and every
filter applied is recorded so the discard rate is visible rather than silent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

__all__ = ["fetch_chain", "ChainSnapshot", "clean_chain"]

# Contracts with wider relative spreads carry little information: the mid is
# a poor estimate of fair value when the bid and ask disagree by more than
# this fraction of the mid.
MAX_RELATIVE_SPREAD = 0.5

# Strikes beyond this distance from spot have negligible vega and their
# implied volatility is not identifiable.
MIN_MONEYNESS = 0.7
MAX_MONEYNESS = 1.3

MIN_OPEN_INTEREST = 10

# Front-month options are distorted by pin risk, tick-size discretisation
# relative to small premiums, and imminent scheduled events. A tenor of a
# month or more is far enough out that the smile reflects the market's view
# of the return distribution rather than expiry mechanics.
DEFAULT_TARGET_TENOR_DAYS = 45


@dataclass
class ChainSnapshot:
    """A cleaned option chain at a point in time.

    Attributes
    ----------
    symbol : Underlying ticker.
    spot : Underlying price at retrieval.
    expiry : Expiration date of this chain.
    tenor : Time to expiry in years, computed on a calendar-day basis.
    options : Cleaned quotes, one row per contract.
    retrieved_at : UTC timestamp of retrieval, recorded because implied
        volatility is only interpretable relative to the spot that produced it.
    filter_log : Rows discarded by each filter, in application order.
    """

    symbol: str
    spot: float
    expiry: str
    tenor: float
    options: pd.DataFrame
    retrieved_at: datetime
    filter_log: dict[str, int] = field(default_factory=dict)

    @property
    def retention_rate(self) -> float:
        """Fraction of the raw chain surviving all filters."""
        discarded = sum(self.filter_log.values())
        total = len(self.options) + discarded
        return len(self.options) / total if total else 0.0


def clean_chain(
    raw: pd.DataFrame,
    spot: float,
    option_type: str = "call",
    max_relative_spread: float = MAX_RELATIVE_SPREAD,
    min_moneyness: float = MIN_MONEYNESS,
    max_moneyness: float = MAX_MONEYNESS,
    min_open_interest: int = MIN_OPEN_INTEREST,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove quotes that cannot support a meaningful implied volatility.

    Filters are applied in order and each records its discard count, so the
    attrition is auditable. A chain retaining very few rows is a signal about
    the underlying's liquidity, not a silent success.

    The mid price is used rather than the last trade. Last trade can be hours
    or days stale on illiquid strikes, while the mid reflects where a market
    maker is currently willing to transact.
    """
    log: dict[str, int] = {}
    df = raw.copy()

    before = len(df)
    df = df[(df["bid"] > 0) & (df["ask"] > 0)]
    log["zero_or_missing_quote"] = before - len(df)

    before = len(df)
    df = df[df["ask"] > df["bid"]]
    log["crossed_or_locked"] = before - len(df)

    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spread"] = df["ask"] - df["bid"]
    df["relative_spread"] = df["spread"] / df["mid"]

    before = len(df)
    df = df[df["relative_spread"] <= max_relative_spread]
    log["spread_too_wide"] = before - len(df)

    df["moneyness"] = spot / df["strike"]

    before = len(df)
    df = df[(df["moneyness"] >= min_moneyness) & (df["moneyness"] <= max_moneyness)]
    log["outside_moneyness_band"] = before - len(df)

    if "openInterest" in df.columns:
        before = len(df)
        df = df[df["openInterest"].fillna(0) >= min_open_interest]
        log["insufficient_open_interest"] = before - len(df)

    before = len(df)
    df = _drop_vertical_spread_violations(df, option_type)
    log["vertical_spread_arbitrage"] = before - len(df)
    df["log_moneyness"] = np.log(df["strike"] / spot)

    return df.sort_values("strike").reset_index(drop=True), log


def fetch_chain(
    symbol: str,
    target_tenor_days: int = DEFAULT_TARGET_TENOR_DAYS,
    option_type: str = "call",
) -> ChainSnapshot:
    """Retrieve and clean one expiry of an option chain.

    Parameters
    ----------
    target_tenor_days : Selects the listed expiry closest to this many days
        out. Chosen by tenor rather than by index because listing schedules
        vary -- SPY lists weekly and daily expiries, so a positional index
        gives an unpredictable tenor across underlyings.

    Notes
    -----
    Tenor is computed on calendar days. Trading-day conventions (252 days)
    are common for realised volatility; the distinction matters for short
    tenors and is left explicit rather than hidden in a constant.
    """
    import yfinance as yf

    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    ticker = yf.Ticker(symbol)
    expiries = ticker.options

    if not expiries:
        raise ValueError(f"no listed option expiries found for {symbol!r}")

    now_naive = pd.Timestamp.now().normalize()
    days_out = np.array([(pd.Timestamp(e) - now_naive).days for e in expiries], dtype=float)
    expiry = expiries[int(np.argmin(np.abs(days_out - target_tenor_days)))]

    chain = ticker.option_chain(expiry)
    raw = chain.calls if option_type == "call" else chain.puts

    history = ticker.history(period="1d")
    if history.empty:
        raise ValueError(f"no price history available for {symbol!r}")
    spot = float(history["Close"].iloc[-1])

    now = datetime.now(timezone.utc)
    expiry_date = pd.Timestamp(expiry).tz_localize("UTC")
    tenor = max((expiry_date - now).days / 365.0, 1.0 / 365.0)

    cleaned, log = clean_chain(raw, spot)

    return ChainSnapshot(
        symbol=symbol,
        spot=spot,
        expiry=expiry,
        tenor=tenor,
        options=cleaned,
        retrieved_at=now,
        filter_log=log,
    )


def _drop_vertical_spread_violations(df: pd.DataFrame, option_type: str) -> pd.DataFrame:
    """Remove quotes inconsistent with neighbouring strikes.

    A call price must be non-increasing in strike, and may fall by at most
    the strike increment: -1 <= dC/dK <= 0. Violating the first bound is a
    call spread arbitrage; violating the second is a box arbitrage. Puts
    obey the mirrored constraint.

    Stale quotes on illiquid strikes violate these routinely. The spread and
    open-interest filters do not catch them, because each quote is
    individually plausible and only the relationship between adjacent
    strikes reveals the staleness.
    """
    if len(df) < 2:
        return df

    df = df.sort_values("strike").reset_index(drop=True)
    dk = df["strike"].diff()
    dc = df["mid"].diff()
    slope = dc / dk

    if option_type == "call":
        valid = (slope >= -1.05) & (slope <= 0.0)
    else:
        valid = (slope >= 0.0) & (slope <= 1.05)

    return df[valid.fillna(True)].reset_index(drop=True)


def fetch_otm_chain(
    symbol: str,
    target_tenor_days: int = DEFAULT_TARGET_TENOR_DAYS,
) -> ChainSnapshot:
    """Retrieve a chain using out-of-the-money contracts on both wings.

    Puts are taken below spot and calls above it, so every retained contract
    is out of the money and its entire premium is time value. In-the-money
    options carry most of their price as intrinsic value, leaving the
    volatility-bearing component as a small residual that quote error
    swamps -- which is why an implied volatility curve built from ITM
    contracts is dominated by microstructure noise rather than by the
    market's view of the return distribution.

    Put-call parity implies the two option types give identical implied
    volatility for the same strike. In practice the out-of-the-money side is
    where the liquidity is, so it is where the parity relation actually holds
    in observed prices.
    """
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    expiries = ticker.options
    if not expiries:
        raise ValueError(f"no listed option expiries found for {symbol!r}")

    now_naive = pd.Timestamp.now().normalize()
    days_out = np.array([(pd.Timestamp(e) - now_naive).days for e in expiries], dtype=float)
    expiry = expiries[int(np.argmin(np.abs(days_out - target_tenor_days)))]

    chain = ticker.option_chain(expiry)

    history = ticker.history(period="1d")
    if history.empty:
        raise ValueError(f"no price history available for {symbol!r}")
    spot = float(history["Close"].iloc[-1])

    now = datetime.now(timezone.utc)
    expiry_date = pd.Timestamp(expiry).tz_localize("UTC")
    tenor = max((expiry_date - now).days / 365.0, 1.0 / 365.0)

    puts, put_log = clean_chain(chain.puts[chain.puts["strike"] < spot], spot, "put")
    calls, call_log = clean_chain(chain.calls[chain.calls["strike"] >= spot], spot, "call")

    puts["option_type"] = "put"
    calls["option_type"] = "call"

    combined = (
        pd.concat([puts, calls], ignore_index=True).sort_values("strike").reset_index(drop=True)
    )

    merged_log = {f"put_{k}": v for k, v in put_log.items()}
    merged_log |= {f"call_{k}": v for k, v in call_log.items()}

    return ChainSnapshot(
        symbol=symbol,
        spot=spot,
        expiry=expiry,
        tenor=tenor,
        options=combined,
        retrieved_at=now,
        filter_log=merged_log,
    )
