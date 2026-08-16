"""
Visualisation of the implied volatility curve.

The plot is the empirical result of the project: a curve that Black-Scholes
predicts should be a horizontal line, and is not.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from ope.data.chains import ChainSnapshot

__all__ = ["plot_smile"]


def plot_smile(
    smile: pd.DataFrame,
    snapshot: ChainSnapshot,
    output_path: str = "docs/volatility_smile.png",
) -> str:
    """Render the implied volatility curve against strike.

    Puts and calls are drawn in separate colours so the seam at spot is
    visible rather than smoothed over. Any discontinuity there is diagnostic:
    put-call parity implies the two types must return the same implied
    volatility at the same strike, so a visible step measures the size of an
    unmodelled effect -- here the omitted dividend yield.

    Only identifiable results are plotted. Drawing a curve through
    volatilities recovered where vega is negligible would present noise with
    the same visual authority as signal.
    """
    usable = smile[smile["status"] == "ok"].copy()
    puts = usable[usable["option_type"] == "put"]
    calls = usable[usable["option_type"] == "call"]

    fig, ax = plt.subplots(figsize=(11, 6.5))

    ax.plot(
        puts["strike"],
        puts["implied_vol"] * 100,
        marker="o",
        markersize=3,
        linewidth=1.4,
        color="#C0392B",
        label="Puts (OTM below spot)",
    )
    ax.plot(
        calls["strike"],
        calls["implied_vol"] * 100,
        marker="o",
        markersize=3,
        linewidth=1.4,
        color="#2471A3",
        label="Calls (OTM above spot)",
    )

    ax.axvline(
        snapshot.spot,
        color="#555555",
        linestyle="--",
        linewidth=1.1,
        label=f"Spot = {snapshot.spot:.2f}",
    )

    flat = usable["implied_vol"].median() * 100
    ax.axhline(
        flat,
        color="#7F8C8D",
        linestyle=":",
        linewidth=1.1,
        label="Black-Scholes prediction (constant sigma)",
    )

    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title(
        f"{snapshot.symbol} implied volatility skew  ·  "
        f"expiry {snapshot.expiry}  ·  T = {snapshot.tenor:.3f}y\n"
        f"Retrieved {snapshot.retrieved_at:%Y-%m-%d %H:%M} UTC",
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return output_path


def smile_summary(smile: pd.DataFrame, snapshot: ChainSnapshot) -> dict:
    """Quantify the skew for reporting.

    The 25-delta risk reversal is the standard desk measure of skew. This is
    a strike-based approximation of it, which is adequate for describing the
    shape but is not the quoted convention.
    """
    usable = smile[smile["status"] == "ok"]
    atm_row = usable.iloc[(usable["strike"] - snapshot.spot).abs().argsort()[:1]]
    atm_vol = float(atm_row["implied_vol"].iloc[0])

    low = usable[usable["moneyness"] > 1.15]["implied_vol"]
    high = usable[usable["moneyness"] < 0.88]["implied_vol"]

    return {
        "spot": snapshot.spot,
        "tenor_years": snapshot.tenor,
        "atm_vol": atm_vol,
        "downside_wing_vol": float(low.mean()) if len(low) else None,
        "upside_wing_vol": float(high.mean()) if len(high) else None,
        "skew_ratio": float(low.mean() / atm_vol) if len(low) else None,
        "strikes_solved": len(usable),
        "strikes_rejected": int((smile["status"] != "ok").sum()),
        "brent_fallback_rate": float((usable["method"] == "brent").mean()),
    }
