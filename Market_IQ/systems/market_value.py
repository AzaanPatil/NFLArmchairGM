"""
Market Value Analysis — Market_IQ/systems/market_value.py

Derives per-position AAV benchmarks from scraped OTC contract data,
then classifies individual contracts as ELITE / STAR / STARTER_1 /
STARTER_2 / BACKUP / MINIMUM and issues a market verdict
(OVERPAID / FAIR / VALUE) for any given AAV + position + age combo.

All thresholds are computed dynamically from live OTC data, so they
age well even as the market inflates.

Usage:
    from Market_IQ.systems.market_value import (
        build_benchmarks, classify_contract, market_verdict,
        positional_market_summary,
    )
    from Market_IQ.data_scraping.scrape_contracts import load_contracts_cache

    df = load_contracts_cache()
    benchmarks = build_benchmarks(df)
    tier = classify_contract("QB", aav=52.0, benchmarks=benchmarks)  # "ELITE"
    verdict = market_verdict("QB", aav=52.0, age=36, benchmarks=benchmarks)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Position normalisation  (mirrors Draft_IQ/systems/positional_value.py)
# ---------------------------------------------------------------------------

_POS_GROUP: dict[str, str] = {
    "QB": "QB",
    "WR": "WR",
    "RB": "RB", "HB": "RB", "FB": "RB",
    "TE": "TE",
    "OT": "OL", "OG": "OL", "C": "OL", "G": "OL", "T": "OL",
    "DT": "DL", "NT": "DL", "DE": "DL",
    "EDGE": "EDGE", "OLB": "EDGE",
    "LB": "LB", "ILB": "LB", "MLB": "LB",
    "CB": "CB",
    "S": "S", "FS": "S", "SS": "S", "DB": "S",
    "K": "K/P", "P": "K/P", "LS": "K/P",
}

ContractTier = Literal["ELITE", "STAR", "STARTER_1", "STARTER_2", "BACKUP", "MINIMUM"]
MarketVerdict = Literal["OVERPAID", "FAIR", "VALUE", "UNKNOWN"]

# Age at which each position typically declines sharply
_PEAK_AGE: dict[str, int] = {
    "QB": 35, "WR": 30, "RB": 28, "TE": 31,
    "OL": 32, "DL": 31, "EDGE": 30, "LB": 30,
    "CB": 30, "S": 31, "K/P": 40,
}

# Benchmark percentile thresholds that define each tier
# (computed from OTC data; these define which percentile rank = which tier)
_TIER_PERCENTILES = {
    "ELITE":     97,    # ~top 1-2 players at each position
    "STAR":      90,    # top 5-ish
    "STARTER_1": 75,    # clear starter, high-end
    "STARTER_2": 50,    # solid starter
    "BACKUP":    20,    # depth / swing player
    # MINIMUM = below 20th percentile
}

# How far above/below fair market (as % of expected AAV) to flip the verdict
_OVERPAID_THRESHOLD = 0.20   # 20% above fair-market → OVERPAID
_VALUE_THRESHOLD    = 0.15   # 15% below fair-market → VALUE


# ---------------------------------------------------------------------------
# Benchmark builder
# ---------------------------------------------------------------------------

def _normalise_pos(raw_pos: str) -> str:
    return _POS_GROUP.get(str(raw_pos).strip().upper(), "OTHER")


def build_benchmarks(df: pd.DataFrame) -> dict[str, dict]:
    """
    Compute per-position AAV benchmarks from a contracts DataFrame.

    Returns a dict keyed by position group:
        {
            "QB": {
                "median_aav": 38.5,
                "mean_aav": 35.2,
                "top5_avg": 52.0,
                "top10_avg": 46.5,
                "top25_avg": 41.2,
                "percentiles": {97: 54.5, 90: 49.2, 75: 43.1, 50: 38.5, 20: 12.3},
                "count": 64,
            },
            ...
        }
    """
    df = df.copy()
    df["pos_group"] = df["position"].apply(_normalise_pos)
    benchmarks: dict[str, dict] = {}

    for pos, group in df.groupby("pos_group"):
        aav = group["aav"].dropna()
        aav = aav[aav > 0].sort_values(ascending=False)
        if len(aav) < 3:
            continue

        pcts = {p: float(np.percentile(aav, p)) for p in _TIER_PERCENTILES.values()}

        benchmarks[pos] = {
            "median_aav":  round(float(aav.median()), 2),
            "mean_aav":    round(float(aav.mean()), 2),
            "top5_avg":    round(float(aav.head(5).mean()), 2),
            "top10_avg":   round(float(aav.head(10).mean()), 2),
            "top25_avg":   round(float(aav.head(25).mean()), 2),
            "percentiles": {k: round(v, 2) for k, v in pcts.items()},
            "count":       len(aav),
        }

    return benchmarks


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_contract(
    position: str,
    aav: float,
    benchmarks: dict[str, dict],
) -> ContractTier:
    """
    Classify a player's contract into a tier based on their AAV vs. the
    current positional market.
    """
    pos = _normalise_pos(position)
    bm = benchmarks.get(pos)
    if bm is None or aav <= 0:
        return "MINIMUM"

    pcts = bm["percentiles"]
    if aav >= pcts[_TIER_PERCENTILES["ELITE"]]:
        return "ELITE"
    if aav >= pcts[_TIER_PERCENTILES["STAR"]]:
        return "STAR"
    if aav >= pcts[_TIER_PERCENTILES["STARTER_1"]]:
        return "STARTER_1"
    if aav >= pcts[_TIER_PERCENTILES["STARTER_2"]]:
        return "STARTER_2"
    if aav >= pcts[_TIER_PERCENTILES["BACKUP"]]:
        return "BACKUP"
    return "MINIMUM"


def expected_aav(
    position: str,
    tier: ContractTier,
    benchmarks: dict[str, dict],
) -> float:
    """
    Return the typical AAV for a given tier at a position.
    Used as the 'fair market' reference in verdict logic.
    """
    pos = _normalise_pos(position)
    bm = benchmarks.get(pos, {})
    pcts = bm.get("percentiles", {})
    mapping: dict[str, int] = {
        "ELITE":     _TIER_PERCENTILES["ELITE"],
        "STAR":      _TIER_PERCENTILES["STAR"],
        "STARTER_1": _TIER_PERCENTILES["STARTER_1"],
        "STARTER_2": _TIER_PERCENTILES["STARTER_2"],
        "BACKUP":    _TIER_PERCENTILES["BACKUP"],
        "MINIMUM":   0,
    }
    return pcts.get(mapping.get(tier, 0), 0.0)


def market_verdict(
    position: str,
    aav: float,
    age: int,
    benchmarks: dict[str, dict],
    years_remaining: int = 0,
) -> MarketVerdict:
    """
    Issue a market verdict for a contract.

    Logic:
      1. Find fair-market AAV for the player's tier at this position.
      2. Apply an age discount if the player is past positional peak age.
      3. Compare signed AAV to age-adjusted fair market:
           > 20% above  → OVERPAID
           < 15% below  → VALUE
           otherwise    → FAIR
    """
    pos = _normalise_pos(position)
    if pos not in benchmarks:
        return "UNKNOWN"

    tier = classify_contract(position, aav, benchmarks)
    fair = expected_aav(position, tier, benchmarks)
    if fair <= 0:
        return "UNKNOWN"

    # Age discount: each year past peak subtracts 8% from fair-market expectation
    peak = _PEAK_AGE.get(pos, 31)
    if age and age > peak:
        discount = min(0.08 * (age - peak), 0.40)   # cap at 40%
        fair = fair * (1 - discount)

    ratio = aav / fair
    if ratio > (1 + _OVERPAID_THRESHOLD):
        return "OVERPAID"
    if ratio < (1 - _VALUE_THRESHOLD):
        return "VALUE"
    return "FAIR"


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def positional_market_summary(
    df: pd.DataFrame,
    benchmarks: dict[str, dict],
) -> pd.DataFrame:
    """
    For each position group in df, return a summary row:
        pos_group, count, median_aav, top5_avg, top10_avg, top25_avg
    """
    rows = []
    for pos, bm in sorted(benchmarks.items()):
        rows.append({
            "position":   pos,
            "contracts":  bm["count"],
            "median_aav": bm["median_aav"],
            "top5_avg":   bm["top5_avg"],
            "top10_avg":  bm["top10_avg"],
            "top25_avg":  bm["top25_avg"],
        })
    return pd.DataFrame(rows)


def annotate_contracts(
    df: pd.DataFrame,
    benchmarks: dict[str, dict],
) -> pd.DataFrame:
    """
    Return a copy of df with 'tier', 'verdict', and 'pos_group' columns added.
    """
    out = df.copy()
    out["pos_group"] = out["position"].apply(_normalise_pos)
    out["tier"] = out.apply(
        lambda r: classify_contract(r["position"], r["aav"], benchmarks), axis=1
    )
    out["verdict"] = out.apply(
        lambda r: market_verdict(
            r["position"], r["aav"],
            int(r["age"]) if pd.notna(r.get("age")) else 0,
            benchmarks,
            int(r.get("years_remaining", 0)) if pd.notna(r.get("years_remaining")) else 0,
        ),
        axis=1,
    )
    return out


def top_contracts_by_position(
    df: pd.DataFrame,
    position: str,
    n: int = 10,
) -> pd.DataFrame:
    """Return the top-N contracts by AAV for a given position group."""
    pos = _normalise_pos(position)
    filtered = df[df["position"].apply(_normalise_pos) == pos]
    cols = [c for c in ["player_name", "team", "position", "age", "years_remaining",
                         "aav", "guaranteed", "total_value", "tier", "verdict"]
            if c in filtered.columns]
    return (
        filtered.sort_values("aav", ascending=False)
        .head(n)[cols]
        .reset_index(drop=True)
    )
