"""
Positional Value Analysis — Draft_IQ/systems/positional_value.py

Answers questions like:
  • Which positions get drafted earliest on average?
  • How much total draft capital (JJ chart) do teams invest in each position?
  • What % of each round goes to each position?
  • Did a specific team's positional investment in a given year match
    what the league typically does?

All analysis is derived from the locally cached actual draft CSVs
(Draft_IQ/data/raw/drafts/*.csv), so no re-scraping is needed.

Usage:
    from Draft_IQ.systems.positional_value import (
        load_historical_picks,
        positional_avg_pick,
        positional_capital,
        positional_frequency_by_round,
        year_vs_historical,
        team_positional_breakdown,
    )

    hist = load_historical_picks()           # 3,836 picks, 2010-2024
    print(positional_avg_pick(hist))
    print(positional_capital(hist))
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Draft_IQ.systems.draft_value import add_jj_values, get_pick_value


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_DRAFTS_CACHE = Path(__file__).parent.parent / "data" / "raw" / "drafts"


def load_historical_picks(start: int = 2010, end: int = 2024) -> pd.DataFrame:
    """
    Load cached actual draft picks for the given year range.
    Returns a combined DataFrame; silently skips missing years.
    """
    frames: list[pd.DataFrame] = []
    for year in range(start, end + 1):
        p = _DRAFTS_CACHE / f"{year}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(
            f"No draft cache found in {_DRAFTS_CACHE}. "
            "Run the actual draft scraper first."
        )
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Position normalisation
# ---------------------------------------------------------------------------

_POS_GROUP: dict[str, str] = {
    # Quarterback
    "QB": "QB",
    # Running backs
    "RB": "RB", "HB": "RB", "FB": "RB",
    # Wide receivers / tight ends
    "WR": "WR", "TE": "TE",
    # Offensive line
    "OT": "OL", "OG": "OL", "C": "OL", "G": "OL", "OL": "OL", "T": "OL",
    # Defensive line (interior)
    "DT": "DL", "NT": "DL", "DE": "DL", "DL": "DL",
    # Edge rushers (some sites use EDGE, some use DE, some use OLB)
    "EDGE": "EDGE", "OLB": "EDGE",
    # Linebackers
    "LB": "LB", "ILB": "LB", "MLB": "LB",
    # Secondary
    "CB": "CB", "S": "S", "FS": "S", "SS": "S", "DB": "S",
    # Special teams / other
    "K": "K/P", "P": "K/P", "LS": "K/P",
}

_DISPLAY_ORDER = ["QB", "WR", "RB", "TE", "OL", "DL", "EDGE", "LB", "CB", "S", "K/P"]


def _normalise_position(pos: str) -> str:
    """Map raw position string to a canonical group."""
    return _POS_GROUP.get(str(pos).strip().upper(), "OTHER")


def _add_pos_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pos_group"] = out["position"].apply(_normalise_position)
    return out


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def positional_avg_pick(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each position group, compute average and median draft slot,
    plus standard deviation and total pick count across all years in df.

    Returns rows sorted by avg_pick (earliest drafted first).
    """
    enriched = _add_pos_group(df)
    result = (
        enriched.groupby("pos_group")
        .agg(
            avg_pick=("pick", "mean"),
            median_pick=("pick", "median"),
            std_pick=("pick", "std"),
            count=("pick", "count"),
        )
        .reset_index()
    )
    result["avg_pick"] = result["avg_pick"].round(1)
    result["median_pick"] = result["median_pick"].round(1)
    result["std_pick"] = result["std_pick"].round(1)

    # Sort by display order, then fall back to avg_pick for unknowns
    order_map = {pos: i for i, pos in enumerate(_DISPLAY_ORDER)}
    result["_sort"] = result["pos_group"].map(lambda p: order_map.get(p, 99))
    return (
        result.sort_values(["_sort", "avg_pick"])
        .drop(columns=["_sort"])
        .reset_index(drop=True)
    )


def positional_capital(df: pd.DataFrame, pick_col: str = "pick") -> pd.DataFrame:
    """
    For each position group, sum the total Jimmy Johnson chart value invested
    across all picks in df. Shows where the league directs its draft capital.

    Returns rows sorted by total JJ value (highest investment first).
    """
    enriched = _add_pos_group(add_jj_values(df, pick_col=pick_col))
    result = (
        enriched.groupby("pos_group")
        .agg(
            total_jj=("jj_value", "sum"),
            pick_count=("jj_value", "count"),
            avg_jj_per_pick=("jj_value", "mean"),
        )
        .reset_index()
    )
    result["total_jj"] = result["total_jj"].round(0).astype(int)
    result["avg_jj_per_pick"] = result["avg_jj_per_pick"].round(1)
    result["pct_of_total"] = (
        (result["total_jj"] / result["total_jj"].sum() * 100).round(1)
    )
    return result.sort_values("total_jj", ascending=False).reset_index(drop=True)


def positional_frequency_by_round(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-tabulation: position group × round, cell = % of that round's picks.

    Each column (round) sums to 100%, showing what share of each round
    goes to each position group.
    """
    enriched = _add_pos_group(df)
    counts = pd.crosstab(enriched["pos_group"], enriched["round"])

    # Convert to percentages within each round (column-wise)
    pct = counts.div(counts.sum(axis=0), axis=1).mul(100).round(1)

    # Reindex rows to canonical order
    existing = [p for p in _DISPLAY_ORDER if p in pct.index]
    other = [p for p in pct.index if p not in existing]
    pct = pct.reindex(existing + other)

    # Rename columns to "R1", "R2", etc.
    pct.columns = [f"R{c}" for c in pct.columns]
    return pct.fillna(0)


def year_vs_historical(
    year_df: pd.DataFrame,
    hist_df: pd.DataFrame,
    pick_col: str = "pick",
) -> pd.DataFrame:
    """
    Compare one year's positional capital investment against the historical average.

    Returns a DataFrame with columns:
        pos_group, year_jj, year_pct, hist_avg_pct, delta_pct
    where delta_pct > 0 means the year over-invested vs. history.
    """
    def _capital_pct(d: pd.DataFrame) -> pd.Series:
        e = _add_pos_group(add_jj_values(d, pick_col=pick_col))
        totals = e.groupby("pos_group")["jj_value"].sum()
        return totals / totals.sum() * 100

    year_pct = _capital_pct(year_df)
    hist_pct = _capital_pct(hist_df)

    all_pos = set(year_pct.index) | set(hist_pct.index)
    rows = []
    for pos in all_pos:
        yp = year_pct.get(pos, 0.0)
        hp = hist_pct.get(pos, 0.0)

        # Actual JJ value this year for this position
        year_enriched = _add_pos_group(add_jj_values(year_df, pick_col=pick_col))
        year_jj = float(year_enriched[year_enriched["pos_group"] == pos]["jj_value"].sum())

        rows.append({
            "pos_group": pos,
            "year_jj": round(year_jj),
            "year_pct": round(yp, 1),
            "hist_avg_pct": round(hp, 1),
            "delta_pct": round(yp - hp, 1),
        })

    result = pd.DataFrame(rows)
    order_map = {pos: i for i, pos in enumerate(_DISPLAY_ORDER)}
    result["_sort"] = result["pos_group"].map(lambda p: order_map.get(p, 99))
    return (
        result.sort_values("_sort")
        .drop(columns=["_sort"])
        .reset_index(drop=True)
    )


def team_positional_breakdown(
    df: pd.DataFrame,
    team: str,
    pick_col: str = "pick",
) -> pd.DataFrame:
    """
    For one team, return their positional draft capital breakdown across all
    years in df. Useful for spotting long-term positional biases.
    """
    team_df = df[df["team"].str.upper() == team.upper()]
    if team_df.empty:
        raise ValueError(f"No picks found for team '{team}' in the provided data.")
    enriched = _add_pos_group(add_jj_values(team_df, pick_col=pick_col))
    result = (
        enriched.groupby("pos_group")
        .agg(
            total_jj=("jj_value", "sum"),
            pick_count=("jj_value", "count"),
        )
        .reset_index()
    )
    result["total_jj"] = result["total_jj"].round(0).astype(int)
    result["pct_of_total"] = (result["total_jj"] / result["total_jj"].sum() * 100).round(1)
    return result.sort_values("total_jj", ascending=False).reset_index(drop=True)


def positional_round_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each position group, show what % of their own picks come from each round.
    Rows sum to 100%. Reveals which positions are primarily found in early vs. late rounds.
    """
    enriched = _add_pos_group(df)
    counts = pd.crosstab(enriched["pos_group"], enriched["round"])
    # Normalise row-wise (per position)
    pct = counts.div(counts.sum(axis=1), axis=0).mul(100).round(1)
    existing = [p for p in _DISPLAY_ORDER if p in pct.index]
    other = [p for p in pct.index if p not in existing]
    pct = pct.reindex(existing + other)
    pct.columns = [f"R{c}" for c in pct.columns]
    return pct.fillna(0)
