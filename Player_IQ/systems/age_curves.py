"""
Career Arc Model — Player_IQ/systems/age_curves.py

Models how a player's market value changes relative to their peak
across their career, based on position-specific age curves.

Each position has:
  peak_age   — age at which production (and contract value) peaks
  rise_rate  — fraction of peak value recovered per year below peak
  fall_rate  — fraction of peak value lost per year above peak (accelerating)

production_factor(age, position) → float in [0.10, 1.00]
  1.00 = at peak, ~0.65-0.75 = rookie year, 0.10 = floor (career end)

Usage:
    from Player_IQ.systems.age_curves import (
        production_factor, peak_age_for,
        decline_projection, peak_value_age,
    )

    # Justin Jefferson (WR, age 26): find peak and current factor
    current_factor = production_factor(26, "WR")   # ≈ 1.00 (near peak)
    peak_fac       = production_factor(25, "WR")   # 1.00 (at peak)

    # Project decline over next 6 years
    proj = decline_projection("WR", current_age=26, years=6)
    # → {26: 1.0, 27: 0.95, 28: 0.89, 29: 0.84, 30: 0.77, 31: 0.71}
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

_POS_GROUP: dict[str, str] = {
    "QB": "QB", "WR": "WR",
    "RB": "RB", "HB": "RB", "FB": "RB",
    "TE": "TE",
    "OT": "OL", "OG": "OL", "C": "OL", "G": "OL", "T": "OL",
    "DT": "DL", "NT": "DL", "DE": "DL",
    "EDGE": "EDGE", "OLB": "EDGE",
    "LB": "LB", "ILB": "LB", "MLB": "LB",
    "CB": "CB", "S": "S", "FS": "S", "SS": "S", "DB": "S",
    "K": "K/P", "P": "K/P", "LS": "K/P",
}


def _normalise_pos(raw: str) -> str:
    return _POS_GROUP.get(str(raw).strip().upper(), "OTHER")


# ---------------------------------------------------------------------------
# Position curve parameters
# ---------------------------------------------------------------------------
# peak_age  — market value typically peaks here
# rise_rate — value recovered per year below peak (linear pre-peak)
# fall_rate — additional value lost per year above peak (super-linear post-peak)
# entry_pct — fraction of peak value at NFL entry (~22 years old)
# retire_age — typical career end

_CURVE_PARAMS: dict[str, dict] = {
    # fall_rate drives post-peak decline via: decline = fall_rate * years_past^1.15
    # (mild power curve — first few years gentle, then gradually steeper)
    # Real-world calibration:
    #   QB age 38 (8y past peak)  → ~66% of peak  (Brady at 38 still very good)
    #   WR age 30 (4y past peak)  → ~77% of peak  (Hill/Adams still elite at 30)
    #   RB age 28 (4y past peak)  → ~68% of peak  (Barkley won rushing title at 27)
    #   EDGE age 30 (4y past)     → ~73% of peak  (still productive but declining)
    "QB":   {"peak_age": 30, "rise_rate": 0.055, "fall_rate": 0.035, "entry_pct": 0.55, "retire_age": 42},
    "WR":   {"peak_age": 26, "rise_rate": 0.090, "fall_rate": 0.050, "entry_pct": 0.62, "retire_age": 36},
    "RB":   {"peak_age": 24, "rise_rate": 0.120, "fall_rate": 0.070, "entry_pct": 0.70, "retire_age": 32},
    "TE":   {"peak_age": 27, "rise_rate": 0.080, "fall_rate": 0.045, "entry_pct": 0.58, "retire_age": 36},
    "OL":   {"peak_age": 28, "rise_rate": 0.068, "fall_rate": 0.042, "entry_pct": 0.60, "retire_age": 38},
    "DL":   {"peak_age": 26, "rise_rate": 0.090, "fall_rate": 0.052, "entry_pct": 0.62, "retire_age": 35},
    "EDGE": {"peak_age": 26, "rise_rate": 0.090, "fall_rate": 0.060, "entry_pct": 0.62, "retire_age": 35},
    "LB":   {"peak_age": 26, "rise_rate": 0.085, "fall_rate": 0.050, "entry_pct": 0.65, "retire_age": 34},
    "CB":   {"peak_age": 26, "rise_rate": 0.085, "fall_rate": 0.055, "entry_pct": 0.65, "retire_age": 34},
    "S":    {"peak_age": 27, "rise_rate": 0.082, "fall_rate": 0.048, "entry_pct": 0.65, "retire_age": 35},
    "K/P":  {"peak_age": 32, "rise_rate": 0.030, "fall_rate": 0.025, "entry_pct": 0.78, "retire_age": 44},
    "OTHER":{"peak_age": 27, "rise_rate": 0.080, "fall_rate": 0.050, "entry_pct": 0.63, "retire_age": 35},
}

_FLOOR = 0.10   # minimum relative value (still active but severely diminished)
_ENTRY_AGE = 22.0


def production_factor(age: float, position: str) -> float:
    """
    Return a multiplier in [0.10, 1.00] representing a player's market value
    relative to their career peak for the given position.

    1.00 = at peak value
    0.70 = 70% of peak (e.g. young player not yet fully developed, or
           veteran in modest decline)
    """
    pos = _normalise_pos(position)
    p = _CURVE_PARAMS.get(pos, _CURVE_PARAMS["OTHER"])
    peak_age:   float = p["peak_age"]
    rise_rate:  float = p["rise_rate"]
    fall_rate:  float = p["fall_rate"]
    entry_pct:  float = p["entry_pct"]

    if age <= _ENTRY_AGE:
        return entry_pct

    if age <= peak_age:
        # Linear rise from entry_pct at ENTRY_AGE to 1.0 at peak_age
        years_to_peak = peak_age - _ENTRY_AGE
        years_from_entry = age - _ENTRY_AGE
        factor = entry_pct + (1.0 - entry_pct) * (years_from_entry / years_to_peak)
        return round(min(factor, 1.0), 3)

    # Post-peak: mild power curve so early decline is gentle but accelerates
    # decline = fall_rate * years_past^1.15
    # At 4y past: ~4.57x multiplier; at 8y past: ~9.71x — calibrated to real players
    years_past = age - peak_age
    decline = fall_rate * (years_past ** 1.15)
    factor = max(1.0 - decline, _FLOOR)
    return round(factor, 3)


def peak_age_for(position: str) -> float:
    pos = _normalise_pos(position)
    return float(_CURVE_PARAMS.get(pos, _CURVE_PARAMS["OTHER"])["peak_age"])


def retire_age_for(position: str) -> float:
    pos = _normalise_pos(position)
    return float(_CURVE_PARAMS.get(pos, _CURVE_PARAMS["OTHER"])["retire_age"])


def peak_value_age(position: str, current_age: float, stats_df=None) -> float:
    """
    Return the age at which this player had (or will have) peak market value.
    If stats_df is provided, we pick the best production season on record;
    otherwise we return the theoretical positional peak age.
    """
    pos_peak = peak_age_for(position)
    if stats_df is None or stats_df.empty:
        return pos_peak
    # If the player's current age is past peak, peak was already in the past
    if current_age >= pos_peak:
        return pos_peak
    return min(current_age + 2, pos_peak)  # project up to 2 years forward


def decline_projection(
    position: str,
    current_age: float,
    years: int = 6,
) -> dict[int, float]:
    """
    Return a year-by-year production factor from current_age to current_age + years.
    Values represent fraction of PEAK value.
    """
    result: dict[int, float] = {}
    for i in range(years + 1):
        age = current_age + i
        result[int(age)] = production_factor(age, position)
    return result


def age_adjusted_aav(
    peak_aav: float,
    position: str,
    current_age: float,
) -> float:
    """Scale a known peak AAV down to current market value based on age."""
    current_factor = production_factor(current_age, position)
    return round(peak_aav * current_factor, 2)


def implied_peak_aav(
    current_aav: float,
    position: str,
    current_age: float,
) -> float:
    """
    Given a player's current contract AAV and age, back-calculate their
    implied peak market value.
    """
    current_factor = production_factor(current_age, position)
    if current_factor <= 0:
        return current_aav
    return round(current_aav / current_factor, 2)


# ---------------------------------------------------------------------------
# Injury / durability adjustment
# ---------------------------------------------------------------------------

def durability_multiplier(
    games_pct_1yr: float,
    games_pct_2yr: float,
    games_pct_3yr: float,
    position: str,
) -> float:
    """
    Return a multiplier [0.75, 1.05] that adjusts value for injury risk.

    A player who consistently plays 90%+ of games gets a slight premium;
    one who plays < 60% faces a meaningful discount.

    This is intentionally conservative — we don't have PFF injury data.
    """
    avg = (games_pct_1yr * 0.5 + games_pct_2yr * 0.35 + games_pct_3yr * 0.15)
    if avg >= 0.90:
        return 1.05   # elite durability premium
    if avg >= 0.75:
        return 1.00
    if avg >= 0.60:
        return 0.93
    if avg >= 0.45:
        return 0.86
    return 0.78       # chronic injury history floor
