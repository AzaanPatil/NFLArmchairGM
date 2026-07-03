"""
Trade Analyzer — Market_IQ/systems/trade_analyzer.py

Converts players to Jimmy Johnson pick-equivalent values so that any
trade — player-for-player, player-for-picks, or picks-for-picks — can
be evaluated on a single common scale.

Player → pick-equivalent logic:
  1. Look up the player's contract tier from live OTC data (ELITE / STAR / etc.)
  2. Map that tier to a base JJ value range for their position
  3. Apply an age discount (each year past positional peak = -8% value)
  4. Apply a contract status bonus for players on cheap rookie deals
  5. Compare both sides of the trade; flag fleeced if gap > threshold

Fleeced thresholds (mirrors how analysts use the JJ chart in practice):
  < 5%   difference → EVEN
  5-15%             → SLIGHT_WIN  (for the better side)
  15-30%            → CLEAR_WIN
  > 30%             → FLEECED

Usage:
    from Market_IQ.systems.trade_analyzer import evaluate_trade, player_pick_value

    result = evaluate_trade(
        side_a={
            "label": "Eagles",
            "players": [{"name": "A.J. Brown", "position": "WR", "age": 27,
                         "aav": 32.0, "years_remaining": 3, "rookie_deal": False}],
            "picks": [23],
        },
        side_b={
            "label": "Titans",
            "players": [],
            "picks": [10, 45],
        },
        benchmarks=benchmarks,
    )
    print_trade_result(result)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Draft_IQ.systems.draft_value import get_pick_value
from Market_IQ.systems.market_value import (
    classify_contract,
    _PEAK_AGE,
    _normalise_pos,
    ContractTier,
)


# ---------------------------------------------------------------------------
# Tier → base JJ value ranges
# The midpoint of each range is used as the player's base pick-equivalent.
# Ranges are defined so ELITE ≈ top-5 pick, STAR ≈ late 1st, etc.
# ---------------------------------------------------------------------------

_TIER_JJ: dict[str, tuple[float, float]] = {
    # tier: (min_jj, max_jj)
    "ELITE":     (1500, 3000),   # top-2 to top-10ish pick equivalent
    "STAR":      (600,  1500),   # late 1st / early 2nd
    "STARTER_1": (270,  600),    # 2nd round
    "STARTER_2": (140,  270),    # 3rd round
    "BACKUP":    (44,   140),    # 4th round
    "MINIMUM":   (1,    44),     # 5th round or later
}

# Position multipliers — premium positions are worth slightly more at equal tiers
# because they impact the game more on a per-snap basis
_POS_MULTIPLIER: dict[str, float] = {
    "QB":   1.30,
    "EDGE": 1.10,
    "CB":   1.05,
    "WR":   1.05,
    "DL":   1.00,
    "OL":   1.00,
    "S":    0.95,
    "LB":   0.95,
    "TE":   0.95,
    "RB":   0.85,   # RBs are historically undervalued in trades
    "K/P":  0.50,
}

_AGE_DISCOUNT_PER_YEAR = 0.08   # 8% per year past positional peak
_ROOKIE_DEAL_BONUS     = 1.25   # 25% premium for cheap rookie contracts
_FLEECED_THRESHOLD     = 0.30   # 30%+ net difference = FLEECED
_CLEAR_WIN_THRESHOLD   = 0.15
_SLIGHT_WIN_THRESHOLD  = 0.05


class PlayerSpec(TypedDict):
    name: str
    position: str
    age: int
    aav: float              # current AAV in millions
    years_remaining: int
    rookie_deal: bool       # True if on a cheap rookie contract


class TradeSide(TypedDict):
    label: str
    players: list[PlayerSpec]
    picks: list[int]        # overall pick numbers


# ---------------------------------------------------------------------------
# Core conversion: player → JJ pick-equivalent value
# ---------------------------------------------------------------------------

def player_pick_value(
    player: PlayerSpec,
    benchmarks: dict[str, dict],
) -> float:
    """
    Convert a player spec to a Jimmy Johnson pick-equivalent value.

    Steps:
      1. Classify contract tier from AAV vs. positional market
      2. Base JJ value = midpoint of tier's JJ range
      3. Apply positional multiplier
      4. Apply age discount for players past their positional peak
      5. Apply rookie deal bonus if applicable
    """
    pos = _normalise_pos(player["position"])
    tier: ContractTier = classify_contract(player["position"], player["aav"], benchmarks)

    lo, hi = _TIER_JJ.get(tier, (1, 44))
    base_jj = (lo + hi) / 2

    # Positional premium
    base_jj *= _POS_MULTIPLIER.get(pos, 1.0)

    # Age discount — each year past positional peak
    peak = _PEAK_AGE.get(pos, 31)
    age = player.get("age", peak)
    if age and age > peak:
        discount = min(_AGE_DISCOUNT_PER_YEAR * (age - peak), 0.40)
        base_jj *= (1 - discount)

    # Rookie deal bonus (player is producing starter/star output at backup price)
    if player.get("rookie_deal", False) and tier in ("STAR", "STARTER_1", "STARTER_2"):
        base_jj *= _ROOKIE_DEAL_BONUS

    return round(base_jj, 1)


def picks_total_value(picks: list[int]) -> float:
    return sum(get_pick_value(p) for p in picks)


# ---------------------------------------------------------------------------
# Trade evaluation
# ---------------------------------------------------------------------------

def evaluate_trade(
    side_a: TradeSide,
    side_b: TradeSide,
    benchmarks: dict[str, dict],
) -> dict:
    """
    Evaluate a trade between two sides.  Each side can include any
    combination of players and draft picks.

    Returns a result dict with:
        side_a / side_b: label, player_values, pick_values, total_value
        net              — side_a_total minus side_b_total
                          positive → side A gives more (side B wins)
                          negative → side B gives more (side A wins)
        winner           — label of the side that received more value
        verdict          — EVEN / SLIGHT_WIN / CLEAR_WIN / FLEECED
        notes            — list of human-readable observations
    """
    def _evaluate_side(side: TradeSide) -> dict:
        player_details = []
        for p in side["players"]:
            jj = player_pick_value(p, benchmarks)
            tier = classify_contract(p["position"], p["aav"], benchmarks)
            player_details.append({
                "name":     p["name"],
                "position": p["position"],
                "age":      p.get("age"),
                "tier":     tier,
                "jj_value": jj,
            })

        pick_details = [(pk, get_pick_value(pk)) for pk in side["picks"]]
        player_total = sum(d["jj_value"] for d in player_details)
        pick_total   = picks_total_value(side["picks"])

        return {
            "label":        side["label"],
            "players":      player_details,
            "picks":        pick_details,
            "player_total": round(player_total, 1),
            "pick_total":   round(pick_total, 1),
            "total":        round(player_total + pick_total, 1),
        }

    result_a = _evaluate_side(side_a)
    result_b = _evaluate_side(side_b)

    net = result_a["total"] - result_b["total"]   # positive → A gives more

    if abs(net) == 0:
        winner  = "Even"
        verdict = "EVEN"
    else:
        winner = result_b["label"] if net > 0 else result_a["label"]
        pct_diff = abs(net) / max(result_a["total"], result_b["total"])
        if pct_diff >= _FLEECED_THRESHOLD:
            verdict = "FLEECED"
        elif pct_diff >= _CLEAR_WIN_THRESHOLD:
            verdict = "CLEAR_WIN"
        elif pct_diff >= _SLIGHT_WIN_THRESHOLD:
            verdict = "SLIGHT_WIN"
        else:
            verdict = "EVEN"

    notes = _generate_notes(result_a, result_b, net, winner, verdict)

    return {
        "side_a":  result_a,
        "side_b":  result_b,
        "net":     round(net, 1),
        "winner":  winner,
        "verdict": verdict,
        "notes":   notes,
    }


def _generate_notes(ra: dict, rb: dict, net: float, winner: str, verdict: str) -> list[str]:
    notes: list[str] = []

    if verdict == "EVEN":
        notes.append("Both sides received roughly equal value.")
        return notes

    loser  = rb["label"] if net > 0 else ra["label"]
    margin = abs(net)
    notes.append(
        f"{winner} wins this trade by ~{margin:.0f} JJ pts "
        f"({verdict.replace('_', ' ').title()})."
    )

    # Flag if an elite player is being undervalued
    for side, other in [(ra, rb), (rb, ra)]:
        for p in side["players"]:
            if p["tier"] in ("ELITE", "STAR") and side["label"] == loser:
                notes.append(
                    f"Warning: {p['name']} ({p['tier']}) may be undervalued — "
                    "elite players are historically difficult to replace."
                )

    # Flag old player risk
    for side in (ra, rb):
        for p in side["players"]:
            pos = _normalise_pos(p["position"])
            peak = _PEAK_AGE.get(pos, 31)
            age = p.get("age") or 0
            if age and age >= peak + 2:
                notes.append(
                    f"{p['name']} is {age} years old — "
                    f"2+ years past typical peak for {p['position']}. "
                    "Age discount applied."
                )

    # Warn on RB premium
    for side in (ra, rb):
        for p in side["players"]:
            if _normalise_pos(p["position"]) == "RB" and p.get("tier") in ("ELITE", "STAR"):
                notes.append(
                    f"RB valuations carry extra uncertainty — "
                    "running backs are historically undervalued in trades."
                )

    return notes


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------

def print_trade_result(result: dict) -> None:
    def _side_block(s: dict) -> None:
        print(f"\n  {s['label']} gives:")
        for p in s["players"]:
            print(f"    {p['name']:<28} ({p['position']}, age {p['age']}, "
                  f"{p['tier']})  ~{p['jj_value']:.0f} JJ pts")
        for pick, val in s["picks"]:
            print(f"    Pick #{pick:<4}                            {val:.0f} JJ pts")
        print(f"  Total {s['label']}: {s['total']:,.0f} pts "
              f"(players {s['player_total']:.0f} + picks {s['pick_total']:.0f})")

    print(f"\n{'-'*60}")
    print("  TRADE EVALUATION  (JJ Chart + Market Value)")
    print(f"{'-'*60}")
    _side_block(result["side_a"])
    _side_block(result["side_b"])

    net = result["net"]
    winner = result["winner"]
    verdict = result["verdict"]
    sign = "+" if net >= 0 else ""
    print(f"\n  Net (Side A gives more): {sign}{net:,.0f} pts")
    print(f"  Winner:  {winner}")
    print(f"  Verdict: {verdict}")
    if result["notes"]:
        print(f"\n  Notes:")
        for note in result["notes"]:
            print(f"    - {note}")
    print(f"{'-'*60}\n")
