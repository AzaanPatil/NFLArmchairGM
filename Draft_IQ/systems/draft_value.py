"""
Jimmy Johnson Draft Value Chart — Draft_IQ/systems/draft_value.py

The original trade value chart popularized by Cowboys coach Jimmy Johnson
in the early 1990s. Still the most widely cited reference for evaluating
draft-pick trades, even though modern empirical alternatives exist.

Key values:
  Pick 1  → 3,000 pts   (a once-in-a-generation asset)
  Pick 32 →   590 pts   (late first)
  Pick 64 →   270 pts   (late second ≈ 9 % of pick 1)
  Pick 128 →   44 pts   (late fourth)
  Pick 224 →    1 pt    (final pick in standard chart)

Picks beyond 224 (NFL compensatory picks) return 1 point.

Usage:
    from Draft_IQ.systems.draft_value import (
        get_pick_value, trade_summary, add_jj_values, team_draft_capital,
    )

    # Evaluate a trade
    print(trade_summary(team_a_gives=[1], team_b_gives=[4, 37]))
    # → {'team_a': 3000, 'team_b': 2330, 'net': 670, 'winner': 'team_a'}

    # Annotate an actual draft DataFrame
    capsule_df = add_jj_values(capsule.actual_draft)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# The chart — picks 1-224 mapped to point values
# ---------------------------------------------------------------------------

_JJ_CHART: dict[int, float] = {
    # Round 1
    1: 3000, 2: 2600, 3: 2200, 4: 1800, 5: 1700, 6: 1600, 7: 1500, 8: 1400,
    9: 1350, 10: 1300, 11: 1250, 12: 1200, 13: 1150, 14: 1100, 15: 1050, 16: 1000,
    17:  950, 18:  900, 19:  875, 20:  850, 21:  800, 22:  780, 23:  760, 24:  740,
    25:  720, 26:  700, 27:  680, 28:  660, 29:  640, 30:  620, 31:  600, 32:  590,
    # Round 2
    33:  580, 34:  560, 35:  550, 36:  540, 37:  530, 38:  520, 39:  510, 40:  500,
    41:  490, 42:  480, 43:  470, 44:  460, 45:  450, 46:  440, 47:  430, 48:  420,
    49:  410, 50:  400, 51:  390, 52:  380, 53:  370, 54:  360, 55:  350, 56:  340,
    57:  330, 58:  320, 59:  310, 60:  300, 61:  292, 62:  284, 63:  276, 64:  270,
    # Round 3
    65:  265, 66:  260, 67:  255, 68:  250, 69:  245, 70:  240, 71:  235, 72:  230,
    73:  225, 74:  220, 75:  215, 76:  210, 77:  205, 78:  200, 79:  195, 80:  190,
    81:  185, 82:  182, 83:  179, 84:  176, 85:  173, 86:  170, 87:  167, 88:  164,
    89:  161, 90:  158, 91:  155, 92:  152, 93:  149, 94:  146, 95:  143, 96:  140,
    # Round 4
    97:  137, 98:  134, 99:  131, 100: 128, 101: 125, 102: 122, 103: 119, 104: 116,
    105: 113, 106: 110, 107: 107, 108: 104, 109: 101, 110:  98, 111:  95, 112:  92,
    113:  89, 114:  86, 115:  83, 116:  80, 117:  77, 118:  74, 119:  71, 120:  68,
    121:  65, 122:  62, 123:  59, 124:  56, 125:  53, 126:  50, 127:  47, 128:  44,
    # Round 5
    129:  42, 130:  40, 131:  38, 132:  36, 133:  34, 134:  32, 135:  30, 136:  28,
    137:  26, 138:  24, 139:  22, 140:  21, 141:  20, 142:  19, 143:  18, 144:  17,
    145:  16, 146:  15, 147:  14, 148:  13, 149:  12, 150:  11, 151:  10, 152:   9,
    153:   8, 154:   7, 155:   6, 156:   5, 157:   4, 158:   3, 159:   2, 160:   2,
    # Round 6
    161:   2, 162:   2, 163:   2, 164:   2, 165:   2, 166:   1, 167:   1, 168:   1,
    169:   1, 170:   1, 171:   1, 172:   1, 173:   1, 174:   1, 175:   1, 176:   1,
    177:   1, 178:   1, 179:   1, 180:   1, 181:   1, 182:   1, 183:   1, 184:   1,
    185:   1, 186:   1, 187:   1, 188:   1, 189:   1, 190:   1, 191:   1, 192:   1,
    # Round 7
    193:   1, 194:   1, 195:   1, 196:   1, 197:   1, 198:   1, 199:   1, 200:   1,
    201:   1, 202:   1, 203:   1, 204:   1, 205:   1, 206:   1, 207:   1, 208:   1,
    209:   1, 210:   1, 211:   1, 212:   1, 213:   1, 214:   1, 215:   1, 216:   1,
    217:   1, 218:   1, 219:   1, 220:   1, 221:   1, 222:   1, 223:   1, 224:   1,
}

# Picks beyond 224 (NFL compensatory selections) get a floor value of 1.
_JJ_FLOOR = 1.0


# ---------------------------------------------------------------------------
# Core lookup
# ---------------------------------------------------------------------------

def get_pick_value(pick: int) -> float:
    """Return the Jimmy Johnson chart value for an overall pick number."""
    return float(_JJ_CHART.get(pick, _JJ_FLOOR))


# ---------------------------------------------------------------------------
# Trade evaluation
# ---------------------------------------------------------------------------

def trade_summary(
    team_a_gives: Sequence[int],
    team_b_gives: Sequence[int],
    label_a: str = "Team A",
    label_b: str = "Team B",
) -> dict:
    """
    Evaluate a hypothetical pick trade using the JJ chart.

    Parameters
    ----------
    team_a_gives : list of overall pick numbers team A is sending
    team_b_gives : list of overall pick numbers team B is sending
    label_a / label_b : optional display names

    Returns
    -------
    dict with keys:
        label_a, label_b,
        value_a_gives, value_b_gives,
        net              — value_a_gives minus value_b_gives
                          positive → A is giving more (B wins the trade)
                          negative → B is giving more (A wins the trade)
        winner           — label of the team that received more value
        picks_a_gives    — annotated list of (pick, value) tuples
        picks_b_gives    — annotated list of (pick, value) tuples
    """
    a_annotated = [(p, get_pick_value(p)) for p in team_a_gives]
    b_annotated = [(p, get_pick_value(p)) for p in team_b_gives]

    val_a = sum(v for _, v in a_annotated)
    val_b = sum(v for _, v in b_annotated)
    net = val_a - val_b

    if net > 0:
        winner = label_b  # A gives more, B wins
    elif net < 0:
        winner = label_a  # B gives more, A wins
    else:
        winner = "Even"

    return {
        "label_a": label_a,
        "label_b": label_b,
        "value_a_gives": val_a,
        "value_b_gives": val_b,
        "net": net,
        "winner": winner,
        "picks_a_gives": a_annotated,
        "picks_b_gives": b_annotated,
    }


def print_trade(result: dict) -> None:
    """Pretty-print a trade_summary result."""
    a, b = result["label_a"], result["label_b"]
    print(f"\n{'-'*50}")
    print(f"  TRADE EVALUATION  (Jimmy Johnson Chart)")
    print(f"{'-'*50}")
    print(f"  {a} gives:")
    for pick, val in result["picks_a_gives"]:
        print(f"    Pick #{pick:<4}  {val:>6.0f} pts")
    print(f"  Total {a}: {result['value_a_gives']:,.0f} pts")

    print(f"\n  {b} gives:")
    for pick, val in result["picks_b_gives"]:
        print(f"    Pick #{pick:<4}  {val:>6.0f} pts")
    print(f"  Total {b}: {result['value_b_gives']:,.0f} pts")

    net = result["net"]
    sign = "+" if net >= 0 else ""
    print(f"\n  Net advantage {b}: {sign}{net:,.0f} pts")
    print(f"  Winner: {result['winner']}")
    print(f"{'-'*50}\n")


# ---------------------------------------------------------------------------
# DataFrame utilities
# ---------------------------------------------------------------------------

def add_jj_values(df: pd.DataFrame, pick_col: str = "pick") -> pd.DataFrame:
    """
    Return a copy of df with a 'jj_value' column added.
    pick_col should be the overall pick number column.
    """
    out = df.copy()
    out["jj_value"] = out[pick_col].apply(get_pick_value)
    return out


def team_draft_capital(df: pd.DataFrame, pick_col: str = "pick") -> pd.DataFrame:
    """
    Aggregate total Jimmy Johnson chart value by team for a given draft DataFrame.

    Returns a DataFrame sorted by descending total JJ value with columns:
        team, total_jj_value, pick_count, avg_jj_per_pick
    """
    valued = add_jj_values(df, pick_col=pick_col)
    result = (
        valued.groupby("team")
        .agg(
            total_jj_value=("jj_value", "sum"),
            pick_count=("jj_value", "count"),
        )
        .reset_index()
    )
    result["avg_jj_per_pick"] = (result["total_jj_value"] / result["pick_count"]).round(1)
    return result.sort_values("total_jj_value", ascending=False).reset_index(drop=True)


def pick_value_table(start: int = 1, end: int = 256) -> pd.DataFrame:
    """Return a DataFrame of pick → JJ value for the given range."""
    picks = range(start, end + 1)
    return pd.DataFrame({
        "pick": list(picks),
        "jj_value": [get_pick_value(p) for p in picks],
    })
