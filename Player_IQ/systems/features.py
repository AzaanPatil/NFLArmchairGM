"""
Feature Engineering — Player_IQ/systems/features.py

Joins contract records (from Market_IQ OTC cache) with player season
stats (from Player_IQ ESPN cache) to produce a training-ready feature
matrix for the GBR value model.

For each contract, we look up the player's stats from the season(s)
immediately before signing and compute per-game rates.  All features
are numeric; position is label-encoded.

Feature vector layout (35 features):
  [0]  pos_encoded           — integer 0-9
  [1]  age_at_signing
  [2]  draft_round           — 1-7, 8 = UDFA
  [3]  years_in_league
  [4]  games_pct_1yr         — fraction of 17-game season played
  [5]  games_pct_2yr         — 2-year rolling avg
  [6]  games_pct_3yr         — 3-year rolling avg
  # Passing
  [7]  pass_yds_pg_1yr
  [8]  pass_td_pg_1yr
  [9]  pass_int_pg_1yr
  [10] comp_pct_1yr
  [11] qbr_1yr
  # Rushing
  [12] rush_yds_pg_1yr
  [13] rush_td_pg_1yr
  [14] rush_ypc_1yr
  # Receiving
  [15] rec_yds_pg_1yr
  [16] rec_pg_1yr
  [17] rec_td_pg_1yr
  [18] rec_avg_1yr
  # Defense
  [19] sacks_pg_1yr
  [20] tfl_pg_1yr
  [21] tackles_pg_1yr
  [22] int_pg_1yr
  [23] pd_pg_1yr
  [24] ff_pg_1yr
  # 2-year rolling averages for key per-position stats
  [25] pass_yds_pg_2yr
  [26] rec_yds_pg_2yr
  [27] rush_yds_pg_2yr
  [28] sacks_pg_2yr
  # Peak season indicators
  [29] career_best_pass_yds_pg
  [30] career_best_rec_yds_pg
  [31] career_best_rush_yds_pg
  [32] career_best_sacks_pg

Target: aav (float, millions USD)
"""

from __future__ import annotations

import sys
import difflib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Position group mapping (mirrors Market_IQ)
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

_POS_ORDER = ["QB", "WR", "RB", "TE", "OL", "DL", "EDGE", "LB", "CB", "S", "K/P", "OTHER"]
_POS_ENCODE = {p: i for i, p in enumerate(_POS_ORDER)}

FEATURE_NAMES = [
    "pos_encoded", "age_at_signing", "draft_round", "years_in_league",
    "games_pct_1yr", "games_pct_2yr", "games_pct_3yr",
    "pass_yds_pg_1yr", "pass_td_pg_1yr", "pass_int_pg_1yr", "comp_pct_1yr", "qbr_1yr",
    "rush_yds_pg_1yr", "rush_td_pg_1yr", "rush_ypc_1yr",
    "rec_yds_pg_1yr", "rec_pg_1yr", "rec_td_pg_1yr", "rec_avg_1yr",
    "sacks_pg_1yr", "tfl_pg_1yr", "tackles_pg_1yr", "int_pg_1yr", "pd_pg_1yr", "ff_pg_1yr",
    "pass_yds_pg_2yr", "rec_yds_pg_2yr", "rush_yds_pg_2yr", "sacks_pg_2yr",
    "career_best_pass_yds_pg", "career_best_rec_yds_pg",
    "career_best_rush_yds_pg", "career_best_sacks_pg",
]


def _normalise_pos(raw: str) -> str:
    return _POS_GROUP.get(str(raw).strip().upper(), "OTHER")


def _normalise_name(name: str) -> str:
    """Lowercase, strip punctuation for fuzzy matching."""
    import re
    return re.sub(r"[^a-z ]", "", name.lower().strip())


def _match_player(name: str, stats_index: dict[str, list[str]]) -> Optional[str]:
    """
    Find the best matching player name in stats_index.
    stats_index maps normalised_name → [original_name, ...].
    Returns the original name of the best match, or None.
    """
    norm = _normalise_name(name)
    if norm in stats_index:
        return stats_index[norm][0]
    # Fuzzy fallback
    candidates = list(stats_index.keys())
    close = difflib.get_close_matches(norm, candidates, n=1, cutoff=0.82)
    if close:
        return stats_index[close[0]][0]
    return None


def _pg(df_row: pd.Series, col: str) -> float:
    """Per-game rate from a stats row, or 0 if column missing/games=0."""
    val = df_row.get(col, None)
    games = df_row.get("games", 0)
    if val is None or pd.isna(val) or games == 0:
        return 0.0
    return float(val) / float(games)


def _safe(df_row: pd.Series, col: str, default: float = 0.0) -> float:
    val = df_row.get(col, default)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return float(val)


# ---------------------------------------------------------------------------
# Per-season feature extractor
# ---------------------------------------------------------------------------

def _season_features(season_row: pd.Series) -> dict:
    """Extract per-game stat features from a single PlayerSeason row."""
    g = float(season_row.get("games", 0)) or 1  # avoid divide-by-zero

    def pg(col: str) -> float:
        v = season_row.get(col)
        if v is None or pd.isna(v):
            return 0.0
        return float(v) / g

    return {
        "games_pct":      min(float(season_row.get("games", 0)) / 17, 1.0),
        "pass_yds_pg":    pg("pass_yards"),
        "pass_td_pg":     pg("pass_tds"),
        "pass_int_pg":    pg("pass_ints"),
        "comp_pct":       _safe(season_row, "comp_pct"),
        "qbr":            _safe(season_row, "qbr"),
        "rush_yds_pg":    pg("rush_yards"),
        "rush_td_pg":     pg("rush_tds"),
        "rush_ypc":       _safe(season_row, "rush_avg"),
        "rec_yds_pg":     pg("rec_yards"),
        "rec_pg":         pg("receptions"),
        "rec_td_pg":      pg("rec_tds"),
        "rec_avg":        _safe(season_row, "rec_avg"),
        "sacks_pg":       pg("sacks"),
        "tfl_pg":         pg("tfl"),
        "tackles_pg":     pg("tackles"),
        "int_pg":         pg("interceptions"),
        "pd_pg":          pg("pass_defenses"),
        "ff_pg":          pg("forced_fumbles"),
    }


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_training_matrix(
    contracts_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    current_year: int = 2024,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Join contracts with stats and build (X, y, player_names).

    contracts_df columns required: player_name, position, age, aav, signed_year
    stats_df columns required:     player_name, season, games, <stat cols>

    Returns:
        X — (N, 33) float32 feature matrix
        y — (N,) float32 AAV targets (millions)
        names — list of player_name strings for diagnostics
    """
    # Build stats lookup: name → DataFrame of seasons indexed by season year
    stats_df = stats_df.copy()
    stats_df["_norm_name"] = stats_df["player_name"].apply(_normalise_name)

    # Group stats by normalised name
    stats_by_name: dict[str, pd.DataFrame] = {
        name: grp.set_index("season")
        for name, grp in stats_df.groupby("_norm_name")
    }
    stats_index: dict[str, list[str]] = {
        norm: list(grp["player_name"].unique())
        for norm, grp in stats_df.groupby("_norm_name")
    }

    X_rows: list[list[float]] = []
    y_vals: list[float] = []
    names: list[str] = []
    skipped = 0

    for _, row in contracts_df.iterrows():
        aav = float(row.get("aav", 0) or 0)
        if aav <= 0:
            continue

        player_name: str = str(row.get("player_name", ""))
        pos_raw: str = str(row.get("position", ""))
        pos = _normalise_pos(pos_raw)
        age = float(row.get("age") or 0)
        signed_year = int(row.get("signed_year") or current_year)

        # Age at signing (OTC shows current age; back-calculate)
        years_since_signing = current_year - signed_year
        age_at_signing = max(21.0, age - years_since_signing)
        draft_round = 8  # default = UDFA
        years_in_league = max(1, int(age_at_signing - 21))

        # Look up player stats
        match_name = _match_player(player_name, stats_index)
        norm_key = _normalise_name(match_name) if match_name else _normalise_name(player_name)
        player_stats: Optional[pd.DataFrame] = stats_by_name.get(norm_key)

        # Build per-season feature dicts for years -1, -2, -3 relative to signing
        def _get_season(yr: int) -> Optional[pd.Series]:
            if player_stats is None or yr not in player_stats.index:
                return None
            return player_stats.loc[yr]

        s1 = _get_season(signed_year - 1)
        s2 = _get_season(signed_year - 2)
        s3 = _get_season(signed_year - 3)

        f1 = _season_features(s1) if s1 is not None else {}
        f2 = _season_features(s2) if s2 is not None else {}
        f3 = _season_features(s3) if s3 is not None else {}

        def _yr(feat: str, *seasons) -> float:
            """Average of feat across provided season dicts, ignoring empty."""
            vals = [s[feat] for s in seasons if s and feat in s and s[feat] != 0]
            return float(np.mean(vals)) if vals else 0.0

        # Career best from all available seasons
        def _career_best(col: str) -> float:
            if player_stats is None:
                return 0.0
            vals = []
            for _, sr in player_stats.iterrows():
                sf = _season_features(sr)
                vals.append(sf.get(col, 0.0))
            return float(max(vals)) if vals else 0.0

        games_pct_1yr = f1.get("games_pct", 0.0)
        games_pct_2yr = _yr("games_pct", f1, f2)
        games_pct_3yr = _yr("games_pct", f1, f2, f3)

        # Skip players with zero games (no meaningful stats)
        # but keep them if we have contract data (OL/specials often missing stats)
        feature_vec = [
            float(_POS_ENCODE.get(pos, len(_POS_ORDER) - 1)),
            age_at_signing,
            float(draft_round),
            float(years_in_league),
            games_pct_1yr,
            games_pct_2yr,
            games_pct_3yr,
            # Passing
            f1.get("pass_yds_pg", 0.0),
            f1.get("pass_td_pg", 0.0),
            f1.get("pass_int_pg", 0.0),
            f1.get("comp_pct", 0.0),
            f1.get("qbr", 0.0),
            # Rushing
            f1.get("rush_yds_pg", 0.0),
            f1.get("rush_td_pg", 0.0),
            f1.get("rush_ypc", 0.0),
            # Receiving
            f1.get("rec_yds_pg", 0.0),
            f1.get("rec_pg", 0.0),
            f1.get("rec_td_pg", 0.0),
            f1.get("rec_avg", 0.0),
            # Defense
            f1.get("sacks_pg", 0.0),
            f1.get("tfl_pg", 0.0),
            f1.get("tackles_pg", 0.0),
            f1.get("int_pg", 0.0),
            f1.get("pd_pg", 0.0),
            f1.get("ff_pg", 0.0),
            # 2-year rolling
            _yr("pass_yds_pg", f1, f2),
            _yr("rec_yds_pg", f1, f2),
            _yr("rush_yds_pg", f1, f2),
            _yr("sacks_pg", f1, f2),
            # Career bests
            _career_best("pass_yds_pg"),
            _career_best("rec_yds_pg"),
            _career_best("rush_yds_pg"),
            _career_best("sacks_pg"),
        ]

        X_rows.append(feature_vec)
        y_vals.append(aav)
        names.append(player_name)

    if skipped:
        print(f"  Skipped {skipped} contracts with zero AAV.")

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_vals, dtype=np.float32)
    return X, y, names


def player_to_features(
    player_name: str,
    position: str,
    age: float,
    stats_df: pd.DataFrame,
    current_year: int = 2024,
    age_override: Optional[float] = None,
) -> np.ndarray:
    """
    Build a single feature vector for an active player for inference.
    age_override lets the caller ask "what would this player be worth at age X?"
    (used for peak value estimation).
    """
    from Player_IQ.systems.age_curves import production_factor, _normalise_pos as _np

    pos = _normalise_pos(position)
    effective_age = age_override if age_override is not None else age

    # Look up player stats
    stats_df = stats_df.copy()
    stats_df["_norm_name"] = stats_df["player_name"].apply(_normalise_name)
    norm_key = _normalise_name(player_name)

    # Try exact then fuzzy
    player_stats: Optional[pd.DataFrame] = None
    if norm_key in {n for n in stats_df["_norm_name"].unique()}:
        player_stats = stats_df[stats_df["_norm_name"] == norm_key].set_index("season")
    else:
        stats_index = {
            n: list(g["player_name"].unique())
            for n, g in stats_df.groupby("_norm_name")
        }
        match = _match_player(player_name, stats_index)
        if match:
            mk = _normalise_name(match)
            player_stats = stats_df[stats_df["_norm_name"] == mk].set_index("season")

    def _get_season(yr: int) -> Optional[pd.Series]:
        if player_stats is None or yr not in player_stats.index:
            return None
        return player_stats.loc[yr]

    s1 = _get_season(current_year - 1)
    s2 = _get_season(current_year - 2)
    f1 = _season_features(s1) if s1 is not None else {}
    f2 = _season_features(s2) if s2 is not None else {}

    def _yr(*seasons) -> dict[str, float]:
        merged: dict[str, float] = {}
        for s in seasons:
            if s:
                for k, v in s.items():
                    if v != 0.0:
                        merged[k] = v
        return merged

    def _career_best(col: str) -> float:
        if player_stats is None:
            return 0.0
        return max((_season_features(r).get(col, 0.0) for _, r in player_stats.iterrows()), default=0.0)

    years_in_league = max(1, int(effective_age - 21))
    feat = [
        float(_POS_ENCODE.get(pos, len(_POS_ORDER) - 1)),
        effective_age,
        4.0,  # assume mid-round if unknown
        float(years_in_league),
        f1.get("games_pct", 0.85),
        (f1.get("games_pct", 0.85) + f2.get("games_pct", 0.85)) / 2,
        f1.get("games_pct", 0.85),
        f1.get("pass_yds_pg", 0.0),
        f1.get("pass_td_pg", 0.0),
        f1.get("pass_int_pg", 0.0),
        f1.get("comp_pct", 0.0),
        f1.get("qbr", 0.0),
        f1.get("rush_yds_pg", 0.0),
        f1.get("rush_td_pg", 0.0),
        f1.get("rush_ypc", 0.0),
        f1.get("rec_yds_pg", 0.0),
        f1.get("rec_pg", 0.0),
        f1.get("rec_td_pg", 0.0),
        f1.get("rec_avg", 0.0),
        f1.get("sacks_pg", 0.0),
        f1.get("tfl_pg", 0.0),
        f1.get("tackles_pg", 0.0),
        f1.get("int_pg", 0.0),
        f1.get("pd_pg", 0.0),
        f1.get("ff_pg", 0.0),
        (f1.get("pass_yds_pg", 0.0) + f2.get("pass_yds_pg", 0.0)) / 2,
        (f1.get("rec_yds_pg", 0.0) + f2.get("rec_yds_pg", 0.0)) / 2,
        (f1.get("rush_yds_pg", 0.0) + f2.get("rush_yds_pg", 0.0)) / 2,
        (f1.get("sacks_pg", 0.0) + f2.get("sacks_pg", 0.0)) / 2,
        _career_best("pass_yds_pg"),
        _career_best("rec_yds_pg"),
        _career_best("rush_yds_pg"),
        _career_best("sacks_pg"),
    ]
    return np.array([feat], dtype=np.float32)
