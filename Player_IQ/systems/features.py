"""
Feature Engineering — Player_IQ/systems/features.py

Joins contract records (from Market_IQ OTC cache) with player season
stats (from Player_IQ ESPN cache) to produce a training-ready feature
matrix for the GBR value model.

Year convention: OTC position pages don't expose the true signing year,
so the model answers "what is this player worth TODAY given their most
recent seasons?"  Both training and inference anchor the stat lookup on
the most recent completed season in the stats cache (s1 = anchor,
s2 = anchor-1, s3 = anchor-2), guaranteeing the feature distributions
match between the two paths.

Feature vector layout (33 features):
  [0]  pos_encoded           — integer index into _POS_ORDER
  [1]  age_at_signing
  [2]  draft_round           — 1-7, 8 = UDFA/unknown, 4 = assumed mid-round
  [3]  years_in_league
  [4]  games_pct_1yr         — fraction of 17-game season played
  [5]  games_pct_2yr         — 2-year avg of non-missing seasons
  [6]  games_pct_3yr         — 3-year avg of non-missing seasons
  [7-11]  passing:   pass_yds_pg, pass_td_pg, pass_int_pg, comp_pct, qbr
  [12-14] rushing:   rush_yds_pg, rush_td_pg, rush_ypc
  [15-18] receiving: rec_yds_pg, rec_pg, rec_td_pg, rec_avg
  [19-24] defense:   sacks_pg, tfl_pg, tackles_pg, int_pg, pd_pg, ff_pg
  [25-28] 2yr avgs:  pass_yds_pg, rec_yds_pg, rush_yds_pg, sacks_pg
  [29-32] career bests: pass_yds_pg, rec_yds_pg, rush_yds_pg, sacks_pg

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
# Shared stats lookup + feature vector builder
# ---------------------------------------------------------------------------

def _player_season_table(
    stats_df: pd.DataFrame,
    player_name: str,
    pos: str,
) -> Optional[pd.DataFrame]:
    """
    Return the player's seasons indexed by season year, or None.

    Expects stats_df to already carry a _norm_name column. Filters by
    position group when possible so same-name players at different
    positions (e.g. Josh Allen QB vs Josh Allen EDGE) don't merge, and
    keeps the highest-games row when a season is still duplicated.
    """
    norm_key = _normalise_name(player_name)
    grp = stats_df[stats_df["_norm_name"] == norm_key]

    if grp.empty:
        # Fuzzy fallback
        stats_index = {
            n: list(g["player_name"].unique())
            for n, g in stats_df.groupby("_norm_name")
        }
        match = _match_player(player_name, stats_index)
        if not match:
            return None
        grp = stats_df[stats_df["_norm_name"] == _normalise_name(match)]
        if grp.empty:
            return None

    # Disambiguate same-name players by position group when we can
    if pos and pos != "OTHER":
        pos_match = grp[grp["position"].apply(_normalise_pos) == pos]
        if not pos_match.empty:
            grp = pos_match

    # One row per season: keep the row with the most games played
    grp = grp.sort_values("games", ascending=False).drop_duplicates("season")
    return grp.set_index("season")


def build_feature_vector(
    pos: str,
    effective_age: float,
    player_stats: Optional[pd.DataFrame],
    anchor_year: int,
    draft_round: float = 8.0,
) -> list[float]:
    """
    The single source of truth for the 33-feature vector, used by BOTH
    training and inference so their distributions always match.

    s1 = anchor_year (most recent completed season), s2/s3 = prior years.
    """
    def _get_season(yr: int) -> Optional[pd.Series]:
        if player_stats is None or yr not in player_stats.index:
            return None
        return player_stats.loc[yr]

    s1 = _get_season(anchor_year)
    s2 = _get_season(anchor_year - 1)
    s3 = _get_season(anchor_year - 2)

    f1 = _season_features(s1) if s1 is not None else {}
    f2 = _season_features(s2) if s2 is not None else {}
    f3 = _season_features(s3) if s3 is not None else {}

    def _yr(feat: str, *seasons) -> float:
        """Average of feat across non-missing seasons (zeros ignored)."""
        vals = [s[feat] for s in seasons if s and feat in s and s[feat] != 0]
        return float(np.mean(vals)) if vals else 0.0

    def _career_best(col: str) -> float:
        if player_stats is None:
            return 0.0
        return max(
            (_season_features(r).get(col, 0.0) for _, r in player_stats.iterrows()),
            default=0.0,
        )

    years_in_league = max(1, int(effective_age - 21))

    return [
        float(_POS_ENCODE.get(pos, len(_POS_ORDER) - 1)),
        effective_age,
        float(draft_round),
        float(years_in_league),
        f1.get("games_pct", 0.0),
        _yr("games_pct", f1, f2),
        _yr("games_pct", f1, f2, f3),
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


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_training_matrix(
    contracts_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    current_year: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Join contracts with stats and build (X, y, player_names).

    contracts_df columns required: player_name, position, age, aav
    stats_df columns required:     player_name, season, games, <stat cols>

    Returns:
        X — (N, 33) float32 feature matrix
        y — (N,) float32 AAV targets (millions)
        names — list of player_name strings for diagnostics
    """
    stats_df = stats_df.copy()
    stats_df["_norm_name"] = stats_df["player_name"].apply(_normalise_name)

    # Anchor stat lookups on the most recent completed season we have
    anchor_year = int(stats_df["season"].max())

    X_rows: list[list[float]] = []
    y_vals: list[float] = []
    names: list[str] = []

    for _, row in contracts_df.iterrows():
        aav = float(row.get("aav", 0) or 0)
        if aav <= 0:
            continue

        player_name: str = str(row.get("player_name", ""))
        pos = _normalise_pos(str(row.get("position", "")))
        age_raw = row.get("age")
        age = 26.0 if age_raw is None or pd.isna(age_raw) or float(age_raw) <= 0 else float(age_raw)

        player_stats = _player_season_table(stats_df, player_name, pos)
        feature_vec = build_feature_vector(pos, age, player_stats, anchor_year)

        X_rows.append(feature_vec)
        y_vals.append(aav)
        names.append(player_name)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_vals, dtype=np.float32)
    return X, y, names


def player_to_features(
    player_name: str,
    position: str,
    age: float,
    stats_df: pd.DataFrame,
    current_year: Optional[int] = None,
    age_override: Optional[float] = None,
) -> np.ndarray:
    """
    Build a single feature vector for an active player for inference.
    age_override lets the caller ask "what would this player be worth at age X?"
    (used for peak value estimation).

    current_year, when given, caps the anchor season (useful for "value
    this player as of year Y" queries); by default the anchor is the most
    recent season in the stats cache — identical to training.
    """
    pos = _normalise_pos(position)
    effective_age = age_override if age_override is not None else age

    stats_df = stats_df.copy()
    stats_df["_norm_name"] = stats_df["player_name"].apply(_normalise_name)

    anchor_year = int(stats_df["season"].max())
    if current_year is not None:
        anchor_year = min(anchor_year, current_year - 1)

    player_stats = _player_season_table(stats_df, player_name, pos)
    feat = build_feature_vector(pos, effective_age, player_stats, anchor_year)
    return np.array([feat], dtype=np.float32)
