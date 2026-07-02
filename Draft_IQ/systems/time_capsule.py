"""
NFL Time Capsule — Draft_IQ/systems/time_capsule.py

Seals and reopens pre-draft snapshots so every analysis can be anchored
to what was known *before* the draft happened, with no hindsight.

On-disk layout:
    Draft_IQ/data/snapshots/{year}/
        manifest.json   — provenance metadata
        actual.csv      — actual picks (from ESPN scraper)
        mock.csv        — pre-draft mock projections (if available)
        combine.csv     — combine measurables (future layer)
        team_needs.csv  — positional needs per team (future layer)

Usage:
    from Draft_IQ.systems.time_capsule import open_capsule, seal_capsule

    capsule = open_capsule(2020)          # load or build
    print(capsule)                        # DraftSnapshot(2020, actual=255 picks, mock=64 picks)
    df = compare_mock_to_actual(capsule)  # pick accuracy analysis
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from Core.snapshot import DraftSnapshot

logger = logging.getLogger(__name__)

_SNAPSHOT_ROOT = Path(__file__).parent.parent / "data" / "snapshots"


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _capsule_dir(year: int) -> Path:
    return _SNAPSHOT_ROOT / str(year)


def _read_csv_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def capsule_exists(year: int) -> bool:
    """True if a sealed Time Capsule exists for this year."""
    return (_capsule_dir(year) / "manifest.json").exists()


def seal_capsule(snapshot: DraftSnapshot) -> None:
    """Write a DraftSnapshot to disk as a sealed Time Capsule."""
    d = _capsule_dir(snapshot.year)
    d.mkdir(parents=True, exist_ok=True)

    if not snapshot.actual_draft.empty:
        snapshot.actual_draft.to_csv(d / "actual.csv", index=False)
    if not snapshot.mock_draft.empty:
        snapshot.mock_draft.to_csv(d / "mock.csv", index=False)
    if not snapshot.combine.empty:
        snapshot.combine.to_csv(d / "combine.csv", index=False)
    if not snapshot.team_needs.empty:
        snapshot.team_needs.to_csv(d / "team_needs.csv", index=False)

    manifest = {
        "year": snapshot.year,
        "created_at": snapshot.created_at or datetime.now().isoformat(),
        "source_actual": snapshot.source_actual,
        "source_mock": snapshot.source_mock,
        "picks_actual": len(snapshot.actual_draft),
        "picks_mock": len(snapshot.mock_draft),
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[{snapshot.year}] Time Capsule sealed -> {d}")


def load_capsule(year: int) -> DraftSnapshot:
    """Load a previously sealed Time Capsule from disk."""
    d = _capsule_dir(year)
    if not (d / "manifest.json").exists():
        raise FileNotFoundError(
            f"No Time Capsule found for {year}. "
            f"Call open_capsule({year}) or build_capsule({year}) first."
        )

    manifest = json.loads((d / "manifest.json").read_text())

    return DraftSnapshot(
        year=year,
        actual_draft=_read_csv_optional(d / "actual.csv"),
        mock_draft=_read_csv_optional(d / "mock.csv"),
        combine=_read_csv_optional(d / "combine.csv"),
        team_needs=_read_csv_optional(d / "team_needs.csv"),
        source_actual=manifest.get("source_actual", "ESPN"),
        source_mock=manifest.get("source_mock", ""),
        created_at=manifest.get("created_at", ""),
    )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_capsule(year: int) -> DraftSnapshot:
    """
    Scrape / load all available data for a draft year and return a DraftSnapshot.
    Does NOT seal to disk — call seal_capsule() or open_capsule() for that.
    """
    from Draft_IQ.data_scraping.data_scraping_drafts import (
        scrape_draft_year,
        picks_to_dataframe,
    )

    print(f"[{year}] Building Time Capsule...")

    # --- Actual draft (uses per-year cache; no re-scrape if cached) ---
    actual_picks = scrape_draft_year(year)
    actual_df = picks_to_dataframe(actual_picks)
    print(f"  Actual draft: {len(actual_df)} picks")

    # --- Mock draft (optional; no error if unavailable) ---
    mock_df = pd.DataFrame()
    source_mock = ""
    try:
        from Draft_IQ.data_scraping.data_scraping_mock_drafts import scrape_mock_draft
        mock_df = scrape_mock_draft(year)
        source_mock = "CBS Sports (Wayback Machine)" if not mock_df.empty else ""
        print(f"  Mock draft: {len(mock_df)} picks ({source_mock})")
    except ImportError:
        logger.debug("Mock draft scraper not yet available.")
    except Exception as e:
        logger.warning(f"  Mock draft unavailable for {year}: {e}")

    return DraftSnapshot(
        year=year,
        actual_draft=actual_df,
        mock_draft=mock_df,
        source_actual="ESPN",
        source_mock=source_mock,
        created_at=datetime.now().isoformat(),
    )


def open_capsule(year: int, force: bool = False) -> DraftSnapshot:
    """
    The main entry point.

    - If a sealed capsule exists on disk and force=False: load it instantly.
    - Otherwise: build it from scratch, seal it, and return it.

    This is the Time Capsule equivalent of scrape_draft_year's cache check.
    """
    if not force and capsule_exists(year):
        print(f"[{year}] Opening Time Capsule...")
        return load_capsule(year)

    snapshot = build_capsule(year)
    seal_capsule(snapshot)
    return snapshot


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compare_mock_to_actual(snapshot: DraftSnapshot) -> pd.DataFrame:
    """
    Merge mock and actual draft picks to compute accuracy metrics.

    Returns a DataFrame with columns:
        player_name, mock_pick, actual_pick, pick_delta,
        mock_team, actual_team, position, college
        abs_error  — absolute pick difference
        fell       — True if picked later than mocked (fell in draft)
        rose       — True if picked earlier than mocked (rose in draft)
    """
    if not snapshot.has_mock:
        raise ValueError(
            f"Time Capsule for {snapshot.year} has no mock draft data. "
            "Run the mock draft scraper first."
        )

    mock = snapshot.mock_draft.rename(columns={"pick": "mock_pick", "team": "mock_team"})
    actual = snapshot.actual_draft.rename(columns={"pick": "actual_pick", "team": "actual_team"})

    shared_cols = {"player_name"}
    mock_cols = shared_cols | {"mock_pick", "mock_team", "position"}
    actual_cols = shared_cols | {"actual_pick", "actual_team", "round", "college", "position"}

    mock_trimmed = mock[[c for c in mock_cols if c in mock.columns]]
    actual_trimmed = actual[[c for c in actual_cols if c in actual.columns]]

    # Inner join on player name
    merged = mock_trimmed.merge(actual_trimmed, on="player_name", suffixes=("", "_actual"))
    if "position_actual" in merged.columns:
        merged["position"] = merged["position"].fillna(merged["position_actual"])
        merged.drop(columns=["position_actual"], inplace=True)

    # CBS Sports publishes multiple analysts' mocks on the same page, so a player
    # can appear several times. Collapse to a consensus by taking the median
    # projected pick across all analysts for each player.
    agg: dict = {"mock_pick": "median", "actual_pick": "first"}
    for col in ("mock_team", "position", "round", "college"):
        if col in merged.columns:
            agg[col] = "first"
    merged = merged.groupby("player_name", as_index=False).agg(agg)

    merged["pick_delta"] = (merged["actual_pick"] - merged["mock_pick"]).round().astype(int)
    merged["abs_error"] = merged["pick_delta"].abs()
    merged["fell"] = merged["pick_delta"] > 0
    merged["rose"] = merged["pick_delta"] < 0

    return merged.sort_values("actual_pick").reset_index(drop=True)


def mock_accuracy_summary(snapshot: DraftSnapshot) -> dict:
    """High-level accuracy stats for a mock draft."""
    df = compare_mock_to_actual(snapshot)
    return {
        "year": snapshot.year,
        "players_matched": len(df),
        "mean_abs_error": round(df["abs_error"].mean(), 1),
        "median_abs_error": round(df["abs_error"].median(), 1),
        "exact_pick": int((df["pick_delta"] == 0).sum()),
        "within_5_picks": int((df["abs_error"] <= 5).sum()),
        "within_10_picks": int((df["abs_error"] <= 10).sum()),
        "fell_pct": round((df["fell"].sum() / len(df)) * 100, 1),
        "rose_pct": round((df["rose"].sum() / len(df)) * 100, 1),
    }
