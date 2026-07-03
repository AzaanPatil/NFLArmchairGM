"""
ESPN Player Stats Scraper — Player_IQ/data_scraping/scrape_player_stats.py

Fetches per-season player statistics from ESPN's stats pages for all
skill and defensive positions. ESPN renders these tables client-side
(React), so we use Playwright to render fully before parsing.

ESPN uses a split-table layout:
  LEFT table  → RK | Player Name (with /nfl/player/_/id/{ID}/... link) | Team
  RIGHT table → Stat columns (G, YDS, TD, …)
We zip them by row index to produce one record per player.

Cache layout:
  Player_IQ/data/raw/stats/{year}/{category}.csv

Usage:
    from Player_IQ.data_scraping.scrape_player_stats import (
        scrape_stats_category, scrape_all_stats, load_stats_cache,
    )
    df = scrape_all_stats(start=2015, end=2024)
"""

from __future__ import annotations

import re
import sys
import time
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from Core.player_season import PlayerSeason

logger = logging.getLogger(__name__)

_ESPN_STATS_BASE = (
    "https://www.espn.com/nfl/stats/player/_/{path}"
    "/season/{year}/seasontype/2"
)
_ESPN_PLAYER_ID_RE = re.compile(r"/nfl/player/_/id/(\d+)/")

_CACHE_DIR = Path(__file__).parent.parent / "data" / "raw" / "stats"

# Stat category → (ESPN URL path segment, sort suffix appended after season).
# Defense is a "view" (one page covers tackles/sacks/INTs/PD/FF), not a
# "stat" like the offense categories. The default defense sort is tackles,
# which misses low-tackle pass rushers and ballhawks — so we make two extra
# passes sorted by sacks and interceptions and merge by espn_id.
_CATEGORIES: dict[str, tuple[str, str]] = {
    "passing":       ("stat/passing", ""),
    "rushing":       ("stat/rushing", ""),
    "receiving":     ("stat/receiving", ""),
    "defense":       ("view/defense", ""),
    "defense_sacks": ("view/defense", "/table/defensive/sort/sacks/dir/desc"),
    "defense_ints":  ("view/defense", "/table/defensiveInterceptions/sort/interceptions/dir/desc"),
}

# ESPN paginates via a "Show More" button that XHR-appends 50 rows to the
# same table (URL-based /start/N pagination silently returns page 1).
# 4 clicks → up to 250 players per category/year.
_SHOW_MORE_CLICKS = 4

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}


# ---------------------------------------------------------------------------
# Playwright fetch
# ---------------------------------------------------------------------------

def _fetch_rendered(url: str, show_more_clicks: int = _SHOW_MORE_CLICKS) -> str:
    """
    Render an ESPN stats page and click "Show More" up to show_more_clicks
    times so the table accumulates additional 50-row pages before parsing.
    """
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=_HEADERS["User-Agent"],
            viewport={"width": 1920, "height": 1080},
        )
        page = ctx.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            try:
                page.wait_for_selector("table", timeout=10_000)
            except Exception:
                pass
            page.wait_for_timeout(1500)
            for _ in range(show_more_clicks):
                btn = page.query_selector("a:has-text('Show More')")
                if not btn:
                    break   # full list already loaded
                btn.click()
                page.wait_for_timeout(1800)
            return page.content()
        finally:
            page.close()
            ctx.close()
            browser.close()


# ---------------------------------------------------------------------------
# HTML parser
# ---------------------------------------------------------------------------

def _safe_float(text: str) -> Optional[float]:
    t = text.strip().replace(",", "")
    if not t or t in ("--", "-", "N/A", ""):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _extract_espn_id(href: str) -> str:
    m = _ESPN_PLAYER_ID_RE.search(href)
    return m.group(1) if m else ""


def _parse_stats_page(html: str, category: str, season: int) -> list[dict]:
    """
    Parse one ESPN stats page HTML into a list of raw stat dicts.

    ESPN uses a two-table layout: left table has name/team, right has stats.
    We zip rows by index.  Returns list of {player_name, espn_id, team, season,
    category, <stat_col>: value, ...}.
    """
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")
    if len(tables) < 2:
        logger.debug(f"[{category}/{season}] Found {len(tables)} table(s) — skipping page.")
        return []

    name_table = tables[0]
    stat_table = tables[1]

    # Stat column headers — the defense view has a group-header row
    # ("Tackles | Sacks | Interceptions | Fumbles") above the real one,
    # so take only the LAST row of the thead.
    stat_headers: list[str] = []
    thead = stat_table.find("thead")
    if thead:
        header_rows = thead.find_all("tr")
        if header_rows:
            stat_headers = [
                th.get_text(strip=True).upper()
                for th in header_rows[-1].find_all(["th", "td"])
            ]
    if not stat_headers:
        stat_headers = [th.get_text(strip=True).upper()
                        for th in stat_table.find_all("th")]

    def _tbody_rows(table) -> list:
        tbody = table.find("tbody")
        return tbody.find_all("tr") if tbody else []

    name_rows = _tbody_rows(name_table)
    stat_rows = _tbody_rows(stat_table)

    records: list[dict] = []
    now = datetime.now().isoformat()

    for name_row, stat_row in zip(name_rows, stat_rows):
        # Skip sub-header rows ESPN sometimes inserts
        if "colhead" in " ".join(name_row.get("class", [])):
            continue

        # ---- Player info (from left table) ----
        link = name_row.find("a", href=_ESPN_PLAYER_ID_RE)
        if not link:
            continue
        player_name = link.get_text(strip=True)
        espn_id = _extract_espn_id(link.get("href", ""))

        # Team abbreviation is a span next to the name link in the athlete cell
        team = ""
        team_span = name_row.find("span", class_=re.compile(r"team", re.I))
        if team_span:
            team = team_span.get_text(strip=True)

        # ---- Stats (from right table) ----
        # POS is a text column inside the stats table; everything else is numeric
        position = ""
        stat_vals: dict[str, Optional[float]] = {}
        stat_cells = stat_row.find_all("td")
        for header, cell in zip(stat_headers, stat_cells):
            txt = cell.get_text(strip=True)
            if header == "POS":
                position = txt
                continue
            stat_vals[header] = _safe_float(txt)

        records.append({
            "player_name": player_name,
            "espn_id": espn_id,
            "team": team,
            "position": position,
            "season": season,
            "category": category,
            "fetched_at": now,
            **stat_vals,
        })

    return records


# ---------------------------------------------------------------------------
# Category → PlayerSeason assembler
# ---------------------------------------------------------------------------

def _records_to_player_seasons(
    records_by_category: dict[str, list[dict]],
    season: int,
) -> list[PlayerSeason]:
    """
    Merge records from different stat categories (passing, rushing, receiving,
    defensive, interceptions) by ESPN player ID into one PlayerSeason per player.
    """
    # Index all records by espn_id
    by_id: dict[str, dict] = {}

    def _merge(espn_id: str, data: dict) -> None:
        if espn_id not in by_id:
            by_id[espn_id] = {
                "player_name": data.get("player_name", ""),
                "espn_id": espn_id,
                "team": data.get("team", ""),
                "position": data.get("position", ""),
                "season": season,
                "fetched_at": data.get("fetched_at", ""),
            }
        else:
            # Backfill identity fields a later category may have filled in
            for key in ("team", "position"):
                if not by_id[espn_id].get(key) and data.get(key):
                    by_id[espn_id][key] = data[key]
        by_id[espn_id].update(
            {k: v for k, v in data.items()
             if k not in ("player_name", "team", "position", "season", "category", "fetched_at")
             and v is not None}
        )

    for category, records in records_by_category.items():
        for rec in records:
            eid = rec.get("espn_id", "")
            if eid:
                _merge(eid, rec)

    seasons: list[PlayerSeason] = []
    for eid, d in by_id.items():
        def g(key: str) -> Optional[float]:
            return d.get(key)

        games = int(g("GP") or 0)
        ps = PlayerSeason(
            player_name=d.get("player_name", ""),
            espn_id=eid,
            team=d.get("team", ""),
            position=d.get("position", ""),
            season=season,
            games=games,
            games_started=games,   # ESPN doesn't split starts/games on these pages
            fetched_at=d.get("fetched_at", ""),

            # Passing (bare names = passing stats by convention)
            pass_yards=g("YDS"),
            pass_tds=g("TD"),
            pass_ints=g("INT"),
            completions=g("CMP"),
            attempts=g("ATT"),
            comp_pct=g("CMP%"),
            pass_avg=g("AVG"),
            qbr=g("QBR"),
            passer_rating=g("RTG"),

            # Rushing (renamed per-category to avoid collisions)
            rush_yards=g("RUSH_YDS"),
            rush_tds=g("RUSH_TD"),
            rush_attempts=g("CAR"),
            rush_avg=g("RUSH_AVG"),

            # Receiving
            receptions=g("REC"),
            targets=g("TGTS"),
            rec_yards=g("REC_YDS"),
            rec_tds=g("REC_TD"),
            rec_avg=g("REC_AVG"),

            # Defense
            tackles=g("TOT"),
            sacks=g("SACKS"),
            tfl=g("TFL"),
            interceptions=g("INT_DEF"),
            pass_defenses=g("PD"),
            forced_fumbles=g("FF"),
            qb_hits=g("QBH"),
        )
        seasons.append(ps)

    return seasons


# ---------------------------------------------------------------------------
# Category-aware stat parsing  (rename duplicate column names by category)
# ---------------------------------------------------------------------------

# ESPN reuses "YDS", "TD", "AVG", "ATT", "SACK", "INT" across categories.
# We prefix them before merging so they don't collide.
# (Passing keeps the bare names: YDS/TD/INT/ATT/SACK mean passing stats.)
_CATEGORY_RENAMES: dict[str, dict[str, str]] = {
    "rushing": {
        "YDS": "RUSH_YDS",
        "TD":  "RUSH_TD",
        "AVG": "RUSH_AVG",
        "ATT": "CAR",        # carries — passing also has ATT (attempts)
        "LNG": "RUSH_LNG",
    },
    "receiving": {
        "YDS": "REC_YDS",
        "TD":  "REC_TD",
        "AVG": "REC_AVG",
        "LNG": "REC_LNG",
    },
    "defense": {
        "SACK": "SACKS",     # passing SACK = sacks taken by the QB
        "INT":  "INT_DEF",   # passing INT = interceptions thrown
        "YDS":  "DEF_YDS",
        "TD":   "DEF_TD",
        "LNG":  "DEF_LNG",
    },
}
# The sack- and INT-sorted defense passes share the defense renames
_CATEGORY_RENAMES["defense_sacks"] = _CATEGORY_RENAMES["defense"]
_CATEGORY_RENAMES["defense_ints"] = _CATEGORY_RENAMES["defense"]


def _rename_cols(records: list[dict], category: str) -> list[dict]:
    renames = _CATEGORY_RENAMES.get(category, {})
    if not renames:
        return records
    out = []
    for r in records:
        row = dict(r)
        for old, new in renames.items():
            if old in row:
                row[new] = row.pop(old)
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------

def _cache_path(year: int, category: str) -> Path:
    return _CACHE_DIR / str(year) / f"{category}.csv"


def _is_cached(year: int, category: str) -> bool:
    return _cache_path(year, category).exists()


def _save_cache(records: list[dict], year: int, category: str) -> None:
    p = _cache_path(year, category)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(p, index=False)


def _load_cache(year: int, category: str) -> list[dict]:
    p = _cache_path(year, category)
    if not p.exists():
        return []
    return pd.read_csv(p).to_dict("records")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrape_stats_category(
    category: str,
    year: int,
    force: bool = False,
    page_delay: float = 1.5,
) -> list[dict]:
    """
    Scrape or load cached stats for one ESPN stat category and year.
    Returns a list of raw stat dicts (one per player row).
    """
    if not force and _is_cached(year, category):
        cached = _load_cache(year, category)
        print(f"[{category}/{year}] Loaded {len(cached)} rows from cache.")
        return cached

    print(f"[{category}/{year}] Scraping ESPN stats...")

    # One rendered page with "Show More" clicked accumulates all rows
    path, suffix = _CATEGORIES.get(category, (f"stat/{category}", ""))
    url = _ESPN_STATS_BASE.format(path=path, year=year) + suffix
    records: list[dict] = []
    for attempt in range(2):   # ESPN renders are occasionally flaky — retry once
        html = _fetch_rendered(url)
        records = _rename_cols(_parse_stats_page(html, category, year), category)
        if records:
            break
        time.sleep(3)

    # Dedupe by espn_id (Show More re-renders can rarely duplicate rows)
    seen_ids: set[str] = set()
    all_records: list[dict] = []
    for r in records:
        eid = r.get("espn_id", "")
        if eid and eid not in seen_ids:
            seen_ids.add(eid)
            all_records.append(r)

    if all_records:
        _save_cache(all_records, year, category)
        print(f"[{category}/{year}] {len(all_records)} players cached.")
    else:
        logger.warning(f"[{category}/{year}] No data parsed — ESPN table structure may differ.")

    return all_records


def scrape_all_stats(
    start: int = 2015,
    end: int = 2024,
    force: bool = False,
    category_delay: float = 2.0,
) -> pd.DataFrame:
    """
    Scrape all stat categories for all years in [start, end].
    Returns a merged DataFrame with one row per (player, season).
    """
    all_seasons: list[PlayerSeason] = []

    for year in range(start, end + 1):
        by_category: dict[str, list[dict]] = {}
        for cat in _CATEGORIES:
            recs = scrape_stats_category(cat, year, force=force)
            by_category[cat] = recs
            time.sleep(category_delay)

        seasons = _records_to_player_seasons(by_category, year)
        all_seasons.extend(seasons)
        print(f"[{year}] {len(seasons)} player-seasons assembled.")

    if not all_seasons:
        return pd.DataFrame()

    df = pd.DataFrame([asdict(s) for s in all_seasons])
    out_path = _CACHE_DIR.parent / "all_stats.csv"
    df.to_csv(out_path, index=False)
    print(f"\nAll stats: {len(df)} player-seasons -> {out_path}")
    return df


def load_stats_cache() -> pd.DataFrame:
    """Load the merged stats cache. Raise if not found."""
    p = _CACHE_DIR.parent / "all_stats.csv"
    if not p.exists():
        raise FileNotFoundError(
            "No stats cache found. Run the scraper first:\n"
            "  python -m Player_IQ.main --scrape"
        )
    return pd.read_csv(p)
