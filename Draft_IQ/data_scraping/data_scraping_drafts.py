"""
Actual NFL draft results scraper — source: ESPN draft rounds pages.

ESPN is used instead of Pro Football Reference because PFR is behind
Cloudflare Managed Challenge which cannot be bypassed by headless browsers.
ESPN's draft pages are fully server-rendered and accessible without challenge.

URL pattern:
  https://www.espn.com/nfl/draft/rounds/_/season/{year}/round/{round}

Caching:
  Each successfully scraped year is stored as a CSV at:
    Draft_IQ/data/raw/drafts/{year}.csv
  Subsequent calls for the same year load from disk rather than re-scraping.
  Pass force=True to bypass the cache and re-scrape.
"""

import sys
import re
import time
import random
import logging
from pathlib import Path
from contextlib import contextmanager
from dataclasses import asdict, fields
from typing import Optional, Generator

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, BrowserContext

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from Core.player import DraftPick

logger = logging.getLogger(__name__)

_ESPN_BASE = "https://www.espn.com"
_DRAFT_ROUND_URL = _ESPN_BASE + "/nfl/draft/rounds/_/season/{year}/round/{round}"

_NFL_ROUNDS = 7
_TEAM_LOGO_RE = re.compile(r"/scoreboard/(\w+)\.png", re.IGNORECASE)

# Per-year cache directory: Draft_IQ/data/raw/drafts/
_CACHE_DIR = Path(__file__).parent.parent / "data" / "raw" / "drafts"


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------

def _cache_path(year: int) -> Path:
    return _CACHE_DIR / f"{year}.csv"


def _load_cache(year: int) -> Optional[list[DraftPick]]:
    path = _cache_path(year)
    if not path.exists():
        return None

    df = pd.read_csv(path)
    int_fields = {"year", "round", "pick", "career_av", "draft_av", "games_played"}
    float_fields = {"age"}
    picks: list[DraftPick] = []

    for _, row in df.iterrows():
        kwargs: dict = {}
        for f in fields(DraftPick):
            val = row.get(f.name)
            if pd.isna(val) if isinstance(val, float) else val != val:
                kwargs[f.name] = None
            elif f.name in int_fields:
                kwargs[f.name] = int(val) if pd.notna(val) else None
            elif f.name in float_fields:
                kwargs[f.name] = float(val) if pd.notna(val) else None
            else:
                kwargs[f.name] = str(val) if pd.notna(val) else ""
        picks.append(DraftPick(**kwargs))

    return picks


def _save_cache(picks: list[DraftPick], year: int) -> None:
    path = _cache_path(year)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(p) for p in picks]).to_csv(path, index=False)
    logger.debug(f"Cached {len(picks)} picks -> {path.name}")


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

def _extract_team(li) -> str:
    img = li.find("img")
    if img:
        m = _TEAM_LOGO_RE.search(img.get("src", ""))
        if m:
            return m.group(1).upper()
    return ""


def _parse_picks_from_html(html: str, year: int, round_num: int) -> list[DraftPick]:
    soup = BeautifulSoup(html, "lxml")
    pick_elements = soup.find_all(
        "li", class_=lambda c: c and "draftTable__data" in c if c else False
    )

    picks: list[DraftPick] = []
    for li in pick_elements:
        m = re.match(r"pick-(\d+)", li.get("data-key", ""))
        if not m:
            continue
        overall_pick = int(m.group(1))

        def _text(cls_fragment: str) -> str:
            el = li.find(
                "span",
                class_=lambda c: c and cls_fragment in c if c else False,
            )
            return el.get_text(strip=True) if el else ""

        player_name = _text("draftTable__headline--player")
        if not player_name:
            continue

        picks.append(DraftPick(
            year=year,
            round=round_num,
            pick=overall_pick,
            team=_extract_team(li),
            player_name=player_name,
            position=_text("draftTable__headline--pos"),
            college=_text("draftTable__headline--school"),
        ))

    return picks


# ---------------------------------------------------------------------------
# Browser context
# ---------------------------------------------------------------------------

@contextmanager
def _browser_context() -> Generator[BrowserContext, None, None]:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
        )
        try:
            yield ctx
        finally:
            ctx.close()
            browser.close()


def _fetch(url: str, ctx: BrowserContext, timeout: int = 20_000) -> str:
    page = ctx.new_page()
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=timeout)
        page.wait_for_selector("li.draftTable__data", timeout=timeout)
        return page.content()
    finally:
        page.close()


def _scrape_year_live(year: int, ctx: BrowserContext) -> list[DraftPick]:
    """Hit ESPN for all 7 rounds of a single draft year (no cache check)."""
    year_picks: list[DraftPick] = []
    for round_num in range(1, _NFL_ROUNDS + 1):
        url = _DRAFT_ROUND_URL.format(year=year, round=round_num)
        try:
            html = _fetch(url, ctx)
            picks = _parse_picks_from_html(html, year, round_num)
            year_picks.extend(picks)
            print(f"  R{round_num}: {len(picks)} picks", flush=True)
        except Exception as e:
            logger.warning(f"  R{round_num}: skipped ({e})")
        if round_num < _NFL_ROUNDS:
            time.sleep(random.uniform(1.0, 2.5))
    return year_picks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrape_draft_year(year: int, force: bool = False) -> list[DraftPick]:
    """
    Return all picks for a single NFL draft year.

    Loads from the local per-year CSV cache if available.
    Pass force=True to re-scrape even when the cache exists.
    """
    if not force:
        cached = _load_cache(year)
        if cached is not None:
            print(f"[{year}] Loaded {len(cached)} picks from cache.")
            return cached

    print(f"[{year}] Scraping from ESPN...", flush=True)
    with _browser_context() as ctx:
        picks = _scrape_year_live(year, ctx)

    _save_cache(picks, year)
    print(f"[{year}] {len(picks)} picks scraped and cached.", flush=True)
    return picks


def scrape_draft_range(
    start_year: int,
    end_year: int,
    delay: tuple[float, float] = (3.0, 6.0),
    force: bool = False,
) -> list[DraftPick]:
    """
    Return all picks for a range of draft years.

    Years with existing cache files are loaded from disk; only uncached years
    trigger a browser session and network requests. Pass force=True to
    re-scrape all years regardless of cache.
    """
    all_picks: list[DraftPick] = []
    years_to_scrape: list[int] = []

    for year in range(start_year, end_year + 1):
        if not force:
            cached = _load_cache(year)
            if cached is not None:
                print(f"[{year}] Cache hit — {len(cached)} picks.", flush=True)
                all_picks.extend(cached)
                continue
        years_to_scrape.append(year)

    if not years_to_scrape:
        print("All years loaded from cache. No scraping needed.")
        return all_picks

    print(f"\nScraping {len(years_to_scrape)} year(s) from ESPN: {years_to_scrape}", flush=True)
    with _browser_context() as ctx:
        for i, year in enumerate(years_to_scrape):
            print(f"\n[{year}] Scraping...", flush=True)
            picks = _scrape_year_live(year, ctx)
            _save_cache(picks, year)
            all_picks.extend(picks)
            print(f"[{year}] {len(picks)} picks cached.", flush=True)

            if i < len(years_to_scrape) - 1:
                sleep_sec = random.uniform(*delay)
                print(f"  Waiting {sleep_sec:.1f}s...", flush=True)
                time.sleep(sleep_sec)

    return all_picks


def picks_to_dataframe(picks: list[DraftPick]) -> pd.DataFrame:
    return pd.DataFrame([asdict(p) for p in picks])


def save_draft_data(picks: list[DraftPick], output_path: str) -> None:
    """Write combined picks list to a CSV (used for the merged actual_drafts.csv)."""
    df = picks_to_dataframe(picks)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(picks)} picks -> {path}")


def load_all_cached(start_year: int, end_year: int) -> pd.DataFrame:
    """Load all cached year files into a single DataFrame without scraping."""
    frames = []
    for year in range(start_year, end_year + 1):
        path = _cache_path(year)
        if path.exists():
            frames.append(pd.read_csv(path))
        else:
            logger.warning(f"No cache for {year} — run scrape_draft_range() first.")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def backfill_cache_from_combined(combined_csv: str) -> None:
    """
    Split an existing combined actual_drafts.csv into per-year cache files.

    Call this once after a legacy scrape run that saved a single combined file,
    to populate the per-year cache without re-scraping.
    """
    path = Path(combined_csv)
    if not path.exists():
        raise FileNotFoundError(f"Combined CSV not found: {path}")

    df = pd.read_csv(path)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for year, group in df.groupby("year"):
        out = _cache_path(int(year))
        if out.exists():
            print(f"[{year}] Cache already exists, skipping.")
            continue
        group.to_csv(out, index=False)
        print(f"[{year}] Wrote {len(group)} picks -> {out.name}")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    picks = scrape_draft_range(2010, 2024)
    output = Path(__file__).parent.parent / "data" / "raw" / "actual_drafts.csv"
    save_draft_data(picks, str(output))
    print(f"\nTotal: {len(picks)} picks across all years.")
