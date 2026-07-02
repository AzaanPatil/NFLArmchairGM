"""
Pre-draft mock draft scraper — source: CBS Sports via Wayback Machine.

Strategy:
  • Current/upcoming draft year: scrape cbssports.com/nfl/draft/mock-draft/ directly.
  • Historical years: use the archive.org CDX API to find the most recent
    archived snapshot of that page from the week before the draft, then
    fetch the Wayback Machine copy.

This approach is architecturally correct for the NFL Time Capsule — we retrieve
the page exactly as it existed before the draft happened, not with hindsight.

CBS Sports table structure (server-rendered, no JS required for data rows):
  <tr>
    <td>N          ← overall pick number
    <td class="cell-player-info">
      <a>Player Name</a>
      <span class="player-details">College, Year</span>
    </td>
    <td class="cell-pos">POS
    <td class="cell-team">TEAM (projected)
  </tr>
  (alternating commentary rows are skipped)

Caching:
  Scraped mock drafts are cached to Draft_IQ/data/raw/mocks/{year}.csv.
"""

import sys
import time
import random
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import requests
import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

_CBS_MOCK_URL = "https://www.cbssports.com/nfl/draft/mock-draft/"
_CDX_API      = "https://web.archive.org/cdx/search/cdx"
_WBM_BASE     = "https://web.archive.org/web"

_CACHE_DIR = Path(__file__).parent.parent / "data" / "raw" / "mocks"

# First day of each NFL Draft (YYYYMMDD) — used to find pre-draft archive snapshots.
_DRAFT_DATES: dict[int, str] = {
    2024: "20240425",
    2023: "20230427",
    2022: "20220428",
    2021: "20210429",
    2020: "20200423",
    2019: "20190425",
    2018: "20180426",
    2017: "20170427",
    2016: "20160428",
    2015: "20150430",
    2014: "20140508",
    2013: "20130425",
    2012: "20120426",
    2011: "20110428",
    2010: "20100422",
}

_CDX_TIMEOUT = 60   # archive.org can be slow


# ---------------------------------------------------------------------------
# Mock pick model
# ---------------------------------------------------------------------------

@dataclass
class MockPick:
    year: int
    pick: int
    player_name: str
    position: str
    college: str
    projected_team: str
    source: str


# ---------------------------------------------------------------------------
# CBS Sports parser
# ---------------------------------------------------------------------------

def _parse_cbs_html(html: str, year: int, source: str) -> list[MockPick]:
    soup = BeautifulSoup(html, "lxml")
    picks: list[MockPick] = []

    for row in soup.find_all("tr"):
        tds = row.find_all("td")
        if not tds:
            continue

        # First cell must be a plain pick number
        pick_text = tds[0].get_text(strip=True)
        if not pick_text.isdigit():
            continue
        pick_num = int(pick_text)

        # Player info cell
        player_td = row.find("td", class_=lambda c: c and "player-info" in c if c else False)
        if not player_td:
            continue

        player_name = ""
        a = player_td.find("a")
        if a:
            player_name = a.get_text(strip=True)
        if not player_name:
            continue

        # College is in a <span class="player-details"> or similar sibling text
        college = ""
        details_span = player_td.find("span", class_=lambda c: c and "player-details" in c if c else False)
        if details_span:
            raw = details_span.get_text(strip=True)
            # Strip trailing year annotation like ", Jr" / ", Sr" / ", So"
            college = raw.split(",")[0].strip()

        # Position — td with class containing "pos"
        pos_td = row.find("td", class_=lambda c: c and "pos" in c.lower() if c else False)
        position = pos_td.get_text(strip=True) if pos_td else ""

        # Projected team — td with class containing "team"
        team_td = row.find("td", class_=lambda c: c and "team" in c.lower() if c else False)
        projected_team = team_td.get_text(strip=True) if team_td else ""

        picks.append(MockPick(
            year=year,
            pick=pick_num,
            player_name=player_name,
            position=position,
            college=college,
            projected_team=projected_team,
            source=source,
        ))

    return picks


# ---------------------------------------------------------------------------
# Wayback Machine lookup
# ---------------------------------------------------------------------------

def _find_wayback_snapshot(draft_date: str) -> Optional[str]:
    """
    Return a Wayback Machine URL for the CBS Sports mock draft page
    from the week leading up to the draft, or None if not found.
    """
    window_start = str(int(draft_date) - 7).zfill(8)
    try:
        r = requests.get(
            _CDX_API,
            params={
                "url": "cbssports.com/nfl/draft/mock-draft/",
                "output": "json",
                "from": window_start,
                "to": draft_date,
                "limit": 1,
                "fl": "timestamp,statuscode",
                "filter": "statuscode:200",
                "matchType": "prefix",
            },
            timeout=_CDX_TIMEOUT,
        )
        data = r.json()
        if len(data) < 2:
            return None
        ts = data[1][0]
        return f"{_WBM_BASE}/{ts}/{_CBS_MOCK_URL}"
    except Exception as e:
        logger.warning(f"CDX lookup failed: {e}")
        return None


_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _fetch_static(url: str, timeout: int = 30) -> str:
    """Fetch a static/server-rendered page with plain requests (fast, no browser)."""
    r = requests.get(url, headers=_REQUEST_HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.text


def _fetch_via_playwright(url: str) -> str:
    """Fetch a JS-rendered page through a headless browser."""
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=_REQUEST_HEADERS["User-Agent"],
            viewport={"width": 1920, "height": 1080},
        )
        page = ctx.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=25_000)
            time.sleep(3)
            return page.content()
        finally:
            page.close()
            ctx.close()
            browser.close()


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------

def _cache_path(year: int) -> Path:
    return _CACHE_DIR / f"{year}.csv"


def _load_mock_cache(year: int) -> Optional[pd.DataFrame]:
    path = _cache_path(year)
    if not path.exists():
        return None
    return pd.read_csv(path)


def _save_mock_cache(picks: list[MockPick], year: int) -> None:
    path = _cache_path(year)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(p) for p in picks]).to_csv(path, index=False)
    logger.debug(f"Mock cache saved: {path.name}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrape_mock_draft(year: int, force: bool = False) -> pd.DataFrame:
    """
    Return a DataFrame of pre-draft mock picks for the given year.

    For historical years, fetches the Wayback Machine snapshot of CBS Sports
    from the week before the draft. Caches results to avoid re-scraping.
    Returns an empty DataFrame if no data is available.
    """
    if not force:
        cached = _load_mock_cache(year)
        if cached is not None:
            print(f"[{year}] Mock draft loaded from cache ({len(cached)} picks).")
            return cached

    import datetime
    current_year = datetime.date.today().year

    if year >= current_year:
        # Current/upcoming draft — CBS Sports is JS-rendered, needs browser
        print(f"[{year}] Scraping CBS Sports mock draft directly...")
        source = _CBS_MOCK_URL
        html = _fetch_via_playwright(_CBS_MOCK_URL)
    else:
        # Historical — Wayback Machine archives are static HTML, use plain requests
        draft_date = _DRAFT_DATES.get(year)
        if not draft_date:
            logger.warning(f"No known draft date for {year}. Cannot find archive.")
            return pd.DataFrame()

        print(f"[{year}] Looking up Wayback Machine snapshot (pre-draft archive)...")
        wayback_url = _find_wayback_snapshot(draft_date)
        if not wayback_url:
            logger.warning(f"[{year}] No Wayback Machine snapshot found.")
            return pd.DataFrame()

        print(f"[{year}] Fetching archived page: {wayback_url}")
        source = wayback_url
        html = _fetch_static(wayback_url, timeout=45)

    picks = _parse_cbs_html(html, year, source)
    if not picks:
        logger.warning(f"[{year}] No picks parsed from mock draft page.")
        return pd.DataFrame()

    _save_mock_cache(picks, year)
    print(f"[{year}] Mock draft: {len(picks)} picks scraped and cached.")
    return pd.DataFrame([asdict(p) for p in picks])
