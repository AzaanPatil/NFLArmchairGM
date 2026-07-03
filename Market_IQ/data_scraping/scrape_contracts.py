"""
OverTheCap contract scraper — Market_IQ/data_scraping/scrape_contracts.py

Fetches current NFL contract data by position from overthecap.com.
OTC renders its contract tables server-side so plain requests works,
but we use Playwright as our fetch layer to be resilient against any
dynamic enhancements OTC may add.

Cache layout:
    Market_IQ/data/raw/contracts/{position}.csv   — per-position contract tables
    Market_IQ/data/raw/contracts/all.csv          — merged, deduplicated view

Usage:
    from Market_IQ.data_scraping.scrape_contracts import (
        scrape_all_positions, load_contracts_cache,
    )
    df = scrape_all_positions(force=False)   # uses cache if < 24h old
"""

from __future__ import annotations

import re
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import requests
import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from Core.contract import Contract

logger = logging.getLogger(__name__)

_OTC_BASE = "https://overthecap.com"
_OTC_CONTRACTS = _OTC_BASE + "/contracts"

# OTC position page slug → canonical position group
_POSITIONS: dict[str, str] = {
    "quarterback":      "QB",
    "wide-receiver":    "WR",
    "running-back":     "RB",
    "tight-end":        "TE",
    "offensive-tackle": "OT",
    "offensive-guard":  "OG",
    "center":           "C",
    "defensive-tackle": "DT",
    "edge-rusher":      "EDGE",
    "linebacker":       "LB",
    "cornerback":       "CB",
    "safety":           "S",
    "kicker":           "K",
    "punter":           "P",
}

_CACHE_DIR = Path(__file__).parent.parent / "data" / "raw" / "contracts"
_CACHE_MAX_AGE_HOURS = 24   # re-scrape if cache older than this

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.9",
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------------------------------------------------------------------------
# Dollar / year parsing helpers
# ---------------------------------------------------------------------------

def _parse_dollars(text: str) -> float:
    """Parse '$123,456,789' or '$123.5M' into a float in millions."""
    text = text.strip().lstrip("$").replace(",", "").strip()
    if not text or text in ("-", "N/A", ""):
        return 0.0
    # Handle 'M' suffix: "$15.5M" → 15.5
    if text.upper().endswith("M"):
        return float(text[:-1])
    # Handle 'K' suffix
    if text.upper().endswith("K"):
        return float(text[:-1]) / 1000
    # Plain number in dollars: convert to millions
    val = float(text)
    if val > 1_000_000:
        return val / 1_000_000
    return val


def _parse_int(text: str) -> Optional[int]:
    t = text.strip()
    if not t or t in ("-", "N/A"):
        return None
    digits = re.sub(r"[^\d]", "", t)
    return int(digits) if digits else None


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

def _parse_otc_table(html: str, position_group: str, source_url: str) -> list[Contract]:
    """
    Parse one OTC position page into a list of Contract objects.

    OTC table headers (as of 2024-2025):
        Player | Pos | Team | Age | Yrs | Total | Gtd | APY | Type
    Dollar columns contain formatted strings like "$45,000,000".
    """
    soup = BeautifulSoup(html, "lxml")
    contracts: list[Contract] = []

    # Find the contracts table — OTC uses a standard <table> with sortable headers
    table = soup.find("table")
    if not table:
        logger.warning(f"[{position_group}] No <table> found on {source_url}")
        return contracts

    # Map header names to column indices
    headers = []
    header_row = table.find("thead")
    if header_row:
        headers = [th.get_text(strip=True).lower() for th in header_row.find_all("th")]
    if not headers:
        # Some pages have headers in first tbody row
        first_row = table.find("tr")
        if first_row:
            headers = [td.get_text(strip=True).lower() for td in first_row.find_all(["th", "td"])]

    # Build a flexible column index mapping
    col = {}
    for i, h in enumerate(headers):
        if "player" in h:             col["player"] = i
        elif h in ("pos", "position"): col["pos"] = i
        elif h in ("team", "tm"):      col["team"] = i
        elif "age" in h:               col["age"] = i
        elif "yr" in h or "year" in h: col["years"] = i
        elif "total" in h:             col["total"] = i
        elif "gtd" in h or "guar" in h: col["guaranteed"] = i
        elif "apy" in h or "aav" in h:  col["aav"] = i
        elif "type" in h:              col["type"] = i

    now_str = datetime.now().isoformat()
    current_year = datetime.now().year
    tbody = table.find("tbody") or table

    for row in tbody.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 5:
            continue

        def _cell(key: str) -> str:
            idx = col.get(key)
            if idx is None or idx >= len(cells):
                return ""
            return cells[idx].get_text(strip=True)

        player_name = _cell("player")
        if not player_name or player_name.lower() in ("player", "name", ""):
            continue

        pos_raw = _cell("pos") or position_group
        team = _cell("team")
        age = _parse_int(_cell("age"))
        years_raw = _parse_int(_cell("years")) or 0
        total = _parse_dollars(_cell("total"))
        gtd = _parse_dollars(_cell("guaranteed"))
        aav = _parse_dollars(_cell("aav"))
        ctype = _cell("type").lower()

        # Skip obviously malformed rows
        if aav <= 0 and total <= 0:
            continue

        # If APY wasn't parsed but total + years are available, derive it
        if aav <= 0 and total > 0 and years_raw > 0:
            aav = total / years_raw

        contracts.append(Contract(
            player_name=player_name,
            team=team,
            position=pos_raw or position_group,
            age=age,
            years=years_raw,
            years_remaining=years_raw,   # OTC shows remaining years by default
            total_value=total,
            guaranteed=gtd,
            aav=aav,
            signed_year=current_year,
            contract_type=ctype,
            is_active=True,
            source=source_url,
            fetched_at=now_str,
        ))

    return contracts


# ---------------------------------------------------------------------------
# Fetch layer  (requests → Playwright fallback)
# ---------------------------------------------------------------------------

def _fetch_static(url: str) -> str:
    r = requests.get(url, headers=_HEADERS, timeout=20)
    r.raise_for_status()
    return r.text


def _fetch_playwright(url: str) -> str:
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
            # Wait for the table to appear
            try:
                page.wait_for_selector("table", timeout=10_000)
            except Exception:
                pass
            time.sleep(1)
            return page.content()
        finally:
            page.close()
            ctx.close()
            browser.close()


def _fetch_page(url: str, use_playwright: bool = False) -> str:
    if use_playwright:
        return _fetch_playwright(url)
    try:
        html = _fetch_static(url)
        # Quick sanity check: does it contain a real table with data?
        if "<tbody>" in html and "<td" in html:
            return html
        logger.debug(f"Static fetch returned no table for {url}; falling back to Playwright")
        return _fetch_playwright(url)
    except Exception as e:
        logger.warning(f"Static fetch failed ({e}); falling back to Playwright")
        return _fetch_playwright(url)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(position_slug: str) -> Path:
    return _CACHE_DIR / f"{position_slug}.csv"


def _cache_is_fresh(position_slug: str) -> bool:
    p = _cache_path(position_slug)
    if not p.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)
    return age < timedelta(hours=_CACHE_MAX_AGE_HOURS)


def _save_cache(contracts: list[Contract], position_slug: str) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(c) for c in contracts]).to_csv(
        _cache_path(position_slug), index=False
    )


def _load_cache(position_slug: str) -> pd.DataFrame:
    p = _cache_path(position_slug)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrape_position(position_slug: str, force: bool = False) -> pd.DataFrame:
    """
    Scrape or load cached contracts for one position.
    position_slug examples: "quarterback", "wide-receiver"
    """
    if not force and _cache_is_fresh(position_slug):
        df = _load_cache(position_slug)
        pos_group = _POSITIONS.get(position_slug, position_slug.upper())
        print(f"[{pos_group}] Contracts loaded from cache ({len(df)} rows).")
        return df

    pos_group = _POSITIONS.get(position_slug, position_slug.upper())
    url = f"{_OTC_BASE}/position/{position_slug}"
    print(f"[{pos_group}] Fetching contracts from {url} ...")

    html = _fetch_page(url)
    contracts = _parse_otc_table(html, pos_group, url)

    if not contracts:
        logger.warning(f"[{pos_group}] No contracts parsed — page structure may have changed.")
        return pd.DataFrame()

    _save_cache(contracts, position_slug)
    print(f"[{pos_group}] {len(contracts)} contracts scraped and cached.")
    return pd.DataFrame([asdict(c) for c in contracts])


def scrape_all_positions(force: bool = False, delay: float = 1.5) -> pd.DataFrame:
    """
    Scrape contracts for all tracked positions and return a merged DataFrame.
    Cached results are used unless force=True or cache is older than 24 hours.
    """
    frames: list[pd.DataFrame] = []
    for slug in _POSITIONS:
        df = scrape_position(slug, force=force)
        if not df.empty:
            frames.append(df)
        time.sleep(delay)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)

    # Write combined cache
    all_path = _CACHE_DIR / "all.csv"
    merged.to_csv(all_path, index=False)
    print(f"\nAll contracts: {len(merged)} rows -> {all_path}")
    return merged


def load_contracts_cache() -> pd.DataFrame:
    """Load the combined contracts cache. Raise if not found."""
    all_path = _CACHE_DIR / "all.csv"
    if not all_path.exists():
        raise FileNotFoundError(
            "No contracts cache found. Run scrape_all_positions() first:\n"
            "  python -m Market_IQ.main --update"
        )
    return pd.read_csv(all_path)
