"""
NFL Transactions scraper — Market_IQ/data_scraping/scrape_transactions.py

Source: ESPN transactions feed (server-rendered, no JS required).
URL pattern: https://www.espn.com/nfl/transactions/_/date/{YYYYMMDD}

Fetches all roster moves for a given date range, classifies each one
(SIGNED, RELEASED, TRADED, WAIVED, RETIRED, IR, ACTIVATED, etc.),
and caches results as per-date CSVs.

Cache layout:
    Market_IQ/data/raw/transactions/{YYYY-MM-DD}.csv

Usage:
    from Market_IQ.data_scraping.scrape_transactions import (
        fetch_transactions_range, load_recent_transactions,
    )
    df = fetch_transactions_range("2026-06-01", "2026-07-03")
"""

from __future__ import annotations

import re
import sys
import time
import logging
from datetime import date, timedelta
from pathlib import Path
from dataclasses import asdict

import requests
import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from Core.transaction import Transaction, classify_transaction

logger = logging.getLogger(__name__)

_ESPN_BASE = "https://www.espn.com/nfl/transactions/_/date/{date}"
_CACHE_DIR = Path(__file__).parent.parent / "data" / "raw" / "transactions"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.9",
    "Accept-Language": "en-US,en;q=0.9",
}

# Regex to pull position abbreviation from description
_POS_RE = re.compile(
    r"\b(QB|WR|RB|TE|OT|OG|OL|C|DT|DE|NT|EDGE|OLB|ILB|LB|CB|S|FS|SS|DB|K|P|LS)\b"
)

# Regex to detect player name from ESPN "Team signed [POS] Player Name to ..."
_PLAYER_RE = re.compile(
    r"""
    (?:
        signed|released|waived|traded|retired|placed|activated|claimed|
        extended|restructured|tagged
    )
    \s+
    (?:[A-Z]{1,5}\s+)?          # optional position abbreviation
    ([A-Z][a-z]+(?:[\s\-'][A-Z][a-z]+)+)  # player name (Title Case)
    """,
    re.VERBOSE,
)

# ESPN abbreviation → standard (ESPN uses a slightly different set for team names)
_ESPN_TEAM_MAP: dict[str, str] = {
    "WSH": "WSH", "WAS": "WSH",
    "LAR": "LAR", "LA": "LAR",
    "JAC": "JAC", "JAX": "JAC",
    "KC": "KC",
}


def _normalise_team(team: str) -> str:
    t = team.strip().upper()
    return _ESPN_TEAM_MAP.get(t, t)


# ---------------------------------------------------------------------------
# HTML parser
# ---------------------------------------------------------------------------

def _parse_transactions_page(html: str, date_str: str) -> list[Transaction]:
    """
    Parse one ESPN transactions page into a list of Transaction objects.

    ESPN page structure (as of 2024-2025):
      <div class="transactions">
        <article>
          <h4 class="teamHeader">Team Name</h4>
          <ul>
            <li>Description text for one transaction</li>
            ...
          </ul>
        </article>
        ...
      </div>

    Fallback: look for any table with transaction rows.
    """
    soup = BeautifulSoup(html, "lxml")
    transactions: list[Transaction] = []

    # Primary: article-based layout (one article per team)
    articles = soup.find_all("article")
    for article in articles:
        # Try to find team name from header
        header = article.find(["h4", "h3", "h2"], class_=re.compile(r"team|header", re.I))
        team_name = header.get_text(strip=True) if header else ""

        # Look for team abbreviation from a logo <img> or link
        team_abbr = ""
        logo = article.find("img", src=re.compile(r"/nfl/", re.I))
        if logo:
            m = re.search(r"/nfl/(\w+)\.", logo.get("src", ""))
            if m:
                team_abbr = m.group(1).upper()

        for li in article.find_all("li"):
            desc = li.get_text(separator=" ", strip=True)
            if not desc:
                continue

            t_type = classify_transaction(desc)

            # Extract position
            pos_match = _POS_RE.search(desc)
            position = pos_match.group(1) if pos_match else ""

            # Extract player name (heuristic)
            player_name = _extract_player_name(desc)

            transactions.append(Transaction(
                date=date_str,
                team=_normalise_team(team_abbr or team_name),
                player_name=player_name,
                position=position,
                transaction_type=t_type,
                description=desc,
                source=_ESPN_BASE.format(date=date_str.replace("-", "")),
            ))

    if transactions:
        return transactions

    # Fallback: table-based layout
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        team_cell = cells[0].get_text(strip=True)
        desc_cell = cells[-1].get_text(strip=True)
        if not desc_cell or len(desc_cell) < 10:
            continue

        t_type = classify_transaction(desc_cell)
        pos_match = _POS_RE.search(desc_cell)
        position = pos_match.group(1) if pos_match else ""
        player_name = _extract_player_name(desc_cell)

        transactions.append(Transaction(
            date=date_str,
            team=_normalise_team(team_cell),
            player_name=player_name,
            position=position,
            transaction_type=t_type,
            description=desc_cell,
            source=_ESPN_BASE.format(date=date_str.replace("-", "")),
        ))

    return transactions


def _extract_player_name(desc: str) -> str:
    """Heuristic: extract the most likely player name from a transaction description."""
    # Try explicit pattern first
    m = _PLAYER_RE.search(desc)
    if m:
        return m.group(1).strip()

    # Fall back: look for "FirstName LastName" (Title Case pair) anywhere
    name_matches = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z']+){1,2}", desc)
    # Filter out known non-name words
    _skip = {"The", "New", "Los", "San", "Green", "Bay", "Kansas", "City",
              "Las", "Las Vegas", "New England", "New Orleans"}
    for nm in name_matches:
        if nm not in _skip and len(nm) > 4:
            return nm

    return ""


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------

def _cache_path(date_str: str) -> Path:
    return _CACHE_DIR / f"{date_str}.csv"


def _is_cached(date_str: str) -> bool:
    return _cache_path(date_str).exists()


def _save_cache(transactions: list[Transaction], date_str: str) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(t) for t in transactions]).to_csv(
        _cache_path(date_str), index=False
    )


def _load_cache(date_str: str) -> pd.DataFrame:
    p = _cache_path(date_str)
    # Zero-byte files mark days with no transactions — pd.read_csv chokes on them
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_transactions_day(date_str: str, force: bool = False) -> pd.DataFrame:
    """
    Fetch and cache all transactions for one date (YYYY-MM-DD).
    Returns an empty DataFrame if ESPN returns no data for that date.
    """
    if not force and _is_cached(date_str):
        return _load_cache(date_str)

    url = _ESPN_BASE.format(date=date_str.replace("-", ""))
    try:
        r = requests.get(url, headers=_HEADERS, timeout=20)
        r.raise_for_status()
        html = r.text
    except Exception as e:
        logger.warning(f"[{date_str}] Transactions fetch failed: {e}")
        return pd.DataFrame()

    transactions = _parse_transactions_page(html, date_str)
    if transactions:
        _save_cache(transactions, date_str)
        logger.debug(f"[{date_str}] {len(transactions)} transactions cached.")
    else:
        # Cache a header-only CSV so we don't re-fetch a day with no activity
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        from dataclasses import fields
        header = ",".join(f.name for f in fields(Transaction))
        _cache_path(date_str).write_text(header + "\n")
        logger.debug(f"[{date_str}] No transactions found.")

    return pd.DataFrame([asdict(t) for t in transactions]) if transactions else pd.DataFrame()


def fetch_transactions_range(
    start: str,
    end: str,
    force: bool = False,
    delay: float = 0.75,
) -> pd.DataFrame:
    """
    Fetch transactions for all dates in [start, end] (YYYY-MM-DD strings).
    Skips already-cached dates unless force=True.
    """
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    frames: list[pd.DataFrame] = []

    d = start_d
    while d <= end_d:
        ds = d.isoformat()
        df = fetch_transactions_day(ds, force=force)
        if not df.empty:
            frames.append(df)
        d += timedelta(days=1)
        time.sleep(delay)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_recent_transactions(days: int = 30) -> pd.DataFrame:
    """
    Load the last N days of cached transactions (no network call).
    """
    today = date.today()
    frames: list[pd.DataFrame] = []
    for i in range(days):
        ds = (today - timedelta(days=i)).isoformat()
        df = _load_cache(ds)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def new_transactions_since(last_checked: str) -> pd.DataFrame:
    """
    Return all cached transactions strictly after last_checked (YYYY-MM-DD).
    Does not make any network calls — call fetch_transactions_range first.
    """
    last_d = date.fromisoformat(last_checked)
    today = date.today()
    frames: list[pd.DataFrame] = []
    d = last_d + timedelta(days=1)
    while d <= today:
        df = _load_cache(d.isoformat())
        if not df.empty:
            frames.append(df)
        d += timedelta(days=1)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
