"""
OverTheCap contract HISTORY scraper — Market_IQ/data_scraping/scrape_contract_history.py

Fetches every historical contract by position from overthecap.com's
contract-history pages. Unlike the position pages (current deals only),
these include the actual YEAR SIGNED and — crucially — "APY as % of cap
at signing", which is the era-normalized value target the Player_IQ
model trains on.

The pages are server-rendered (plain requests works; ~1MB each).

Cache layout:
    Market_IQ/data/raw/contract_history/{position}.csv
    Market_IQ/data/raw/contract_history/all.csv
    Market_IQ/data/raw/contract_history/cap_by_year.csv

Usage:
    from Market_IQ.data_scraping.scrape_contract_history import (
        scrape_contract_history, load_history_cache, load_cap_by_year,
    )
    df = scrape_contract_history()        # uses cache if < 7 days old
    caps = load_cap_by_year()             # {year: cap in $M}
"""

from __future__ import annotations

import re
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

_OTC_BASE = "https://overthecap.com/contract-history"

# Same slug scheme as the current-contract position pages
_POSITIONS: dict[str, str] = {
    "quarterback":      "QB",
    "wide-receiver":    "WR",
    "running-back":     "RB",
    "tight-end":        "TE",
    "left-tackle":      "OT",
    "right-tackle":     "OT",
    "guard":            "OG",
    "center":           "C",
    "defensive-tackle": "DT",
    "edge-rusher":      "EDGE",
    "linebacker":       "LB",
    "cornerback":       "CB",
    "safety":           "S",
    "kicker":           "K",
    "punter":           "P",
}

_CACHE_DIR = Path(__file__).parent.parent / "data" / "raw" / "contract_history"
_CACHE_MAX_AGE_DAYS = 7   # history changes slowly

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.9",
}


def _parse_dollars(text: str) -> float:
    """'$150,815,000' → 150.815 (millions)."""
    t = text.strip().lstrip("$").replace(",", "")
    if not t or t in ("-", "N/A"):
        return 0.0
    try:
        val = float(t)
    except ValueError:
        return 0.0
    return val / 1_000_000 if val >= 10_000 else val


def _parse_pct(text: str) -> float:
    """'24.1%' → 24.1 (percentage points of cap)."""
    t = text.strip().rstrip("%")
    if not t or t in ("-", "N/A"):
        return 0.0
    try:
        return float(t)
    except ValueError:
        return 0.0


def _parse_history_table(html: str, position_group: str) -> list[dict]:
    """
    Parse one contract-history page.

    Headers (2026): Player | Team | YearSigned | Years | (sp) | Value |
      APY | Guaranteed | (sp) | APY as % Of Cap At Signing | (sp) |
      InflatedValue | InflatedAPY | InflatedGuaranteed
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if not table:
        return []

    thead = table.find("thead")
    header_cells = (thead.find_all("tr")[-1].find_all(["th", "td"])
                    if thead and thead.find_all("tr") else [])
    col: dict[str, int] = {}
    for i, th in enumerate(header_cells):
        h = re.sub(r"[^a-z]", "", th.get_text(strip=True).lower())
        if h == "player":                col.setdefault("player", i)
        elif h == "team":                col.setdefault("team", i)
        elif h == "yearsigned":          col.setdefault("signed_year", i)
        elif h == "years":               col.setdefault("years", i)
        elif h == "value":               col.setdefault("total", i)
        elif h == "apy":                 col.setdefault("aav", i)
        elif h == "guaranteed":          col.setdefault("guaranteed", i)
        elif h.startswith("apyas"):      col.setdefault("cap_pct", i)

    tbody = table.find("tbody") or table
    records: list[dict] = []
    for row in tbody.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 6:
            continue

        def _cell(key: str) -> str:
            idx = col.get(key)
            return cells[idx].get_text(strip=True) if idx is not None and idx < len(cells) else ""

        player = _cell("player")
        if not player:
            continue
        try:
            signed_year = int(_cell("signed_year"))
        except ValueError:
            continue

        aav = _parse_dollars(_cell("aav"))
        cap_pct = _parse_pct(_cell("cap_pct"))
        if aav <= 0 or cap_pct <= 0:
            continue

        try:
            years = int(_cell("years"))
        except ValueError:
            years = 0

        records.append({
            "player_name": player,
            "team": _cell("team"),
            "position": position_group,
            "signed_year": signed_year,
            "years": years,
            "total_value": _parse_dollars(_cell("total")),
            "guaranteed": _parse_dollars(_cell("guaranteed")),
            "aav": aav,
            "cap_pct": cap_pct,
        })
    return records


# ---------------------------------------------------------------------------
# Cache + public API
# ---------------------------------------------------------------------------

def _all_path() -> Path:
    return _CACHE_DIR / "all.csv"


def _cache_is_fresh() -> bool:
    p = _all_path()
    if not p.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)
    return age < timedelta(days=_CACHE_MAX_AGE_DAYS)


def scrape_contract_history(force: bool = False, delay: float = 1.5) -> pd.DataFrame:
    """Scrape (or load cached) contract history for every position."""
    if not force and _cache_is_fresh():
        df = pd.read_csv(_all_path())
        print(f"[history] Loaded {len(df)} contracts from cache.")
        return df

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for slug, pos_group in _POSITIONS.items():
        url = f"{_OTC_BASE}/{slug}"
        print(f"[history/{pos_group}] Fetching {url} ...")
        try:
            r = requests.get(url, headers=_HEADERS, timeout=60)
            r.raise_for_status()
        except Exception as e:
            logger.warning(f"[history/{slug}] fetch failed: {e}")
            continue
        records = _parse_history_table(r.text, pos_group)
        if not records:
            logger.warning(f"[history/{slug}] no contracts parsed.")
            continue
        df = pd.DataFrame(records)
        df.to_csv(_CACHE_DIR / f"{slug}.csv", index=False)
        print(f"[history/{pos_group}] {len(df)} contracts.")
        frames.append(df)
        time.sleep(delay)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(_all_path(), index=False)
    print(f"\nContract history: {len(merged)} contracts -> {_all_path()}")

    _derive_cap_table(merged)
    return merged


def _derive_cap_table(history_df: pd.DataFrame) -> None:
    """
    Back out the league salary cap per year from the data itself:
    cap = APY / (cap_pct/100) for each contract; median per signing year.
    Keeps us exactly consistent with OTC's own cap accounting.
    """
    df = history_df[(history_df.aav > 0) & (history_df.cap_pct > 0)].copy()
    df["implied_cap"] = df["aav"] / (df["cap_pct"] / 100.0)
    caps = (df.groupby("signed_year")["implied_cap"].median()
              .round(1).reset_index()
              .rename(columns={"implied_cap": "cap_millions"}))
    caps.to_csv(_CACHE_DIR / "cap_by_year.csv", index=False)
    recent = caps[caps.signed_year >= caps.signed_year.max() - 3]
    print("Derived salary cap (recent):",
          {int(r.signed_year): float(r.cap_millions) for r in recent.itertuples()})


def load_history_cache() -> pd.DataFrame:
    if not _all_path().exists():
        raise FileNotFoundError(
            "No contract history cache. Run:\n"
            "  py -m Market_IQ.data_scraping.scrape_contract_history"
        )
    return pd.read_csv(_all_path())


def load_cap_by_year() -> dict[int, float]:
    """Return {signing_year: league cap in $M} derived from OTC data."""
    p = _CACHE_DIR / "cap_by_year.csv"
    if not p.exists():
        raise FileNotFoundError("No cap table. Run scrape_contract_history() first.")
    df = pd.read_csv(p)
    return {int(r.signed_year): float(r.cap_millions) for r in df.itertuples()}


if __name__ == "__main__":
    scrape_contract_history(force=True)
