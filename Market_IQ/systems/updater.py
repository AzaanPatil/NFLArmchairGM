"""
Updater / diff engine — Market_IQ/systems/updater.py

Manages the state of Market_IQ's data and orchestrates incremental refreshes.
On each run it:
  1. Loads the last-known state from state.json
  2. Fetches transactions since the last check date
  3. Classifies new transactions and logs them
  4. If new SIGNED / EXTENSION / RESTRUCTURED events detected, marks contracts
     cache as stale so the next market_value call re-scrapes OTC
  5. Writes updated state.json

State file: Market_IQ/data/state.json
    {
        "last_transaction_date": "2026-07-02",   # last date we fetched transactions for
        "last_contract_fetch":   "2026-07-02T12:00:00",  # ISO datetime of last OTC scrape
        "total_transactions":    5821,
        "last_run":              "2026-07-03T08:00:00"
    }

Usage:
    from Market_IQ.systems.updater import run_update, load_state

    changes = run_update()
    print(changes["summary"])
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

_DATA_DIR   = Path(__file__).parent.parent / "data"
_STATE_FILE = _DATA_DIR / "state.json"

# Minimum hours between contract cache refreshes
_CONTRACT_REFRESH_HOURS = 24

# Transaction types that invalidate the contracts cache
_CONTRACT_INVALIDATING = {"SIGNED", "EXTENSION", "RESTRUCTURED", "FRANCHISE_TAG", "RELEASED"}


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _default_state() -> dict:
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    return {
        "last_transaction_date": yesterday,
        "last_contract_fetch":   "",
        "total_transactions":    0,
        "last_run":              "",
    }


def load_state() -> dict:
    if not _STATE_FILE.exists():
        return _default_state()
    try:
        return json.loads(_STATE_FILE.read_text())
    except Exception:
        return _default_state()


def save_state(state: dict) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(state, indent=2))


def _contracts_cache_is_stale(state: dict) -> bool:
    last = state.get("last_contract_fetch", "")
    if not last:
        return True
    try:
        age = datetime.now() - datetime.fromisoformat(last)
        return age > timedelta(hours=_CONTRACT_REFRESH_HOURS)
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Main update cycle
# ---------------------------------------------------------------------------

def run_update(force_contracts: bool = False) -> dict:
    """
    Execute one full update cycle.

    Returns a change summary dict:
        {
            "new_transactions": int,
            "by_type": {"SIGNED": 4, "RELEASED": 2, ...},
            "contracts_refreshed": bool,
            "summary": str,          # human-readable one-liner
            "transactions": DataFrame or None,
        }
    """
    from Market_IQ.data_scraping.scrape_transactions import (
        fetch_transactions_range,
        new_transactions_since,
    )
    from Market_IQ.data_scraping.scrape_contracts import scrape_all_positions

    state = load_state()
    today_str = date.today().isoformat()
    last_checked = state.get("last_transaction_date", "")

    changes: dict = {
        "new_transactions": 0,
        "by_type": {},
        "contracts_refreshed": False,
        "summary": "",
        "transactions": None,
    }

    # --- Step 1: Fetch new transactions ---
    if not last_checked or last_checked < today_str:
        start = last_checked or (date.today() - timedelta(days=7)).isoformat()
        print(f"[Updater] Fetching transactions from {start} to {today_str}...")
        fetch_transactions_range(start, today_str)

    new_tx = new_transactions_since(last_checked) if last_checked else pd.DataFrame()

    if not new_tx.empty:
        changes["new_transactions"] = len(new_tx)
        changes["by_type"] = new_tx["transaction_type"].value_counts().to_dict()
        changes["transactions"] = new_tx

        # Log notable moves
        signings = new_tx[new_tx["transaction_type"].isin({"SIGNED", "EXTENSION"})]
        for _, row in signings.head(10).iterrows():
            print(f"  [NEW] {row['transaction_type']}: {row['player_name']} "
                  f"({row.get('position', '?')}) - {row['team']}")

    # --- Step 2: Decide whether to refresh contracts ---
    new_types = set(changes.get("by_type", {}).keys())
    needs_refresh = (
        force_contracts
        or _contracts_cache_is_stale(state)
        or bool(new_types & _CONTRACT_INVALIDATING)
    )

    if needs_refresh:
        print("[Updater] Refreshing contract data from OverTheCap...")
        try:
            scrape_all_positions(force=True)
            state["last_contract_fetch"] = datetime.now().isoformat()
            changes["contracts_refreshed"] = True
            print("[Updater] Contracts refreshed.")
        except Exception as e:
            logger.warning(f"[Updater] Contract refresh failed: {e}")

    # --- Step 3: Update state ---
    state["last_transaction_date"] = today_str
    state["total_transactions"] = (
        state.get("total_transactions", 0) + changes["new_transactions"]
    )
    state["last_run"] = datetime.now().isoformat()
    save_state(state)

    n = changes["new_transactions"]
    by_type_str = ", ".join(f"{v}x {k}" for k, v in changes["by_type"].items()) or "none"
    changes["summary"] = (
        f"{n} new transaction(s) since {last_checked or 'never'} "
        f"[{by_type_str}]. "
        f"Contracts refreshed: {changes['contracts_refreshed']}."
    )
    print(f"[Updater] {changes['summary']}")
    return changes


# ---------------------------------------------------------------------------
# Transaction log helpers
# ---------------------------------------------------------------------------

def recent_signings(days: int = 14) -> pd.DataFrame:
    """Return SIGNED + EXTENSION transactions from the last N days."""
    from Market_IQ.data_scraping.scrape_transactions import load_recent_transactions
    df = load_recent_transactions(days)
    if df.empty:
        return df
    return df[df["transaction_type"].isin({"SIGNED", "EXTENSION"})].reset_index(drop=True)


def recent_releases(days: int = 14) -> pd.DataFrame:
    """Return RELEASED + WAIVED transactions from the last N days."""
    from Market_IQ.data_scraping.scrape_transactions import load_recent_transactions
    df = load_recent_transactions(days)
    if df.empty:
        return df
    return df[df["transaction_type"].isin({"RELEASED", "WAIVED"})].reset_index(drop=True)


def recent_trades(days: int = 14) -> pd.DataFrame:
    """Return TRADED transactions from the last N days."""
    from Market_IQ.data_scraping.scrape_transactions import load_recent_transactions
    df = load_recent_transactions(days)
    if df.empty:
        return df
    return df[df["transaction_type"] == "TRADED"].reset_index(drop=True)
