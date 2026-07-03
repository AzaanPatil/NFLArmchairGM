"""
Auto-update scheduler — Market_IQ/scheduler.py

Runs the Market_IQ updater on a self-adjusting schedule based on the
NFL calendar. During free agency and the season it checks frequently;
during the dead offseason it backs off to avoid unnecessary requests.

NFL Calendar awareness:
  Free agency  (Mar 1 - Apr 15):   every 2 hours
  Draft week   (Apr 20 - May 5):   every 4 hours
  Regular season (Sep - Jan):      every 12 hours
  Training camp (Jul - Aug):       every 24 hours
  Dead offseason (May, Jun):       every 48 hours

Usage:
    # Long-running watcher process (keep terminal open or run as a service)
    python -m Market_IQ.scheduler

    # Or set this up in Windows Task Scheduler to run
    # "python -m Market_IQ.main --update" on a daily/hourly basis.

Press Ctrl+C to stop gracefully.
"""

from __future__ import annotations

import sys
import signal
import logging
import time
from datetime import datetime, date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import schedule
except ImportError:
    print("ERROR: 'schedule' package not installed. Run: pip install schedule>=1.2.0")
    sys.exit(1)

from Market_IQ.systems.updater import run_update, load_state

# ---------------------------------------------------------------------------
# Logging — write to file so the scheduler can run unattended
# ---------------------------------------------------------------------------

_LOG_DIR = Path(__file__).parent / "data"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(_LOG_DIR / "scheduler.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NFL calendar logic
# ---------------------------------------------------------------------------

def _current_interval_hours() -> int:
    """Return how many hours to wait between update runs, based on today's date."""
    today = date.today()
    m, d = today.month, today.day

    # Free agency (March 1 – April 15): very active
    if m == 3 or (m == 4 and d <= 15):
        return 2

    # Draft week (April 20 – May 5)
    if (m == 4 and d >= 20) or (m == 5 and d <= 5):
        return 4

    # Regular season + playoffs (September – January)
    if m in (9, 10, 11, 12, 1):
        return 12

    # Training camp / preseason (July – August)
    if m in (7, 8):
        return 24

    # Dead offseason (May 6 onwards, June)
    return 48


def _season_label() -> str:
    today = date.today()
    m, d = today.month, today.day
    if m == 3 or (m == 4 and d <= 15):    return "Free Agency"
    if (m == 4 and d >= 20) or (m == 5 and d <= 5): return "Draft Week"
    if m in (9, 10, 11, 12, 1):           return "Regular Season"
    if m in (7, 8):                        return "Training Camp"
    return "Offseason"


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

_running = True


def _update_job() -> None:
    label = _season_label()
    logger.info(f"[{label}] Starting scheduled update...")
    try:
        changes = run_update()
        logger.info(f"[{label}] {changes['summary']}")
    except Exception as e:
        logger.error(f"Update failed: {e}", exc_info=True)


def _reschedule(interval_hours: int) -> None:
    """Clear all jobs and register a fresh one at the new interval."""
    schedule.clear()
    schedule.every(interval_hours).hours.do(_update_job)
    logger.info(
        f"Scheduled next update every {interval_hours}h "
        f"[{_season_label()} mode]."
    )


def _handle_signal(sig, frame) -> None:
    global _running
    logger.info("Shutdown signal received — stopping scheduler.")
    _running = False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> None:
    global _running

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    state = load_state()
    last_run = state.get("last_run", "")
    logger.info(
        f"Market_IQ Scheduler starting. "
        f"Last run: {last_run or 'never'}. "
        f"Mode: {_season_label()}."
    )

    # Run once immediately on start, then hand off to schedule
    _update_job()

    interval = _current_interval_hours()
    _reschedule(interval)

    # Main loop: re-check interval every hour so the schedule adapts
    # when the NFL calendar crosses a boundary (e.g., free agency starts)
    last_interval = interval
    while _running:
        schedule.run_pending()

        new_interval = _current_interval_hours()
        if new_interval != last_interval:
            logger.info(
                f"NFL calendar shifted: "
                f"{last_interval}h -> {new_interval}h interval."
            )
            _reschedule(new_interval)
            last_interval = new_interval

        time.sleep(60)   # check every minute for pending jobs

    logger.info("Scheduler stopped cleanly.")


if __name__ == "__main__":
    run()
