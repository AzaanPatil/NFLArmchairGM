"""
Draft_IQ pipeline entry point.

Usage:
    # Open Time Capsule for a year (default 2024)
    python -m Draft_IQ.main --year 2024

    # Force rebuild even if capsule already exists on disk
    python -m Draft_IQ.main --year 2020 --rebuild

    # Show historical positional analysis (2010-2024 by default)
    python -m Draft_IQ.main --positional
    python -m Draft_IQ.main --positional --start 2015 --end 2024

    # Evaluate a hypothetical trade using the Jimmy Johnson chart
    #   Team A sends picks 1 and 64, Team B sends picks 4, 33, and 100
    python -m Draft_IQ.main --trade 1 64 --for 4 33 100
    python -m Draft_IQ.main --trade 1 64 --for 4 33 100 --team-a CLE --team-b NYG
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from Draft_IQ.systems.time_capsule import (
    open_capsule,
    compare_mock_to_actual,
    mock_accuracy_summary,
)
from Draft_IQ.systems.draft_value import (
    team_draft_capital,
    trade_summary,
    print_trade,
)
from Draft_IQ.systems.positional_value import (
    load_historical_picks,
    positional_avg_pick,
    positional_capital,
    positional_frequency_by_round,
    year_vs_historical,
)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _header(title: str) -> None:
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")


def _show_capsule(capsule, hist_df=None) -> None:
    """Print all Time Capsule analyses for one year."""
    year = capsule.year

    _header(f"NFL Time Capsule - {year} Draft")
    print(f"\n{capsule}\n")

    # --- Actual draft first 10 picks ---
    _header(f"Actual Draft: first 10 picks")
    print(
        capsule.actual_draft[
            ["round", "pick", "team", "player_name", "position", "college"]
        ]
        .head(10)
        .to_string(index=False)
    )

    # --- Mock vs actual comparison ---
    if capsule.has_mock:
        _header("Mock vs Actual comparison (first 15 matched players)")
        comparison = compare_mock_to_actual(capsule)
        print(
            comparison[
                ["player_name", "mock_pick", "actual_pick", "pick_delta", "position"]
            ]
            .head(15)
            .to_string(index=False)
        )

        _header("Mock Draft Accuracy Summary")
        summary = mock_accuracy_summary(capsule)
        for k, v in summary.items():
            print(f"  {k:<22} {v}")
    else:
        print("\n  (No mock draft data in this capsule - run the mock scraper to add it.)")

    # --- JJ draft capital by team for this year ---
    _header(f"Jimmy Johnson Draft Capital by Team - {year}")
    cap = team_draft_capital(capsule.actual_draft)
    print(
        cap[["team", "total_jj_value", "pick_count", "avg_jj_per_pick"]]
        .head(10)
        .rename(columns={
            "team": "Team",
            "total_jj_value": "Total JJ Pts",
            "pick_count": "Picks",
            "avg_jj_per_pick": "Avg JJ/Pick",
        })
        .to_string(index=False)
    )

    # --- This year vs historical positional investment ---
    if hist_df is not None:
        _header(f"Positional Capital Investment - {year} vs 2010-2024 Average")
        comparison_df = year_vs_historical(capsule.actual_draft, hist_df)
        print(
            comparison_df.rename(columns={
                "pos_group": "Position",
                "year_jj": "JJ Pts",
                "year_pct": "Year %",
                "hist_avg_pct": "Hist Avg %",
                "delta_pct": "Delta %",
            })
            .to_string(index=False)
        )
        print("\n  Delta % > 0 means over-invested vs. historical average.")


def _show_positional(start: int, end: int) -> None:
    """Print full positional analysis for the given year range."""
    hist = load_historical_picks(start, end)

    _header(f"Average Draft Slot by Position ({start}-{end})")
    print(positional_avg_pick(hist).to_string(index=False))

    _header(f"Total JJ Draft Capital by Position ({start}-{end})")
    print(positional_capital(hist).to_string(index=False))

    _header(f"Positional Frequency by Round - % of round ({start}-{end})")
    freq = positional_frequency_by_round(hist)
    print(freq.to_string())

    _header(f"Positional Round Distribution - where each position is found ({start}-{end})")
    from Draft_IQ.systems.positional_value import positional_round_heatmap
    print(positional_round_heatmap(hist).to_string())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NFL Time Capsule - Draft_IQ")
    sub = parser.add_subparsers(dest="cmd")

    # Default mode: open a capsule year
    cap_p = sub.add_parser("capsule", help="Open a Time Capsule for a specific year")
    cap_p.add_argument("--year", type=int, default=2024)
    cap_p.add_argument("--rebuild", action="store_true")
    cap_p.add_argument(
        "--no-historical",
        action="store_true",
        help="Skip year-vs-historical comparison (faster)",
    )

    # Positional analysis
    pos_p = sub.add_parser("positional", help="Historical positional value analysis")
    pos_p.add_argument("--start", type=int, default=2010)
    pos_p.add_argument("--end", type=int, default=2024)

    # Trade evaluation
    trade_p = sub.add_parser("trade", help="Evaluate a trade using the JJ chart")
    trade_p.add_argument("gives", nargs="+", type=int, metavar="PICK",
                         help="Picks that Team A is giving")
    trade_p.add_argument("--for", nargs="+", type=int, metavar="PICK",
                         dest="receives", required=True,
                         help="Picks that Team B is giving in return")
    trade_p.add_argument("--team-a", default="Team A")
    trade_p.add_argument("--team-b", default="Team B")

    # Legacy flat flags so the old --year / --rebuild invocation still works
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--positional", action="store_true")
    parser.add_argument("--start", type=int, default=2010)
    parser.add_argument("--end", type=int, default=2024)
    parser.add_argument("--trade", nargs="+", type=int, metavar="PICK")
    parser.add_argument("--for", nargs="+", type=int, metavar="PICK", dest="receives")
    parser.add_argument("--team-a", default="Team A")
    parser.add_argument("--team-b", default="Team B")

    args = parser.parse_args()

    # Subcommand routing (subparser wins; fall back to flat flags)
    if args.cmd == "positional" or args.positional:
        start = getattr(args, "start", 2010)
        end = getattr(args, "end", 2024)
        _show_positional(start, end)
        return

    if args.cmd == "trade" or args.trade:
        gives = args.gives if args.cmd == "trade" else args.trade
        receives = args.receives
        if not receives:
            parser.error("--trade requires --for PICK [PICK ...]")
        result = trade_summary(gives, receives, args.team_a, args.team_b)
        print_trade(result)
        return

    # Default: open capsule
    year = getattr(args, "year", 2024)
    rebuild = getattr(args, "rebuild", False)

    print(f"\n{'='*60}")
    print(f"  NFL Time Capsule - {year} Draft")
    print(f"{'='*60}")

    capsule = open_capsule(year, force=rebuild)

    hist_df = None
    no_historical = getattr(args, "no_historical", False)
    if not no_historical:
        try:
            hist_df = load_historical_picks(2010, 2024)
        except FileNotFoundError:
            pass

    _show_capsule(capsule, hist_df=hist_df)


if __name__ == "__main__":
    main()
