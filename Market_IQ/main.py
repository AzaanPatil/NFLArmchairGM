"""
Market_IQ pipeline entry point.

Usage:
    # Pull latest data and update local cache
    python -m Market_IQ.main --update

    # Show positional market summary (requires contracts cache)
    python -m Market_IQ.main --market
    python -m Market_IQ.main --market --position QB

    # Evaluate a specific player's contract against the market
    python -m Market_IQ.main --player "Lamar Jackson" --position QB --aav 52.0 --age 29

    # Evaluate a trade (player-for-picks, player-for-player, or picks-for-picks)
    python -m Market_IQ.main --trade \\
        --side-a "Eagles" --players "A.J. Brown:WR:27:32.0:3" --picks 23 \\
        --side-b "Titans" --picks 10 45

    # Show recent transactions (from local cache)
    python -m Market_IQ.main --transactions --days 7
    python -m Market_IQ.main --transactions --type SIGNED --days 14

    # Start the background auto-update scheduler
    python -m Market_IQ.scheduler
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _header(title: str) -> None:
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_update() -> None:
    from Market_IQ.systems.updater import run_update
    _header("Market_IQ - Updating data")
    run_update(force_contracts=True)


def cmd_market(position: str | None) -> None:
    from Market_IQ.data_scraping.scrape_contracts import load_contracts_cache
    from Market_IQ.systems.market_value import (
        build_benchmarks, positional_market_summary, annotate_contracts,
        top_contracts_by_position,
    )

    df = load_contracts_cache()
    benchmarks = build_benchmarks(df)

    if position:
        _header(f"Market_IQ - {position.upper()} Contract Market")
        annotated = annotate_contracts(df, benchmarks)
        top = top_contracts_by_position(annotated, position, n=15)
        print(top.to_string(index=False))
    else:
        _header("Market_IQ - Positional Market Summary (AAV in $M)")
        summary = positional_market_summary(df, benchmarks)
        print(summary.to_string(index=False))


def cmd_player(name: str, position: str, aav: float, age: int) -> None:
    from Market_IQ.data_scraping.scrape_contracts import load_contracts_cache
    from Market_IQ.systems.market_value import (
        build_benchmarks, classify_contract, market_verdict, expected_aav,
    )

    df = load_contracts_cache()
    benchmarks = build_benchmarks(df)

    tier    = classify_contract(position, aav, benchmarks)
    verdict = market_verdict(position, aav, age, benchmarks)
    fair    = expected_aav(position, tier, benchmarks)

    _header(f"Market_IQ - Contract Evaluation: {name}")
    print(f"  Position : {position.upper()}")
    print(f"  Age      : {age}")
    print(f"  AAV      : ${aav:.1f}M")
    print(f"  Tier     : {tier}")
    print(f"  Fair mkt : ${fair:.1f}M")
    diff = aav - fair
    sign = "+" if diff >= 0 else ""
    print(f"  Delta    : {sign}${diff:.1f}M vs. fair market")
    print(f"  Verdict  : {verdict}")


def cmd_trade(args: argparse.Namespace) -> None:
    from Market_IQ.data_scraping.scrape_contracts import load_contracts_cache
    from Market_IQ.systems.market_value import build_benchmarks
    from Market_IQ.systems.trade_analyzer import evaluate_trade, print_trade_result

    df = load_contracts_cache()
    benchmarks = build_benchmarks(df)

    def _parse_players(raw_list: list[str] | None) -> list[dict]:
        """Parse "Name:POS:Age:AAV:Years" strings into PlayerSpec dicts."""
        if not raw_list:
            return []
        players = []
        for item in raw_list:
            parts = item.split(":")
            if len(parts) < 4:
                print(f"  Warning: could not parse player spec '{item}' "
                      "(expected Name:POS:Age:AAV[:Years])")
                continue
            name, pos, age_s, aav_s = parts[0], parts[1], parts[2], parts[3]
            years = int(parts[4]) if len(parts) > 4 else 2
            players.append({
                "name": name,
                "position": pos,
                "age": int(age_s),
                "aav": float(aav_s),
                "years_remaining": years,
                "rookie_deal": False,
            })
        return players

    side_a = {
        "label":   args.side_a or "Side A",
        "players": _parse_players(getattr(args, "players_a", None)),
        "picks":   getattr(args, "picks_a", []) or [],
    }
    side_b = {
        "label":   args.side_b or "Side B",
        "players": _parse_players(getattr(args, "players_b", None)),
        "picks":   getattr(args, "picks_b", []) or [],
    }

    result = evaluate_trade(side_a, side_b, benchmarks)
    print_trade_result(result)


def cmd_transactions(days: int, tx_type: str | None) -> None:
    from Market_IQ.data_scraping.scrape_transactions import load_recent_transactions

    df = load_recent_transactions(days)
    if df.empty:
        print(f"\nNo cached transactions found for the last {days} days.")
        print("Run: python -m Market_IQ.main --update")
        return

    if tx_type:
        df = df[df["transaction_type"] == tx_type.upper()]

    label = f"{tx_type.upper()} " if tx_type else ""
    _header(f"Recent {label}Transactions - last {days} days ({len(df)} total)")

    cols = [c for c in ["date", "team", "player_name", "position",
                         "transaction_type", "description"]
            if c in df.columns]
    # Truncate description for readability
    if "description" in df.columns:
        df = df.copy()
        df["description"] = df["description"].str[:70]

    print(df[cols].sort_values("date", ascending=False).head(50).to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Market_IQ - NFL Contract & Trade Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--update",       action="store_true",
                        help="Pull latest transactions + contracts and update local cache")
    parser.add_argument("--market",       action="store_true",
                        help="Show positional contract market summary")
    parser.add_argument("--position",     type=str,
                        help="Filter market view or player eval to this position (e.g. QB, WR)")

    # Player contract evaluation
    parser.add_argument("--player",       type=str,   help="Player name for contract eval")
    parser.add_argument("--aav",          type=float, help="Contract AAV in $M")
    parser.add_argument("--age",          type=int,   help="Player age")

    # Trade evaluation
    parser.add_argument("--trade",        action="store_true", help="Evaluate a trade")
    parser.add_argument("--side-a",       type=str,   default="Side A")
    parser.add_argument("--side-b",       type=str,   default="Side B")
    parser.add_argument("--players-a",    nargs="+",  metavar="NAME:POS:AGE:AAV[:YRS]",
                        help="Players Side A gives (format: 'A.J. Brown:WR:27:32.0:3')")
    parser.add_argument("--players-b",    nargs="+",  metavar="NAME:POS:AGE:AAV[:YRS]",
                        help="Players Side B gives")
    parser.add_argument("--picks-a",      nargs="*",  type=int, metavar="PICK",
                        help="Overall pick numbers Side A gives")
    parser.add_argument("--picks-b",      nargs="*",  type=int, metavar="PICK",
                        help="Overall pick numbers Side B gives")

    # Recent transactions
    parser.add_argument("--transactions", action="store_true", help="Show recent transactions")
    parser.add_argument("--type",         type=str,
                        help="Filter transactions by type (SIGNED, RELEASED, TRADED, etc.)")
    parser.add_argument("--days",         type=int, default=14,
                        help="How many days back to look for transactions (default: 14)")

    args = parser.parse_args()

    if args.update:
        cmd_update()
    elif args.market or args.position:
        cmd_market(args.position)
    elif args.player:
        if not args.position or not args.aav or not args.age:
            parser.error("--player requires --position, --aav, and --age")
        cmd_player(args.player, args.position, args.aav, args.age)
    elif args.trade:
        cmd_trade(args)
    elif args.transactions:
        cmd_transactions(args.days, args.type)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
