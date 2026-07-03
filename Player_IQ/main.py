"""
Player_IQ pipeline entry point.

Usage:
    # Scrape ESPN stats for 2015-2024 (run once; ~20-30 min)
    python -m Player_IQ.main --scrape
    python -m Player_IQ.main --scrape --start 2019 --end 2024  # narrower range

    # Train the value model (requires contracts + stats cache)
    python -m Player_IQ.main --train

    # Get a player's current and peak value report
    python -m Player_IQ.main --player "Justin Jefferson" --position WR --age 26
    python -m Player_IQ.main --player "Lamar Jackson" --position QB --age 28

    # Evaluate a player-for-picks trade using market value
    python -m Player_IQ.main --trade \\
        --side-a Eagles --players-a "A.J. Brown:WR:27" \\
        --side-b Titans --picks-b 10 45

    # Model diagnostics
    python -m Player_IQ.main --model-info
"""

from __future__ import annotations

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


def _print_value_report(report: dict) -> None:
    """Pretty-print a value_report() dict."""
    _header(
        f"Player_IQ Value Report - "
        f"{report['player_name']} ({report['position']}, age {report['current_age']:.0f})"
    )
    print(f"  Current AAV       : ${report['current_aav']:.1f}M")
    print(f"  Peak AAV          : ${report['peak_aav']:.1f}M  (age {report['peak_age']:.0f})")
    print(f"  Durability adj.   : {report['durability_mult']:.0%}")
    print()
    print(f"  Current trade val : ~{report['current_jj_value']:.0f} JJ pts"
          f"  (~Pick #{report['current_pick_eq']} equivalent)")
    print(f"  Peak trade val    : ~{report['peak_jj_value']:.0f} JJ pts"
          f"  (~Pick #{report['peak_pick_eq']} equivalent)")
    print()
    print("  Projected AAV by age:")
    for age, aav in sorted(report["aav_decline_curve"].items()):
        bar_len = int(aav / report["peak_aav"] * 30)
        bar = "#" * bar_len
        marker = " <-- NOW" if age == int(report["current_age"]) else ""
        print(f"    Age {age:2d}  ${aav:5.1f}M  {bar}{marker}")
    if report.get("notes"):
        print()
        print("  Notes:")
        for note in report["notes"]:
            print(f"    - {note}")


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_scrape(start: int, end: int, force: bool) -> None:
    from Player_IQ.data_scraping.scrape_player_stats import scrape_all_stats
    _header(f"Player_IQ - Scraping ESPN Stats ({start}-{end})")
    print("This scrapes ESPN using Playwright. Expect ~2-3 min per year.")
    df = scrape_all_stats(start=start, end=end, force=force)
    print(f"\nDone. {len(df)} total player-seasons cached.")


def cmd_train() -> None:
    from Player_IQ.train import run_training
    run_training()


def cmd_player(
    player_name: str,
    position: str,
    age: float,
    games_pct: float,
    assumed_tier: str,
) -> None:
    # Try ML model first; fall back to heuristic
    from Player_IQ.systems.value_model import PlayerValueModel, heuristic_value_report
    from Market_IQ.data_scraping.scrape_contracts import load_contracts_cache
    from Market_IQ.systems.market_value import build_benchmarks

    contracts_df = load_contracts_cache()
    benchmarks   = build_benchmarks(contracts_df)

    try:
        model = PlayerValueModel.load()
        from Player_IQ.data_scraping.scrape_player_stats import load_stats_cache
        stats_df = load_stats_cache()
        report = model.value_report(
            player_name, position, age, stats_df, benchmarks,
            games_pct_1yr=games_pct,
            games_pct_2yr=games_pct,
            games_pct_3yr=games_pct,
        )
    except FileNotFoundError as e:
        print(f"\n  [{type(e).__name__}] {e}")
        print("  Falling back to market-benchmark heuristic...\n")
        report = heuristic_value_report(
            player_name, position, age, benchmarks,
            games_pct_1yr=games_pct,
            games_pct_2yr=games_pct,
            games_pct_3yr=games_pct,
            assumed_tier=assumed_tier,
        )

    _print_value_report(report)


def cmd_trade_with_players(args: argparse.Namespace) -> None:
    """
    Trade evaluation where player values come from Player_IQ model
    rather than manual AAV entry.
    """
    from Player_IQ.systems.value_model import PlayerValueModel, heuristic_value_report
    from Market_IQ.data_scraping.scrape_contracts import load_contracts_cache
    from Market_IQ.systems.market_value import build_benchmarks
    from Market_IQ.systems.trade_analyzer import evaluate_trade, print_trade_result

    contracts_df = load_contracts_cache()
    benchmarks   = build_benchmarks(contracts_df)

    def _get_report(spec: str) -> dict:
        """spec = "Name:POS:Age[:GamesPct]" """
        parts = spec.split(":")
        name, pos, age_s = parts[0], parts[1], parts[2]
        gp = float(parts[3]) if len(parts) > 3 else 0.85
        try:
            model = PlayerValueModel.load()
            from Player_IQ.data_scraping.scrape_player_stats import load_stats_cache
            stats_df = load_stats_cache()
            return model.value_report(name, pos, float(age_s), stats_df, benchmarks,
                                      games_pct_1yr=gp, games_pct_2yr=gp, games_pct_3yr=gp)
        except FileNotFoundError:
            return heuristic_value_report(name, pos, float(age_s), benchmarks,
                                          games_pct_1yr=gp, games_pct_2yr=gp, games_pct_3yr=gp)

    def _spec_to_player_dict(spec: str, report: dict) -> dict:
        parts = spec.split(":")
        return {
            "name": parts[0],
            "position": parts[1],
            "age": float(parts[2]),
            "aav": report["current_aav"],
            "years_remaining": 3,
            "rookie_deal": False,
        }

    players_a = [_spec_to_player_dict(s, _get_report(s))
                 for s in (getattr(args, "players_a", None) or [])]
    players_b = [_spec_to_player_dict(s, _get_report(s))
                 for s in (getattr(args, "players_b", None) or [])]

    side_a = {
        "label": args.side_a or "Side A",
        "players": players_a,
        "picks": getattr(args, "picks_a", []) or [],
    }
    side_b = {
        "label": args.side_b or "Side B",
        "players": players_b,
        "picks": getattr(args, "picks_b", []) or [],
    }

    result = evaluate_trade(side_a, side_b, benchmarks)
    print_trade_result(result)


def cmd_model_info() -> None:
    from Player_IQ.systems.value_model import PlayerValueModel
    try:
        model = PlayerValueModel.load()
        _header("Player_IQ - Model Info")
        print(f"  {model.eval_summary()}")
    except FileNotFoundError as e:
        print(f"\n  No model found: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Player_IQ - ML-based NFL Player Value Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--scrape",      action="store_true",
                        help="Scrape ESPN stats (Playwright, ~20-30 min for full range)")
    parser.add_argument("--start",       type=int, default=2015, help="First season to scrape")
    parser.add_argument("--end",         type=int, default=2024, help="Last season to scrape")
    parser.add_argument("--force",       action="store_true",  help="Re-scrape even if cached")

    parser.add_argument("--train",       action="store_true",  help="Train the value model")

    parser.add_argument("--player",      type=str,   help="Player name for value report")
    parser.add_argument("--position",    type=str,   help="Position (QB, WR, RB, ...)")
    parser.add_argument("--age",         type=float, help="Current age")
    parser.add_argument("--games-pct",   type=float, default=0.85,
                        help="Fraction of season played (durability; default 0.85)")
    parser.add_argument("--tier",        type=str, default="STARTER_1",
                        choices=["ELITE", "STAR", "STARTER_1", "STARTER_2", "BACKUP"],
                        help="Assumed tier for heuristic fallback (default: STARTER_1)")

    # Trade with player specs
    parser.add_argument("--trade",       action="store_true")
    parser.add_argument("--side-a",      type=str, default="Side A")
    parser.add_argument("--side-b",      type=str, default="Side B")
    parser.add_argument("--players-a",   nargs="+", metavar="NAME:POS:AGE[:GP]")
    parser.add_argument("--players-b",   nargs="+", metavar="NAME:POS:AGE[:GP]")
    parser.add_argument("--picks-a",     nargs="*", type=int, metavar="PICK")
    parser.add_argument("--picks-b",     nargs="*", type=int, metavar="PICK")

    parser.add_argument("--model-info",  action="store_true",
                        help="Show trained model metadata")

    args = parser.parse_args()

    if args.scrape:
        cmd_scrape(args.start, args.end, args.force)
    elif args.train:
        cmd_train()
    elif args.player:
        if not args.position or not args.age:
            parser.error("--player requires --position and --age")
        cmd_player(args.player, args.position, args.age, args.games_pct, args.tier)
    elif args.trade:
        cmd_trade_with_players(args)
    elif args.model_info:
        cmd_model_info()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
