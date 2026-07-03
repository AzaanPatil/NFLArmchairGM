"""
Training Pipeline — Player_IQ/train.py

End-to-end pipeline that:
  1. Loads the ESPN stats cache (Player_IQ/data/raw/all_stats.csv)
  2. Loads the OTC contracts cache (Market_IQ/data/raw/contracts/all.csv)
  3. Joins them via feature engineering
  4. Trains a GradientBoostingRegressor
  5. Saves the model to Player_IQ/data/models/value_model.joblib

Run once after scraping data:
    python -m Player_IQ.train

Or trigger from main.py:
    python -m Player_IQ.main --train
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_training(verbose: bool = True) -> None:
    from Player_IQ.data_scraping.scrape_player_stats import load_stats_cache
    from Market_IQ.data_scraping.scrape_contracts import load_contracts_cache
    from Player_IQ.systems.features import build_training_matrix
    from Player_IQ.systems.value_model import PlayerValueModel

    print("=" * 60)
    print("  Player_IQ — Value Model Training")
    print("=" * 60)

    # --- Load data ---
    print("\n[1/4] Loading contracts...")
    contracts_df = load_contracts_cache()
    print(f"  {len(contracts_df)} contracts loaded.")

    print("\n[2/4] Loading player stats...")
    try:
        stats_df = load_stats_cache()
        print(f"  {len(stats_df)} player-seasons loaded.")
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        print(
            "  Run the stats scraper first:\n"
            "    python -m Player_IQ.main --scrape\n"
            "  Or train with contracts only (lower accuracy, no stats features):"
        )
        _train_contracts_only(contracts_df)
        return

    # --- Feature engineering ---
    print("\n[3/4] Building feature matrix...")
    X, y, names = build_training_matrix(contracts_df, stats_df)
    print(f"  Feature matrix: {X.shape}  |  Targets: {y.shape}")
    print(f"  AAV range: ${y.min():.1f}M – ${y.max():.1f}M  |  "
          f"Mean: ${y.mean():.1f}M  |  Median: ${float(__import__('numpy').median(y)):.1f}M")

    if len(y) < 50:
        print(
            f"\n  WARNING: Only {len(y)} matched samples — model accuracy will be limited.\n"
            "  The more stats data you have (--scrape), the better."
        )

    # --- Train ---
    print("\n[4/4] Training GradientBoostingRegressor...")
    model = PlayerValueModel.train(X, y, names)
    model.save()

    print(f"\n  {model.eval_summary()}")
    print("\nTraining complete. Run predictions with:")
    print("  python -m Player_IQ.main --player \"Justin Jefferson\" --position WR --age 26")


def _train_contracts_only(contracts_df) -> None:
    """Minimal fallback: train on age + position + no stats. Lower accuracy."""
    import numpy as np
    from Player_IQ.systems.features import _POS_ENCODE, _normalise_pos
    from Player_IQ.systems.value_model import PlayerValueModel, FEATURE_NAMES

    print("\n  Training on contracts-only (no stats features)...")
    rows, targets = [], []
    for _, row in contracts_df.iterrows():
        aav = float(row.get("aav", 0) or 0)
        if aav <= 0:
            continue
        pos = _normalise_pos(str(row.get("position", "")))
        age = float(row.get("age") or 28)
        feat = [float(_POS_ENCODE.get(pos, 10))] + [age] + [0.0] * (len(FEATURE_NAMES) - 2)
        rows.append(feat)
        targets.append(aav)

    if not rows:
        print("  No valid contracts found. Aborting.")
        return

    X = np.array(rows, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    model = PlayerValueModel.train(X, y, [], n_estimators=100)
    model.save()
    print(f"\n  {model.eval_summary()}")


if __name__ == "__main__":
    run_training()
