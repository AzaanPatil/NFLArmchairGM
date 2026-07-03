"""
Player Value Model — Player_IQ/systems/value_model.py

Gradient Boosting Regressor that predicts a player's market AAV (in $M)
from their stats, age, position, and durability.  Trained once on the
combined OTC contracts + ESPN stats dataset; updated whenever new data
arrives via Player_IQ/train.py.

The model emits four values per player:
  current_aav       — what the market would pay them TODAY
  peak_aav          — their career-peak market value (best season on record,
                       or age-curve-implied peak if data is limited)
  current_jj        — JJ chart pick-equivalent for current value
  peak_jj           — JJ chart pick-equivalent for peak value
  decline_curve     — dict of {age: factor} for next 6 years

Usage:
    from Player_IQ.systems.value_model import PlayerValueModel
    model = PlayerValueModel.load()   # load trained model
    result = model.value_report("Justin Jefferson", "WR", 26, stats_df, benchmarks)
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Player_IQ.systems.age_curves import (
    production_factor, peak_age_for, decline_projection,
    durability_multiplier, implied_peak_aav, age_adjusted_aav,
)
from Player_IQ.systems.features import player_to_features, FEATURE_NAMES

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).parent.parent / "data" / "models"
_MODEL_PATH = _MODEL_DIR / "value_model.joblib"
_META_PATH  = _MODEL_DIR / "value_model_meta.json"

# JJ pick value lookup (mirrors Draft_IQ)
# We import lazily to avoid circular deps at module load time
def _get_pick_value(v: float) -> int:
    """Return the closest overall pick number for a given JJ value."""
    from Draft_IQ.systems.draft_value import _JJ_CHART
    best_pick, best_diff = 1, float("inf")
    for pick, jj in _JJ_CHART.items():
        diff = abs(jj - v)
        if diff < best_diff:
            best_diff, best_pick = diff, pick
    return best_pick


def _jj_from_aav(aav: float, benchmarks: dict) -> tuple[float, int]:
    """
    Convert a predicted AAV to an approximate JJ chart value + pick equivalent.

    We scale relative to the QB market (highest AAV = top picks) using
    the benchmark data as an anchor.  The mapping is:
      ELITE tier (top-3 at pos) → ~pick 1-10  (1800-3000 JJ pts)
      STAR        → ~pick 11-32 (590-1700)
      STARTER_1   → ~pick 33-64 (270-590)
      STARTER_2   → ~pick 65-128 (44-265)
      BACKUP      → <44 JJ pts
    """
    from Market_IQ.systems.market_value import classify_contract, _TIER_JJ
    from Draft_IQ.systems.draft_value import _JJ_CHART

    # We need position to classify; use QB benchmark as a universal reference
    # This is approximate — the trade analyzer does the precise version
    # Here we just provide a rough pick range for the value report

    # Normalise AAV to a 0-100 score using QB market as reference
    # (QB is the most expensive position and anchors the top of the scale)
    qb_bm = benchmarks.get("QB", {})
    qb_top5 = qb_bm.get("top5_avg", 55.0)

    # Linear mapping: top QB AAV → 3000 JJ pts, 0 → 0
    jj_value = max(1.0, (aav / qb_top5) * 3000)
    pick_eq = _get_pick_value(jj_value)
    return round(jj_value, 1), pick_eq


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class PlayerValueModel:
    """
    Thin wrapper around a sklearn GradientBoostingRegressor with
    age-curve integration for peak/current/decline reporting.
    """

    def __init__(self, gbr=None, meta: dict | None = None):
        self._gbr = gbr
        self._meta = meta or {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        import joblib, json
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._gbr, _MODEL_PATH)
        _META_PATH.write_text(json.dumps(self._meta, indent=2))
        print(f"Model saved -> {_MODEL_PATH}")

    @classmethod
    def load(cls) -> "PlayerValueModel":
        import joblib, json
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No trained model found at {_MODEL_PATH}.\n"
                "Run:  python -m Player_IQ.main --train"
            )
        gbr = joblib.load(_MODEL_PATH)
        meta = json.loads(_META_PATH.read_text()) if _META_PATH.exists() else {}
        return cls(gbr=gbr, meta=meta)

    @property
    def is_trained(self) -> bool:
        return self._gbr is not None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        names: list[str],
        n_estimators: int = 300,
        learning_rate: float = 0.04,
        max_depth: int = 4,
        subsample: float = 0.8,
        random_state: int = 42,
    ) -> "PlayerValueModel":
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_val_score

        print(f"Training on {len(y)} samples, {X.shape[1]} features...")

        gbr = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            min_samples_leaf=3,
            random_state=random_state,
        )
        gbr.fit(X, y)

        # Cross-validated MAE
        cv_scores = cross_val_score(
            gbr, X, y, cv=5, scoring="neg_mean_absolute_error"
        )
        mae = -cv_scores.mean()
        print(f"5-fold CV MAE: ${mae:.2f}M AAV")

        # Feature importance
        importances = sorted(
            zip(FEATURE_NAMES, gbr.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )
        print("\nTop 10 feature importances:")
        for fname, imp in importances[:10]:
            print(f"  {fname:<30} {imp:.4f}")

        meta = {
            "n_samples": len(y),
            "cv_mae_millions": round(float(mae), 3),
            "feature_names": FEATURE_NAMES,
            "trained_at": __import__("datetime").datetime.now().isoformat(),
        }
        return cls(gbr=gbr, meta=meta)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_aav(self, X: np.ndarray) -> np.ndarray:
        """Raw model prediction — returns AAV in $M for each row in X."""
        if self._gbr is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        preds = self._gbr.predict(X)
        return np.maximum(preds, 1.0)   # floor at $1M (minimum contract)

    def value_report(
        self,
        player_name: str,
        position: str,
        current_age: float,
        stats_df: pd.DataFrame,
        benchmarks: dict,
        games_pct_1yr: float = 0.85,
        games_pct_2yr: float = 0.85,
        games_pct_3yr: float = 0.85,
        current_year: int = 2024,
    ) -> dict:
        """
        Full value report for one player:
          current_aav, peak_aav, current_jj, peak_jj,
          decline_curve, durability_mult, notes
        """
        # ---- Current value ----
        X_current = player_to_features(
            player_name, position, current_age, stats_df,
            current_year=current_year,
        )
        current_aav = float(self.predict_aav(X_current)[0])

        # Durability adjustment
        dur_mult = durability_multiplier(
            games_pct_1yr, games_pct_2yr, games_pct_3yr, position
        )
        current_aav_adj = round(current_aav * dur_mult, 2)

        # ---- Peak value (best season on record or age-curve implied) ----
        pos_peak = peak_age_for(position)
        peak_age = min(current_age, pos_peak)   # peak can't be in the future

        X_peak = player_to_features(
            player_name, position, current_age, stats_df,
            current_year=current_year,
            age_override=peak_age,
        )
        raw_peak = float(self.predict_aav(X_peak)[0])

        # Scale current → peak using age curve ratio
        cf = production_factor(current_age, position)
        pf = production_factor(peak_age, position)
        if cf > 0:
            peak_aav = round(raw_peak * (pf / cf), 2)
        else:
            peak_aav = raw_peak
        peak_aav = max(peak_aav, current_aav_adj)  # peak >= current always

        # ---- JJ chart equivalents ----
        current_jj, current_pick_eq = _jj_from_aav(current_aav_adj, benchmarks)
        peak_jj, peak_pick_eq = _jj_from_aav(peak_aav, benchmarks)

        # ---- Decline curve ----
        curve = decline_projection(position, current_age, years=6)
        aav_curve = {
            age: round(peak_aav * factor, 2)
            for age, factor in curve.items()
        }

        # ---- Notes ----
        notes = _build_notes(
            player_name, position, current_age, current_aav_adj,
            peak_aav, dur_mult, peak_age, games_pct_1yr,
        )

        return {
            "player_name":      player_name,
            "position":         position,
            "current_age":      current_age,
            "current_aav":      current_aav_adj,
            "peak_aav":         peak_aav,
            "peak_age":         peak_age,
            "current_jj_value": current_jj,
            "current_pick_eq":  current_pick_eq,
            "peak_jj_value":    peak_jj,
            "peak_pick_eq":     peak_pick_eq,
            "durability_mult":  round(dur_mult, 3),
            "aav_decline_curve": aav_curve,
            "notes":            notes,
        }

    def eval_summary(self) -> str:
        m = self._meta
        return (
            f"Model trained on {m.get('n_samples', '?')} samples | "
            f"CV MAE: ${m.get('cv_mae_millions', '?')}M | "
            f"Trained: {m.get('trained_at', 'unknown')}"
        )


def _build_notes(
    name: str, position: str, age: float,
    current_aav: float, peak_aav: float,
    dur_mult: float, peak_age: float,
    games_pct: float,
) -> list[str]:
    notes: list[str] = []
    from Player_IQ.systems.age_curves import retire_age_for, peak_age_for

    pos_peak = peak_age_for(position)
    retire = retire_age_for(position)
    years_left = max(0, retire - age)

    if age < pos_peak - 1:
        notes.append(
            f"{name} is {age:.0f}, still {pos_peak - age:.0f} year(s) from "
            f"projected peak — value should continue rising."
        )
    elif age <= pos_peak + 1:
        notes.append(f"{name} is at or near their peak production window.")
    else:
        pct_decline = round((1 - current_aav / peak_aav) * 100, 1)
        notes.append(
            f"{name} is {age - pos_peak:.0f} year(s) past peak; "
            f"~{pct_decline}% decline from career high ({peak_aav:.1f}M → {current_aav:.1f}M)."
        )

    if dur_mult < 0.90:
        notes.append(
            f"Durability discount applied ({dur_mult:.0%}) — "
            f"recent games-played rate below 75%."
        )
    elif dur_mult > 1.02:
        notes.append("Elite durability premium: consistently healthy.")

    if years_left <= 2:
        notes.append(
            f"Approaching typical retirement window for {position} "
            f"(~age {int(retire)}) — factor into multi-year deal risk."
        )

    return notes


# ---------------------------------------------------------------------------
# Heuristic fallback (no trained model)
# ---------------------------------------------------------------------------

def heuristic_value_report(
    player_name: str,
    position: str,
    current_age: float,
    benchmarks: dict,
    games_pct_1yr: float = 0.85,
    games_pct_2yr: float = 0.85,
    games_pct_3yr: float = 0.85,
    assumed_tier: str = "STARTER_1",
) -> dict:
    """
    Fallback when no ML model is trained yet. Uses positional market benchmarks
    and the age curve to estimate current and peak value.

    assumed_tier: ELITE / STAR / STARTER_1 / STARTER_2 / BACKUP
    """
    from Market_IQ.systems.market_value import expected_aav as _expected
    pos = position

    # Get fair-market AAV for the assumed tier
    fair_aav = _expected(pos, assumed_tier, benchmarks)  # type: ignore
    if fair_aav <= 0:
        fair_aav = 10.0   # fallback if no benchmark data

    # Current factor vs peak
    cf = production_factor(current_age, pos)
    pf = production_factor(peak_age_for(pos), pos)

    peak_aav    = round(fair_aav / cf * pf, 2)
    current_aav = round(fair_aav, 2)

    dur_mult = durability_multiplier(games_pct_1yr, games_pct_2yr, games_pct_3yr, pos)
    current_aav_adj = round(current_aav * dur_mult, 2)

    curve = decline_projection(pos, current_age, years=6)
    aav_curve = {age: round(peak_aav * factor, 2) for age, factor in curve.items()}

    current_jj, current_pick = _jj_from_aav(current_aav_adj, benchmarks)
    peak_jj,    peak_pick    = _jj_from_aav(peak_aav, benchmarks)

    notes = _build_notes(
        player_name, pos, current_age, current_aav_adj,
        peak_aav, dur_mult, peak_age_for(pos), games_pct_1yr,
    )
    notes.insert(0, "Note: ML model not trained — using market-benchmark heuristic.")

    return {
        "player_name":      player_name,
        "position":         pos,
        "current_age":      current_age,
        "current_aav":      current_aav_adj,
        "peak_aav":         peak_aav,
        "peak_age":         peak_age_for(pos),
        "current_jj_value": current_jj,
        "current_pick_eq":  current_pick,
        "peak_jj_value":    peak_jj,
        "peak_pick_eq":     peak_pick,
        "durability_mult":  round(dur_mult, 3),
        "aav_decline_curve": aav_curve,
        "notes":            notes,
    }
