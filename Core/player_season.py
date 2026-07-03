"""
PlayerSeason data model — Core/player_season.py

One player's statistics for one NFL regular season.
Fields are optional so a single dataclass covers every position —
populate what's available and leave the rest as None.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class PlayerSeason:
    player_name: str
    espn_id: str
    team: str
    position: str
    season: int
    age: Optional[float] = None
    games: int = 0
    games_started: int = 0

    # ---------- Passing (QB) ----------
    pass_yards: Optional[float] = None
    pass_tds: Optional[float] = None
    pass_ints: Optional[float] = None
    completions: Optional[float] = None
    attempts: Optional[float] = None
    comp_pct: Optional[float] = None
    pass_avg: Optional[float] = None    # yards per attempt
    qbr: Optional[float] = None
    passer_rating: Optional[float] = None

    # ---------- Rushing (QB / RB) ----------
    rush_yards: Optional[float] = None
    rush_tds: Optional[float] = None
    rush_attempts: Optional[float] = None
    rush_avg: Optional[float] = None    # yards per carry

    # ---------- Receiving (WR / TE / RB) ----------
    receptions: Optional[float] = None
    targets: Optional[float] = None
    rec_yards: Optional[float] = None
    rec_tds: Optional[float] = None
    rec_avg: Optional[float] = None     # yards per reception

    # ---------- Defense (EDGE / LB / DL / CB / S) ----------
    tackles: Optional[float] = None
    sacks: Optional[float] = None
    tfl: Optional[float] = None         # tackles for loss
    interceptions: Optional[float] = None
    pass_defenses: Optional[float] = None
    forced_fumbles: Optional[float] = None
    qb_hits: Optional[float] = None

    source: str = "ESPN"
    fetched_at: str = ""

    # ------------------------------------------------------------------
    # Derived per-game rates (all safe to call on any PlayerSeason)
    # ------------------------------------------------------------------

    def _pg(self, val: Optional[float]) -> float:
        """Return val / games, or 0 if val is None or games == 0."""
        if val is None or self.games == 0:
            return 0.0
        return round(val / self.games, 3)

    @property
    def games_pct(self) -> float:
        """Fraction of a 17-game season played (durability proxy)."""
        return round(self.games / 17, 3)

    # Passing per-game
    @property
    def pass_yds_pg(self) -> float: return self._pg(self.pass_yards)
    @property
    def pass_td_pg(self) -> float:  return self._pg(self.pass_tds)
    @property
    def pass_int_pg(self) -> float: return self._pg(self.pass_ints)

    # Rushing per-game
    @property
    def rush_yds_pg(self) -> float: return self._pg(self.rush_yards)
    @property
    def rush_td_pg(self) -> float:  return self._pg(self.rush_tds)

    # Receiving per-game
    @property
    def rec_yds_pg(self) -> float:  return self._pg(self.rec_yards)
    @property
    def rec_pg(self) -> float:      return self._pg(self.receptions)
    @property
    def rec_td_pg(self) -> float:   return self._pg(self.rec_tds)
    @property
    def tgt_pg(self) -> float:      return self._pg(self.targets)

    # Defense per-game
    @property
    def sacks_pg(self) -> float:    return self._pg(self.sacks)
    @property
    def tfl_pg(self) -> float:      return self._pg(self.tfl)
    @property
    def tackles_pg(self) -> float:  return self._pg(self.tackles)
    @property
    def int_pg(self) -> float:      return self._pg(self.interceptions)
    @property
    def pd_pg(self) -> float:       return self._pg(self.pass_defenses)
    @property
    def ff_pg(self) -> float:       return self._pg(self.forced_fumbles)
