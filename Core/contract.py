"""
Contract data model — Core/contract.py
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Contract:
    player_name: str
    team: str
    position: str
    age: Optional[int]
    years: int              # total years on deal
    years_remaining: int    # years left
    total_value: float      # millions USD
    guaranteed: float       # guaranteed money, millions
    aav: float              # average annual value, millions
    signed_year: int
    contract_type: str = ""   # "extension", "rookie", "free_agent", "restructure", etc.
    is_active: bool = True
    source: str = "OverTheCap"
    fetched_at: str = ""

    @property
    def guaranteed_pct(self) -> float:
        """Percentage of total value that is guaranteed."""
        if self.total_value <= 0:
            return 0.0
        return round(self.guaranteed / self.total_value * 100, 1)

    @property
    def is_rookie_deal(self) -> bool:
        return self.contract_type.lower() in ("rookie", "rookie_4th_year", "5th_year_option")
