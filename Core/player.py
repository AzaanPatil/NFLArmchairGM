from dataclasses import dataclass
from typing import Optional


@dataclass
class DraftPick:
    year: int
    round: int
    pick: int          # overall pick number
    team: str          # PFR 3-letter abbreviation
    player_name: str
    position: str
    college: str
    age: Optional[float] = None
    career_av: Optional[int] = None   # Career Approximate Value (PFR metric)
    draft_av: Optional[int] = None    # AV accrued with drafting team
    games_played: Optional[int] = None
