from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class DraftSnapshot:
    """
    A sealed record of everything known about an NFL draft year.

    'actual_draft'  — the real picks as they happened.
    'mock_draft'    — pre-draft analyst projections (may be empty if not yet scraped).
    'combine'       — pre-draft combine measurables (future: populated by combine scraper).
    'team_needs'    — positional needs per team entering the draft (future).

    All DataFrames default to empty so callers can check .empty rather than None.
    Metadata fields record provenance so the system knows what was scraped and when.
    """

    year: int

    # Core data layers — each is a DataFrame so they can be queried with pandas
    actual_draft: pd.DataFrame = field(default_factory=pd.DataFrame)
    mock_draft: pd.DataFrame = field(default_factory=pd.DataFrame)
    combine: pd.DataFrame = field(default_factory=pd.DataFrame)
    team_needs: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Provenance
    source_actual: str = "ESPN"
    source_mock: str = ""
    created_at: str = ""

    # -----------------------------------------------------------------------
    # Convenience properties
    # -----------------------------------------------------------------------

    @property
    def has_mock(self) -> bool:
        return not self.mock_draft.empty

    @property
    def has_combine(self) -> bool:
        return not self.combine.empty

    @property
    def pick_count(self) -> int:
        return len(self.actual_draft)

    def __repr__(self) -> str:
        layers = []
        if not self.actual_draft.empty:
            layers.append(f"actual={len(self.actual_draft)} picks")
        if not self.mock_draft.empty:
            layers.append(f"mock={len(self.mock_draft)} picks")
        if not self.combine.empty:
            layers.append(f"combine={len(self.combine)} players")
        return f"DraftSnapshot({self.year}, {', '.join(layers) or 'empty'})"
