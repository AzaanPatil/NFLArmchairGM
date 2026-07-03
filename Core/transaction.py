"""
Transaction data model — Core/transaction.py

Represents a single NFL roster move: signing, release, trade, waiver,
retirement, IR placement, or activation.
"""
import re
from dataclasses import dataclass
from typing import Optional


# Ordered most-specific first so the first match wins
_TYPE_PATTERNS = [
    (re.compile(r"\bfranchise\s+tag\b", re.I),                    "FRANCHISE_TAG"),
    (re.compile(r"\bdesignated.{0,20}franchise\b", re.I),         "FRANCHISE_TAG"),
    (re.compile(r"\btransition\s+tag\b", re.I),                   "TRANSITION_TAG"),
    (re.compile(r"\bretired?\b", re.I),                           "RETIRED"),
    # "placed [player] on injured reserve" or "placed on IR"
    (re.compile(r"\bplaced\b.{0,80}(?:injur|\bon\s+IR\b)", re.I), "IR"),
    (re.compile(r"\bactivat(ed|ion)\b", re.I),                    "ACTIVATED"),
    (re.compile(r"\bclaim(ed|s)\b", re.I),                        "CLAIMED"),
    (re.compile(r"\bwai(?:ved|ver|ving)\b", re.I),                "WAIVED"),
    (re.compile(r"\btrad(ed|ing|e)\b", re.I),                     "TRADED"),
    (re.compile(r"\breleas(ed|ing)\b|\bcut\b", re.I),             "RELEASED"),
    # EXTENSION before SIGNED — "signed ... extension" should resolve to EXTENSION
    (re.compile(r"\bextend(ed|s|ing)\b", re.I),                   "EXTENSION"),
    (re.compile(r"\bextension\b", re.I),                          "EXTENSION"),
    (re.compile(r"\brestructur(ed|ing)\b", re.I),                 "RESTRUCTURED"),
    (re.compile(r"\bsign(ed|ing|s)\b", re.I),                     "SIGNED"),
]


def classify_transaction(description: str) -> str:
    """Infer transaction type from a raw description string."""
    for pattern, t_type in _TYPE_PATTERNS:
        if pattern.search(description):
            return t_type
    return "OTHER"


@dataclass
class Transaction:
    date: str               # YYYY-MM-DD
    team: str               # primary team involved (ESPN abbreviation)
    player_name: str
    position: str           # may be empty if not in source
    transaction_type: str   # see _TYPE_PATTERNS values above
    description: str        # raw text from source
    from_team: Optional[str] = None   # for trades: the team sending the player
    to_team: Optional[str] = None     # for trades: the team receiving the player
    source: str = "ESPN"
