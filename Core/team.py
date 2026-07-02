from dataclasses import dataclass
from typing import Optional


@dataclass
class Team:
    abbreviation: str
    name: str
    city: str
    conference: str
    division: str


# PFR 3-letter abbreviation -> Team
# Includes historical franchises that appear in older draft records
PFR_TEAM_MAP: dict[str, Team] = {
    "ARI": Team("ARI", "Cardinals",     "Arizona",       "NFC", "West"),
    "ATL": Team("ATL", "Falcons",        "Atlanta",       "NFC", "South"),
    "BAL": Team("BAL", "Ravens",         "Baltimore",     "AFC", "North"),
    "BUF": Team("BUF", "Bills",          "Buffalo",       "AFC", "East"),
    "CAR": Team("CAR", "Panthers",       "Carolina",      "NFC", "South"),
    "CHI": Team("CHI", "Bears",          "Chicago",       "NFC", "North"),
    "CIN": Team("CIN", "Bengals",        "Cincinnati",    "AFC", "North"),
    "CLE": Team("CLE", "Browns",         "Cleveland",     "AFC", "North"),
    "DAL": Team("DAL", "Cowboys",        "Dallas",        "NFC", "East"),
    "DEN": Team("DEN", "Broncos",        "Denver",        "AFC", "West"),
    "DET": Team("DET", "Lions",          "Detroit",       "NFC", "North"),
    "GNB": Team("GNB", "Packers",        "Green Bay",     "NFC", "North"),
    "HOU": Team("HOU", "Texans",         "Houston",       "AFC", "South"),
    "IND": Team("IND", "Colts",          "Indianapolis",  "AFC", "South"),
    "JAX": Team("JAX", "Jaguars",        "Jacksonville",  "AFC", "South"),
    "KAN": Team("KAN", "Chiefs",         "Kansas City",   "AFC", "West"),
    "LAC": Team("LAC", "Chargers",       "Los Angeles",   "AFC", "West"),
    "LAR": Team("LAR", "Rams",           "Los Angeles",   "NFC", "West"),
    "LVR": Team("LVR", "Raiders",        "Las Vegas",     "AFC", "West"),
    "MIA": Team("MIA", "Dolphins",       "Miami",         "AFC", "East"),
    "MIN": Team("MIN", "Vikings",        "Minnesota",     "NFC", "North"),
    "NOR": Team("NOR", "Saints",         "New Orleans",   "NFC", "South"),
    "NWE": Team("NWE", "Patriots",       "New England",   "AFC", "East"),
    "NYG": Team("NYG", "Giants",         "New York",      "NFC", "East"),
    "NYJ": Team("NYJ", "Jets",           "New York",      "AFC", "East"),
    "PHI": Team("PHI", "Eagles",         "Philadelphia",  "NFC", "East"),
    "PIT": Team("PIT", "Steelers",       "Pittsburgh",    "AFC", "North"),
    "SEA": Team("SEA", "Seahawks",       "Seattle",       "NFC", "West"),
    "SFO": Team("SFO", "49ers",          "San Francisco", "NFC", "West"),
    "TAM": Team("TAM", "Buccaneers",     "Tampa Bay",     "NFC", "South"),
    "TEN": Team("TEN", "Titans",         "Tennessee",     "AFC", "South"),
    "WAS": Team("WAS", "Commanders",     "Washington",    "NFC", "East"),
    # Historical franchises
    "OAK": Team("OAK", "Raiders",        "Oakland",       "AFC", "West"),
    "SDG": Team("SDG", "Chargers",       "San Diego",     "AFC", "West"),
    "STL": Team("STL", "Rams",           "St. Louis",     "NFC", "West"),
    "RAM": Team("RAM", "Rams",           "Los Angeles",   "NFC", "West"),
    "HTX": Team("HTX", "Texans",         "Houston",       "AFC", "South"),
}


def lookup_team(abbreviation: str) -> Optional[Team]:
    return PFR_TEAM_MAP.get(abbreviation.upper())
