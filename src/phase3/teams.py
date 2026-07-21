"""NBA team tricode -> full name and primary colour.

Factual reference data (team names and widely published brand colours) used to
render a polished scoreboard header and colour the win-probability chart. No
image assets are embedded; the dashboard uses these colours plus text so the
project stays free of third-party logo licensing.
"""

from __future__ import annotations

# tricode: (full name, primary colour, secondary colour)
TEAMS: dict[str, tuple[str, str, str]] = {
    "ATL": ("Atlanta Hawks", "#e03a3e", "#c1d32f"),
    "BOS": ("Boston Celtics", "#007a33", "#ba9653"),
    "BKN": ("Brooklyn Nets", "#000000", "#ffffff"),
    "CHA": ("Charlotte Hornets", "#1d1160", "#00788c"),
    "CHI": ("Chicago Bulls", "#ce1141", "#000000"),
    "CLE": ("Cleveland Cavaliers", "#860038", "#fdbb30"),
    "DAL": ("Dallas Mavericks", "#00538c", "#002b5e"),
    "DEN": ("Denver Nuggets", "#0e2240", "#fec524"),
    "DET": ("Detroit Pistons", "#c8102e", "#1d42ba"),
    "GSW": ("Golden State Warriors", "#1d428a", "#ffc72c"),
    "HOU": ("Houston Rockets", "#ce1141", "#c4ced4"),
    "IND": ("Indiana Pacers", "#002d62", "#fdbb30"),
    "LAC": ("LA Clippers", "#c8102e", "#1d428a"),
    "LAL": ("Los Angeles Lakers", "#552583", "#fdb927"),
    "MEM": ("Memphis Grizzlies", "#5d76a9", "#12173f"),
    "MIA": ("Miami Heat", "#98002e", "#f9a01b"),
    "MIL": ("Milwaukee Bucks", "#00471b", "#eee1c6"),
    "MIN": ("Minnesota Timberwolves", "#0c2340", "#236192"),
    "NOP": ("New Orleans Pelicans", "#0c2340", "#c8102e"),
    "NYK": ("New York Knicks", "#006bb6", "#f58426"),
    "OKC": ("Oklahoma City Thunder", "#007ac1", "#ef3b24"),
    "ORL": ("Orlando Magic", "#0077c0", "#c4ced4"),
    "PHI": ("Philadelphia 76ers", "#006bb6", "#ed174c"),
    "PHX": ("Phoenix Suns", "#1d1160", "#e56020"),
    "POR": ("Portland Trail Blazers", "#e03a3e", "#000000"),
    "SAC": ("Sacramento Kings", "#5a2d81", "#63727a"),
    "SAS": ("San Antonio Spurs", "#c4ced4", "#000000"),
    "TOR": ("Toronto Raptors", "#ce1141", "#000000"),
    "UTA": ("Utah Jazz", "#002b5c", "#00471b"),
    "WAS": ("Washington Wizards", "#002b5c", "#e31837"),
}


def team_name(tricode: str) -> str:
    """Full team name for a tricode, falling back to the tricode itself."""
    t = TEAMS.get((tricode or "").upper())
    return t[0] if t else (tricode or "—")


def team_color(tricode: str, fallback: str = "#888888") -> str:
    """Primary brand colour for a tricode."""
    t = TEAMS.get((tricode or "").upper())
    return t[1] if t else fallback


def team_initials(tricode: str) -> str:
    """The tricode itself, used as a compact 'logo' chip."""
    return (tricode or "?").upper()
