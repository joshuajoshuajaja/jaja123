"""
MLB Board — data feeds.
statsapi.mlb.com (free, no key, reachable from Actions) supplies schedule,
probable pitchers, rosters, season rates and final results.
The Odds API supplies prices.
"""
from datetime import datetime, timezone

from core import (TEAMS, get_json, norm_person, team_id_from_name)

STATS = "https://statsapi.mlb.com/api/v1"
ODDS = "https://api.the-odds-api.com/v4"
SPORT = "baseball_mlb"


# ---------------------------------------------------------------- statsapi

def schedule(date_str):
    """All MLB games on a US-Eastern date, with probable pitchers + lineups."""
    j, code, _ = get_json(
        f"{STATS}/schedule",
        {"sportId": 1, "date": date_str,
         "hydrate": "probablePitcher,team,linescore,lineups"},
    )
    print(f"  schedule {date_str}: HTTP {code}")
    if not j:
        return []
    games = []
    for d in j.get("dates", []):
        for g in d.get("games", []):
            if g.get("gameType") not in ("R", "F", "D", "L", "W"):
                continue
            h = g["teams"]["home"]
            a = g["teams"]["away"]
            games.append({
                "gamePk": g["gamePk"],
                "gameDate": g.get("gameDate"),
                "state": g.get("status", {}).get("abstractGameState"),
                "detailed": g.get("status", {}).get("detailedState"),
                "home_id": h["team"]["id"],
                "away_id": a["team"]["id"],
                "home_name": h["team"]["name"],
                "away_name": a["team"]["name"],
                "home_score": h.get("score"),
                "away_score": a.get("score"),
                "home_sp": (h.get("probablePitcher") or {}).get("id"),
                "away_sp": (a.get("probablePitcher") or {}).get("id"),
                "home_sp_name": (h.get("probablePitcher") or {}).get("fullName"),
                "away_sp_name": (a.get("probablePitcher") or {}).get("fullName"),
                "lineups": g.get("lineups") or {},
                "venue": (g.get("venue") or {}).get("name"),
            })
    print(f"  schedule {date_str}: {len(games)} games")
    return games


def season_stats(season, group):
    """Every player's season line. group = 'hitting' | 'pitching'."""
    out, offset, limit = {}, 0, 1000
    while True:
        j, code, _ = get_json(
            f"{STATS}/stats",
            {"stats": "season", "group": group, "season": season,
             "sportId": 1, "limit": limit, "offset": offset,
             "playerPool": "All"},
        )
        if not j:
            print(f"  season {group}: HTTP {code} at offset {offset}")
            break
        splits = []
        for blk in j.get("stats", []):
            splits.extend(blk.get("splits", []))
        for s in splits:
            pid = (s.get("player") or {}).get("id")
            if pid:
                out[pid] = s.get("stat", {})
        if len(splits) < limit:
            break
        offset += limit
        if offset > 6000:
            break
    print(f"  season {group} {season}: {len(out)} players")
    return out


def roster(team_id):
    j, code, _ = get_json(f"{STATS}/teams/{team_id}/roster",
                          {"rosterType": "active"})
    if not j:
        print(f"  roster {team_id}: HTTP {code}")
        return {}
    out = {}
    for e in j.get("roster", []):
        p = e.get("person", {})
        if p.get("id"):
            out[norm_person(p.get("fullName"))] = p["id"]
    return out


def boxscore_hr(game_pk):
    """{player_id: home_runs} for a finished game."""
    j, code, _ = get_json(f"{STATS}/game/{game_pk}/boxscore")
    if not j:
        print(f"  boxscore {game_pk}: HTTP {code}")
        return None
    out = {}
    for side in ("home", "away"):
        for key, pl in (j.get("teams", {}).get(side, {})
                        .get("players", {}) or {}).items():
            pid = pl.get("person", {}).get("id")
            bat = (pl.get("stats", {}) or {}).get("batting", {}) or {}
            if pid is not None and "homeRuns" in bat:
                out[pid] = int(bat.get("homeRuns") or 0)
    return out


def league_hr_rate(hitting):
    hr = sum(int(s.get("homeRuns") or 0) for s in hitting.values())
    pa = sum(int(s.get("plateAppearances") or 0) for s in hitting.values())
    return (hr / pa) if pa > 5000 else None


# ---------------------------------------------------------------- odds api

class Odds:
    def __init__(self, key, regions="us,us2"):
        self.key = key
        self.regions = regions
        self.remaining = None
        self.used = None

    def _note(self, headers):
        try:
            self.remaining = int(headers.get("x-requests-remaining", -1))
            self.used = int(headers.get("x-requests-used", -1))
        except Exception:
            pass

    def events(self):
        """Free endpoint (0 credits): upcoming events with ids + start times."""
        j, code, h = get_json(f"{ODDS}/sports/{SPORT}/events",
                              {"apiKey": self.key})
        self._note(h)
        print(f"  odds events: HTTP {code}, {len(j or [])} events, "
              f"credits left {self.remaining}")
        return j or []

    def moneylines(self):
        """One call for the whole sport. Cost = 1 market x regions."""
        j, code, h = get_json(
            f"{ODDS}/sports/{SPORT}/odds",
            {"apiKey": self.key, "regions": self.regions,
             "markets": "h2h", "oddsFormat": "decimal"},
        )
        self._note(h)
        print(f"  odds h2h: HTTP {code}, {len(j or [])} events, "
              f"credits left {self.remaining}")
        return j or []

    def hr_props(self, event_id):
        """Cost = 1 market x regions, per event."""
        j, code, h = get_json(
            f"{ODDS}/sports/{SPORT}/events/{event_id}/odds",
            {"apiKey": self.key, "regions": self.regions,
             "markets": "batter_home_runs", "oddsFormat": "decimal"},
        )
        self._note(h)
        if code != 200:
            print(f"    props {event_id}: HTTP {code}")
        return j


def event_start(ev):
    try:
        return datetime.fromisoformat(
            ev["commence_time"].replace("Z", "+00:00"))
    except Exception:
        return datetime.now(timezone.utc)


def link_events_to_games(events, games):
    """
    Pair Odds API events with statsapi games by (home team, away team).
    Team identity is resolved to a stable MLB team id on both sides first,
    so a rename or a nickname-only spelling can't break the pairing.
    """
    by_pair = {}
    for g in games:
        by_pair[(g["home_id"], g["away_id"])] = g
    linked, misses = [], []
    for ev in events:
        h = team_id_from_name(ev.get("home_team"))
        a = team_id_from_name(ev.get("away_team"))
        g = by_pair.get((h, a))
        if g is None:
            misses.append(f"{ev.get('away_team')} @ {ev.get('home_team')}")
            continue
        linked.append((ev, g))
    if misses:
        print(f"  unlinked events ({len(misses)}): {'; '.join(misses[:6])}")
    return linked
