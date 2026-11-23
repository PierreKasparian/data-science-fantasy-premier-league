"""Match Wyscout 2017/18 players to FPL players (2017/18 and 2018/19).

Usage (from the repo root):  python3 matching/match_players.py

Inputs : premier-league-2017-18/{players,lineups,substitutions,teams}.csv
         fpl/2017-18/players_raw.csv, fpl/2018-19/players_raw.csv
Outputs: matching/player_id_mapping.csv   one row per Wyscout player who played
         matching/unmatched.csv            players left without a match, for manual review

Method
 1. Minutes and team(s) per Wyscout player from line-ups + substitutions.
 2. Map each FPL team id to a Wyscout teamId: the Wyscout team whose players
    share the most surnames with that FPL team.
 3. Within each team, score every (Wyscout, FPL) pair by name similarity:
    shared name parts (accents removed, lower case) plus a string-similarity
    bonus. Pairs are taken greedily from the best score down, one-to-one.
 4. Players not matched within their team (e.g. moved club mid-season) get a
    second pass against all FPL players, requiring a stronger name match.
    A last pass catches spelling variants (Hegazy / Hegazi): same team,
    very similar surname and near-identical minutes.
 5. Each match is checked by comparing Wyscout minutes with FPL minutes.
 6. FPL 2018/19 ids are added through FPL's permanent player `code`.
"""
import csv
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WY = os.path.join(ROOT, "premier-league-2017-18")
FPL = os.path.join(ROOT, "fpl")
OUT = os.path.join(ROOT, "matching")


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def decode(s):
    """Wyscout names contain JSON escapes like 'Ag\\u00fcero' -> 'Agüero'."""
    if "\\u" in s:
        s = json.loads('"' + s.replace('"', '\\"') + '"')
    return s.strip()


# letters that are not "accented" in Unicode, so NFKD below does not simplify them
TRANSLIT = str.maketrans({"ø": "o", "Ø": "O", "ð": "d", "Ð": "D", "æ": "ae", "Æ": "Ae",
                          "ß": "ss", "ł": "l", "Ł": "L", "þ": "th", "đ": "d", "ı": "i"})


def norm(s):
    """'Agüero del Castillo' -> 'aguero del castillo', 'Jørgensen' -> 'jorgensen'"""
    s = unicodedata.normalize("NFKD", decode(s).translate(TRANSLIT))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z ]+", " ", s.replace("-", " ")).strip()


def tokens(*parts):
    """Name parts longer than one letter (drops initials like 'h.')."""
    return {t for p in parts for t in norm(p).split() if len(t) > 1}


# ---------- 1. Wyscout players: minutes and teams ----------
players = {r["playerId"]: r for r in read(f"{WY}/players.csv")}
teams = {r["teamId"]: r["name"] for r in read(f"{WY}/teams.csv")}
subs = read(f"{WY}/substitutions.csv")
on = {(s["matchId"], s["playerIn"]): int(s["minute"]) for s in subs}
off = {(s["matchId"], s["playerOut"]): int(s["minute"]) for s in subs}

minutes, team_minutes = Counter(), defaultdict(Counter)
for r in read(f"{WY}/lineups.csv"):
    key = (r["matchId"], r["playerId"])
    if r["status"] == "lineup":
        start, end = 0, off.get(key, 90)
    elif key in on:
        start, end = on[key], off.get(key, 90)
    else:
        continue                      # unused substitute
    red = int(r["redCards"])          # minute of the red card, 0 = none
    if red:
        end = min(end, red)
    m = max(0, min(end, 90) - start)
    minutes[r["playerId"]] += m
    team_minutes[r["playerId"]][r["teamId"]] += m

wy = {}
for pid, mins in minutes.items():
    if mins == 0:
        continue
    p = players.get(pid, {})
    wy[pid] = {
        "name": decode(p.get("shortName", f"unknown {pid}")),
        "full": f'{decode(p.get("firstName", ""))} {decode(p.get("lastName", ""))}'.strip(),
        "tok": tokens(p.get("firstName", ""), p.get("lastName", ""), p.get("shortName", "")),
        "last": tokens(p.get("lastName", "")),
        "teams": set(team_minutes[pid]),
        "team": team_minutes[pid].most_common(1)[0][0],
        "minutes": mins,
        "role": p.get("role", ""),
    }

# ---------- FPL players ----------
fpl = {}
for r in read(f"{FPL}/2017-18/players_raw.csv"):
    fpl[r["id"]] = {
        "name": r["web_name"], "full": f'{r["first_name"]} {r["second_name"]}',
        "tok": tokens(r["first_name"], r["second_name"], r["web_name"]),
        "last": tokens(r["second_name"], r["web_name"]),
        "team": r["team"], "code": r["code"], "minutes": int(r["minutes"]),
        "pos": {"1": "GKP", "2": "DEF", "3": "MID", "4": "FWD"}[r["element_type"]],
        "points": int(r["total_points"]),
    }
fpl_1819 = {r["code"]: r["id"] for r in read(f"{FPL}/2018-19/players_raw.csv")}

# ---------- 2. FPL team id -> Wyscout teamId ----------
surnames_by_wy_team = defaultdict(Counter)
for p in wy.values():
    surnames_by_wy_team[p["team"]].update(p["last"])
team_map = {}
for t in sorted({q["team"] for q in fpl.values()}, key=int):
    fpl_names = Counter(x for q in fpl.values() if q["team"] == t for x in q["last"])
    overlap = {w: sum((fpl_names & c).values()) for w, c in surnames_by_wy_team.items()}
    team_map[t] = max(overlap, key=overlap.get)
assert len(set(team_map.values())) == 20, "team mapping is not one-to-one"


# ---------- 3./4. scoring and greedy one-to-one matching ----------
def score(w, f):
    shared = w["tok"] & f["tok"]
    if not shared:
        return 0.0
    s = len(shared) + SequenceMatcher(None, norm(w["full"]), norm(f["full"])).ratio()
    if w["last"] & f["last"]:
        s += 1                         # surname agreement counts extra
    return s


def assign(pairs, threshold, method, result):
    used_w, used_f = set(result), {m["fpl_id"] for m in result.values()}
    for s, pid, fid in sorted(pairs, reverse=True):
        if s >= threshold and pid not in used_w and fid not in used_f:
            result[pid] = {"fpl_id": fid, "score": round(s, 2), "method": method}
            used_w.add(pid)
            used_f.add(fid)


matches = {}
same_team = [(score(w, f), pid, fid) for pid, w in wy.items() for fid, f in fpl.items()
             if team_map[f["team"]] in w["teams"]]
assign(same_team, 2.0, "same team", matches)
left = [(score(w, f), pid, fid) for pid, w in wy.items() if pid not in matches
        for fid, f in fpl.items()]
assign(left, 3.0, "any team", matches)


# spelling variants (Hegazy / Hegazi): same team, near-identical surname AND minutes
def surname_sim(w, f):
    return max((SequenceMatcher(None, a, b).ratio() for a in w["last"] for b in f["last"]), default=0)


left = [(surname_sim(w, f), pid, fid) for pid, w in wy.items() if pid not in matches
        for fid, f in fpl.items()
        if team_map[f["team"]] in w["teams"]
        and abs(w["minutes"] - f["minutes"]) <= max(90, 0.1 * w["minutes"])]
assign(left, 0.8, "similar surname + minutes", matches)

# ---------- 5./6. output ----------
rows = []
for pid, w in sorted(wy.items(), key=lambda kv: -kv[1]["minutes"]):
    m = matches.get(pid)
    f = fpl[m["fpl_id"]] if m else None
    diff = (w["minutes"] - f["minutes"]) if f else None
    rows.append({
        "wyscout_playerId": pid, "wyscout_name": w["name"], "wyscout_team": teams[w["team"]],
        "wyscout_role": w["role"], "wyscout_minutes": w["minutes"],
        "fpl_1718_id": m["fpl_id"] if m else "", "fpl_code": f["code"] if f else "",
        "fpl_1819_id": fpl_1819.get(f["code"], "") if f else "",
        "fpl_name": f["full"] if f else "", "fpl_web_name": f["name"] if f else "",
        "fpl_team": teams[team_map[f["team"]]] if f else "", "fpl_position": f["pos"] if f else "",
        "fpl_1718_points": f["points"] if f else "", "fpl_minutes": f["minutes"] if f else "",
        "match_method": m["method"] if m else "unmatched", "match_score": m["score"] if m else "",
        # minutes should agree; a big gap suggests a wrong match
        "minutes_check": "" if not f else ("ok" if abs(diff) <= max(90, 0.1 * w["minutes"]) else "CHECK"),
    })

with open(f"{OUT}/player_id_mapping.csv", "w", newline="", encoding="utf-8") as fh:
    wr = csv.DictWriter(fh, fieldnames=rows[0].keys())
    wr.writeheader()
    wr.writerows(rows)

matched_fpl = {r["fpl_1718_id"] for r in rows}
unmatched = [{"side": "wyscout", "id": r["wyscout_playerId"], "name": r["wyscout_name"],
              "team": r["wyscout_team"], "minutes": r["wyscout_minutes"]}
             for r in rows if r["match_method"] == "unmatched"]
unmatched += [{"side": "fpl", "id": fid, "name": f["full"], "team": teams[team_map[f["team"]]],
               "minutes": f["minutes"]}
              for fid, f in fpl.items() if f["minutes"] > 0 and fid not in matched_fpl]
with open(f"{OUT}/unmatched.csv", "w", newline="", encoding="utf-8") as fh:
    wr = csv.DictWriter(fh, fieldnames=["side", "id", "name", "team", "minutes"])
    wr.writeheader()
    wr.writerows(sorted(unmatched, key=lambda r: -r["minutes"]))

n = len(rows)
by = Counter(r["match_method"] for r in rows)
mins_total = sum(r["wyscout_minutes"] for r in rows)
mins_matched = sum(r["wyscout_minutes"] for r in rows if r["fpl_1718_id"])
print(f"Wyscout players who played: {n}")
print(f"  matched: {n - by['unmatched']} ({by['same team']} same team, {by['any team']} any team,"
      f" {by['similar surname + minutes']} similar surname + minutes)"
      f" = {mins_matched / mins_total:.1%} of all minutes")
print(f"  minutes check flagged: {sum(r['minutes_check'] == 'CHECK' for r in rows)}")
print(f"  with an FPL 2018/19 id: {sum(bool(r['fpl_1819_id']) for r in rows)}")
print(f"FPL 2017/18 players with minutes left unmatched: {sum(u['side'] == 'fpl' for u in unmatched)}")
