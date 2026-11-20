"""
props_market.py: the market layer under the props grid (mbm-nfl).

    snapshot   pull player prop prices once per build into data/odds/props_{season}_w{week}_{ts}.csv
    names      Odds API player names -> nflverse gsis_id through the season roster
    consensus  main line per player/market (mode, ties -> median) and the devigged mean P(over)
    anchor     solve proj_targets so the model's own P(over consensus line) equals the market's,
               cr and ypc held, ladders regenerated through the model's simulate() untouched
    board      one row per (player, market, line, side, book) with a price, model_pct, edge, EV

Design (2026-09-15). The Week 1 audit found the model's SHAPE right (realised MAE 1.03x what
its ladders implied, 50% band caught 54.5%, 80% band caught 81.8%) and its CENTRE compressed:
raw medians ran 1 yard under the closing line for small players and 17 under for stars. The
published grid therefore takes its centre from the market and its shape from the model.
The raw grid is kept beside it so the model's own centre stays auditable and Phase 2 can fix it.
"""
import csv, datetime as dt, glob, io, json, math, os, re, sys, unicodedata
from collections import Counter, defaultdict

import math
import numpy as np, pandas as pd, requests

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import props_model as M

API = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
MARKETS_DEFAULT = ["player_reception_yds", "player_receptions", "player_reception_yds_alternate", "player_receptions_alternate"]
MKT_SHORT = {"player_reception_yds": "rec_yds", "player_reception_yds_alternate": "rec_yds",
             "player_receptions": "receptions", "player_receptions_alternate": "receptions"}
TEAMS = {"Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF", "Carolina Panthers": "CAR",
         "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL", "Denver Broncos": "DEN",
         "Detroit Lions": "DET", "Green Bay Packers": "GB", "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
         "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
         "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG", "New York Jets": "NYJ",
         "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA",
         "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS"}
SNAP_COLS = ["snap_ts", "event_id", "commence", "home", "away", "book", "book_update", "market", "outcome", "player", "point", "price_dec", "price_am"]


# --- names -------------------------------------------------------------------------------

def norm_name(s):
    """accents stripped, apostrophes/periods/hyphens dropped, suffix dropped, letters only.
    'Ja'Marr Chase' -> 'jamarrchase', 'Amon-Ra St. Brown' -> 'amonrastbrown',
    'Kenneth Walker III' -> 'kennethwalker', 'Chris Godwin Jr.' -> 'chrisgodwin', 'Tre' Harris' -> 'treharris'."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z ]", "", s).lower().strip()
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s)
    return re.sub(r"[^a-z]", "", s)


def american(dec):
    """American odds for a decimal price. A 100% model probability (the under for a player ruled
    out, whose ladders are zeros, or on a line past the ladder's top rung) means a price of 1.0,
    which has no finite American number and crashed the Nov 12 and Nov 15 runs with a
    ZeroDivisionError. Anything above 99.9% prints as -100000."""
    dec = float(dec)
    return f"+{round((dec - 1) * 100):d}" if dec >= 2.0 else f"{round(-100 / max(dec - 1, 0.001)):d}"


def name_map(roster):
    """{normalized name -> [gsis_id,...]} and {(normalized name, team) -> gsis_id} from the roster."""
    by_name, by_name_team = defaultdict(list), {}
    for r in roster.itertuples():
        n = norm_name(r.full_name)
        by_name[n].append(r.gsis_id); by_name_team[(n, r.team)] = r.gsis_id
    return by_name, by_name_team


def match_player(name, team_codes, by_name, by_name_team):
    """Team-qualified first (the event's two teams), then unique name, else None."""
    n = norm_name(name)
    for t in team_codes:
        if (n, t) in by_name_team:
            return by_name_team[(n, t)]
    ids = by_name.get(n, [])
    return ids[0] if len(ids) == 1 else None


# --- snapshot ----------------------------------------------------------------------------

def _get(url, params, tries=3):
    for i in range(tries):
        try:
            return requests.get(url, params=params, timeout=60)
        except requests.exceptions.RequestException:
            import time; time.sleep(5 * (i + 1))
    return None


def snapshot(season, week, games, key, bookmakers, markets=MARKETS_DEFAULT, out_dir=None):
    """One odds pull for the week's games; never overwrites (ts in the filename).
    Cost: len(markets) x ceil(books/10) credits per game. Returns the path, or None."""
    out_dir = out_dir or os.path.join(HERE, "data", "odds"); os.makedirs(out_dir, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc); ts = now.strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(out_dir, f"props_{season}_w{week}_{ts}.csv")
    r = _get(f"{API}/sports/{SPORT}/events", dict(apiKey=key))
    if r is None or r.status_code != 200:
        print(f"odds: events list failed ({None if r is None else r.status_code})"); return None
    inv = {v: k for k, v in TEAMS.items()}
    want = {(inv.get(h), inv.get(a)) for h, a in zip(games.home_team, games.away_team)}
    evs = [e for e in r.json() if (e.get("home_team"), e.get("away_team")) in want]
    print(f"odds: {len(evs)} of {len(want)} games found on the feed; pulling {','.join(markets)} ({len(markets) * math.ceil(len(bookmakers.split(',')) / 10)} credits per game)")
    n = 0; rem = None
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(SNAP_COLS)
        for e in evs:
            rr = _get(f"{API}/sports/{SPORT}/events/{e['id']}/odds", dict(apiKey=key, bookmakers=bookmakers, markets=",".join(markets), oddsFormat="decimal"))
            if rr is None or rr.status_code != 200:
                code = None if rr is None else rr.status_code
                print(f"  {e['away_team']} @ {e['home_team']}: HTTP {code} {(rr.text[:120] if rr is not None else '')}")
                if code in (401, 403, 429):
                    # 401 with a valid key means the month's credits are gone (quota resets on the
                    # 1st); 429 is rate limiting. The rest of the slate would fail the same way,
                    # so stop rather than spend 16 requests on error pages. The grid publishes raw.
                    print("  odds pull aborted: key rejected or monthly quota exhausted")
                    break
                continue
            rem = rr.headers.get("x-requests-remaining", rem)
            d = rr.json()
            for bk in d.get("bookmakers", []):
                for mk in bk.get("markets", []):
                    for o in mk.get("outcomes", []):
                        if o.get("price") in (None, 0):
                            continue
                        w.writerow([now.isoformat(timespec="seconds"), d["id"], d["commence_time"], d["home_team"], d["away_team"], bk["key"], bk.get("last_update", ""),
                                    mk["key"], o.get("name"), o.get("description", ""), o.get("point", ""), float(o["price"]), american(o["price"])]); n += 1
    print(f"odds: {n} rows -> {os.path.relpath(path, HERE)}; {rem} credits left")
    return path


def latest_snapshot(season, week, out_dir=None):
    out_dir = out_dir or os.path.join(HERE, "data", "odds")
    files = sorted(glob.glob(os.path.join(out_dir, f"props_{season}_w{week}_*.csv")))
    return files[-1] if files else None


def load_snapshot(path, roster, season, week, out_dir=None):
    """Snapshot rows with pid attached; unmatched names written out; build fails if an unmatched
    player carries a receiving-yards line of 30 or more."""
    s = pd.read_csv(path, low_memory=False)
    s = s[s.market.isin(MKT_SHORT)].copy()
    s["price_dec"] = pd.to_numeric(s.price_dec, errors="coerce"); s["point"] = pd.to_numeric(s.point, errors="coerce")
    s = s.dropna(subset=["price_dec", "point", "player"])
    s["home_code"] = s.home.map(TEAMS); s["away_code"] = s.away.map(TEAMS)
    by_name, by_name_team = name_map(roster)
    s["pid"] = [match_player(p, (h, a), by_name, by_name_team) for p, h, a in zip(s.player, s.home_code, s.away_code)]
    s["mkt"] = s.market.map(MKT_SHORT); s["is_alt"] = s.market.str.endswith("_alternate")
    un = s[s.pid.isna()].groupby("player").agg(team_home=("home_code", "first"), team_away=("away_code", "first"), n_rows=("price_dec", "size"),
                                                 max_rec_yds_line=("point", lambda x: float(x[s.loc[x.index, "mkt"] == "rec_yds"].max()) if (s.loc[x.index, "mkt"] == "rec_yds").any() else np.nan)).reset_index()
    out_dir = out_dir or os.path.join(HERE, "data", "odds"); os.makedirs(out_dir, exist_ok=True)
    up = os.path.join(out_dir, f"unmatched_{season}_w{week}.csv"); un.to_csv(up, index=False)
    print(f"names: {s.pid.notna().sum()} rows matched, {len(un)} unmatched player name(s) -> {os.path.relpath(up, HERE)}")
    bad = un[un.max_rec_yds_line >= 30]
    if len(bad):
        sys.exit(f"UNMATCHED player with a receiving-yards line >= 30: {bad.player.tolist()}; fix the roster map before publishing")
    return s[s.pid.notna()].copy(), un


# --- consensus ---------------------------------------------------------------------------

def consensus(s):
    """Per (pid, mkt): the most common MAIN line across books (ties -> median), and the mean
    devigged P(over) across books quoting both sides at that line."""
    out = {}
    main = s[~s.is_alt]
    for (pid, mkt), g in main.groupby(["pid", "mkt"]):
        lines = g.groupby("book").point.first()
        cnt = Counter(lines.values); top = max(cnt.values()); cands = sorted(k for k, v in cnt.items() if v == top)
        line = float(cands[len(cands) // 2]) if len(cands) > 1 else float(cands[0])
        ps = []
        for bk, gb in g[g.point == line].groupby("book"):
            o = gb[gb.outcome == "Over"].price_dec; u = gb[gb.outcome == "Under"].price_dec
            if len(o) and len(u):
                io_, iu = 1 / float(o.iloc[0]), 1 / float(u.iloc[0]); ps.append(io_ / (io_ + iu))
        if ps:
            out[(pid, mkt)] = dict(line=line, p_over=float(np.mean(ps)), n_books=len(ps))
    return out


# --- anchoring ---------------------------------------------------------------------------

def ladder_at(row, mkt, line):
    """The model's P(over line) read off its own ladders (yards: interpolation; receptions:
    P(rec >= k+1) for a k.5 line, and for a whole-number line the average of the two adjacent
    counts, a push-excluded approximation)."""
    if mkt == "rec_yds":
        ps = np.array([row[f"p_over_{int(x)}"] for x in M.YGRID], float)
        return float(np.interp(line, M.YGRID, ps))
    k = math.floor(line)
    if line != k:
        return float(row.get(f"p_rec_{k + 1}", 0.0)) if k + 1 <= 10 else 0.0
    a = float(row.get(f"p_rec_{k}", 1.0)) if k >= 1 else 1.0; b = float(row.get(f"p_rec_{k + 1}", 0.0)) if k + 1 <= 10 else 0.0
    return (a + b) / 2


def anchor_row(row, line, target, spec, seed):
    """Bisection on proj_targets in [0.5x, 2.0x] so the model's P(yards > line) equals the
    market's. The simulation is reseeded identically at every evaluation, so P(over) is
    monotone in proj_targets and the search converges. Returns (proj_targets, clipped, sim)."""
    share, cr, ypc = float(row["share"]), float(row["cr"]), float(row["ypc"])
    att_mu = float(row["proj_targets"]) / max(share, 1e-6)
    def p_over(pt):
        M.RNG = np.random.default_rng(seed)
        tg, rc, y = M.simulate(att_mu, min(pt / att_mu, 0.6), cr, ypc, spec["kappa"], spec["alpha"], n=3000)
        return float(np.mean(y > line)), (tg, rc, y)
    lo, hi = 0.5 * float(row["proj_targets"]), 2.0 * float(row["proj_targets"])
    plo, _ = p_over(lo); phi, _ = p_over(hi)
    if target <= plo:
        return lo, True, p_over(lo)[1]
    if target >= phi:
        return hi, True, p_over(hi)[1]
    for _ in range(40):
        mid = (lo + hi) / 2; p, sim = p_over(mid)
        if abs(p - target) < 0.005:
            return mid, False, sim
        if p < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, False, p_over((lo + hi) / 2)[1]


def anchor_grid(raw, cons, spec):
    """Anchored copy of the raw grid. Players without a receiving-yards main line keep their
    raw projection. Ladders, medians and means are regenerated from the anchored simulation."""
    g = raw.copy()
    for c in ("market_line", "market_p_over", "gap_vs_market", "p_over_at_line",
              "yds_line", "yds_p_over", "rec_line", "rec_p_over", "mean_rec"):
        g[c] = np.nan
    g["anchored"] = False; g["anchor_clipped"] = False; g["rec_anchored"] = False
    ycols = [f"p_over_{int(x)}" for x in M.YGRID]; rcols = [f"p_rec_{k}" for k in M.RGRID]
    for i, row in g.iterrows():
        c = cons.get((row["pid"], "rec_yds"))
        if c is None or float(row.get("proj_targets", 0)) <= 0:
            continue
        line, target = c["line"], c["p_over"]
        g.loc[i, ["market_line", "market_p_over"]] = [line, round(target, 4)]
        g.loc[i, "gap_vs_market"] = round(float(row["med_yds"]) - line, 1)
        g.loc[i, "p_over_at_line"] = round(ladder_at(row, "rec_yds", line), 4)
        pt, clipped, (tg, rc, y) = anchor_row(row, line, target, spec, seed=abs(hash(str(row["pid"]))) % (2 ** 32))
        g.loc[i, "proj_targets"] = round(pt, 2); g.loc[i, "share"] = round(pt / (float(row["proj_targets"]) / max(float(row["share"]), 1e-6)), 3)
        g.loc[i, "med_yds"] = float(np.median(y)); g.loc[i, "mean_yds"] = float(y.mean()); g.loc[i, "med_rec"] = float(np.median(rc))
        g.loc[i, ycols] = [round(float(np.mean(y > x)), 4) for x in M.YGRID]
        g.loc[i, rcols] = [round(float(np.mean(rc >= k)), 4) for k in M.RGRID]
        g.loc[i, "anchored"] = True; g.loc[i, "anchor_clipped"] = bool(clipped)
        # per-market anchor columns for the Unabated export: the exact P(over) at each posted
        # line from the anchored simulation itself, no ladder interpolation
        g.loc[i, "yds_line"] = line; g.loc[i, "yds_p_over"] = round(float(np.mean(y > line)), 4)
        g.loc[i, "mean_rec"] = round(float(rc.mean()), 3)
        cr_ = cons.get((row["pid"], "receptions"))
        if cr_ is not None:
            rl = float(cr_["line"]); need = int(math.floor(rl)) + 1
            g.loc[i, "rec_line"] = rl; g.loc[i, "rec_p_over"] = round(float(np.mean(rc >= need)), 4)
            g.loc[i, "rec_anchored"] = True
    return g


def team_volume_report(raw, anchored):
    r = raw.groupby("team").proj_targets.sum(); a = anchored.groupby("team").proj_targets.sum()
    t = pd.DataFrame({"raw": r, "anchored": a}).fillna(0).round(1)
    t["flag"] = np.where(t.anchored > 40, "OVER 40", np.where(t.anchored < 22, "UNDER 22", ""))
    return t


# --- board -------------------------------------------------------------------------------

def build_board(s, grid, cons, out_dir, season, week, meta_extra):
    """One row per (player, market, line, side, book) with a price. model_pct from the anchored
    ladders. Devig two-way where the book posts both sides at that line, else the book's hold
    on the player's main line (devig_method=main_hold_proxy)."""
    gi = grid.set_index("pid")
    hold = {}
    for (bk, pid, mkt), g in s[~s.is_alt].groupby(["book", "pid", "mkt"]):
        o = g[g.outcome == "Over"].price_dec; u = g[g.outcome == "Under"].price_dec
        if len(o) and len(u):
            hold[(bk, pid, mkt)] = 1 / float(o.iloc[0]) + 1 / float(u.iloc[0])
    rows = []
    for (pid, mkt, line, side, bk), g in s.groupby(["pid", "mkt", "point", "outcome", "book"]):
        if pid not in gi.index:
            continue
        row = gi.loc[pid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        dec = float(g.price_dec.max()); imp = 1 / dec
        other = s[(s.pid == pid) & (s.mkt == mkt) & (s.point == line) & (s.book == bk) & (s.outcome != side)].price_dec
        if len(other):
            fair = imp / (imp + 1 / float(other.iloc[0])); method = "two_way"
        elif (bk, pid, mkt) in hold:
            fair = imp / hold[(bk, pid, mkt)]; method = "main_hold_proxy"
        else:
            fair = imp; method = "none"
        p_over = ladder_at(row, mkt, float(line)); model = p_over if side == "Over" else 1 - p_over
        rows.append(dict(player=g.player.iloc[0], pid=pid, team=row["team"], opp=row["opp"], kickoff_utc=row.get("kickoff_utc", ""), inj=row.get("inj", ""),
                         status_applied=row.get("status_applied", False), play_prob=row.get("play_prob", 1.0), anchored=row.get("anchored", False),
                         anchor_clipped=row.get("anchor_clipped", False), market=mkt, line=float(line), side=side, book=bk, book_odds=american(dec),
                         book_implied_pct=round(100 * imp, 1), book_fair_pct=round(100 * fair, 1), devig_method=method, model_pct=round(100 * model, 1),
                         fair_odds=american(1 / max(model, 1e-6)), edge_pp=round(100 * (model - fair), 1), ev_per_dollar=round(model * dec - 1, 4)))
    b = pd.DataFrame(rows)
    if len(b):
        b["rungs_same_player"] = b.groupby(["pid", "market"]).line.transform("size")
        b = b.sort_values("ev_per_dollar", ascending=False).reset_index(drop=True)
    b.to_csv(os.path.join(out_dir, "props_board.csv"), index=False)
    meta = dict(build_ts=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), n_rows=int(len(b)), **meta_extra)
    json.dump(meta, open(os.path.join(out_dir, "board_meta.json"), "w"), indent=1)
    md = [f"# NFL receiving props board, {season} week {week}", "",
          f"odds {meta.get('odds_ts', '')} | {meta.get('n_players_lined', 0)} players lined | {meta.get('n_unmatched', 0)} unmatched | {meta.get('n_clipped', 0)} clipped | anchored={meta.get('anchored')}", "",
          "| player | market | line | side | book | book odds | fair odds | edge pp | EV/$ |", "|---|---|---|---|---|---|---|---|---|"]
    for r in b.head(25).itertuples():
        md.append(f"| {r.player} ({r.team}) | {r.market} | {r.line:g} | {r.side} | {r.book} | {r.book_odds} | {r.fair_odds} | {r.edge_pp:+.1f} | {r.ev_per_dollar:+.3f} |")
    open(os.path.join(out_dir, "props_board.md"), "w").write("\n".join(md) + "\n")
    return b


def empty_board(out_dir, season, week, reason, meta_extra):
    pd.DataFrame(columns=["player", "market", "line", "side", "book"]).to_csv(os.path.join(out_dir, "props_board.csv"), index=False)
    json.dump(dict(build_ts=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), anchored=False, reason=reason, **meta_extra), open(os.path.join(out_dir, "board_meta.json"), "w"), indent=1)
    open(os.path.join(out_dir, "props_board.md"), "w").write(f"# NFL receiving props board, {season} week {week}\n\nno board: {reason}\n")


# --- loose ends --------------------------------------------------------------------------

def dropped_report(prev, cur, roster, out_path):
    """Every player in last week's grid is in this week's or in dropped_*.csv with a reason."""
    if prev is None or not len(prev):
        return pd.DataFrame(columns=["pid", "name", "team", "reason"])
    have = set(cur.pid); rows = []
    rmap = {r.gsis_id: (r.team, r.status) for r in roster.itertuples()} if roster is not None and "status" in roster.columns else {}
    rteam = {r.gsis_id: r.team for r in roster.itertuples()} if roster is not None else {}
    for r in prev.itertuples():
        if r.pid in have:
            continue
        if r.pid not in rteam:
            reason = "released_or_not_on_roster"
        elif rteam[r.pid] != r.team:
            reason = "traded"
        elif rmap.get(r.pid, ("", "ACT"))[1] != "ACT":
            reason = "IR_or_inactive_status"
        else:
            reason = "depth_cut"
        rows.append(dict(pid=r.pid, name=r.name, team=r.team, reason=reason))
    d = pd.DataFrame(rows, columns=["pid", "name", "team", "reason"]); d.to_csv(out_path, index=False)
    return d
