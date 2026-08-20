"""
MBM NFL daily runner (rebuild v1, frozen spec).
- Refreshes nflverse games + current-season pbp EPA aggregates
- Expected starters from depth charts + injury report (fallback: last actual starter)
- Boards upcoming games (7-day horizon): p(home), model margin, model total
- EARLY WINDOW: boards all of Week 1 from the moment the window opens, on a weekly
  cadence until seven days out, then the normal twice-daily runs. Self-reverting.
- CALIBRATION: weeks 1 to 4 publish the full priced line, tagged as forward test
- Grades logged predictions (ties push), rolling metrics + last-150 tripwire
Usage: python daily_nfl.py [--date YYYY-MM-DD]
"""
import argparse, io, json, os, re, sys, math, datetime as dt
import numpy as np, pandas as pd, requests
from collections import defaultdict
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tg_delta
import board_format as BF
import notify
import spread
import engine_nfl as E

# Shared files must be at least this new. A minimum rather than an exact match, so a later
# improvement to a shared file does not force an edit here.
#
# spread.py is deliberately NOT in this loop. Its VERSION is an integer with a different
# job: the runner compares it against sp_ver in the log and rebackfills history when they
# differ. Folding it in would raise on the string comparison, and bumping it for freshness
# reasons would trigger a full rebackfill for no reason.
SHARED_MIN = "2026-08-03"
for _mod in (BF, notify, tg_delta):
    _have = getattr(_mod, "VERSION", "")
    if _have < SHARED_MIN:
        raise ImportError(
            f"{_mod.__name__}.py is version {_have or 'unstamped'}, this runner needs "
            f"{SHARED_MIN} or newer. Copy the current shared files into this repo.")

# A capability check, because SHARED_MIN cannot do this job. One date gates three files, so
# raising it to demand the push-aware board_format also demanded a newer notify.py and
# tg_delta.py that were never changed, and the workflow failed on notify at 2026-08-09.
# Asking for the functions this runner actually calls names the missing thing exactly and
# stays true no matter what the other two shared files are stamped.
for _mod, _fns in ((BF, ("spread_threshold_push", "handicap")), (spread, ("cover3",))):
    _missing = [f for f in _fns if not hasattr(_mod, f)]
    if _missing:
        raise ImportError(
            f"{_mod.__name__}.py is missing {', '.join(_missing)}. This runner prices whole"
            f" numbers, which needs the 2026-08-19 copy of board_format.py and spread.py."
            f" notify.py and tg_delta.py are unchanged, leave them alone.")

H = {"User-Agent": "Mozilla/5.0"}
NV = "https://github.com/nflverse/nflverse-data/releases/download"
LOGS = os.path.join(HERE, "logs")
SPEC = json.load(open(os.path.join(HERE, "spec.json")))
CFG, C = SPEC["cfg"], SPEC["C"]
TRAIN_LO, TRAIN_HI = 2018, 2025          # frozen; 2026 = forward test
CAL_UNLOCK_WEEK = SPEC.get("cal_unlock_week", 5)   # value lines suppressed through week 4
# Tracks board_format rather than restating it. This is the console-side copy, and a
# local 0.05 here while board_format ran at 0.00 meant the Actions log and Discord
# quoted different prices for the same game.
MIN_EDGE = BF.MIN_EDGE
HORIZON_DAYS = SPEC.get("horizon_days", 7)

SP_LINES = [-7.5, -3.5, -0.5, 3.5, 7.5]     # margin thresholds logged for grading
SP_COLS = ["p_sp_m75", "p_sp_m35", "p_sp_m05", "p_sp_p35", "p_sp_p75"]

# Half points AND whole numbers. The old grid was np.arange(-40.5, 41.0, 1.0), half points
# only, so the board could never publish -3, -7, -10 or -14. Measured on 2,494 games from
# 2017 to 2025, an NFL margin lands exactly on one of those four in 33.3% of games, 15.0%
# on 3 alone. Skipping them meant skipping the third of the distribution where the push
# does the work. Priced through spread.cover3, which separates the push out instead of
# handing it to the away side. See board_format.spread_threshold_push.
KEY_GRID = np.arange(-40.0, 41.0, 0.5)

# --- Week 1 early window -------------------------------------------------------------
# A date rather than a wider horizon. A horizon big enough to reach Week 1 in August would
# still be that big in November, where it would park every game on the board for a month.
# This reaches forward to one fixed date and then stops mattering: once today passes it the
# board is back on HORIZON_DAYS with no edit, no revert step, nothing to forget.
EARLY_UNTIL = SPEC.get("early_until")            # "2026-09-14", the last Week 1 game
EARLY_WEEKDAY = SPEC.get("early_weekday", 1)     # Monday=0, so 1 is Tuesday
EARLY_DENSE_DAYS = SPEC.get("early_dense_days", 7)

# Which cron fired, set by the workflow. "manual" for workflow_dispatch, which always runs.
SLOT = os.environ.get("MBM_SLOT", "manual")

# ESPN, because nflverse injuries_2026.parquet 404s until the regular season and the QB
# layer has therefore been applying no injury filter at all. Probed 2026-08-19: 800 entries
# live, an athlete id on every one, and all 75 QB rows joining to the depth chart. The
# join key is espn_id, which nflverse already carries in depth_charts beside gsis_id, so
# there is no name matching anywhere in this path.
ESPN_INJ = "https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
# site.api.espn.com returns 403 on this route; site.web.api is the one that answers.
#
# Questionable is deliberately absent. It is the largest bucket by far (193 of 800 live)
# and most of it plays. Treating it as out would have benched Mahomes off a Questionable
# knee three weeks before a game he is expected to start.
ESPN_OUT = {"out", "injured reserve", "suspension", "doubtful"}


def spread_ladder(i, ph, Ph, Pa, exp, home, away, span=4):
    """Console-only ladder, now stepping every half point so the whole numbers appear.

    A push column comes with them. At -3 the push is the third of the row that decides
    whether the line is a bet, and a ladder that prints cover and no-cover without it
    reads as though 3 and 3.5 are the same proposition.
    """
    out = [f"    {'line':>7}  {home + ' cov':>12} {'needs':>7}  {'push':>6}  "
           f"{away + ' cov':>12} {'needs':>7}"]
    lo = math.floor(exp[i]) - span + 0.5
    for n in range(int(4 * span) + 1):
        k = lo + n * 0.5
        h, pu, a = spread.cover3(ph[i:i+1], Ph[i:i+1], Pa[i:i+1], k)
        h, pu, a = float(h[0]), float(pu[0]), float(a[0])
        live = max(h + a, 1e-9)
        out.append(f"    {-k:>+7.1f}  {h:>12.3f} {min_odds(h / live):>7.2f}  "
                   f"{pu:>6.3f}  {a:>12.3f} {min_odds(a / live):>7.2f}")
    return out


def min_odds(p):
    p = min(max(float(p), 0.01), 0.99)
    return math.ceil((1 + MIN_EDGE) / p * 100) / 100.0

def fetch_games():
    r = requests.get("https://github.com/nflverse/nfldata/raw/master/data/games.csv",
                     headers=H, timeout=120)
    g = pd.read_csv(io.StringIO(r.text))
    open(os.path.join(HERE, "data", "games.csv"), "w").write(r.text)
    return g

def refresh_current_pbp(season):
    cols = ["game_id","season","week","posteam","defteam","epa","pass","rush",
            "qb_dropback","passer_player_id","qb_epa"]
    try:
        r = requests.get(f"{NV}/pbp/play_by_play_{season}.parquet", headers=H, timeout=300)
        if r.status_code != 200:
            return None, None
        d = pd.read_parquet(io.BytesIO(r.content), columns=cols)
    except Exception:
        return None, None
    d = d[d.posteam.notna() & ((d["pass"] == 1) | (d["rush"] == 1))]
    t = d.groupby(["game_id","posteam"]).agg(plays=("epa","size"), epa_sum=("epa","sum"),
                                             dropbacks=("qb_dropback","sum")).reset_index()
    q = d[(d.qb_dropback == 1) & d.passer_player_id.notna()].groupby(
        ["game_id","passer_player_id"]).agg(db=("qb_epa","size"), qb_epa_sum=("qb_epa","sum")).reset_index()
    return t, q

def espn_outs():
    """ESPN athlete ids listed as unavailable. Empty set on any failure, so a bad response
    degrades to the nflverse path rather than benching nobody loudly or everybody quietly.

    The id is not a field on the payload. It sits in the playercard URL, which is stable
    and present on every entry; 800 of 800 parsed on the probe run.
    """
    try:
        j = requests.get(ESPN_INJ, headers={**H, "Accept": "application/json"},
                         timeout=60).json()
        out = set()
        for team in j.get("injuries", []):
            for it in team.get("injuries", []):
                if str(it.get("status", "")).strip().lower() not in ESPN_OUT:
                    continue
                for link in (it.get("athlete") or {}).get("links", []):
                    m = re.search(r"/id/(\d+)/", link.get("href", ""))
                    if m:
                        out.add(m.group(1))
                        break
        print(f"ESPN injuries: {len(out)} player(s) unavailable "
              f"({'/'.join(sorted(ESPN_OUT))})")
        return out
    except Exception as ex:
        print("ESPN injury feed unavailable:", type(ex).__name__)
        return set()


def expected_starters(season):
    """team -> qb gsis id, from depth charts minus everyone ruled out; {} on any failure."""
    try:
        r = requests.get(f"{NV}/injuries/injuries_{season}.parquet", headers=H, timeout=120)
        inj = pd.read_parquet(io.BytesIO(r.content)) if r.status_code == 200 else pd.DataFrame()
        outs = set()
        if len(inj) and "report_status" in inj.columns:
            w = inj[inj.week == inj.week.max()]
            outs = set(w.loc[w.report_status.astype(str).str.lower() == "out", "gsis_id"].dropna())
        else:
            # Not an error. This file does not exist until the regular season injury
            # reports start, which is exactly the stretch the early window runs in.
            print(f"nflverse injuries_{season} not published yet, ESPN is carrying this")
        eouts = espn_outs()
        r = requests.get(f"{NV}/depth_charts/depth_charts_{season}.parquet", headers=H, timeout=120)
        if r.status_code != 200:
            return {}
        dc = pd.read_parquet(io.BytesIO(r.content))
        pos = next(c for c in ("position", "pos_abb", "pos_name") if c in dc.columns)
        dep = next(c for c in ("depth_team", "pos_rank", "depth_position") if c in dc.columns)
        tcol = next(c for c in ("club_code", "team") if c in dc.columns)
        idc = next(c for c in ("gsis_id", "player_gsis_id") if c in dc.columns)
        qb = dc[dc[pos].astype(str).str.upper() == "QB"].copy()
        wk = next((c for c in ("week", "dt") if c in qb.columns), None)
        if wk:
            # Latest snapshot PER TEAM, not one global max. The file is a daily scrape and
            # every team currently shares a timestamp, so the two agree today. On the day
            # one team is scraped a second late, a global max keeps that team and silently
            # drops the other 31 to the last-starter fallback, which is a board-wide
            # mispricing that prints no error.
            qb = qb[qb[wk] == qb.groupby(tcol)[wk].transform("max")]
        qb[dep] = pd.to_numeric(qb[dep], errors="coerce")
        eid = "espn_id" if "espn_id" in qb.columns else None
        out, benched = {}, []
        for tm, grp in qb.sort_values(dep).groupby(tcol):
            for _, row in grp.iterrows():
                gid = row[idc]
                if gid != gid:
                    continue
                if gid in outs or (eid and str(row[eid]) in eouts):
                    benched.append(f"{tm} {row.get('player_name', gid)}")
                    continue
                out[E.REN.get(tm, tm)] = gid
                break
        print(f"expected starters: {len(out)}/32 teams resolved from the depth chart")
        if benched:
            print("  ruled out, next man up: " + ", ".join(benched))
        return out
    except Exception as ex:
        print("depth-chart feed unavailable, using last-starter fallback:", type(ex).__name__)
        return {}


def board_until(today):
    """Last game date to board. HORIZON_DAYS normally, reaching to EARLY_UNTIL while the
    early window is open. max() rather than a swap, so the window can only ever widen the
    board and never shorten it."""
    hi = today + pd.Timedelta(days=HORIZON_DAYS)
    if EARLY_UNTIL and today <= pd.Timestamp(EARLY_UNTIL):
        hi = max(hi, pd.Timestamp(EARLY_UNTIL))
    return hi


def cadence_skip(today, first_game):
    """True when this scheduled run should do nothing.

    Both crons fire every day and always will; GitHub Actions cannot express "weekly until
    a date, then twice daily", and a workflow that rewrites its own schedule is a worse
    problem than an early exit. So the cadence lives here, where it can be read.

    Weekly while the first boarded game is more than EARLY_DENSE_DAYS away, because nothing
    moves a Week 1 line in August except a depth chart change, and those are worth one
    look a week. Inside that, both runs, which is the normal in-season behaviour arriving
    on its own.
    """
    if SLOT == "manual" or not EARLY_UNTIL:
        return False
    if today > pd.Timestamp(EARLY_UNTIL) or first_game is None:
        return False
    days = (first_game - today).days
    if days <= EARLY_DENSE_DAYS:
        return False
    return not (SLOT == "am" and today.weekday() == EARLY_WEEKDAY)


def run(today_str):
    today = pd.Timestamp(today_str)
    season = today.year if today.month >= 3 else today.year - 1
    g = fetch_games()
    tcur, qcur = refresh_current_pbp(season)
    T = pd.read_parquet(os.path.join(HERE, "data", "team_game_epa.parquet"))
    Q = pd.read_parquet(os.path.join(HERE, "data", "qb_game_epa.parquet"))
    if tcur is not None and len(tcur):
        T = pd.concat([T[~T.game_id.isin(tcur.game_id)], tcur], ignore_index=True)
        Q = pd.concat([Q[~Q.game_id.isin(qcur.game_id)], qcur], ignore_index=True)
        T.to_parquet(os.path.join(HERE, "data", "team_game_epa.parquet"))
        Q.to_parquet(os.path.join(HERE, "data", "qb_game_epa.parquet"))

    g = g[(g.season >= 2017)].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(E.REN)
    g["date"] = pd.to_datetime(g.gameday)
    hi = board_until(today)
    upcoming = g[g.result.isna() & (g.date >= today) & (g.date <= hi)].copy()

    # Before the fit and before any feed is touched, so a skipped run costs seconds.
    first_game = upcoming.date.min() if len(upcoming) else None
    if cadence_skip(today, first_game):
        d = (first_game - today).days
        print(f"weekly cadence: {len(upcoming)} game(s) boarded, first in {d} days, "
              f"next scheduled work is the {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][EARLY_WEEKDAY]}"
              f" morning run. Twice daily from {d - EARLY_DENSE_DAYS} day(s) out.")
        return
    if EARLY_UNTIL and today <= pd.Timestamp(EARLY_UNTIL):
        print(f"early window open: boarding through {pd.Timestamp(EARLY_UNTIL):%Y-%m-%d} "
              f"({len(upcoming)} game(s)); reverts to a {HORIZON_DAYS}-day horizon by itself")

    exp = expected_starters(season)
    last_starter = {}
    for r in g[g.result.notna()].sort_values("date").itertuples():
        if r.home_qb_id == r.home_qb_id: last_starter[r.home_team] = r.home_qb_id
        if r.away_qb_id == r.away_qb_id: last_starter[r.away_team] = r.away_qb_id
    for i, r in upcoming.iterrows():
        g.loc[i, "home_qb_id"] = exp.get(r.home_team, last_starter.get(r.home_team))
        g.loc[i, "away_qb_id"] = exp.get(r.away_team, last_starter.get(r.away_team))

    tmap = {(r.game_id, r.posteam): r for r in T.assign(posteam=T.posteam.replace(E.REN)).itertuples()}
    qmap = defaultdict(list)
    for r in Q.itertuples(): qmap[r.game_id].append(r)
    g_sorted = g.sort_values(["date", "game_id"]).reset_index(drop=True)
    f = E.add_diffs(E.build(g_sorted, tmap, qmap, CFG))
    tr = (f.season >= TRAIN_LO) & (f.season <= TRAIN_HI) & f.margin.notna() & (~f.tie)
    sc = StandardScaler().fit(f.loc[tr, E.W].values)
    mdl = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(f.loc[tr, E.W].values),
                                                     f.loc[tr, "y"].values)
    trm = f.loc[(f.season >= TRAIN_LO) & (f.season <= TRAIN_HI) & f.margin.notna()]
    scm = StandardScaler().fit(trm[E.MG].values)
    mm = Ridge(alpha=5.0).fit(scm.transform(trm[E.MG].values), trm.margin.values)
    sct = StandardScaler().fit(trm[E.TT].values)
    mt = Ridge(alpha=5.0).fit(sct.transform(trm[E.TT].values), trm.total.values)
    base = float(f.loc[tr, "y"].mean())
    # Spread head. Same construction as NBA; the fitted weights come back spiky here
    # because NFL margins really do pile up on 3, 7 and 6.
    SH = spread.fit(f, E.W, (f.season >= TRAIN_LO) & (f.season <= TRAIN_HI)
                    & f.margin.notna() & (f.margin != 0), C)

    alerts = []
    os.makedirs(LOGS, exist_ok=True)
    pred_path = os.path.join(LOGS, "predictions.csv")
    log = pd.read_csv(pred_path) if os.path.exists(pred_path) else \
        pd.DataFrame(columns=["date","game_id","week","season","home","away","p_home","p_elo","model"])


    # One-time backfill: fill the spread columns for games logged before the spread head
    # existed, so the dashboard has history to grade instead of waiting a season. Self
    # healing, runs once, then the condition is false forever after.
    stale = ("sp_ver" not in log.columns or
             pd.to_numeric(log.get("sp_ver"), errors="coerce").max() != spread.VERSION)
    need_bf = stale or [c for c in SP_COLS + ["sp_margin", "sp_line", "p_sp_elo"]
               if c not in log.columns
               or log[c].isna().all()]
    if need_bf and len(log):
        try:
            have = f[f.game_id.astype(str).isin(log.game_id.astype(str))]
            if len(have):
                bph, bPh, bPa = spread.price(SH, have)
                # Historical rows already carry the moneyline this board published.
                # Anchor on it so the backfilled line agrees with what is logged.
                _lp = pd.Series(pd.to_numeric(log.p_home, errors="coerce").values,
                                index=log.game_id.astype(str))
                _lp = _lp[~_lp.index.duplicated(keep="last")]
                _ov = have.game_id.astype(str).map(_lp).astype(float).values
                bph = np.where(np.isnan(_ov), bph, _ov)
                _le = pd.Series(pd.to_numeric(log.p_elo, errors="coerce").values,
                                index=log.game_id.astype(str))
                _le = _le[~_le.index.duplicated(keep="last")]
                _be = have.game_id.astype(str).map(_le).astype(float).values
                _be = np.where(np.isnan(_be), bph, _be)
                _, ePh, ePa = spread.price(SH, have, ph=_be)
                bf = pd.DataFrame({"game_id": have.game_id.astype(str).values,
                                   "sp_margin": spread.expected_margin(bph, bPh, bPa),
                                   "sp_line": spread.median_line(bph, bPh, bPa),
                                   "p_sp_elo": spread.cover(_be, ePh, ePa, 3.5),
                                   "sp_ver": spread.VERSION})
                for nm, k in zip(SP_COLS, SP_LINES):
                    bf[nm] = spread.cover(bph, bPh, bPa, k)
                log = log.drop(columns=[c for c in bf.columns if c != "game_id"
                                        and c in log.columns], errors="ignore")
                log["game_id"] = log.game_id.astype(str)
                log = log.merge(bf, on="game_id", how="left")
                print(f"spread backfill v{spread.VERSION}: "
                      f"filled {bf.sp_margin.notna().sum()} historical rows")
        except Exception as ex:
            print(f"spread backfill skipped: {type(ex).__name__}: {ex}")

    board = None
    if len(upcoming):
        rows = f[f.game_id.isin(upcoming.game_id)].copy()
        p = mdl.predict_proba(sc.transform(rows[E.W].values))[:, 1]
        margins = mm.predict(scm.transform(rows[E.MG].values))
        totals = mt.predict(sct.transform(rows[E.TT].values))
        pe = 1/(1+10**(-((rows.h_elo + CFG["HA"]) - rows.a_elo)/400))
        cal = rows.week < CAL_UNLOCK_WEEK
        # anchor on the moneyline this runner actually logs, not a refit
        sp_ph, sp_Ph, sp_Pa = spread.price(SH, rows, ph=p)
        sp_exp = spread.expected_margin(sp_ph, sp_Ph, sp_Pa)
        sp_line = spread.median_line(sp_ph, sp_Ph, sp_Pa)
        # Elo floor for the spread: the rating's own win probability pushed through the
        # same magnitude model. Isolates what the feature layer adds on this market, the
        # way p_elo does for the moneyline.
        _pe = pe.values if hasattr(pe, "values") else pe
        _, sp_ePh, sp_ePa = spread.price(SH, rows, ph=_pe)
        board = pd.DataFrame({"date": rows.date.dt.strftime("%Y-%m-%d"), "game_id": rows.game_id,
                              "week": rows.week, "season": rows.season, "home": rows.home,
                              "away": rows.away, "p_home": p, "p_elo": pe.values,
                              "sp_margin": sp_exp, "sp_line": sp_line,
                              "p_sp_elo": spread.cover(_pe, sp_ePh, sp_ePa, 3.5),
                              "sp_ver": spread.VERSION, "model": "v1"})
        for nm, k in zip(SP_COLS, SP_LINES):
            board[nm] = spread.cover(sp_ph, sp_Ph, sp_Pa, k)
        log = pd.concat([log[~log.game_id.isin(board.game_id)], board], ignore_index=True)
        print("\n--- NFL detail: model margin, total, spread ladder ---")
        for bi, ((idx, r), tot, incal) in enumerate(zip(board.iterrows(), totals, cal.values)):
            mgn = sp_exp[bi]
            side = f"model: {r.home} by {mgn:.1f}" if mgn > 0 else f"model: {r.away} by {-mgn:.1f}"
            vals = f"value: {r.home} >= {min_odds(r.p_home):.2f} / {r.away} >= {min_odds(1-r.p_home):.2f}"
            if incal:
                vals += f"  [wk {int(r.week)} calibration]"
            print(f"{r.away:>3} @ {r.home:<3}  p(home)={r.p_home:.3f}  {side}, total {tot:.1f}  | {vals}")
            for ln in spread_ladder(bi, sp_ph, sp_Ph, sp_Pa, sp_exp, r.home, r.away):
                print(ln)
    else:
        print(f"no games on or before {hi:%Y-%m-%d}")

    fin = g.set_index("game_id")["result"]
    log["result"] = log.game_id.map(fin)
    _m = f.dropna(subset=["margin"]).drop_duplicates("game_id")
    log["final_margin"] = log.game_id.astype(str).map(
        pd.Series(_m.margin.values, index=_m.game_id.astype(str)))
    log.to_csv(pred_path, index=False)
    dec = log[log.result.notna() & (log.result != 0)]
    if len(dec) >= 30:
        y = (dec.result > 0).astype(int).values
        print(f"\nrolling grade, decided only (pushes: {(log.result == 0).sum():.0f}):")
        for lbl, pp in (("model v1 ", dec.p_home), ("elo floor", dec.p_elo),
                        ("base rate", pd.Series(np.full(len(y), base)))):
            pc = np.clip(pp.values.astype(float), 1e-6, 1-1e-6)
            print(f"{lbl}: n={len(y)}  LL={log_loss(y, pc):.4f}  Brier={brier_score_loss(y, pc):.4f}  "
                  f"hit={np.mean((pc > .5) == (y == 1)):.3f}")
        w = dec.tail(150)
        if len(w) >= 150:
            yy = (w.result > 0).astype(int).values
            edge = log_loss(yy, np.full(len(yy), base)) - \
                log_loss(yy, np.clip(w.p_home.values.astype(float), 1e-6, 1-1e-6))
            print(f"last-150 edge vs base: {edge:+.4f}  [{'OK' if edge > 0 else 'TRIPWIRE: bench review'}]")
            if edge <= 0:
                alerts.append("NFL winner")
        dec.to_csv(os.path.join(LOGS, "graded.csv"), index=False)

    push, skipped, cal_n = {}, 0, 0
    if board is not None:
        _c3 = lambda k: spread.cover3(sp_ph, sp_Ph, sp_Pa, k)
        bd = []
        for bi, (_, r) in enumerate(board.iterrows()):
            # Weeks 1 to 4 now publish the same priced line every other week does, carrying
            # a tag instead of a blank. Suppressing the number never protected anyone: it
            # removed the record. A forward test that prints nothing for its first four
            # weeks cannot be graded on its first four weeks.
            incal = bool(r.week < CAL_UNLOCK_WEEK)
            kh, ka = BF.spread_threshold_push(_c3, bi, KEY_GRID)
            # Two actionable lines per game, one each side. Together they bound the
            # zone where neither side is worth taking at standard juice.
            parts = []
            if kh is not None:
                parts.append(f"{r.home} {BF.handicap(-kh)}")
            if ka is not None:
                parts.append(f"{r.away} {BF.handicap(ka)}")
            if not parts:
                skipped += 1
                continue
            bd.append((str(r.date)[:10], f"{r.away} @ {r.home}", " / ".join(parts),
                       r.game_id, float(p[bi]), incal, int(r.week)))
        if bd:
            wg = max([len(x[1]) for x in bd] + [10])
            wp = max([len(x[2]) for x in bd] + [8])
            for d_, gm, pk, gid, ph_i, incal, wk in bd:
                # The probability passed here exists only to make price() print the
                # standard spread number. It has to track MIN_EDGE: hardcoding 1.05
                # printed 1.91 while the edge was 5% and 1.82 once it went to zero,
                # and 1.82 on a 52.36% cover is -4.7% EV.
                ln = BF.line(d_, gm, pk,
                             (1 + BF.MIN_EDGE) / BF.STD_SPREAD_ODDS,
                             w_game=wg, w_pick=wp)
                notes = []
                if incal:
                    cal_n += 1
                    notes.append(f"      week {wk} calibration, forward test until "
                                 f"Week {CAL_UNLOCK_WEEK}")
                # Live-entry note keyed off the moneyline favourite rather than the
                # spread. Only fires where the pregame moneyline is too short to take.
                notes += BF.live_note("NFL", max(ph_i, 1 - ph_i))
                if notes:
                    ln = "\n".join([ln] + notes)
                push[gid] = {"line": ln,
                             "min_odds": BF.STD_SPREAD_ODDS,
                             # min_odds is a module constant on this board, identical on
                             # every row forever, so the gate's price check can never fire.
                             # The pick string is what actually moves: a QB ruled out on
                             # Sunday morning shifts the threshold and rewrites this.
                             "match": pk,
                             "cal": incal}
    print("\n=== MBM NFL board ===")
    for _v in push.values():
        print(_v["line"])
    if skipped:
        print(f"({skipped} game{'s' if skipped != 1 else ''} skipped, "
              f"no line qualifies at {BF.STD_SPREAD_ODDS:.2f} / "
              f"{BF.american(BF.STD_SPREAD_ODDS)})")

    mode, lines = tg_delta.gate(LOGS, dt.date.today().strftime("%Y%m%d"), push)
    if lines or alerts:
        body = list(lines) if lines else ["no changes since the last push"]
        foot = (f"{skipped} game{'s' if skipped != 1 else ''} skipped, no line qualifies "
                f"at {BF.STD_SPREAD_ODDS:.2f} / {BF.american(BF.STD_SPREAD_ODDS)}")\
               if skipped else None
        title = "MBM NFL" + (" · changes since the last push"
                             if mode == "delta" and lines else "")
        # GREY is the calibration colour. Only when every row is inside the window, so a
        # week straddling the unlock reads as a live board with some rows marked.
        colour = notify.GREY if push and cal_n == len(push) else notify.GREEN
        if alerts:
            body = body + ["", "TRIPWIRE: " + ", ".join(alerts)
                           + " below base rate, stop betting this board"]
            colour = notify.RED
        notify.send(title, body, foot, colour)
    elif not push:
        print("no board to send")
    else:
        print("board unchanged since the last push, nothing sent")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    run(ap.parse_args().date)
