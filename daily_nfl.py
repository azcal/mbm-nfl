"""
MBM NFL daily runner (rebuild v1, frozen spec).
- Refreshes nflverse games + current-season pbp EPA aggregates
- Expected starters from depth charts + injury report (fallback: last actual starter)
- Boards upcoming games (8-day horizon): p(home), model margin, model total
- CALIBRATION GATE: value lines suppressed through Week 4; unlock Week 5
- Grades logged predictions (ties push), rolling metrics + last-150 tripwire
Usage: python daily_nfl.py [--date YYYY-MM-DD]
"""
import argparse, io, json, os, sys, math, datetime as dt
import numpy as np, pandas as pd, requests
from collections import defaultdict
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tg_delta
import spread
import engine_nfl as E

H = {"User-Agent": "Mozilla/5.0"}
NV = "https://github.com/nflverse/nflverse-data/releases/download"
LOGS = os.path.join(HERE, "logs")
SPEC = json.load(open(os.path.join(HERE, "spec.json")))
CFG, C = SPEC["cfg"], SPEC["C"]
TRAIN_LO, TRAIN_HI = 2018, 2025          # frozen; 2026 = forward test
CAL_UNLOCK_WEEK = 5                       # value lines suppressed through week 4
MIN_EDGE = 0.05

SP_LINES = [-7.5, -3.5, -0.5, 3.5, 7.5]     # margin thresholds logged for grading
SP_COLS = ["p_sp_m75", "p_sp_m35", "p_sp_m05", "p_sp_p35", "p_sp_p75"]


def spread_ladder(i, ph, Ph, Pa, exp, home, away, span=4):
    out = [f"    {'line':>7}  {home + ' covers':>14} {'needs':>7}   {away + ' covers':>14} {'needs':>7}"]
    for q, p, o1, p2, o2 in spread.ladder(ph[i], Ph[i], Pa[i], exp[i], min_odds, span=span):
        out.append(f"    {q:>+7.1f}  {p:>14.3f} {o1:>7.2f}   {p2:>14.3f} {o2:>7.2f}")
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

def expected_starters(season):
    """team -> qb gsis id, from depth charts minus injury Outs; {} on any failure."""
    try:
        r = requests.get(f"{NV}/injuries/injuries_{season}.parquet", headers=H, timeout=120)
        inj = pd.read_parquet(io.BytesIO(r.content)) if r.status_code == 200 else pd.DataFrame()
        outs = set()
        if len(inj) and "report_status" in inj.columns:
            w = inj[inj.week == inj.week.max()]
            outs = set(w.loc[w.report_status.astype(str).str.lower() == "out", "gsis_id"].dropna())
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
            qb = qb[qb[wk] == qb[wk].max()]
        qb[dep] = pd.to_numeric(qb[dep], errors="coerce")
        out = {}
        for tm, grp in qb.sort_values(dep).groupby(tcol):
            for gid in grp[idc]:
                if gid == gid and gid not in outs:
                    out[E.REN.get(tm, tm)] = gid
                    break
        return out
    except Exception as ex:
        print("depth-chart feed unavailable, using last-starter fallback:", type(ex).__name__)
        return {}

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
    upcoming = g[g.result.isna() & (g.date >= today) & (g.date <= today + pd.Timedelta(days=8))].copy()
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
    need_bf = [c for c in SP_COLS + ["sp_margin", "sp_line"] if c not in log.columns
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
                bf = pd.DataFrame({"game_id": have.game_id.astype(str).values,
                                   "sp_margin": spread.expected_margin(bph, bPh, bPa),
                                   "sp_line": spread.median_line(bph, bPh, bPa)})
                for nm, k in zip(SP_COLS, SP_LINES):
                    bf[nm] = spread.cover(bph, bPh, bPa, k)
                log = log.drop(columns=[c for c in bf.columns if c != "game_id"
                                        and c in log.columns], errors="ignore")
                log["game_id"] = log.game_id.astype(str)
                log = log.merge(bf, on="game_id", how="left")
                print(f"spread backfill: filled {bf.sp_margin.notna().sum()} historical rows")
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
        board = pd.DataFrame({"date": rows.date.dt.strftime("%Y-%m-%d"), "game_id": rows.game_id,
                              "week": rows.week, "season": rows.season, "home": rows.home,
                              "away": rows.away, "p_home": p, "p_elo": pe.values,
                              "sp_margin": sp_exp, "sp_line": sp_line,
                              "model": "v1"})
        for nm, k in zip(SP_COLS, SP_LINES):
            board[nm] = spread.cover(sp_ph, sp_Ph, sp_Pa, k)
        log = pd.concat([log[~log.game_id.isin(board.game_id)], board], ignore_index=True)
        print(f"\n=== MBM NFL board (week {int(rows.week.min())}) ===")
        for bi, ((idx, r), tot, incal) in enumerate(zip(board.iterrows(), totals, cal.values)):
            mgn = sp_exp[bi]
            side = f"model: {r.home} by {mgn:.1f}" if mgn > 0 else f"model: {r.away} by {-mgn:.1f}"
            vals = (f"value: {r.home} >= {min_odds(r.p_home):.2f} / {r.away} >= {min_odds(1-r.p_home):.2f}"
                    if not incal else "CALIBRATION thru Wk4: predictions only, no betting")
            print(f"{r.away:>3} @ {r.home:<3}  p(home)={r.p_home:.3f}  {side}, total {tot:.1f}  | {vals}")
            if not incal:
                for ln in spread_ladder(bi, sp_ph, sp_Ph, sp_Pa, sp_exp, r.home, r.away):
                    print(ln)
    else:
        print("no games in the next 8 days")

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

    tg, ch = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT")
    if tg and ch and (board is not None or alerts):
        push = {}
        if board is not None:
            for _, r in board.iterrows():
                line = (f"{r.away} @ {r.home}: {r.p_home:.3f} | CALIBRATION, no betting thru Wk4"
                        if r.week < CAL_UNLOCK_WEEK else
                        f"{r.away} @ {r.home}: {r.p_home:.3f} | "
                        f"{r.home} >= {min_odds(r.p_home):.2f} / {r.away} >= {min_odds(1-r.p_home):.2f}")
                push[r.game_id] = {"line": line, "min_odds": min_odds(r.p_home)}
        mode, lines = tg_delta.gate(LOGS, dt.date.today().strftime("%Y%m%d"), push)
        if not (lines or alerts):
            return
        txt = "\n".join(lines) if lines else "no price changes since the last push"
        if mode == "delta" and lines:
            txt = "changes since the last push:\n" + txt
        if alerts:
            txt += "\n\u26a0 TRIPWIRE: " + ", ".join(alerts) + " below base rate, stop betting this board"
        requests.post(f"https://api.telegram.org/bot{tg}/sendMessage",
                      data={"chat_id": ch, "text": f"MBM NFL\n{txt}"}, timeout=20)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    run(ap.parse_args().date)
