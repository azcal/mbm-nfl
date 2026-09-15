"""
props_weekly.py: Sunday-morning receiving projections for this week's games (mbm-nfl).

Slate comes from the nflverse schedule and nothing else: one target week, every game in it
whether scheduled, in progress or final, every team present. The odds feed never decides
who plays whom. Injury status comes from ESPN (keyed by name AND team) with the nflverse
report as backup; Out/Doubtful/IR/PUP/Suspended are removed from the active set and written
with zeros and status_applied=True; Questionable stays projected and flagged.

Writes logs/props_grid.csv. Build-time assertions fail the run on a bad slate or an
unapplied status. Set PROPS_WEEK=<n> to force a week (tests).
"""
import datetime as dt, io, json, os, sys
import numpy as np, pandas as pd, requests
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import props_model as M
import props_market as PM
PLAY_PROB = {"": 1.0, "Questionable": 0.8}     # configurable; zeroed rows get 0.0; never scales the ladders
BOOK_ABBR = {"draftkings": "DK", "fanduel": "FD", "betmgm": "MGM", "betrivers": "BR", "pinnacle": "PIN", "betonlineag": "BOL",
             "bovada": "BOV", "mybookieag": "MB", "betus": "BU", "lowvig": "LV", "caesars": "CZR", "fanatics": "FAN"}
MIN_EDGE_PP, MIN_EV = 3.0, 0.05


def format_board(bd, week):
    """Player header, then one line per prop: side, line, market, the fair price, and the books
    currently at or better than fair. Filter: edge >= 3pp and EV >= 5% at build time, a real
    devig, no Questionables. Players ordered by their best EV."""
    if not len(bd):
        return ["no priced rows this week"]
    ok = bd[(bd.devig_method != "none") & (bd.play_prob >= 1.0) & (bd.edge_pp >= MIN_EDGE_PP) & (bd.ev_per_dollar >= MIN_EV)].copy()
    if not len(ok):
        return ["no prop clears 3pp edge and 5% EV at build time"]
    def dec_str(am):
        am = int(str(am).replace("+", "")); d = 1 + am / 100 if am > 0 else 1 + 100 / (-am); return f"{d:.2f}"
    ok["kick_day"] = pd.to_datetime(ok.kickoff_utc, utc=True).dt.tz_convert("America/Denver").dt.strftime("%a")
    ok["best_ev"] = ok.groupby("pid").ev_per_dollar.transform("max")
    lines = [f"week {week} props board. Fair = the model's price; take the prop anywhere the book is at fair or better. Books listed are at or better right now, best first.", ""]
    for pid, g in sorted(ok.groupby("pid"), key=lambda kv: -kv[1].best_ev.iloc[0]):
        r0 = g.iloc[0]
        lines.append(f"{r0.player} ({r0.team} vs {r0.opp}, {r0.kick_day})")
        for (mk, line, side), q in g.groupby(["market", "line", "side"]):
            q = q.sort_values("book_dec" if "book_dec" in q.columns else "ev_per_dollar", ascending=False)
            fair_am = str(q.fair_odds.iloc[0]); fair_dec = dec_str(fair_am)
            books = ", ".join(f"{BOOK_ABBR.get(b, b)} {('+' if str(o).lstrip('-').isdigit() and int(o) > 0 else '')}{o}" for b, o in zip(q.book, q.book_odds))
            mkt = "rec yds" if mk == "rec_yds" else "receptions"
            lines.append(f"  {side} {line:g} {mkt:<10} fair {fair_am if fair_am.startswith('-') else '+' + fair_am} ({fair_dec})   find {fair_dec} or better   now: {books}")
        lines.append("")
    lines.append("edge = model minus the book's fair (devigged) probability; every row above clears 3pp and 5% EV. Prices move; re-check before betting.")
    return lines


SCHED_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
ESPN_INJ = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
TEAM_NORM = {"LAR": "LA", "STL": "LA", "JAC": "JAX", "WSH": "WAS", "OAK": "LV", "SD": "LAC", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU"}
norm_team = lambda t: TEAM_NORM.get(str(t).strip().upper(), str(t).strip().upper())
norm_name = lambda s: M.re.sub(r"[^a-z]", "", str(s).lower())


def schedule(season=None):
    r = M._get(SCHED_URL)
    if r is None or r.status_code != 200:
        sys.exit("could not download the nflverse schedule")
    g = pd.read_csv(io.StringIO(r.text))
    g = g[g.game_type == "REG"].copy()
    g["home_team"] = g.home_team.map(norm_team); g["away_team"] = g.away_team.map(norm_team)
    g["kickoff_utc"] = pd.to_datetime(g.gameday + " " + g.gametime.fillna("13:00"), errors="coerce").dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    return g if season is None else g[g.season == season]


def target_week(g, now):
    """The week that is on: the earliest week with a game not yet final, unless every game of
    the current week has kicked off more than 8 hours ago (then it is over and the next week
    is the slate)."""
    for wk in sorted(g.week.unique()):
        w = g[g.week == wk]
        if w.result.isna().any() or (w.kickoff_utc.max() + pd.Timedelta(hours=8)) > now:
            return int(wk)
    return int(g.week.max())


def slate_for(g, week, now):
    w = g[g.week == week].copy()
    def status(r):
        if pd.notna(r.result):
            return "final"
        return "in_progress" if r.kickoff_utc <= now else "scheduled"
    w["game_status"] = w.apply(status, axis=1)
    rows = []
    for r in w.itertuples():
        rows.append(dict(season=int(r.season), week=int(r.week), team=r.home_team, opponent_team=r.away_team, home_away="home", game_id=r.game_id, kickoff_utc=r.kickoff_utc.isoformat(), game_status=r.game_status))
        rows.append(dict(season=int(r.season), week=int(r.week), team=r.away_team, opponent_team=r.home_team, home_away="away", game_id=r.game_id, kickoff_utc=r.kickoff_utc.isoformat(), game_status=r.game_status))
    return pd.DataFrame(rows), w


def assert_slate(up, w):
    opp = dict(zip(up.team, up.opponent_team))
    assert all(opp[opp[t]] == t for t in opp), "opp symmetry broken"
    pairs = set(zip(w.home_team, w.away_team)) | set(zip(w.away_team, w.home_team))
    assert all((t, o) in pairs for t, o in zip(up.team, up.opponent_team)), "a (team, opp) pair is not on the schedule"
    assert up.team.nunique() == len(set(w.home_team) | set(w.away_team)), "team count differs from the schedule"


def espn_injuries(max_age_days=6):
    """{(normalized name, team code): status} from ESPN's injuries feed.

    Root cause of the Week 1 false Outs: this feed is a news feed. Every item carries a
    status ('Active', 'Out', 'Questionable', ...) and a date, and old items linger. The old
    code keyed on name alone and ignored the date, so a stale 'Out' note tagged a healthy
    starter. Now: team comes from the item's own athlete.team, items older than
    max_age_days are ignored, and only injury statuses are kept.
    """
    out = {}
    try:
        ej = requests.get(ESPN_INJ, timeout=60).json()          # ESPN 403s any custom User-Agent; no headers
    except Exception as ex:
        print("ESPN injuries unavailable:", type(ex).__name__); return out, None
    now = pd.Timestamp.now(tz="UTC")
    keep = {"Out": "Out", "Doubtful": "Doubtful", "Questionable": "Questionable", "Injured Reserve": "IR",
            "Suspension": "Suspended", "Physically Unable to Perform": "PUP"}
    for tm in ej.get("injuries", []):
        for it in tm.get("injuries", []):
            st = keep.get(str(it.get("status", "")))
            if not st:
                continue
            when = pd.to_datetime(it.get("date"), utc=True, errors="coerce")
            if pd.isna(when) or (now - when).days > max_age_days:
                continue
            ath = it.get("athlete") or {}
            code = norm_team(((ath.get("team") or {}).get("abbreviation")) or "")
            out[(norm_name(ath.get("displayName", "")), code)] = st
    return out, now.isoformat(timespec="seconds")


def main():
    spec = json.load(open(os.path.join(HERE, "props_spec.json")))
    now = pd.Timestamp.now(tz="UTC")
    g = schedule()
    season = int(g[g.kickoff_utc <= now + pd.Timedelta(days=7)].season.max()) if (g.kickoff_utc <= now + pd.Timedelta(days=7)).any() else int(g.season.max())
    gs = g[g.season == season]
    week = int(os.environ["PROPS_WEEK"]) if os.environ.get("PROPS_WEEK") else target_week(gs, now)
    up, w = slate_for(gs, week, now)
    assert_slate(up, w)
    print(f"{season} week {week}: {len(w)} games, {up.team.nunique()} teams; status: {w.game_status.value_counts().to_dict()}")
    first_kick = w.kickoff_utc.min()
    if now > first_kick:
        print(f"WARNING: build time {now.isoformat(timespec='minutes')} is after the earliest kickoff {first_kick.isoformat()}; move the cron earlier")
    roster = M.load_roster(season, live=True)
    print(f"roster: {'none' if roster is None else str(len(roster)) + ' active receivers'}")
    inj = M._read("injuries", season, True)
    inj_status = {(int(r.season), int(r.week), r.gsis_id): r.report_status for r in inj.itertuples() if pd.notna(r.report_status)} if inj is not None else {}
    n_nv = sum(1 for k in inj_status if k[1] == week)
    espn, espn_ts = espn_injuries()
    n_espn = 0
    if roster is not None and espn:
        for rr in roster.itertuples():
            st = espn.get((norm_name(rr.full_name), norm_team(rr.team)))
            if st:
                inj_status[(season, week, rr.gsis_id)] = st; n_espn += 1
    print(f"injury statuses: nflverse {n_nv}, ESPN matched by name+team {n_espn} ({len(espn)} on ESPN's list)")
    M.RNG = np.random.default_rng(int(now.strftime("%Y%m%d")))
    M.SEASON_CARRY = spec.get("season_carry", M.SEASON_CARRY)
    df = M.run(range(2019, season + 1), kappa=spec["kappa"], alpha=spec["alpha"], H=spec["H"], upcoming=up[["season", "week", "team", "opponent_team"]],
               live=True, vol_mult=spec.get("volume_mult"), roster=roster, inj_status=inj_status, min_proj_targets=2.5)
    df = df[(df.season == season) & (df.week == week)].copy()
    meta = up.set_index("team")
    for c in ("home_away", "game_id", "kickoff_utc", "game_status"):
        df[c] = df.team.map(meta[c])
    df["inj_source"] = np.where(df.inj.astype(str) != "", "espn+nflverse", ""); df["inj_ts"] = espn_ts or ""
    bad = df[df.inj.astype(str).isin(M.REMOVE_STATUSES) & (df.proj_targets > 0) & (~df.status_applied.astype(bool))]
    assert bad.empty, f"status not applied for {list(bad.name)}"
    df["play_prob"] = [0.0 if bool(a) else PLAY_PROB.get(str(i), 1.0) for a, i in zip(df.status_applied, df.inj)]
    # MOVE LOG (Phase 2, item 11): where every share and target projection came from this week
    logs0 = os.path.join(HERE, "logs"); os.makedirs(logs0, exist_ok=True)
    prev0 = pd.read_csv(os.path.join(logs0, "props_grid.csv"), low_memory=False) if os.path.exists(os.path.join(logs0, "props_grid.csv")) else None
    prev_med = dict(zip(prev0.pid, prev0.med_yds)) if prev0 is not None and "pid" in prev0.columns else {}
    prev_team = dict(zip(prev0.pid, prev0.team)) if prev0 is not None and "pid" in prev0.columns else {}
    mv = df[df.proj_targets > 0].copy()
    mv["data_update_delta"] = mv.blend_share - mv.prior_share
    mv["redistribution_delta"] = mv.final_share - mv.blend_share
    for c in ("prior_share", "data_update_delta", "redistribution_delta", "final_share"):
        mv[c.replace("share", "targets") if "share" in c else c + "_targets"] = (mv[c] * mv.att_mu).round(2)
    mv["last_week_med_yds"] = mv.pid.map(prev_med); mv["this_week_med_yds"] = mv.med_yds.round(1)
    def reason(r):
        parts = [f"n_games_team={int(r.n_games_team)} w_data={r.w_data:.2f}", str(r.prior_src)]
        if r.pid in prev_team and prev_team[r.pid] != r.team:
            parts.append(f"team change {prev_team[r.pid]}->{r.team}")
        if abs(r.redistribution_delta) > 0.01:
            parts.append("active set changed (status/roster) -> redistribution")
        if str(r.inj):
            parts.append(f"inj={r.inj}")
        return "; ".join(parts)
    mv["move_reason"] = [reason(r) for r in mv.itertuples()]
    mv[["season", "week", "team", "pid", "name", "pos", "prior_share", "data_update_delta", "redistribution_delta", "final_share",
        "prior_targets", "data_update_delta_targets", "redistribution_delta_targets", "final_targets", "last_week_med_yds", "this_week_med_yds", "move_reason"]] \
        .to_csv(os.path.join(logs0, "props_moves.csv"), index=False)
    print(f"move log: {len(mv)} players -> logs/props_moves.csv")
    ident = ["season", "week", "team", "opp", "home_away", "game_id", "kickoff_utc", "game_status", "pid", "name", "pos",
             "inj", "inj_source", "inj_ts", "status_applied", "play_prob", "depth_prior", "prior_share", "w_data", "n_games_team"]
    centre = ["proj_targets", "share", "cr", "ypc", "med_yds", "mean_yds", "med_rec"]
    out = df[ident + centre].copy()
    for i, x in enumerate(M.YGRID):
        out[f"p_over_{int(x)}"] = [round(p[i], 4) for p in df.p_over]
    for k in M.RGRID:
        out[f"p_rec_{k}"] = [round(p[k - 1], 4) for p in df.p_rec]
    out["ts"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    logs = os.path.join(HERE, "logs"); os.makedirs(logs, exist_ok=True)
    prev = pd.read_csv(os.path.join(logs, "props_grid.csv"), low_memory=False) if os.path.exists(os.path.join(logs, "props_grid.csv")) else None
    # ---- market layer: snapshot once per week, then anchor and board from the snapshot
    odds_path = PM.latest_snapshot(season, week)
    key = os.environ.get("ODDS_API_KEY", "").strip()
    # Pull fresh prices when there is no snapshot for the week or the latest is older than
    # PROPS_SNAPSHOT_MAX_AGE_H (default 12h). PROPS_WINDOW_H limits the pull to games kicking off
    # within that many hours (Thursday run: the Thursday game only, 2 credits). Games already
    # final are never pulled. Every pull is a new timestamped file; nothing is overwritten.
    max_age = float(os.environ.get("PROPS_SNAPSHOT_MAX_AGE_H", "12"))
    stale = True
    if odds_path is not None:
        try:
            ts_s = os.path.basename(odds_path).split("_")[-1].replace(".csv", "")
            stale = (now - dt.datetime.strptime(ts_s, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)).total_seconds() > max_age * 3600
        except Exception:
            stale = True
    if stale and key and os.environ.get("PROPS_NO_ODDS") != "1":
        books = os.environ.get("PROPS_BOOKS", "pinnacle,betonlineag,lowvig,bovada,mybookieag,betus,draftkings,fanduel,betmgm,betrivers")
        markets = PM.MARKETS_DEFAULT if os.environ.get("PROPS_ALTS", "0") == "1" else PM.MARKETS_DEFAULT[:2]
        pull = w[w.game_status != "final"]
        if os.environ.get("PROPS_WINDOW_H"):
            pull = pull[pull.kickoff_utc <= now + dt.timedelta(hours=float(os.environ["PROPS_WINDOW_H"]))]
        print(f"odds: pulling {len(pull)} game(s) ({'window ' + os.environ['PROPS_WINDOW_H'] + 'h' if os.environ.get('PROPS_WINDOW_H') else 'all not yet final'})")
        newp = PM.snapshot(season, week, pull, key, books, markets=markets) if len(pull) else None
        # a partial pull (Thursday) is merged with the week's earlier snapshot so the board still
        # covers every lined game; the newest price per (book, player, market, line) wins
        if newp and odds_path:
            old_s = pd.read_csv(odds_path, low_memory=False); new_s = pd.read_csv(newp, low_memory=False)
            keep = old_s[~old_s.event_id.isin(set(new_s.event_id))]
            merged = pd.concat([keep, new_s], ignore_index=True); merged.to_csv(newp, index=False)
        odds_path = newp or odds_path
    raw = out.copy()
    market_cols = ["market_line", "market_p_over", "gap_vs_market", "p_over_at_line", "anchored", "anchor_clipped", "odds_ts"]
    for c in market_cols:
        raw[c] = np.nan if c not in ("anchored", "anchor_clipped") else False
    raw["odds_ts"] = ""
    meta_extra = dict(season=season, week=week, first_kickoff_utc=first_kick.isoformat(), n_players_lined=0, n_unmatched=0, n_clipped=0)
    if odds_path and roster is not None:
        snap, un = PM.load_snapshot(odds_path, roster, season, week)
        cons = PM.consensus(snap)
        odds_ts = str(pd.read_csv(odds_path, nrows=1).snap_ts.iloc[0]) if os.path.getsize(odds_path) > 50 else ""
        pub = PM.anchor_grid(raw, cons, spec); pub["odds_ts"] = odds_ts
        vol = PM.team_volume_report(raw, pub)
        print("team target volume, raw vs anchored (flag = outside 22..40):"); print(vol[vol.flag != ""].to_string() if (vol.flag != "").any() else "  all teams inside 22..40")
        meta_extra.update(odds_ts=odds_ts, anchored=True, n_players_lined=int(pub.anchored.sum()), n_unmatched=int(len(un)), n_clipped=int(pub.anchor_clipped.sum()))
        print(f"anchored {meta_extra['n_players_lined']} lined players ({meta_extra['n_clipped']} clipped to the 0.5x..2.0x bound); {int((~pub.anchored).sum())} unlined keep raw")
    else:
        pub = raw.copy(); snap = None; cons = {}
        meta_extra.update(anchored=False, odds_ts="")
    order = ident + ["market_line", "market_p_over", "gap_vs_market", "p_over_at_line", "anchored", "anchor_clipped", "odds_ts"] + centre + \
            [f"p_over_{int(x)}" for x in M.YGRID] + [f"p_rec_{k}" for k in M.RGRID] + ["ts"]
    for c in order:
        if c not in raw.columns: raw[c] = np.nan
        if c not in pub.columns: pub[c] = np.nan
    raw = raw[order]; pub = pub[order]
    raw.to_csv(os.path.join(logs, "props_grid_raw.csv"), index=False)
    pub.to_csv(os.path.join(logs, "props_grid.csv"), index=False)
    # ---- loose ends: every player from last week is here or in dropped_*.csv with a reason
    dropped = PM.dropped_report(prev, pub, roster, os.path.join(logs, f"dropped_{season}_w{week}.csv"))
    if prev is not None and len(prev):
        assert set(prev.pid) <= (set(pub.pid) | set(dropped.pid)), "a player from last week is neither in the grid nor in dropped_*.csv"
        print(f"dropped since last week: {len(dropped)} -> logs/dropped_{season}_w{week}.csv ({dropped.reason.value_counts().to_dict() if len(dropped) else ''})")
    # ---- board
    if snap is not None and meta_extra["anchored"]:
        b = PM.build_board(snap, pub, cons, logs, season, week, meta_extra)
        print(f"board: {len(b)} priced rows -> logs/props_board.csv, props_board.md")
    else:
        PM.empty_board(logs, season, week, "no odds snapshot for this week (ODDS_API_KEY missing, PROPS_NO_ODDS=1, or the pull failed)", meta_extra)
        print("board: empty (no odds snapshot)")
    out = pub
    print(f"props grid: {len(out)} rows ({int(out.status_applied.astype(bool).sum())} removed by status) -> logs/props_grid.csv (+ props_grid_raw.csv)")
    live = out[(out.inj.astype(str) == "") & (out.game_status != "final") & (out.proj_targets > 0)].sort_values("med_yds", ascending=False)
    lines = [f"{season} week {week}: {len(out)} receivers, {int((out.inj.astype(str) != '').sum())} on an injury list (removed or flagged), "
             f"{int((out.game_status == 'final').sum())} already final. median yards | P(over 50 / 75 / 100) | median rec | P(rec >= 5 / 7)"]
    for r in live.head(40).itertuples():
        lines.append(f"  {r.team:<3} {r.name:<22} {r.pos:<2} {r.med_yds:>6.1f} | {r.p_over_50:.2f} / {r.p_over_75:.2f} / {r.p_over_100:.2f} | {r.med_rec:>4.1f} | {r.p_rec_5:.2f} / {r.p_rec_7:.2f}")
    lines.append("full grid in logs/props_grid.csv; fair decimal odds = 1 / probability")
    print("\n".join(lines))
    board_path = os.path.join(HERE, "logs", "props_board.csv")
    blines = []
    if os.path.exists(board_path):
        bd = pd.read_csv(board_path)
        blines = format_board(bd, week)
        open(os.path.join(HERE, "logs", "props_board.md"), "w").write("\n".join([f"# NFL receiving props board, {season} week {week}", ""] + blines) + "\n")
        print("\n".join(blines))
    try:
        import notify
        notify.send(f"NFL receiving projections, week {week}", lines, "anchored to consensus lines where a line exists; model shape; calibrated on 2023-25 within 3 points per band", getattr(notify, "GREY", None))
        if blines:
            notify.send(f"NFL receiving props board, week {week}", blines, "paper rule: 5% EV at the best book, graded Tuesday", getattr(notify, "GREY", None))
    except Exception as ex:
        print("discord post skipped:", type(ex).__name__)


if __name__ == "__main__":
    main()
