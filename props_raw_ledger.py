"""
props_raw_ledger.py: PAPER ledger for two raw-model rules on receiving props (mbm-nfl).

    python props_raw_ledger.py            # after props_weekly.py and props_rushpass.py: flag, grade, post
    python props_raw_ledger.py --test     # unit tests only

RULE 1, RECEIVING YARDS (written 2026-11-26, before any live row was graded)
  For every player with a consensus pregame receiving-yards line, compare the RAW model median
  (logs/props_grid_raw.csv, no market input) to that line.
      gap = |raw median - line| / max(line, 10)
  gap >= 0.30  -> tier "rule":  take the model's side (Over if the median is above the line)
  gap >= 0.15  -> tier "watch": logged the same way, not part of the bar

RULE 2, RECEPTIONS (written 2026-11-27, before any live row was graded)
  For every player with a consensus half-point receptions line and a two-sided price, compare
  the RAW model's chance of going over to the market's own chance with the juice removed.
      edge = |model P(over) - market P(over)|
  edge >= 0.15 -> tier "rule":  take the side the model favours
  edge >= 0.10 -> tier "watch"

BOTH RULES
  One unit, at the best price on that side at the consensus line among the books pulled.
  If a later run prices the same player before kickoff, that later look replaces the earlier one.
  A player with no stat line in a finished game is void, as a book would void it.

EXCLUSIONS (judgment rules; no prop history exists to test either one properly)
  An OVER is logged with a reason in the `excluded` column, graded and reported on its own, and
  kept out of the bar, when either holds. Unders always stand.
    weather    the game is tagged poor weather (props_weather.py: outdoors with 15+ mph wind,
               30+ mph gusts, a 60%+ chance of precipitation, or snow). Added 2026-11-26.
    backup QB  the team's expected starter (props_rushpass.py) is not the passer who threw the
               most over the team's last four games, or the starter is listed Questionable or
               Doubtful. Added 2026-11-27. The receiving model does not know who is throwing.
               On 2026 pregame lines, 30%+ yardage overs behind a backup went 8-9 and the
               unders went 12-1; small samples, and that test knew who really started.

THE BAR, per rule
  150 graded, non-excluded rule-tier flags from live runs: hit rate 56% or better AND positive
  ROI at the best price. Short of either, that rule is retired. source=backfill does not count.

WHERE THEY CAME FROM (props_backtest_replay.py, 2026 weeks 2-9 and 11, pregame lines only)
  Yards:      rule tier 83-61 (57.6%), +14.5u. Unders 57-38, overs 26-23. Week 11 was +9.6u of it.
  Receptions: 15-point edge 107-76 (58.5%), ROI +13.6%. Unders 89-62, overs 18-14.
  Both cuts were chosen after seeing those results. That is why this is a paper ledger.

Reads logs/props_grid_raw.csv (raw model), logs/props_grid.csv (consensus lines),
logs/props_board.csv (book prices) and logs/rushpass_grid.csv (expected starters). Writes
logs/raw_gap_ledger.csv. Never changes the grid, the board or the Unabated file.
"""
import argparse, math, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
LOGS = os.path.join(HERE, "logs")
LEDGER = os.path.join(LOGS, "raw_gap_ledger.csv")
RULE_GAP, WATCH_GAP = 0.30, 0.15          # receiving yards: share of the line
RULE_EDGE, WATCH_EDGE = 0.15, 0.10        # receptions: probability points
BAR_N, BAR_HIT = 150, 0.56
STALE_H = 36.0          # a line older than this at flag time is marked stale: re-check it before acting
QB_GAMES = 4
MARKETS = {"rec_yds": ("receiving_yards", "rec yds"), "receptions": ("receptions", "receptions")}
COLS = ["flag_ts", "source", "season", "week", "pid", "player", "pos", "team", "opp", "kickoff_utc", "inj", "tier", "side", "line",
        "raw_median", "gap", "best_price", "best_book", "n_books", "odds_ts", "line_age_h", "actual", "result", "pnl", "graded",
        "weather", "excluded", "market", "model_p", "market_p", "qb"]


def dec(american):
    a = float(american)
    return 1 + a / 100 if a > 0 else 1 + 100 / -a


def qb_tags(rush_grid, stats, season, week):
    """{team: reason} for teams whose expected starter is a backup or is in doubt. stats holds
    this season's weekly player stats with team already in the grid's team codes."""
    out = {}
    if rush_grid is None or not len(rush_grid) or "starter" not in rush_grid.columns:
        return out
    s = rush_grid[(rush_grid.season.astype(int) == season) & (rush_grid.week.astype(int) == week) & rush_grid.starter.astype(bool)]
    p = stats[(stats.season.astype(int) == season) & (stats.season_type == "REG") & (stats.week.astype(int) < week) & (stats.attempts > 0)]
    for r in s.itertuples():
        t = p[p.team == r.team]
        if not len(t):
            continue
        last = sorted(t.week.unique())[-QB_GAMES:]
        thr = t[t.week.isin(last)].groupby("player_id").attempts.sum().sort_values(ascending=False)
        usual = thr.index[0]
        if usual != r.gsis_id:
            nm = t[t.player_id == usual].player_display_name.iloc[0]
            out[r.team] = f"backup QB: {r.name} starts, {nm} threw most of the last {len(last)} games"
        elif isinstance(r.inj, str) and r.inj in ("Questionable", "Doubtful"):
            out[r.team] = f"QB {r.inj.lower()}: {r.name}"
    return out


def _row(x, now, backfill, market, tier, side, line, gap, best, n_books, tags, qbs, raw_median=np.nan, model_p=np.nan, market_p=np.nan):
    k = pd.Timestamp(x.kickoff_utc); o = pd.Timestamp(x.odds_ts)
    age = (min(now, k) - o).total_seconds() / 3600 if not backfill else (pd.Timestamp(x.ts) - o).total_seconds() / 3600
    why = []
    if side == "Over" and (tags or {}).get(x.team, {}).get("status") == "poor":
        why.append("weather: " + tags[x.team]["reason"])
    if side == "Over" and (qbs or {}).get(x.team):
        why.append(qbs[x.team])
    return dict(flag_ts=now.strftime("%Y-%m-%dT%H:%M:%SZ"), source="backfill" if backfill else "live", season=int(x.season), week=int(x.week),
                pid=x.pid, player=x.display_name, pos=x.pos, team=x.team, opp=x.opp, kickoff_utc=x.kickoff_utc,
                inj=x.inj if isinstance(x.inj, str) else "", tier=tier, side=side, line=float(line),
                raw_median=raw_median, gap=round(float(gap), 3), best_price=round(float(best.price), 3), best_book=best.book,
                n_books=int(n_books), odds_ts=x.odds_ts, line_age_h=round(age, 1), actual=np.nan, result="", pnl=np.nan, graded="",
                weather=(tags or {}).get(x.team, {}).get("detail", ""), excluded="; ".join(why), market=market,
                model_p=model_p, market_p=market_p, qb=(qbs or {}).get(x.team, ""))


def build_flags(raw, grid, board, now, backfill=False, team_ts=None, tags=None, qbs=None):
    """Rows to flag from one run's files, both markets. now is a UTC Timestamp. team_ts: {team:
    time that team's game was last priced}; the grid's own odds_ts is the first row of a merged
    file, so on a week with a Thursday pull it reports Thursday for every game."""
    keep = [c for c in ("market_line", "rec_line", "odds_ts") if c in grid.columns]
    g = grid[["pid"] + keep].drop_duplicates("pid")
    r = raw.drop(columns=[c for c in ("market_line", "rec_line", "odds_ts") if c in raw.columns]).merge(g, on="pid")
    for c in ("market_line", "rec_line"):
        if c not in r.columns:
            r[c] = np.nan
    r = r[~r.status_applied.astype(bool) & (r.proj_targets > 0) & (r.market_line.notna() | r.rec_line.notna())].copy()
    if team_ts:
        r["odds_ts"] = [team_ts.get(t, o) for t, o in zip(r.team, r.odds_ts)]
    r = r[r.odds_ts.notna()]
    kick = pd.to_datetime(r.kickoff_utc, utc=True); ots = pd.to_datetime(r.odds_ts, utc=True)
    built = pd.to_datetime(r.ts, utc=True) if "ts" in r.columns else pd.Series(now, index=r.index)
    # the line and the projection must both predate kickoff; a live run also needs the game still ahead
    ok = (ots < kick) & (built < kick)
    if not backfill:
        ok &= kick > now
    r = r[ok].copy()
    if not len(r):
        return pd.DataFrame(columns=COLS)
    b = board.copy()
    b["price"] = b.book_odds.map(dec)
    by, br = b[b.market == "rec_yds"], b[b.market == "receptions"]
    rows = []
    for x in r.itertuples():
        # rule 1: receiving yards
        if pd.notna(x.market_line):
            gap = abs(x.med_yds - x.market_line) / max(x.market_line, 10)
            if gap >= WATCH_GAP:
                side = "Over" if x.med_yds > x.market_line else "Under"
                q = by[(by.pid == x.pid) & (by.line == x.market_line) & (by.side == side)]
                if len(q):                # some book is quoting that side at the consensus line
                    rows.append(_row(x, now, backfill, "rec_yds", "rule" if gap >= RULE_GAP else "watch", side, x.market_line, gap,
                                     q.sort_values("price", ascending=False).iloc[0], q.book.nunique(), tags, qbs,
                                     raw_median=round(float(x.med_yds), 1)))
        # rule 2: receptions
        L = x.rec_line
        if pd.notna(L) and float(L) % 1 == 0.5 and 1 <= math.ceil(L) <= 10 and "book_fair_pct" in br.columns:
            q = br[(br.pid == x.pid) & (br.line == L) & (br.devig_method == "two_way")]
            ov, un = q[q.side == "Over"], q[q.side == "Under"]
            if len(ov) or len(un):
                mk = float(ov.book_fair_pct.median()) / 100 if len(ov) else 1 - float(un.book_fair_pct.median()) / 100
                po = float(getattr(x, f"p_rec_{math.ceil(L)}"))
                edge = po - mk
                side = "Over" if edge > 0 else "Under"
                qs = ov if side == "Over" else un
                if abs(edge) >= WATCH_EDGE and len(qs):
                    rows.append(_row(x, now, backfill, "receptions", "rule" if abs(edge) >= RULE_EDGE else "watch", side, L, abs(edge),
                                     qs.sort_values("price", ascending=False).iloc[0], qs.book.nunique(), tags, qbs,
                                     model_p=round(po if side == "Over" else 1 - po, 3), market_p=round(mk if side == "Over" else 1 - mk, 3)))
    return pd.DataFrame(rows, columns=COLS)


def _market(led):
    m = led.market.fillna("").astype(str) if "market" in led.columns else pd.Series("", index=led.index)
    return m.where(m != "", "rec_yds")          # rows written before the receptions rule are receiving yards


def grade(led, stats, now):
    """Fill result and pnl for rows whose game is over and whose week is in the stats file."""
    if not len(led):
        return led, 0
    led = led.copy(); n = 0
    led["result"] = led.result.fillna("").astype(str); led["graded"] = led.graded.fillna("").astype(str)
    mk = _market(led)
    st = stats[stats.season_type == "REG"]
    have = set(zip(st.season.astype(int), st.week.astype(int), st.team))
    act = {m: {(int(s), int(w), p): float(v) for s, w, p, v in zip(st.season, st.week, st.player_id, st[col]) if pd.notna(v)}
           for m, (col, _) in MARKETS.items() if col in st.columns}
    for i, x in led.iterrows():
        if x.result or pd.Timestamp(x.kickoff_utc) + pd.Timedelta(hours=5) > now:
            continue
        if (int(x.season), int(x.week), x.team) not in have or mk[i] not in act:
            continue                      # nflverse has not posted that game yet
        a = act[mk[i]].get((int(x.season), int(x.week), x.pid))
        if a is None:
            res, pnl = "void", 0.0
        elif a == x.line:
            res, pnl = "push", 0.0
        else:
            won = (a > x.line) == (x.side == "Over")
            res, pnl = ("win", float(x.best_price) - 1) if won else ("loss", -1.0)
        led.loc[i, ["actual", "result", "pnl", "graded"]] = [a if a is not None else np.nan, res, round(pnl, 3), now.strftime("%Y-%m-%d")]
        n += 1
    return led, n


def record(led, tier, source="live", excluded=False, market="rec_yds", side=None):
    ex = led.excluded.fillna("").astype(str) != "" if "excluded" in led.columns else pd.Series(False, index=led.index)
    d = led[(led.tier == tier) & (led.source == source) & led.result.isin(["win", "loss"]) & (ex == excluded) & (_market(led) == market)]
    if side:
        d = d[d.side == side]
    w = int((d.result == "win").sum()); l = int((d.result == "loss").sum())
    return w, l, float(d.pnl.sum()) if len(d) else 0.0


def post_lines(new, led, now):
    """Discord body: today's and tomorrow's new rule-tier flags per market under day headers, then the records."""
    lines = ["PAPER ONLY. Raw receiving model vs the pregame line. One weekend proves nothing; the bar is 150 flags per rule.", ""]
    if len(new):
        n2 = new.copy(); n2["local"] = pd.to_datetime(n2.kickoff_utc, utc=True).dt.tz_convert("America/Edmonton")
        today = now.tz_convert("America/Edmonton").normalize()
        n2 = n2[(n2.local >= today) & (n2.local < today + pd.Timedelta(days=2))]
        n2 = n2.assign(_ex=n2.excluded.fillna("").astype(str) != "", _mk=_market(n2))
        for mkt, (_, label) in MARKETS.items():
            m = n2[(n2._mk == mkt) & (n2.tier == "rule")]
            if not len(m):
                continue
            lines.append(f"== {label.upper()} ==" + (" (raw median 30%+ from the line)" if mkt == "rec_yds" else " (model 15+ points off the market's chance)"))
            keep = m[~m._ex].sort_values(["local", "gap"], ascending=[True, False])
            for day, g in keep.groupby(keep.local.dt.strftime("%a %b %-d"), sort=False):
                lines.append(day)
                for x in g.itertuples():
                    stale = f"  [line {x.line_age_h:.0f}h old: re-check]" if x.line_age_h > STALE_H else ""
                    q = f" ({x.inj})" if x.inj else ""
                    why = f"model {x.raw_median:g}" if mkt == "rec_yds" else f"model {100 * x.model_p:.0f}% vs market {100 * x.market_p:.0f}%"
                    qb = "  [backup QB]" if isinstance(x.qb, str) and x.qb else ""
                    lines.append(f"  {x.player}{q} ({x.team} vs {x.opp}) {x.side.upper()} {x.line:g} {label} | {why} | best {x.best_price:.2f} {x.best_book}{qb}{stale}")
            ex = m[m._ex]
            if len(ex):
                lines.append(f"left out: {len(ex)} over(s): " + "; ".join(f"{x.player} ({x.excluded.split(':')[0]})" for x in ex.itertuples()))
            lines.append("")
    for mkt, (_, label) in MARKETS.items():
        w, l, u = record(led, "rule", market=mkt)
        if w + l:
            wu, lu, uu = record(led, "rule", market=mkt, side="Under"); wo, lo, uo = record(led, "rule", market=mkt, side="Over")
            lines.append(f"{label}, rule tier, live: {w}-{l} ({w / (w + l):.1%}), {u:+.1f}u (ROI {u / (w + l):+.1%}) | unders {wu}-{lu} {uu:+.1f}u, overs {wo}-{lo} {uo:+.1f}u"
                         f"  [bar: {BAR_N} flags at {BAR_HIT:.0%}+ and profit]")
        w, l, u = record(led, "rule", market=mkt, excluded=True)
        if w + l:
            lines.append(f"{label}, excluded overs graded for reference: {w}-{l}, {u:+.1f}u (ROI {u / (w + l):+.1%})")
    return lines


def main(backfill=False):
    now = pd.Timestamp.now(tz="UTC")
    led = pd.read_csv(LEDGER) if os.path.exists(LEDGER) else pd.DataFrame(columns=COLS)
    for c in COLS:
        if c not in led.columns:
            led[c] = ""                      # ledgers written before a column existed
    led["market"] = _market(led)
    stats = None

    def load_stats(seasons):
        import props_model as M, props_weekly as W
        frames = [d for d in (M._read("stats", int(y), True) for y in sorted(set(seasons))) if d is not None and len(d)]
        if not frames:
            return None
        st = pd.concat(frames, ignore_index=True)
        st["team"] = st.team.map(W.norm_team)
        return st

    paths = [os.path.join(LOGS, f) for f in ("props_grid_raw.csv", "props_grid.csv", "props_board.csv")]
    new = pd.DataFrame(columns=COLS)
    if all(os.path.exists(p) for p in paths):
        raw, grid, board = (pd.read_csv(p, low_memory=False) for p in paths)
        season, week = int(raw.season.iloc[0]), int(raw.week.iloc[0])
        lined = {"rec_yds": set(grid.pid[grid.market_line.notna()]) if "market_line" in grid.columns else set(),
                 "receptions": set(grid.pid[grid.rec_line.notna()]) if "rec_line" in grid.columns else set()}
        if not lined["rec_yds"] and not lined["receptions"]:
            print("raw-gap ledger: this run has no pregame lines (no price pull), so there is nothing to flag")
        team_ts = {}
        try:
            import props_market as PM
            sp = PM.latest_snapshot(season, week)
            if sp:
                sn = pd.read_csv(sp, usecols=["snap_ts", "home", "away"])
                for (h, a), t in sn.groupby(["home", "away"]).snap_ts.max().items():
                    team_ts[PM.TEAMS.get(h, h)] = t; team_ts[PM.TEAMS.get(a, a)] = t
        except Exception as ex:
            print("raw-gap ledger: price times unavailable,", type(ex).__name__, ex)
        tags, qbs = {}, {}
        if not backfill:
            try:
                import props_weather as PW
                tags = PW.week_tags(raw[pd.to_datetime(raw.kickoff_utc, utc=True) > now])
                poor = sorted({t for t, v in tags.items() if v["status"] == "poor"})
                print(f"raw-gap ledger: weather checked for {len(tags)} team(s); poor weather: {', '.join(poor) if poor else 'none'}")
            except Exception as ex:
                print("raw-gap ledger: weather unavailable, nothing excluded for it,", type(ex).__name__, ex)
            try:
                rp = os.path.join(LOGS, "rushpass_grid.csv")
                stats = load_stats([season])
                if os.path.exists(rp) and stats is not None:
                    qbs = qb_tags(pd.read_csv(rp), stats, season, week)
                print(f"raw-gap ledger: quarterbacks checked; backup or in doubt: {', '.join(sorted(qbs)) if qbs else 'none'}")
                for t in sorted(qbs):
                    print(f"    {t}: {qbs[t]}")
            except Exception as ex:
                print("raw-gap ledger: quarterback check unavailable, nothing excluded for it,", type(ex).__name__, ex)
        cand = build_flags(raw, grid, board, now, backfill=backfill, team_ts=team_ts, tags=tags, qbs=qbs)
        key = lambda d: list(zip(d.season.astype(int), d.week.astype(int), d.pid, _market(d)))
        # The latest pregame look is the one that counts. Every player this run priced in a market,
        # whose game is still ahead, has his earlier ungraded live row in that market replaced by
        # this run's verdict (or removed, if the gap has closed). Graded and backfill rows stay.
        before = {}
        if not backfill and len(led):
            before = {k: (s, float(l)) for k, s, l in zip(key(led), led.side, led.line)}
            ahead = set(raw.pid[pd.to_datetime(raw.kickoff_utc, utc=True) > now])
            priced = [p in ahead and p in lined.get(m, set()) for p, m in zip(led.pid, led.market)]
            drop = (led.source == "live") & (led.result.fillna("") == "") & pd.Series(priced, index=led.index) & (led.week.astype(int) == week)
            led = led[~drop].copy()
        seen = set(key(led)) if len(led) else set()
        cand = cand[[k not in seen for k in key(cand)]] if len(cand) else cand
        new = cand[[before.get(k) != (s, float(l)) for k, s, l in zip(key(cand), cand.side, cand.line)]] if len(cand) else cand
        if len(cand):
            led = pd.concat([led, cand], ignore_index=True) if len(led) else cand.copy()
        for mkt in MARKETS:
            c = cand[_market(cand) == mkt] if len(cand) else cand
            print(f"raw-gap ledger: {mkt}: {len(c)} candidate(s) this run ({int((c.tier == 'rule').sum()) if len(c) else 0} rule, "
                  f"{int((c.tier == 'watch').sum()) if len(c) else 0} watch, {int((c.excluded.astype(str) != '').sum()) if len(c) else 0} excluded)")
    else:
        print("raw-gap ledger: grid files missing, grading only")
    n_graded = 0
    if len(led) and (led.result.fillna("") == "").any():
        st = load_stats(led.season.astype(int).unique())
        if st is not None:
            import props_weekly as W
            led["team"] = led.team.map(W.norm_team)
            led, n_graded = grade(led, st, now)
    if len(led):
        led[COLS].to_csv(LEDGER, index=False)
    for mkt in MARKETS:
        for tier in ("rule", "watch"):
            for src in ("live", "backfill"):
                w, l, u = record(led, tier, src, market=mkt) if len(led) else (0, 0, 0.0)
                if w + l:
                    print(f"  {mkt:10s} {tier:5s} {src:8s} {w}-{l} ({w / (w + l):.1%}) {u:+.1f}u")
    lines = post_lines(new if not backfill else new.iloc[0:0], led, now) if len(led) else []
    for ln in lines:
        print(ln)
    if os.environ.get("RAW_LEDGER_POST") == "1" and (len(new) or n_graded) and not backfill:
        import notify
        notify.send("NFL receiving props, raw-model rules (paper)", lines, "props_raw_ledger.py. Rules and bars are written at the top of the file.", getattr(notify, "GREY", None))


def unit_tests():
    now = pd.Timestamp("2026-11-29T12:00:00Z")
    pr = {f"p_rec_{k}": v for k, v in zip(range(1, 11), [.95, .85, .70, .50, .30, .15, .08, .04, .02, .01])}
    raw = pd.DataFrame(dict(season=2026, week=12, pid=["a", "b", "c", "d"], display_name=["A", "B", "C", "D"], pos="WR", team="X", opp="Y",
                            kickoff_utc=["2026-11-29T18:00:00+00:00"] * 3 + ["2026-11-27T18:00:00+00:00"], inj=np.nan, status_applied=False,
                            proj_targets=5.0, med_yds=[40.0, 52.0, 70.0, 10.0], ts="2026-11-29T11:59:00Z", **pr))
    grid = pd.DataFrame(dict(pid=["a", "b", "c", "d"], market_line=[60.5, 60.5, 60.5, 40.5], rec_line=[4.5, 3.5, np.nan, 2.5], odds_ts="2026-11-29T11:00:00+00:00"))
    board = pd.DataFrame(dict(pid=["a", "a", "a", "b", "c", "d"], market="rec_yds", line=[60.5, 60.5, 60.5, 60.5, 60.5, 40.5],
                              side=["Under", "Under", "Over", "Under", "Over", "Under"], book=["dk", "fd", "dk", "dk", "dk", "dk"],
                              book_odds=[-115, -105, -110, -110, 100, -110], book_fair_pct=50.0, devig_method="two_way"))
    rb = pd.DataFrame(dict(pid=["a", "a", "b", "b"], market="receptions", line=[4.5, 4.5, 3.5, 3.5], side=["Over", "Under", "Over", "Under"], book="dk",
                           book_odds=[-110, -110, 120, -150], book_fair_pct=[50.0, 50.0, 55.0, 45.0], devig_method="two_way"))
    f = build_flags(raw, grid, board, now).set_index("pid")
    assert list(f.index) == ["a", "c"], list(f.index)          # b is 14% off (no flag); d has kicked off
    assert f.at["a", "tier"] == "rule" and f.at["a", "side"] == "Under" and f.at["a", "best_book"] == "fd" and abs(f.at["a", "best_price"] - 1.952) < 1e-3
    assert f.at["c", "tier"] == "watch" and f.at["c", "side"] == "Over" and f.at["c", "best_price"] == 2.0 and f.at["a", "market"] == "rec_yds"
    tg = {"X": dict(status="poor", reason="wind 18 mph", detail="2C, wind 18 mph")}
    fw = build_flags(raw, grid, board, now, tags=tg, qbs={"X": "backup QB: Z starts"}).set_index("pid")
    assert fw.at["c", "excluded"] == "weather: wind 18 mph; backup QB: Z starts" and fw.at["a", "excluded"] == ""   # only the over is excluded
    assert fw.at["a", "qb"] == "backup QB: Z starts" and fw.at["a", "weather"] == "2C, wind 18 mph" and f.at["c", "excluded"] == ""
    # receptions: a is P(5+)=0.30 vs market 0.50 -> under by 20 points (rule); b is P(4+)=0.50 vs 0.55 -> 5 points, no flag
    fr = build_flags(raw, grid, pd.concat([board, rb]), now)
    r2 = fr[fr.market == "receptions"].set_index("pid")
    assert list(r2.index) == ["a"] and r2.at["a", "side"] == "Under" and r2.at["a", "tier"] == "rule" and abs(r2.at["a", "gap"] - 0.20) < 1e-9
    assert abs(r2.at["a", "model_p"] - 0.70) < 1e-9 and abs(r2.at["a", "market_p"] - 0.50) < 1e-9 and abs(r2.at["a", "best_price"] - dec(-110)) < 1e-3
    st = pd.DataFrame(dict(season=2026, week=12, season_type="REG", team=["X", "X"], player_id=["a", "zz"], receiving_yards=[45.0, 5.0], receptions=[5.0, 1.0]))
    g, n = grade(fr[COLS], st, pd.Timestamp("2026-11-30T12:00:00Z"))
    gy = g[g.market == "rec_yds"].set_index("pid"); gr = g[g.market == "receptions"].set_index("pid")
    assert n == 3 and gy.at["a", "result"] == "win" and abs(gy.at["a", "pnl"] - 0.952) < 1e-3 and gy.at["c", "result"] == "void" and gy.at["c", "pnl"] == 0
    assert gr.at["a", "result"] == "loss" and gr.at["a", "pnl"] == -1.0                       # 5 catches beats under 4.5
    assert record(g, "rule") == (1, 0, gy.at["a", "pnl"]) and record(g, "rule", market="receptions") == (0, 1, -1.0)
    assert record(g, "rule", side="Over") == (0, 0, 0.0)
    # an old ledger with no market column is all receiving yards
    assert list(_market(g.drop(columns=["market"]))) == ["rec_yds"] * len(g)
    # quarterback tags: X's expected starter did not throw the last four games; Y's did
    rg = pd.DataFrame(dict(season=2026, week=12, team=["X", "Y", "W"], gsis_id=["q2", "q3", "q4"], name=["Backup", "Usual", "Hurt"], starter=True,
                           inj=[np.nan, np.nan, "Questionable"]))
    ps = pd.DataFrame(dict(season=2026, season_type="REG", week=[8, 9, 10, 11, 11, 10, 11, 12, 11], team=["X"] * 5 + ["Y", "Y", "Y", "W"],
                           player_id=["q1", "q1", "q1", "q1", "q2", "q3", "q3", "q9", "q4"], player_display_name=["Starter"] * 4 + ["Backup"] + ["Usual"] * 2 + ["Other", "Hurt"],
                           attempts=[30, 31, 28, 20, 9, 33, 35, 40, 30]))
    qt = qb_tags(rg, ps, 2026, 12)
    assert set(qt) == {"X", "W"} and qt["X"].startswith("backup QB: Backup starts, Starter") and qt["W"] == "QB questionable: Hurt", qt
    assert qb_tags(rg.assign(week=11), ps, 2026, 12) == {}                                    # a grid from another week tags nothing
    assert abs(dec(150) - 2.5) < 1e-9 and abs(dec(-200) - 1.5) < 1e-9
    print("props_raw_ledger unit tests passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--test", action="store_true"); ap.add_argument("--backfill", action="store_true")
    a = ap.parse_args()
    unit_tests() if a.test else main(backfill=a.backfill)
