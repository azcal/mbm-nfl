"""
props_raw_ledger.py: PAPER ledger for the raw-gap rule on receiving yards (mbm-nfl).

    python props_raw_ledger.py            # after props_weekly.py: flag this run's players, grade finished ones, post
    python props_raw_ledger.py --test     # unit tests only

THE RULE (written 2026-11-26, before any row below was graded)
  For every player with a consensus pregame receiving-yards line, compare the RAW model median
  (logs/props_grid_raw.csv, no market input) to that line.
      gap = |raw median - line| / max(line, 10)
  gap >= 0.30  -> tier "rule":  take the model's side (Over if the median is above the line)
  gap >= 0.15  -> tier "watch": logged the same way, not part of the bar
  One unit, at the best price on that side at the consensus line among the books pulled.
  If a later run prices the same player before kickoff, that later look replaces the earlier one.
  A player with no stat line in a finished game is void, as a book would void it.

THE BAR
  150 graded rule-tier flags from live runs: hit rate 56% or better AND positive ROI at the
  best price. Short of either, the rule is retired. Rows marked source=backfill do not count.

WHERE IT CAME FROM
  props_backtest_replay.py on 2026 weeks 2-9 and 11, pregame lines only: rule tier 83-61
  (57.6%), +14.5u; watch tier 196-161 (54.9%), +15.9u. Week 11 alone was +9.6u of the rule
  tier, the profit was almost all unders, and the 30% cut was chosen after seeing those
  results. That is why this is a paper ledger.

Reads logs/props_grid_raw.csv (raw medians), logs/props_grid.csv (consensus line, price time)
and logs/props_board.csv (book prices). Writes logs/raw_gap_ledger.csv. Never changes the grid,
the board or the Unabated file.
"""
import argparse, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
LOGS = os.path.join(HERE, "logs")
LEDGER = os.path.join(LOGS, "raw_gap_ledger.csv")
RULE_GAP, WATCH_GAP = 0.30, 0.15
BAR_N, BAR_HIT = 150, 0.56
STALE_H = 36.0          # a line older than this at flag time is marked stale: re-check it before acting
COLS = ["flag_ts", "source", "season", "week", "pid", "player", "pos", "team", "opp", "kickoff_utc", "inj", "tier", "side", "line",
        "raw_median", "gap", "best_price", "best_book", "n_books", "odds_ts", "line_age_h", "actual", "result", "pnl", "graded"]


def dec(american):
    a = float(american)
    return 1 + a / 100 if a > 0 else 1 + 100 / -a


def build_flags(raw, grid, board, now, backfill=False, team_ts=None):
    """Rows to flag from one run's files. now is a UTC Timestamp. team_ts: {team: time that team's
    game was last priced}; the grid's own odds_ts is the first row of a merged file, so on a week
    with a Thursday pull it reports Thursday for every game."""
    g = grid[grid.market_line.notna()][["pid", "market_line", "odds_ts"]].drop_duplicates("pid")
    r = raw.drop(columns=[c for c in ("market_line", "odds_ts") if c in raw.columns]).merge(g, on="pid")
    r = r[~r.status_applied.astype(bool) & (r.proj_targets > 0)].copy()
    if team_ts:
        r["odds_ts"] = [team_ts.get(t, o) for t, o in zip(r.team, r.odds_ts)]
    kick = pd.to_datetime(r.kickoff_utc, utc=True); ots = pd.to_datetime(r.odds_ts, utc=True)
    built = pd.to_datetime(r.ts, utc=True) if "ts" in r.columns else pd.Series(now, index=r.index)
    # the line and the projection must both predate kickoff; a live run also needs the game still ahead
    ok = (ots < kick) & (built < kick)
    if not backfill:
        ok &= kick > now
    r = r[ok].copy()
    if not len(r):
        return pd.DataFrame(columns=COLS)
    r["gap"] = (r.med_yds - r.market_line).abs() / r.market_line.clip(lower=10)
    r = r[r.gap >= WATCH_GAP].copy()
    b = board[(board.market == "rec_yds")].copy()
    b["price"] = b.book_odds.map(dec)
    rows = []
    for x in r.itertuples():
        side = "Over" if x.med_yds > x.market_line else "Under"
        q = b[(b.pid == x.pid) & (b.line == x.market_line) & (b.side == side)]
        if not len(q):
            continue                      # no book is quoting that side at the consensus line
        best = q.sort_values("price", ascending=False).iloc[0]
        k = pd.Timestamp(x.kickoff_utc); o = pd.Timestamp(x.odds_ts)
        age = (min(now, k) - o).total_seconds() / 3600 if not backfill else (pd.Timestamp(x.ts) - o).total_seconds() / 3600
        rows.append(dict(flag_ts=now.strftime("%Y-%m-%dT%H:%M:%SZ"), source="backfill" if backfill else "live", season=int(x.season), week=int(x.week),
                         pid=x.pid, player=x.display_name, pos=x.pos, team=x.team, opp=x.opp, kickoff_utc=x.kickoff_utc,
                         inj=x.inj if isinstance(x.inj, str) else "", tier="rule" if x.gap >= RULE_GAP else "watch", side=side,
                         line=float(x.market_line), raw_median=round(float(x.med_yds), 1), gap=round(float(x.gap), 3),
                         best_price=round(float(best.price), 3), best_book=best.book, n_books=int(q.book.nunique()), odds_ts=x.odds_ts,
                         line_age_h=round(age, 1), actual=np.nan, result="", pnl=np.nan, graded=""))
    return pd.DataFrame(rows, columns=COLS)


def grade(led, stats, now):
    """Fill result and pnl for rows whose game is over and whose week is in the stats file."""
    if not len(led):
        return led, 0
    led = led.copy(); n = 0
    led["result"] = led.result.fillna("").astype(str); led["graded"] = led.graded.fillna("").astype(str)
    st = stats[stats.season_type == "REG"]
    have = set(zip(st.season.astype(int), st.week.astype(int), st.team))
    act = {(int(r.season), int(r.week), r.player_id): float(r.receiving_yards) for r in st.itertuples() if pd.notna(r.receiving_yards)}
    for i, x in led.iterrows():
        if x.result or pd.Timestamp(x.kickoff_utc) + pd.Timedelta(hours=5) > now:
            continue
        if (int(x.season), int(x.week), x.team) not in have:
            continue                      # nflverse has not posted that game yet
        a = act.get((int(x.season), int(x.week), x.pid))
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


def record(led, tier, source="live"):
    d = led[(led.tier == tier) & (led.source == source) & led.result.isin(["win", "loss"])]
    w = int((d.result == "win").sum()); l = int((d.result == "loss").sum())
    return w, l, float(d.pnl.sum()) if len(d) else 0.0


def post_lines(new, led, now):
    """Discord body: today's and tomorrow's new rule-tier flags under day headers, then the record."""
    lines = ["PAPER ONLY. Raw receiving model vs the pregame line, 30%+ apart. One weekend proves nothing; the bar is 150 flags.", ""]
    if len(new):
        n2 = new.copy(); n2["local"] = pd.to_datetime(n2.kickoff_utc, utc=True).dt.tz_convert("America/Edmonton")
        today = now.tz_convert("America/Edmonton").normalize()
        n2 = n2[(n2.local >= today) & (n2.local < today + pd.Timedelta(days=2))]
        rule = n2[n2.tier == "rule"].sort_values(["local", "gap"], ascending=[True, False])
        for day, g in rule.groupby(rule.local.dt.strftime("%a %b %-d"), sort=False):
            lines.append(day)
            for x in g.itertuples():
                stale = f"  [line {x.line_age_h:.0f}h old: re-check]" if x.line_age_h > STALE_H else ""
                q = f" ({x.inj})" if x.inj else ""
                lines.append(f"  {x.player}{q} ({x.team} vs {x.opp}) {x.side.upper()} {x.line:g} rec yds | model {x.raw_median:g} | best {x.best_price:.2f} {x.best_book}{stale}")
            lines.append("")
        lines.append(f"new this run: {int((n2.tier == 'rule').sum())} rule flags, {int((n2.tier == 'watch').sum())} watch (15-30%, in the ledger only)")
    for tier in ("rule", "watch"):
        w, l, u = record(led, tier)
        if w + l:
            lines.append(f"{tier} tier, live: {w}-{l} ({w / (w + l):.1%}), {u:+.1f}u" + (f"  [bar: {BAR_N} flags at {BAR_HIT:.0%}+ and profit]" if tier == "rule" else ""))
    return lines


def main(backfill=False):
    now = pd.Timestamp.now(tz="UTC")
    led = pd.read_csv(LEDGER) if os.path.exists(LEDGER) else pd.DataFrame(columns=COLS)
    paths = [os.path.join(LOGS, f) for f in ("props_grid_raw.csv", "props_grid.csv", "props_board.csv")]
    new = pd.DataFrame(columns=COLS)
    if all(os.path.exists(p) for p in paths):
        raw, grid, board = (pd.read_csv(p, low_memory=False) for p in paths)
        team_ts = {}
        try:
            import props_market as PM
            sp = PM.latest_snapshot(int(raw.season.iloc[0]), int(raw.week.iloc[0]))
            if sp:
                sn = pd.read_csv(sp, usecols=["snap_ts", "home", "away"])
                for (h, a), t in sn.groupby(["home", "away"]).snap_ts.max().items():
                    team_ts[PM.TEAMS.get(h, h)] = t; team_ts[PM.TEAMS.get(a, a)] = t
        except Exception as ex:
            print("raw-gap ledger: price times unavailable,", type(ex).__name__, ex)
        cand = build_flags(raw, grid, board, now, backfill=backfill, team_ts=team_ts)
        # The latest pregame look is the one that counts. Every player this run priced, whose
        # game is still ahead, has his earlier ungraded live row replaced by this run's verdict
        # (or removed, if the gap has closed). Graded rows and backfill rows are never touched.
        if not backfill and len(led):
            kick = pd.to_datetime(raw.kickoff_utc, utc=True)
            priced = set(raw.pid[kick > now]) & set(grid.pid[grid.market_line.notna()])
            before = {(int(a), int(b), c): (d, float(e)) for a, b, c, d, e in zip(led.season, led.week, led.pid, led.side, led.line)}
            drop = (led.source == "live") & (led.result.fillna("") == "") & led.pid.isin(priced) & (led.week.astype(int) == int(raw.week.iloc[0]))
            led = led[~drop].copy()
        else:
            before = {}
        seen = set(zip(led.season.astype(int), led.week.astype(int), led.pid)) if len(led) else set()
        cand = cand[[(int(a), int(b), c) not in seen for a, b, c in zip(cand.season, cand.week, cand.pid)]] if len(cand) else cand
        new = cand[[before.get((int(a), int(b), c)) != (d, float(e)) for a, b, c, d, e in zip(cand.season, cand.week, cand.pid, cand.side, cand.line)]] if len(cand) else cand
        if len(cand):
            led = pd.concat([led, cand], ignore_index=True) if len(led) else cand.copy()
        print(f"raw-gap ledger: {len(cand)} candidate(s) this run, {len(new)} new ({int((new.tier == 'rule').sum()) if len(new) else 0} rule, "
              f"{int((new.tier == 'watch').sum()) if len(new) else 0} watch)")
    else:
        print("raw-gap ledger: grid files missing, grading only")
    n_graded = 0
    if len(led) and (led.result.fillna("") == "").any():
        import props_model as M
        frames = [d for d in (M._read("stats", int(y), True) for y in sorted(led.season.astype(int).unique())) if d is not None and len(d)]
        if frames:
            st = pd.concat(frames, ignore_index=True)
            import props_weekly as W
            st["team"] = st.team.map(W.norm_team)
            led["team"] = led.team.map(W.norm_team)
            led, n_graded = grade(led, st, now)
    if len(led):
        led[COLS].to_csv(LEDGER, index=False)
    for tier in ("rule", "watch"):
        for src in ("live", "backfill"):
            w, l, u = record(led, tier, src) if len(led) else (0, 0, 0.0)
            if w + l:
                print(f"  {tier:5s} {src:8s} {w}-{l} ({w / (w + l):.1%}) {u:+.1f}u")
    lines = post_lines(new if not backfill else new.iloc[0:0], led, now) if len(led) else []
    for ln in lines:
        print(ln)
    if os.environ.get("RAW_LEDGER_POST") == "1" and (len(new) or n_graded) and not backfill:
        import notify
        notify.send("NFL receiving raw-gap rule (paper)", lines, "props_raw_ledger.py. Rule and bar are written at the top of the file.", getattr(notify, "GREY", None))


def unit_tests():
    now = pd.Timestamp("2026-11-29T12:00:00Z")
    raw = pd.DataFrame(dict(season=2026, week=12, pid=["a", "b", "c", "d"], display_name=["A", "B", "C", "D"], pos="WR", team="X", opp="Y",
                            kickoff_utc=["2026-11-29T18:00:00+00:00"] * 3 + ["2026-11-27T18:00:00+00:00"], inj=np.nan, status_applied=False,
                            proj_targets=5.0, med_yds=[40.0, 52.0, 70.0, 10.0], ts="2026-11-29T11:59:00Z"))
    grid = pd.DataFrame(dict(pid=["a", "b", "c", "d"], market_line=[60.5, 60.5, 60.5, 40.5], odds_ts="2026-11-29T11:00:00+00:00"))
    board = pd.DataFrame(dict(pid=["a", "a", "a", "b", "c", "d"], market="rec_yds", line=[60.5, 60.5, 60.5, 60.5, 60.5, 40.5],
                              side=["Under", "Under", "Over", "Under", "Over", "Under"], book=["dk", "fd", "dk", "dk", "dk", "dk"],
                              book_odds=[-115, -105, -110, -110, 100, -110]))
    f = build_flags(raw, grid, board, now).set_index("pid")
    assert list(f.index) == ["a", "c"], list(f.index)          # b is 14% off (no flag); d has kicked off
    assert f.at["a", "tier"] == "rule" and f.at["a", "side"] == "Under" and f.at["a", "best_book"] == "fd" and abs(f.at["a", "best_price"] - 1.952) < 1e-3
    assert f.at["c", "tier"] == "watch" and f.at["c", "side"] == "Over" and f.at["c", "best_price"] == 2.0
    st = pd.DataFrame(dict(season=2026, week=12, season_type="REG", team=["X", "X"], player_id=["a", "zz"], receiving_yards=[45.0, 5.0]))
    g, n = grade(f.reset_index()[COLS], st, pd.Timestamp("2026-11-30T12:00:00Z"))
    g = g.set_index("pid")
    assert n == 2 and g.at["a", "result"] == "win" and abs(g.at["a", "pnl"] - 0.952) < 1e-3 and g.at["c", "result"] == "void" and g.at["c", "pnl"] == 0
    assert record(g.reset_index(), "rule") == (1, 0, g.at["a", "pnl"])
    assert abs(dec(150) - 2.5) < 1e-9 and abs(dec(-200) - 1.5) < 1e-9
    print("props_raw_ledger unit tests passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--test", action="store_true"); ap.add_argument("--backfill", action="store_true")
    a = ap.parse_args()
    unit_tests() if a.test else main(backfill=a.backfill)
