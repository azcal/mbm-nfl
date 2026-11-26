"""
props_backtest_replay.py: replay the receiving model week by week through the LIVE roster path
and score it, by how big a role the model gave each player and against archived prop lines.

    python props_backtest_replay.py --data DIR [--set K_UPDATE=2 ...]

DIR holds nflverse files for 2023 on: stats_player_week_YYYY, injuries_YYYY, snap_counts_YYYY,
roster_weekly_YYYY and depth_charts_YYYY (.parquet). 2023-24 are burn-in. 2025 is the tuning
season, 2026 the test season, and only 2026 has prop lines (data/odds/props_2026_w*.csv).

Each past week is projected from the roster as it stood the week before and the depth chart
dated before that week's first game, which is what the Sunday run sees. Without this the
model's history-only path runs instead, and that path is flatter than what goes out live.
"""
import argparse, glob, json, os, re, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import props_model as M

norm = lambda s: re.sub(r"[^a-z]", "", str(s).lower())


def build_replay(data, seasons=(2025, 2026)):
    g = pd.read_csv(os.path.join(HERE, "data", "games.csv"))
    first = g[g.game_type == "REG"].groupby(["season", "week"]).gameday.min()
    out = {}
    for y in seasons:
        ro = pd.read_parquet(os.path.join(data, f"roster_weekly_{y}.parquet"))
        ro = ro[ro.position.isin(M.POS) & (ro.status == "ACT")][["season", "week", "team", "gsis_id", "position", "full_name"]]
        dc = pd.read_parquet(os.path.join(data, f"depth_charts_{y}.parquet"), columns=["dt", "gsis_id", "pos_abb", "pos_rank"])
        dc = dc[dc.pos_abb.isin(M.POS)].copy(); dc["day"] = dc.dt.str[:10]; dc["pos_rank"] = pd.to_numeric(dc.pos_rank, errors="coerce")
        days = np.array(sorted(dc.day.unique()))
        for wk in sorted(ro.week.unique()):
            if (y, wk) not in first.index:
                continue
            rw = ro[ro.week == max(wk - 1, 1)]
            ok = days[days < first[(y, wk)]]
            if len(ok):
                d = dc[dc.day == ok[-1]]
                d = d[d.dt == d.dt.max()].sort_values("pos_rank").drop_duplicates("gsis_id")[["gsis_id", "pos_rank"]]
                rw = rw.merge(d, on="gsis_id", how="left")
            else:
                rw = rw.assign(pos_rank=np.nan)
            for tm, grp in rw.groupby("team"):
                out[(int(y), int(wk), tm)] = grp
    return out


def lines_2026():
    pr = []
    for f in sorted(glob.glob(os.path.join(HERE, "data", "odds", "props_2026_w*.csv"))):
        x = pd.read_csv(f); x["week"] = int(re.search(r"_w(\d+)_", f).group(1)); pr.append(x)
    pr = pd.concat(pr); pr["nm"] = pr.player.map(norm)
    # Pregame quotes only. A snapshot taken on Sunday afternoon holds LIVE lines for the games
    # already under way, and a live line has seen part of the result. Keep each game's last
    # snapshot from before its kickoff.
    pr = pr[pd.to_datetime(pr.snap_ts, utc=True) < pd.to_datetime(pr.commence, utc=True)]
    pr = pr[pr.snap_ts == pr.groupby(["week", "event_id"]).snap_ts.transform("max")]
    o = pr[pr.outcome == "Over"]
    L = o[o.market == "player_reception_yds"].groupby(["week", "nm"]).point.median().rename("line").reset_index()
    R = o[o.market == "player_receptions"].groupby(["week", "nm"]).point.median().rename("rline").reset_index()
    return L.merge(R, on=["week", "nm"], how="outer")


def prices_2026():
    """Median and best pregame price per side at the consensus receiving-yards line."""
    pr = []
    for f in sorted(glob.glob(os.path.join(HERE, "data", "odds", "props_2026_w*.csv"))):
        x = pd.read_csv(f); x["week"] = int(re.search(r"_w(\d+)_", f).group(1)); pr.append(x)
    pr = pd.concat(pr); pr["nm"] = pr.player.map(norm)
    pr = pr[pd.to_datetime(pr.snap_ts, utc=True) < pd.to_datetime(pr.commence, utc=True)]
    pr = pr[pr.snap_ts == pr.groupby(["week", "event_id"]).snap_ts.transform("max")]
    pr = pr[pr.market == "player_reception_yds"]
    med = pr[pr.outcome == "Over"].groupby(["week", "nm"]).point.median().rename("line").reset_index()
    q = pr.merge(med, on=["week", "nm"]); q = q[q.point == q.line]
    a = q.groupby(["week", "nm", "line", "outcome"]).price_dec.agg(["median", "max"]).unstack("outcome")
    a.columns = [f"{o.lower()}_{s}" for s, o in a.columns]
    return a.reset_index()


def gap_table(df, label):
    """The model's side of the pregame line by how far the raw median sits from it."""
    d = df[(df.season == 2026) & df.played].copy(); d["nm"] = d.name.map(norm)
    k = d.merge(prices_2026(), on=["week", "nm"])
    k = k[(k.act_yds != k.line) & k.over_median.notna() & k.under_median.notna()].copy()
    k["side"] = k.med_yds > k.line; k["ok"] = k.side == (k.act_yds > k.line)
    k["rel"] = (k.med_yds - k.line).abs() / k.line.clip(lower=10)
    k["pnl_med"] = np.where(k.ok, np.where(k.side, k.over_median, k.under_median) - 1, -1)
    k["pnl_best"] = np.where(k.ok, np.where(k.side, k.over_max, k.under_max) - 1, -1)
    print(f"\n[{label}] raw median vs the pregame line, {len(k)} player-weeks with a two-sided price; breakeven {((1 / k.over_median + 1 / k.under_median) / 2).mean():.3f}")
    for th in (0, .15, .30, .45):
        m = k[k.rel >= th]
        print(f"  gap {int(th * 100):2d}%+  n={len(m):3d}  side right {m.ok.mean():.3f} (+/-{1.96 * np.sqrt(.25 / len(m)):.3f})  ROI at the median price {m.pnl_med.mean():+.3f}, best price {m.pnl_best.mean():+.3f}"
              f"  | overs {int(m.side.sum())} at {m[m.side].ok.mean():.3f}, unders {int((~m.side).sum())} at {m[~m.side].ok.mean():.3f}")
    return k


def report(df, label):
    out = {}
    for y in (2025, 2026):
        d = df[(df.season == y) & df.played & (df.proj_targets > 0)].copy()
        print(f"\n[{label}] {y}: {len(d)} player-games | median MAE {np.abs(d.act_yds - d.med_yds).mean():.2f} | actual over the model median {np.mean(d.act_yds > d.med_yds):.3f}")
        print(f"  {'projected targets':>18} {'n':>5} {'proj tg':>8} {'act tg':>7} {'model med':>10} {'act med':>8} {'over med':>9} {'MAE':>6}")
        for lo, hi in ((8, 99), (6, 8), (4, 6), (2.5, 4)):
            k = d[(d.proj_targets >= lo) & (d.proj_targets < hi)]
            if len(k):
                print(f"  {f'{lo} to {hi}' if hi < 99 else f'{lo}+':>18} {len(k):>5} {k.proj_targets.mean():>8.2f} {k.act_tg.mean():>7.2f} {k.med_yds.mean():>10.1f} {k.act_yds.median():>8.1f} "
                      f"{np.mean(k.act_yds > k.med_yds):>9.3f} {np.abs(k.act_yds - k.med_yds).mean():>6.1f}")
        out[y] = float(np.abs(d.act_yds - d.med_yds).mean())
    d = df[(df.season == 2026) & df.played].copy(); d["nm"] = d.name.map(norm)
    x = d.merge(lines_2026(), on=["week", "nm"])
    k = x[x.line.notna() & (x.act_yds != x.line)]
    side = (k.med_yds > k.line).values; win = (k.act_yds > k.line).values; rel = np.abs(k.med_yds - k.line).values / np.maximum(k.line.values, 10)
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression().fit(k[["med_yds", "line"]], k.act_yds)
    print(f"\n[{label}] 2026 vs archived receiving-yard lines, {len(k)} player-weeks")
    print(f"  model MAE {np.abs(k.act_yds - k.med_yds).mean():.2f} vs line MAE {np.abs(k.act_yds - k.line).mean():.2f} | spread of medians {k.med_yds.std():.1f} vs lines {k.line.std():.1f} | corr {np.corrcoef(k.med_yds, k.line)[0, 1]:.2f}")
    print(f"  side right {np.mean(side == win):.3f} | gap 15%+ {np.mean((side == win)[rel >= .15]):.3f} (n={int((rel >= .15).sum())}) | gap 30%+ {np.mean((side == win)[rel >= .30]):.3f} (n={int((rel >= .30).sum())})")
    print(f"  weights the results give: {lr.coef_[0]:+.2f} x model, {lr.coef_[1]:+.2f} x line")
    for lab, m in (("line 50+", k.line >= 50), ("line 25 to 50", (k.line >= 25) & (k.line < 50)), ("line under 25", k.line < 25)):
        q = k[m]; print(f"  {lab:14s} n={len(q):3d} line {q.line.mean():5.1f} | model median {q.med_yds.mean():5.1f} | actual median {q.act_yds.median():5.1f} | side right {np.mean((q.med_yds > q.line) == (q.act_yds > q.line)):.3f}")
    r = x[x.rline.notna() & (x.act_rec != x.rline) & (x.med_rec != x.rline)]
    print(f"  receptions: side right {np.mean((r.med_rec > r.rline) == (r.act_rec > r.rline)):.3f} (n={len(r)})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--label", default="run"); ap.add_argument("--save")
    a = ap.parse_args()
    M.DATA = a.data
    for kv in a.set:
        k, v = kv.split("="); setattr(M, k, type(getattr(M, k))(v) if not isinstance(getattr(M, k), bool) else v == "True")
    spec = json.load(open(os.path.join(HERE, "props_spec.json")))
    M.RNG = np.random.default_rng(7)
    df = M.run(range(2023, 2027), kappa=spec["kappa"], alpha=spec["alpha"], H=spec["H"], vol_mult=spec.get("volume_mult"),
               min_proj_targets=2.5, replay=build_replay(a.data))
    if a.save:
        df.to_pickle(a.save)
    report(df, a.label)
    gap_table(df, a.label)
