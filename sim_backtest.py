"""
sim_backtest.py: does the joint sim (sim_nfl.py) describe real games? Walk-forward, same
protocol as the sides model: 2023 is the tuning year, 2024 and 2025 are frozen holdouts, each
scored by heads trained only on earlier seasons.

    python sim_backtest.py --pbp DIR --fit       # print the constants measured on 2016-2023
    python sim_backtest.py --pbp DIR             # totals and team totals on 2023 / 2024 / 2025
    python sim_backtest.py --pbp DIR --players   # rushing and passing ladders

DIR holds nflverse play_by_play_YYYY.parquet files (2016 on). Player runs download nflverse
weekly player stats.

PRE-REGISTERED BARS (written before the holdouts were scored):
  totals       CRPS of the simulated total at or below the empirical-residual floor (the model
               total plus the training seasons' own misses) on 2024 and 2025 pooled, and every
               reliability bucket with n >= 100 within 0.05 of its stated rate.
  team totals  same, per team score.
  players      every reliability bucket with n >= 100 within 0.05, pooled 2024-25.
Fail any bar and the output stays a research file.
"""
import argparse, glob, os, sys
import numpy as np, pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import engine_nfl as E, spread, sim_nfl as S

SPEC = __import__("json").load(open(os.path.join(HERE, "spec.json")))
CFG, C = SPEC["cfg"], SPEC["C"]


def load_pbp_drives(pbp_dir):
    cache = os.path.join(HERE, "data", "team_game_drives.parquet")
    frames = []
    for p in sorted(glob.glob(os.path.join(pbp_dir, "*.parquet"))):
        frames.append(S.drive_table(pd.read_parquet(p, columns=S.PBP_COLS)))
    T = pd.concat(frames, ignore_index=True)
    T.to_parquet(cache, index=False)
    return T


def features():
    g = pd.read_csv(os.path.join(HERE, "data", "games.csv"))
    g = g[g.season >= 2017].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(E.REN)
    g["date"] = pd.to_datetime(g.gameday)
    g = g.sort_values(["date", "game_id"]).reset_index(drop=True)
    Tm = pd.read_parquet(os.path.join(HERE, "data", "team_game_epa.parquet"))
    Tm["posteam"] = Tm.posteam.replace(E.REN)
    tmap = {(r.game_id, r.posteam): r for r in Tm.itertuples()}
    Q = pd.read_parquet(os.path.join(HERE, "data", "qb_game_epa.parquet"))
    qmap = defaultdict(list)
    for r in Q.itertuples(): qmap[r.game_id].append(r)
    f = E.add_diffs(E.build(g, tmap, qmap, CFG))
    return f.merge(g[["game_id", "home_score", "away_score", "total_line", "spread_line"]], on="game_id", how="left")


def heads(f, hi, test):
    """Winner, spread and total heads trained on 2018..hi, priced on season `test`."""
    reg = f.gtype == "REG"
    tr = (f.season >= 2018) & (f.season <= hi) & f.margin.notna()
    te = (f.season == test) & f.margin.notna() & reg
    trw = tr & (~f.tie)
    sc = StandardScaler().fit(f.loc[trw, E.W].values)
    mdl = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(f.loc[trw, E.W].values), f.loc[trw, "y"].values)
    sct = StandardScaler().fit(f.loc[tr, E.TT].values)
    mt = Ridge(alpha=5.0).fit(sct.transform(f.loc[tr, E.TT].values), f.loc[tr, "total"].values)
    SH = spread.fit(f, E.W, tr & (f.margin != 0), C)
    rows = f.loc[te].copy()
    ph = mdl.predict_proba(sc.transform(rows[E.W].values))[:, 1]
    ph, Ph, Pa = spread.price(SH, rows, ph=ph)
    rows["p_home"] = ph; rows["sp_margin"] = spread.expected_margin(ph, Ph, Pa)
    rows["model_total"] = mt.predict(sct.transform(rows[E.TT].values))
    trn = f.loc[tr].copy()
    trn["model_total"] = mt.predict(sct.transform(trn[E.TT].values))
    trn["p_home"] = mdl.predict_proba(sc.transform(trn[E.W].values))[:, 1]
    _, tPh, tPa = spread.price(SH, trn, ph=trn.p_home.values)
    trn["sp_margin"] = spread.expected_margin(trn.p_home.values, tPh, tPa)
    return rows, Ph, Pa, trn


def crps(xs, w, actual, grid):
    """Ranked probability score on an integer grid: sum over thresholds of (F - step)^2."""
    cdf = np.array([w[xs <= k].sum() for k in grid])
    return float(((cdf - (actual <= grid)) ** 2).sum())


def crps_floor(center, resid, actual, grid):
    cdf = np.array([(center + resid <= k + 0.5).mean() for k in grid])
    return float(((cdf - (actual <= grid)) ** 2).sum())


def reliability(p, y, edges=(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001)):
    p, y = np.asarray(p), np.asarray(y); rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (p >= lo) & (p < hi)
        if k.sum():
            rows.append((f"{lo:.1f}-{min(hi,1):.1f}", int(k.sum()), float(p[k].mean()), float(y[k].mean())))
    return pd.DataFrame(rows, columns=["bucket", "n", "stated", "actual"])


def logloss(p, y):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4); y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def run_totals(T, f, n_sims, variants):
    GT = np.arange(0, 121); GS = np.arange(0, 71)
    out = []; rel = defaultdict(list); keep = []
    for test in (2023, 2024, 2025):
        rows, Ph, Pa, trn = heads(f, test - 1, test)
        res_tot = (trn.total - trn.model_total).values
        mu_h_tr = (trn.model_total + trn.sp_margin) / 2
        hs = trn.game_id.map(f.set_index("game_id").home_score); as_ = trn.game_id.map(f.set_index("game_id").away_score)
        res_team = np.concatenate([(hs - mu_h_tr).values, (as_ - (trn.model_total - mu_h_tr)).values])
        res_team = res_team[~np.isnan(res_team)]
        for name, (pw, mw, rk) in variants.items():
            S.K["pace_weight"], S.K["mix_weight"] = pw, mw
            acc = defaultdict(list)
            for i, r in enumerate(rows.itertuples()):
                st, lg_d, lg_m = states_cache(T, r.season, r.week)
                d, mh, ma = S.game_shape(st, lg_d, lg_m, r.home, r.away)
                hp, ap, _, _ = S.simulate_scores((r.model_total + r.sp_margin) / 2, (r.model_total - r.sp_margin) / 2,
                                                 mh, ma, d, r.p_home, n_sims, seed=7 + i)
                if rk:
                    w, ess = S.rake(hp, ap, S.margin_target(r.p_home, Ph[i], Pa[i]), total_mean=r.model_total,
                                    total_sd=float(np.std(res_tot)) if rk == 2 else None)
                else:
                    w, ess = np.ones(n_sims) / n_sims, n_sims
                tot = hp + ap; a_t = r.total
                acc["crps_total"].append(crps(tot, w, a_t, GT))
                acc["crps_total_floor"].append(crps_floor(r.model_total, res_tot, a_t, GT))
                acc["crps_team"].append(0.5 * (crps(hp, w, r.home_score, GS) + crps(ap, w, r.away_score, GS)))
                acc["crps_team_floor"].append(0.5 * (crps_floor((r.model_total + r.sp_margin) / 2, res_team, r.home_score, GS)
                                                     + crps_floor((r.model_total - r.sp_margin) / 2, res_team, r.away_score, GS)))
                acc["ess"].append(ess)
                acc["m3"].append(w[np.abs(hp - ap) == 3].sum()); acc["a3"].append(abs(r.margin) == 3)
                acc["sim_sd"].append(np.sqrt((w * (tot - (w * tot).sum()) ** 2).sum()))
                acc["corr"].append(np.cov(np.vstack([hp, ap]), aweights=w)[0, 1] / np.sqrt(np.cov(hp, aweights=w) * np.cov(ap, aweights=w)))
                # ladder probabilities: lines at model total -7, -3.5, 0, +3.5, +7 (rounded to .5) and the closing line
                for off in (-7, -3.5, 0, 3.5, 7):
                    L = np.floor(r.model_total + off) + 0.5
                    rel[(name, test, "total")].append((w[tot > L].sum(), float(a_t > L), float(((r.model_total + res_tot) > L).mean())))
                for arr, act, mu in ((hp, r.home_score, (r.model_total + r.sp_margin) / 2), (ap, r.away_score, (r.model_total - r.sp_margin) / 2)):
                    for off in (-7, -3.5, 0, 3.5, 7):
                        L = np.floor(mu + off) + 0.5
                        rel[(name, test, "team")].append((w[arr > L].sum(), float(act > L), float(((mu + res_team) > L).mean())))
                if r.total_line == r.total_line and a_t != r.total_line:
                    po, pp = S.over_push(tot, w, r.total_line)
                    rel[(name, test, "close")].append((po / max(1 - pp, 1e-9), float(a_t > r.total_line), 0.5))
                for k in (37, 40, 41, 43, 44, 45, 47, 48, 51):
                    rel[(name, test, "exact")].append((w[tot == k].sum(), float(a_t == k), float((np.rint(r.model_total + res_tot) == k).mean())))
            out.append(dict(variant=name, season=test, n=len(rows),
                            crps_total=np.mean(acc["crps_total"]), floor_total=np.mean(acc["crps_total_floor"]),
                            crps_team=np.mean(acc["crps_team"]), floor_team=np.mean(acc["crps_team_floor"]),
                            ess=np.mean(acc["ess"]), sim_on3=np.mean(acc["m3"]), real_on3=np.mean(acc["a3"]),
                            sim_total_sd=np.mean(acc["sim_sd"]), real_total_sd=float((rows.total - rows.model_total).std()),
                            score_corr=np.mean(acc["corr"])))
    R = pd.DataFrame(out)
    pd.set_option("display.width", 220); pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(R.to_string(index=False))
    return R, rel


_SC = {}
def states_cache(T, season, week):
    if (season, week) not in _SC:
        _SC[(season, week)] = S.team_states(T, season, week)
    return _SC[(season, week)]


def fit_constants(T, pbp_dir):
    t = T[(T.season >= 2016) & (T.season <= 2023)]
    g = pd.read_csv(os.path.join(HERE, "data", "games.csv"))
    g = g[(g.game_type == "REG")]
    sc = pd.concat([g[["game_id", "home_team", "home_score"]].set_axis(["game_id", "team", "pts"], axis=1),
                    g[["game_id", "away_team", "away_score"]].set_axis(["game_id", "team", "pts"], axis=1)])
    sc["team"] = sc.team.replace(S.TEAM_NORM)
    t = t.merge(sc, left_on=["game_id", "posteam"], right_on=["game_id", "team"], how="left")
    w = t.pivot_table(index="game_id", columns=t.groupby("game_id").cumcount(), values="drives")
    ev = dict(xp_att=0, xp_good=0, tp_att=0, tp_good=0)
    for y in range(2016, 2024):
        d = pd.read_parquet(os.path.join(pbp_dir, f"pbp_{y}.parquet") if os.path.exists(os.path.join(pbp_dir, f"pbp_{y}.parquet"))
                            else os.path.join(pbp_dir, f"play_by_play_{y}.parquet"),
                            columns=["season_type", "extra_point_result", "two_point_conv_result"])
        d = d[d.season_type == "REG"]
        ev["xp_att"] += d.extra_point_result.notna().sum(); ev["xp_good"] += (d.extra_point_result == "good").sum()
        ev["tp_att"] += d.two_point_conv_result.notna().sum(); ev["tp_good"] += (d.two_point_conv_result == "success").sum()
    p_two = ev["tp_att"] / (ev["tp_att"] + ev["xp_att"]); p_xp = ev["xp_good"] / ev["xp_att"]; p_tg = ev["tp_good"] / ev["tp_att"]
    vtd = 6 + (1 - p_two) * p_xp + p_two * p_tg * 2
    print(f"lg_drives={t.drives.mean():.2f} drive_sd={t.drives.std():.2f} lg_mix={t.td.sum()/(t.td.sum()+t.fg.sum()):.3f}")
    print(f"p_two={p_two:.3f} p_xp={p_xp:.3f} p_two_good={p_tg:.3f} points_per_td={vtd:.3f}")
    print(f"other_pts={(t.pts - 3*t.fg - vtd*t.td).mean():.2f}")
    diff = (w[1] - w[0]).abs()
    print(f"drive diff: same {np.mean(diff==0):.2f}, one {np.mean(diff==1):.2f}, two or more {np.mean(diff>=2):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbp", required=True); ap.add_argument("--fit", action="store_true")
    ap.add_argument("--players", action="store_true"); ap.add_argument("--sims", type=int, default=8000)
    ap.add_argument("--stats", help="folder of nflverse stats_player_week_YYYY.parquet files")
    a = ap.parse_args()
    cache = os.path.join(HERE, "data", "team_game_drives.parquet")
    T = pd.read_parquet(cache) if os.path.exists(cache) else load_pbp_drives(a.pbp)
    if a.fit:
        fit_constants(T, a.pbp)
    elif a.players:
        import sim_backtest_players as P
        P.main(T, features(), a.sims, a.stats)
    else:
        f = features()
        R, rel = run_totals(T, f, a.sims, {"raw drive sim": (0, 0, 0), "raked margin+mean": (0, 0, 1), "raked +spread": (0, 0, 2),
                                           "raked +spread, team": (1, 1, 2)})
        for kind in ("total", "team", "close", "exact"):
            print(f"\n== {kind}: log loss sim vs floor, and reliability (2024-25 pooled; 2023 was the tuning year) ==")
            for v in ("raw drive sim", "raked margin+mean", "raked +spread", "raked +spread, team"):
                x = np.array(rel[(v, 2024, kind)] + rel[(v, 2025, kind)])
                print(f"{v:22s} n={len(x):5d} sim LL {logloss(x[:,0], x[:,1]):.4f} | floor LL {logloss(x[:,2], x[:,1]):.4f}")
            x = np.array(rel[("raked +spread", 2024, kind)] + rel[("raked +spread", 2025, kind)])
            if kind != "exact":
                print(reliability(x[:, 0], x[:, 1]).to_string(index=False))
