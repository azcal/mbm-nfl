"""
sim_backtest_players.py: rushing and passing ladders from sim_nfl.py against real games.
Run through sim_backtest.py --players --stats DIR (nflverse stats_player_week_YYYY.parquet).

Each week of 2023, 2024 and 2025 is projected with props_rushpass.project() from stats before
that week only, then every game is simulated and each player's ladder is scored against what
he did. 2023 is where the noise constants in sim_nfl.K were measured; 2024 and 2025 are the
holdouts.

ONE PIECE OF HINDSIGHT, stated plainly: the roster for a week is the players who recorded a
stat in it, and the starting QB is the passer with the most attempts. So this scores the
ladders GIVEN a player suits up. Historical depth charts and injury reports are not replayed,
which means the live risk of a wrong starter or a surprise inactive is not in these numbers.
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import sim_nfl as S, spread
import props_rushpass as R
import props_weekly as W


def load_stats(stats_dir, seasons):
    fr = [pd.read_parquet(os.path.join(stats_dir, f"stats_player_week_{y}.parquet")) for y in seasons]
    st = pd.concat(fr, ignore_index=True)
    st = st[st.season_type == "REG"].copy()
    st["team"] = st.team.map(W.norm_team); st["opponent_team"] = st.opponent_team.map(W.norm_team)
    for c in ("carries", "rushing_yards", "attempts", "passing_yards", "receiving_yards"):
        st[c] = pd.to_numeric(st[c], errors="coerce").fillna(0.0)
    return st


def project_season(st, season, margins_by_week):
    out = []
    depth = pd.DataFrame(columns=["team", "gsis_id", "pos_abb", "pos_rank"])
    for wk in sorted(st[st.season == season].week.unique()):
        prior = st[(st.season == season - 1) | ((st.season == season) & (st.week < wk))]
        act = st[(st.season == season) & (st.week == wk) & st.position.isin(R.POSITIONS)]
        if not len(act) or not len(prior):
            continue
        top = act[act.position == "QB"].sort_values("attempts", ascending=False).drop_duplicates("team")
        ros = pd.concat([act[act.position != "QB"], top])
        roster = pd.DataFrame(dict(gsis_id=ros.player_id.values, full_name=ros.player_display_name.values,
                                   position=ros.position.values, team=ros.team.values)).drop_duplicates("gsis_id")
        slate = act.drop_duplicates("team")[["team", "opponent_team", "game_id"]].assign(kickoff_utc="", game_status="final")
        p = R.project(prior, roster, depth, {}, slate, season, wk, margins_by_week.get(wk, {}))
        if not len(p):
            continue
        a = act.set_index("player_id")[["carries", "rushing_yards", "attempts", "passing_yards", "game_id"]]
        a.columns = ["a_car", "a_ry", "a_att", "a_py", "game_id"]
        p = p.merge(a, left_on="gsis_id", right_index=True, how="left")
        tg = act.groupby("team").agg(t_car=("carries", "sum"), t_att=("attempts", "sum"))
        p = p.merge(tg, left_on="team", right_index=True, how="left")
        out.append(p)
    return pd.concat(out, ignore_index=True)


def fit_noise(P, G):
    """Noise constants from one season of projections P and its games G (game_id -> margin info)."""
    P = P.copy()
    P["dm"] = [(G[g]["margin"] - G[g]["sp_margin"]) * (1 if G[g]["home"] == t else -1) if g in G else np.nan
               for g, t in zip(P.game_id, P.team)]
    P = P[P.dm.notna()]
    tm = P.drop_duplicates(["game_id", "team"])
    x = tm.dm.clip(-28, 28).values
    rs = np.polyfit(x, tm.t_car / tm.team_carries - 1, 1)[0]
    q = P[P.starter.astype(bool) & P.proj_att.notna()]
    ps = -np.polyfit(q.dm.clip(-28, 28).values, q.a_att / q.proj_att - 1, 1)[0]
    rb = P[(P.proj_carries >= 8) & ~P.starter.astype(bool)]
    mu = rb.proj_carries * (1 + rs * rb.dm.clip(-28, 28))
    kap = float((mu ** 2).mean() / max(((rb.a_car - mu) ** 2 - mu).mean(), 1e-6))
    # carries: (1-p) NB(mu*up, kappa) + p NB(mu*mult, kappa), by maximum likelihood on a grid
    from scipy.stats import nbinom
    best = None
    c = rb.a_car.values.astype(int); m = np.maximum(mu.values, 0.5)
    for bp in (0.0, 0.03, 0.05, 0.08, 0.11, 0.14):
        for bm in (0.2, 0.3, 0.4):
            for kk in (10, 14, 20, 30, 50, 80):
                up = (1 - bp * bm) / (1 - bp)
                pm = lambda mm: nbinom.pmf(c, kk, kk / (kk + mm))
                ll = float(np.log((1 - bp) * pm(m * up) + bp * pm(m * bm) + 1e-12).sum())
                if best is None or ll > best[0]:
                    best = (ll, bp, bm, kk)
    print(f"carries mixture on this season: bust_p={best[1]} bust_mult={best[2]} car_kappa={best[3]} (log lik {best[0]:.1f})")
    r5 = rb[rb.a_car >= 5]
    z = (r5.a_ry - r5.a_car * (r5.mean_rush_yds / r5.proj_carries)) / np.sqrt(r5.a_car)
    mu_a = q.proj_att * (1 - ps * q.dm.clip(-28, 28))
    q10 = q[q.a_att >= 10]
    zp = (q10.a_py - q10.a_att * (q10.mean_pass_yds / q10.proj_att)) / np.sqrt(q10.a_att)
    sk = lambda v: float(((v - v.mean()) ** 3).mean() / v.std() ** 3)
    return dict(rush_slope=round(float(rs), 4), pass_slope=round(float(ps), 4), car_kappa=round(kap, 1),
                rush_sd=round(float(z.std()), 2), rush_skew=round(sk(z), 2), att_sd=round(float((q.a_att - mu_a).std()), 2),
                pass_sd=round(float(zp.std()), 2), pass_skew=round(sk(zp), 2))


def main(T, f, n_sims, stats_dir):
    import sim_backtest as B
    st = load_stats(stats_dir, range(2022, 2026))
    rel = {}; summ = []
    for season in (2023, 2024, 2025):
        rows, Ph, Pa, trn = B.heads(f, season - 1, season)
        sd_tot = float((trn.total - trn.model_total).std())
        G = {}; mbw = {}
        for i, r in enumerate(rows.itertuples()):
            G[r.game_id] = dict(i=i, home=r.home, away=r.away, margin=r.margin, sp_margin=r.sp_margin, p_home=r.p_home,
                                model_total=r.model_total)
            m = max(-R.GS_CAP, min(R.GS_CAP, r.sp_margin))
            mbw.setdefault(int(r.week), {}).update({r.home: m, r.away: -m})
        P = project_season(st, season, mbw)
        P = P[P.game_id.isin(G)]
        if season == 2023:
            print("noise constants measured on 2023:", fit_noise(P, G))
        recs = []
        for gid, grp in P.groupby("game_id"):
            gi = G[gid]; i = gi["i"]
            hp, ap, hd, ad = S.simulate_scores((gi["model_total"] + gi["sp_margin"]) / 2, (gi["model_total"] - gi["sp_margin"]) / 2,
                                               S.K["lg_mix"], S.K["lg_mix"], S.K["lg_drives"], gi["p_home"], n_sims, seed=11 + i)
            w, _ = S.rake(hp, ap, S.margin_target(gi["p_home"], Ph[i], Pa[i]), total_mean=gi["model_total"], total_sd=sd_tot)
            g = dict(hp=hp, ap=ap, w=w)
            for pi, r in enumerate(grp.to_dict("records")):
                side = "home" if r["team"] == gi["home"] else "away"
                dr = S.player_draws(g, r, side, seed=900 + 37 * i + pi)
                if "rush" in dr and not r["starter"] and r["proj_carries"] >= 8:
                    y = dr["rush"]
                    for L in S.RUSH_LINES:
                        recs.append(("rush", L, float(w[y > L].sum()), float(r["a_ry"] > L)))
                    recs.append(("rush_mean", 0, float((w * y).sum()), r["a_ry"]))
                if "pass" in dr and r["starter"]:
                    y = dr["pass"]
                    for L in S.PASS_LINES:
                        recs.append(("pass", L, float(w[y > L].sum()), float(r["a_py"] > L)))
                    recs.append(("pass_mean", 0, float((w * y).sum()), r["a_py"]))
        rel[season] = pd.DataFrame(recs, columns=["stat", "line", "p", "y"])
        d = rel[season]
        for stat in ("rush", "pass"):
            m = d[d.stat == stat + "_mean"]
            summ.append(dict(season=season, stat=stat, players=len(m), mean_proj=m.p.mean(), mean_actual=m.y.mean(),
                             mae=(m.p - m.y).abs().mean(), mae_constant=(m.y - m.y.mean()).abs().mean(),
                             ladder_ll=B.logloss(d[d.stat == stat].p, d[d.stat == stat].y),
                             ladder_ll_base=B.logloss(d[d.stat == stat].groupby("line").y.transform("mean"), d[d.stat == stat].y)))
    pd.set_option("display.width", 200); pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(pd.DataFrame(summ).to_string(index=False))
    hold = pd.concat([rel[2024], rel[2025]])
    for stat in ("rush", "pass"):
        d = hold[hold.stat == stat]
        print(f"\n== {stat} yards ladder reliability, 2024-25 pooled (n = player-game-lines) ==")
        print(B.reliability(d.p, d.y).to_string(index=False))
        by = d.groupby("line").agg(n=("y", "size"), stated=("p", "mean"), actual=("y", "mean"))
        print(by.round(3).T.to_string())
    return rel
