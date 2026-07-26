"""
MBM NFL rebuild v1 -- standalone, no market inputs.
Winner head: P(home win), ties (~1/yr) excluded as pushes. Margin and total heads as
reference numbers, validated on MAE vs floors. Data: nflverse games + pbp EPA aggregates.
Protocol: burn-in 2017; tune on 2023 only; 2024 and 2025 frozen holdouts (expanding train).
QB layer: rolling EPA/dropback per starter (pregame-knowable via listed starters).
"""
import numpy as np, pandas as pd, json
from collections import defaultdict, deque
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss

REN = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

class Decay:
    def __init__(self, hl): self.d = 0.5**(1.0/hl); self.sw = 0.0; self.sx = 0.0
    def add(self, x): self.sw = self.sw*self.d + 1; self.sx = self.sx*self.d + x
    def brk(self, keep): self.sw *= keep; self.sx *= keep
    def shrunk(self, tau=3.0): return self.sx/(self.sw + tau)

def load():
    g = pd.read_csv("nfl/games.csv")
    g = g[(g.season >= 2017) & g.result.notna()].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(REN)
    g["date"] = pd.to_datetime(g.gameday)
    g = g.sort_values(["date", "game_id"]).reset_index(drop=True)
    T = pd.read_parquet("nfl/team_game_epa.parquet")
    T["posteam"] = T.posteam.replace(REN)
    tmap = {(r.game_id, r.posteam): r for r in T.itertuples()}
    Q = pd.read_parquet("nfl/qb_game_epa.parquet")
    qmap = defaultdict(list)
    for r in Q.itertuples(): qmap[r.game_id].append(r)
    return g, tmap, qmap

def build(g, tmap, qmap, cfg):
    K, HA, CARRY, HLF, HLS, QTAU, QCARRY = (cfg["K"], cfg["HA"], cfg["carry"], cfg["hl_f"],
                                            cfg["hl_s"], cfg["q_tau"], cfg["q_carry"])
    elo = defaultdict(lambda: 1500.0)
    off_f = defaultdict(lambda: Decay(HLF)); off_s = defaultdict(lambda: Decay(HLS))
    deff = defaultdict(lambda: Decay(HLS))
    qb = {}                                     # qb_id -> {"sw","sx","season"}
    recent_qb = defaultdict(lambda: deque(maxlen=4))
    season_of = {}; gp = defaultdict(int)
    lg_off = Decay(300)
    feats = []
    for r in g.itertuples():
        for tm in (r.home_team, r.away_team):
            if season_of.get(tm) != r.season:
                elo[tm] = 1500 + CARRY*(elo[tm]-1500)
                for D in (off_f[tm], off_s[tm], deff[tm]): D.brk(CARRY)
                gp[tm] = 0
                season_of[tm] = r.season
        lgm = lg_off.sx/max(lg_off.sw, 1)
        def qrate(qid, team):
            if qid != qid or qid is None or qid not in qb:
                return 0.0, 0.0, 1
            s = qb[qid]
            usual = max(set(recent_qb[team]) or {None},
                        key=lambda x: list(recent_qb[team]).count(x)) if recent_qb[team] else None
            backup = 1 if (usual is not None and qid != usual) else 0
            return s["sx"]/(s["sw"] + QTAU), min(s["sw"], 400.0), backup
        row = {"game_id": r.game_id, "season": r.season, "week": r.week, "date": r.date,
               "home": r.home_team, "away": r.away_team, "margin": r.result,
               "total": r.total, "div": r.div_game,
               "dome": 1 if str(r.roof) in ("dome", "closed") else 0,
               "gtype": r.game_type}
        for side, tm, qid, rest in (("h", r.home_team, r.home_qb_id, r.home_rest),
                                    ("a", r.away_team, r.away_qb_id, r.away_rest)):
            qr, qexp, qb_backup = qrate(qid, tm)
            row.update({f"{side}_elo": elo[tm],
                        f"{side}_off_f": off_f[tm].shrunk(), f"{side}_off_s": off_s[tm].shrunk(),
                        f"{side}_def": deff[tm].shrunk(),
                        f"{side}_qb": qr, f"{side}_qexp": qexp/400.0, f"{side}_bkp": qb_backup,
                        f"{side}_rest": min(rest, 14) if rest == rest else 7,
                        f"{side}_short": 1 if (rest == rest and rest <= 5) else 0,
                        f"{side}_gp": min(gp[tm], 8)})
        feats.append(row)
        # ---- post-game updates ----
        hb, ab = tmap.get((r.game_id, r.home_team)), tmap.get((r.game_id, r.away_team))
        if hb is None or ab is None: continue
        for tm, mine, opp in ((r.home_team, hb, ab), (r.away_team, ab, hb)):
            oe = mine.epa_sum/max(mine.plays, 1)
            adj_o = oe - (deff[opp.posteam].shrunk() - 0)      # def state is centered
            off_f[tm].add(adj_o - lgm); off_s[tm].add(adj_o - lgm)
            de = opp.epa_sum/max(opp.plays, 1)
            adj_d = de - (off_s[opp.posteam].shrunk())
            deff[tm].add(adj_d - lgm)
            lg_off.add(oe)
            gp[tm] += 1
        for tm, qid in ((r.home_team, r.home_qb_id), (r.away_team, r.away_qb_id)):
            if qid == qid and qid is not None: recent_qb[tm].append(qid)
        for q in qmap.get(r.game_id, []):
            s = qb.setdefault(q.passer_player_id, {"sw": 0.0, "sx": 0.0, "season": r.season})
            if s["season"] != r.season:
                s["sw"] *= QCARRY; s["sx"] *= QCARRY; s["season"] = r.season
            d = 0.5 ** (q.db/120.0)              # ~120-dropback half-life
            s["sw"] = s["sw"]*d + q.db; s["sx"] = s["sx"]*d + q.qb_epa_sum
        res = 1.0 if r.result > 0 else 0.0 if r.result < 0 else 0.5
        e = 1/(1+10**(-((elo[r.home_team]+HA)-elo[r.away_team])/400))
        k = K * (1.4 if r.game_type != "REG" else 1.0)
        elo[r.home_team] += k*(res-e); elo[r.away_team] -= k*(res-e)
    return pd.DataFrame(feats)

W = ["elo_d","off_f_d","off_s_d","def_d","qb_d","h_bkp","a_bkp","h_qexp","a_qexp",
     "rest_d","h_short","a_short","div","h_gp","a_gp"]
MG = W + ["dome"]
TT = ["h_off_s","a_off_s","h_def","a_def","dome","h_qb","a_qb","div"]

def add_diffs(f):
    f = f.copy()
    f["elo_d"] = f.h_elo - f.a_elo
    f["off_f_d"] = f.h_off_f - f.a_off_f; f["off_s_d"] = f.h_off_s - f.a_off_s
    f["def_d"] = f.a_def - f.h_def
    f["qb_d"] = f.h_qb - f.a_qb; f["rest_d"] = f.h_rest - f.a_rest
    f["y"] = (f.margin > 0).astype(int); f["tie"] = f.margin == 0
    return f

def fit_winner(f, tr, te, C):
    trm = tr & (~f.tie); tem = te & (~f.tie)
    sc = StandardScaler().fit(f.loc[trm, W].values)
    m = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(f.loc[trm, W].values), f.loc[trm, "y"].values)
    return m.predict_proba(sc.transform(f.loc[tem, W].values))[:, 1], f.loc[tem, "y"].values, f.loc[tem], (m, sc)

def M(y, p, lbl):
    p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6)
    return dict(model=lbl, n=len(y), log_loss=round(float(log_loss(y, p)), 4),
                brier=round(float(brier_score_loss(y, p)), 4), hit=round(float(np.mean((p > .5) == (y == 1))), 4))

def run():
    g, tmap, qmap = load()
    best = None
    for K in (14, 20):
        for carry in (0.6, 0.75):
            for HA in (40, 55):
                cfg = dict(K=K, HA=HA, carry=carry, hl_f=4, hl_s=10, q_tau=120, q_carry=0.8)
                f = add_diffs(build(g, tmap, qmap, cfg))
                tr = (f.season >= 2018) & (f.season <= 2022); va = f.season == 2023
                for C in (0.05, 0.2, 1.0):
                    p, y, _, _ = fit_winner(f, tr, va, C)
                    ll = log_loss(y, np.clip(p, 1e-6, 1-1e-6))
                    if best is None or ll < best[0]: best = (ll, cfg, C, f)
    ll23, cfg, C, f = best
    print(f"tuned on 2023: LL={ll23:.4f}  K={cfg['K']} HA={cfg['HA']} carry={cfg['carry']} C={C}")
    json.dump({"cfg": cfg, "C": C}, open("nfl/spec.json", "w"))
    f.to_parquet("nfl/features_best.parquet")
    out = []
    preds = []
    for test, hi in ((2024, 2023), (2025, 2024)):
        tr = (f.season >= 2018) & (f.season <= hi); te = f.season == test
        p, y, rows, _ = fit_winner(f, tr, te, C)
        preds.append(rows.assign(p_home=p))
        base = f.loc[tr & (~f.tie), "y"].mean()
        e = 1/(1+10**(-((rows.h_elo + cfg["HA"]) - rows.a_elo)/400)).values
        res = [M(y, np.full_like(y, 0.5, float), "coin flip"),
               M(y, np.full_like(y, base, float), f"base rate ({base:.3f})"),
               M(y, e, "Elo (floor)"),
               {"model": "always Elo favorite", "n": len(y), "log_loss": None, "brier": None,
                "hit": round(float(np.mean((e > .5).astype(int) == y)), 4)},
               M(y, p, "REBUILD v1")]
        R = pd.DataFrame(res); R.insert(0, "holdout", test); out.append(R)
        # margin + total reference heads
        trm = f.loc[tr]; tem = rows
        scm = StandardScaler().fit(trm[MG].values)
        mm = Ridge(alpha=5.0).fit(scm.transform(trm[MG].values), trm.margin.values)
        pm = mm.predict(scm.transform(tem[MG].values))
        elo_m = (tem.h_elo + cfg["HA"] - tem.a_elo).values/25.0
        cm = trm.margin.mean()
        sct = StandardScaler().fit(trm[TT].values)
        mt = Ridge(alpha=5.0).fit(sct.transform(trm[TT].values), trm.total.values)
        pt = mt.predict(sct.transform(tem[TT].values))
        ct = trm.total.mean()
        print(f"\n{test} margin MAE: model {np.mean(np.abs(pm-tem.margin)):.2f} | "
              f"Elo {np.mean(np.abs(elo_m-tem.margin)):.2f} | constant {np.mean(np.abs(cm-tem.margin)):.2f}")
        print(f"{test} total  MAE: model {np.mean(np.abs(pt-tem.total)):.2f} | "
              f"constant {np.mean(np.abs(ct-tem.total)):.2f}")
    allr = pd.concat(out, ignore_index=True)
    allr.to_csv("nfl/comparison.csv", index=False)
    pd.concat(preds).to_csv("nfl/holdout_predictions.csv", index=False)
    pd.set_option("display.width", 160)
    print("\n" + allr.to_string(index=False))
    y_all = pd.concat(preds); yy = y_all[~y_all.tie]
    print(f"\ncombined 2024-25: n={len(yy)}  model LL={log_loss(yy.y, np.clip(yy.p_home,1e-6,1-1e-6)):.4f}")

if __name__ == "__main__":
    run()
