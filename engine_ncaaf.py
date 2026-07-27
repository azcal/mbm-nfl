"""
MBM NCAAF rebuild v1 -- standalone, no market inputs.
Winner head P(home) on FBS-vs-FBS games; margin and total reference heads (MAE floors).
Data: cfbfastR schedules (CFBD-sourced), 2016 burn-in, 2018-2025 modeled.
FCS/lower opponents update states (low Elo init) but are excluded from evaluation.
Protocol: tune on 2023 only; 2024 and 2025 frozen holdouts (expanding train).
Extra floor: CFBD's own pregame Elo where present.
"""
import numpy as np, pandas as pd, json, math
from collections import defaultdict
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss

P4 = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12", "FBS Independents"}

# start_date from CFBD/ESPN is UTC. dt.tz_localize(None) STRIPS the zone rather than
# converting it, which rolls every 7-11pm kickoff into the next calendar day. That was
# 22.9% of games 2018-2025, and it split real Saturdays into a Saturday bucket and a
# phantom Sunday bucket for anything that groups by date. Convert, then drop the zone.
TZ = "America/New_York"

def game_day(series):
    return pd.to_datetime(series, utc=True).dt.tz_convert(TZ).dt.tz_localize(None)


def tier(conf, team):
    if team == "Notre Dame": return 2
    if conf in ("SEC", "Big Ten"): return 2
    if conf in ("Big 12", "ACC", "Pac-12"): return 2
    return 1 if conf else 0

class Decay:
    def __init__(self, hl): self.d = 0.5**(1.0/hl); self.sw = 0.0; self.sx = 0.0
    def add(self, x): self.sw = self.sw*self.d + 1; self.sx = self.sx*self.d + x
    def brk(self, keep): self.sw *= keep; self.sx *= keep
    def shrunk(self, tau=3.0): return self.sx/(self.sw + tau)

def load():
    rows = []
    for y in range(2016, 2026):
        s = pd.read_parquet(f"ncaaf/sched_{y}.parquet")
        s = s[s.completed == True]
        s = s[s.home_points.notna() & s.away_points.notna()]
        s = s[(s.home_division == "fbs") | (s.away_division == "fbs")]
        rows.append(s)
    g = pd.concat(rows, ignore_index=True)
    g["date"] = game_day(g.start_date)
    g = g.sort_values(["date", "game_id"]).reset_index(drop=True)
    g["margin"] = g.home_points - g.away_points
    g["total"] = g.home_points + g.away_points
    g["fbs_both"] = (g.home_division == "fbs") & (g.away_division == "fbs")
    return g

def build(g, cfg):
    K, HA, CARRY, HLF, HLS = cfg["K"], cfg["HA"], cfg["carry"], cfg["hl_f"], cfg["hl_s"]
    elo = {}
    off_f = defaultdict(lambda: Decay(HLF)); off_s = defaultdict(lambda: Decay(HLS))
    deff = defaultdict(lambda: Decay(HLS))
    season_of = {}; gp = defaultdict(int); prev = {}
    lg = Decay(1500)
    feats = []
    for r in g.itertuples():
        for tm, div in ((r.home_team, r.home_division), (r.away_team, r.away_division)):
            if tm not in elo:
                elo[tm] = 1500.0 if div == "fbs" else 1150.0
            if season_of.get(tm) != r.season:
                elo[tm] = 1500 + CARRY*(elo[tm]-1500)
                for D in (off_f[tm], off_s[tm], deff[tm]): D.brk(CARRY)
                gp[tm] = 0; prev.pop(tm, None)
                season_of[tm] = r.season
        lgm = lg.sx/max(lg.sw, 1)
        row = {"game_id": r.game_id, "season": r.season, "week": r.week, "date": r.date,
               "home": r.home_team, "away": r.away_team, "margin": r.margin, "total": r.total,
               "neutral": 1 if r.neutral_site else 0, "fbs_both": r.fbs_both,
               "conf_game": 1 if r.conference_game else 0,
               "cfbd_h_elo": getattr(r, "home_pregame_elo", np.nan),
               "cfbd_a_elo": getattr(r, "away_pregame_elo", np.nan),
               "tier_d": tier(r.home_conference, r.home_team) - tier(r.away_conference, r.away_team)}
        for side, tm in (("h", r.home_team), ("a", r.away_team)):
            pv = prev.get(tm)
            rest = min((r.date - pv).days, 21) if pv is not None else 10
            row.update({f"{side}_elo": elo[tm], f"{side}_off_f": off_f[tm].shrunk(),
                        f"{side}_off_s": off_s[tm].shrunk(), f"{side}_def": deff[tm].shrunk(),
                        f"{side}_rest": rest, f"{side}_gp": min(gp[tm], 8)})
        feats.append(row)
        if r.margin != r.margin:
            continue
        # ---- post-game updates ----
        for tm, pf, pa, opp in ((r.home_team, r.home_points, r.away_points, r.away_team),
                                (r.away_team, r.away_points, r.home_points, r.home_team)):
            adj_o = pf - (lgm - deff[opp].shrunk())
            adj_d = pa - (lgm + off_s[opp].shrunk())
            off_f[tm].add(adj_o); off_s[tm].add(adj_o); deff[tm].add(-adj_d)
            gp[tm] += 1; prev[tm] = r.date
        lg.add(r.home_points); lg.add(r.away_points)
        ha = 0 if r.neutral_site else HA
        e = 1/(1+10**(-((elo[r.home_team]+ha)-elo[r.away_team])/400))
        res = 1.0 if r.margin > 0 else 0.0
        mov = math.log(abs(r.margin)+1) * 2.2/((abs(elo[r.home_team]-elo[r.away_team]))*0.001+2.2)
        d = K * mov * (res - e)
        elo[r.home_team] += d; elo[r.away_team] -= d
    return pd.DataFrame(feats)

W = ["elo_d","off_f_d","off_s_d","def_d","tier_d","rest_d","neutral","conf_game","h_gp","a_gp"]
MG = W
TT = ["h_off_s","a_off_s","h_def","a_def","tier_d","neutral"]

def add_diffs(f):
    f = f.copy()
    f["elo_d"] = f.h_elo - f.a_elo
    f["off_f_d"] = f.h_off_f - f.a_off_f; f["off_s_d"] = f.h_off_s - f.a_off_s
    f["def_d"] = f.h_def - f.a_def
    f["rest_d"] = f.h_rest - f.a_rest
    f["y"] = (f.margin > 0).astype(int)
    return f

def fit_winner(f, tr, te, C):
    sc = StandardScaler().fit(f.loc[tr, W].values)
    m = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(f.loc[tr, W].values), f.loc[tr, "y"].values)
    return m.predict_proba(sc.transform(f.loc[te, W].values))[:, 1], f.loc[te, "y"].values, f.loc[te]

def M(y, p, lbl):
    p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6)
    return dict(model=lbl, n=len(y), log_loss=round(float(log_loss(y, p)), 4),
                brier=round(float(brier_score_loss(y, p)), 4), hit=round(float(np.mean((p > .5) == (y == 1))), 4))

def run():
    g = load()
    print("games loaded:", len(g), "| FBS-vs-FBS:", int(g.fbs_both.sum()))
    best = None
    for K in (28, 38):
        for carry in (0.75, 0.9):
            for HA in (45, 65):
                cfg = dict(K=K, HA=HA, carry=carry, hl_f=4, hl_s=9)
                f = add_diffs(build(g, cfg))
                tr = (f.season >= 2018) & (f.season <= 2022) & f.fbs_both
                va = (f.season == 2023) & f.fbs_both
                for C in (0.05, 0.2, 1.0):
                    p, y, _ = fit_winner(f, tr, va, C)
                    ll = log_loss(y, np.clip(p, 1e-6, 1-1e-6))
                    if best is None or ll < best[0]: best = (ll, cfg, C, f)
    ll23, cfg, C, f = best
    print(f"tuned on 2023: LL={ll23:.4f}  K={cfg['K']} HA={cfg['HA']} carry={cfg['carry']} C={C}")
    json.dump({"cfg": cfg, "C": C}, open("ncaaf/spec.json", "w"))
    f.to_parquet("ncaaf/features_best.parquet")
    out, preds = [], []
    for test, hi in ((2024, 2023), (2025, 2024)):
        tr = (f.season >= 2018) & (f.season <= hi) & f.fbs_both
        te = (f.season == test) & f.fbs_both
        p, y, rows = fit_winner(f, tr, te, C)
        preds.append(rows.assign(p_home=p))
        base = f.loc[tr, "y"].mean()
        ha_v = np.where(rows.neutral == 1, 0, cfg["HA"])
        e = 1/(1+10**(-((rows.h_elo + ha_v) - rows.a_elo)/400))
        res = [M(y, np.full_like(y, 0.5, float), "coin flip"),
               M(y, np.full_like(y, base, float), f"base rate ({base:.3f})"),
               M(y, e, "own Elo (floor)"),
               {"model": "always Elo favorite", "n": len(y), "log_loss": None, "brier": None,
                "hit": round(float(np.mean((e > .5).astype(int) == y)), 4)}]
        m = rows.cfbd_h_elo.notna() & rows.cfbd_a_elo.notna()
        if m.mean() > 0.8:
            ec = 1/(1+10**(-((rows.loc[m, "cfbd_h_elo"] + np.where(rows.loc[m, "neutral"]==1, 0, 55)) - rows.loc[m, "cfbd_a_elo"])/400))
            res.append(M(y[m.values], ec, "CFBD pregame Elo (ref)"))
        res.append(M(y, p, "REBUILD v1"))
        R = pd.DataFrame(res); R.insert(0, "holdout", test); out.append(R)
        trm = f.loc[tr]
        scm = StandardScaler().fit(trm[MG].values)
        mm = Ridge(alpha=5.0).fit(scm.transform(trm[MG].values), trm.margin.values)
        pm = mm.predict(scm.transform(rows[MG].values))
        sct = StandardScaler().fit(trm[TT].values)
        mt = Ridge(alpha=5.0).fit(sct.transform(trm[TT].values), trm.total.values)
        pt = mt.predict(sct.transform(rows[TT].values))
        elo_m = ((rows.h_elo + ha_v) - rows.a_elo).values/28.0
        print(f"\n{test} margin MAE: model {np.mean(np.abs(pm-rows.margin)):.2f} | Elo {np.mean(np.abs(elo_m-rows.margin)):.2f} | constant {np.mean(np.abs(trm.margin.mean()-rows.margin)):.2f}")
        print(f"{test} total  MAE: model {np.mean(np.abs(pt-rows.total)):.2f} | constant {np.mean(np.abs(trm.total.mean()-rows.total)):.2f}")
    allr = pd.concat(out, ignore_index=True)
    allr.to_csv("ncaaf/comparison.csv", index=False)
    pd.concat(preds).to_csv("ncaaf/holdout_predictions.csv", index=False)
    pd.set_option("display.width", 160)
    print("\n" + allr.to_string(index=False))

if __name__ == "__main__":
    run()
