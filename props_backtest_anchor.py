"""
props_backtest_anchor.py: Phase 1 check. Anchor a week's RAW grid to a closing-line fixture and
grade raw vs anchored medians against actuals.

    python props_backtest_anchor.py tests/fixtures/props_grid_w1_graded.csv tests/fixtures/w1_closing_consensus_lines.csv

graded fixture: the week's raw grid rows joined to actuals (columns act_targets, act_receptions,
act_yards; any of act_yds / receiving_yards accepted). lines fixture: player, rec_yds_close_line
(optionally rec_yds_close_p_over; 0.5 assumed when absent, i.e. the line is the market median).

Reports: MAE of raw median, anchored median and the line vs actual yards; PIT deciles for raw
and anchored (flat is the target); anchored P(over line+15) and P(over line+30) vs realised.
Appends both medians' MAE to logs/props_anchor_results.csv so the series accumulates weekly.
"""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import props_model as M, props_market as PM

graded_path, lines_path = sys.argv[1], sys.argv[2]
g = pd.read_csv(graded_path, low_memory=False)
L = pd.read_csv(lines_path)
spec = json.load(open(os.path.join(HERE, "props_spec.json")))
acol = next(c for c in ("act_yards", "act_yds", "receiving_yards") if c in g.columns)
L["key"] = L.player.map(PM.norm_name); g["key"] = g.name.map(PM.norm_name)
pcol = "rec_yds_close_p_over" if "rec_yds_close_p_over" in L.columns else None
cons = {(r.pid, "rec_yds"): dict(line=float(lk), p_over=float(pk) if pcol else 0.5, n_books=1)
        for r in g.itertuples() for lk, pk in [(L.loc[L.key == r.key, "rec_yds_close_line"].iloc[0] if (L.key == r.key).any() else np.nan,
                                              L.loc[L.key == r.key, pcol].iloc[0] if pcol and (L.key == r.key).any() else 0.5)] if lk == lk}
ycols = [f"p_over_{int(x)}" for x in M.YGRID]
raw = g.copy()
anch = PM.anchor_grid(raw, cons, spec)
lined = anch[anch.anchored].copy()
y = lined[acol].astype(float).values
mae_raw = np.mean(np.abs(raw.loc[lined.index, "med_yds"].values - y)); mae_anc = np.mean(np.abs(lined.med_yds.values - y)); mae_line = np.mean(np.abs(lined.market_line.values - y))
print(f"lined players: {len(lined)} of {len(g)}")
print(f"MAE vs actual yards: raw median {mae_raw:.1f} | anchored median {mae_anc:.1f} | line {mae_line:.1f}")
def pit(df):
    out = []
    for r, a in zip(df[ycols].values, df[acol].astype(float).values):
        out.append(1 - float(np.interp(a, M.YGRID, r)))
    return np.array(out)
for lab, d in (("raw", raw.loc[lined.index]), ("anchored", lined)):
    p = pit(d); hist = np.histogram(p, bins=np.linspace(0, 1, 11))[0]
    print(f"PIT deciles {lab:<9}: {' '.join(f'{h:>3}' for h in hist)}   (flat target ~{len(p)/10:.0f} each; over-median {np.mean(p < .5):.2f})")
for off in (15, 30):
    ps = np.array([float(np.interp(r.market_line + off, M.YGRID, r[ycols].values.astype(float))) for _, r in lined.iterrows()])
    hit = (y > lined.market_line.values + off).mean()
    print(f"anchored P(over line+{off}): model {ps.mean():.3f} vs realised {hit:.3f}")
gap = raw.loc[lined.index, "med_yds"].values - lined.market_line.values
tiers = pd.cut(lined.market_line.values, [0, 20, 35, 50, 999], labels=["<20", "20-35", "35-50", "50+"])
print("raw median minus line by line tier:", pd.Series(gap).groupby(tiers).mean().round(1).to_dict())
os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
rp = os.path.join(HERE, "logs", "props_anchor_results.csv")
row = pd.DataFrame([dict(season=int(g.season.iloc[0]), week=int(g.week.iloc[0]), n=len(lined), mae_raw=round(mae_raw, 2), mae_anchored=round(mae_anc, 2), mae_line=round(mae_line, 2))])
row.to_csv(rp, mode="a", index=False, header=not os.path.exists(rp)); print(f"appended -> logs/props_anchor_results.csv")
