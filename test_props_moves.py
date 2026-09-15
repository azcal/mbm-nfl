"""Regression for the Wan'Dale Robinson case: adding or removing a teammate from the GRID (the
display floor) must not move anyone else's projection. Only a change to the active set
(status or roster) may, and that shows up as redistribution_delta. Run: python tests/test_props_moves.py
Needs the backtest data under props_model.DATA (2019-2025)."""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import props_model as M
spec = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "props_spec.json")))
st = pd.read_parquet(f"{M.DATA}/stats_player_week_2025.parquet"); wk = st[(st.season_type == "REG") & (st.week == 10)].drop_duplicates("team")[["season", "week", "team", "opponent_team"]]
orig = M._read
def trimmed(kind, y, live):
    d = orig(kind, y, live)
    return d[d.week < 10] if (d is not None and y == 2025 and kind in ("stats", "snap_counts")) else d
M._read = trimmed
outs = []
for floor in (2.5, 0.0):
    M.RNG = np.random.default_rng(3)
    d = M.run(range(2019, 2026), kappa=spec["kappa"], alpha=spec["alpha"], H=spec["H"], upcoming=wk, vol_mult=spec["volume_mult"], min_proj_targets=floor)
    outs.append(d[(d.season == 2025) & (d.week == 10)].set_index("pid"))
a, b = outs
common = a.index.intersection(b.index)
diff = (a.loc[common, "final_share"] - b.loc[common, "final_share"]).abs().max()
assert diff < 1e-9, f"display floor changed a projection: max share diff {diff}"
assert len(b) > len(a), "floor 0 should show more players"
print(f"PASS: {len(common)} shared players identical across display floors 2.5 and 0.0 ({len(b) - len(a)} extra rows shown at floor 0); redistribution only moves with the active set")
