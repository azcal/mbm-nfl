"""
props_validate.py: post-game check of the injury tags. Joins a week's grid to nflverse snap
counts and weekly stats and prints (1) players flagged Out/Doubtful/IR/PUP/Suspended who
played, (2) players with no flag who took zero offensive snaps. Runs every Tuesday in the
props workflow so the injury sources' accuracy is tracked all season.

    python props_validate.py [path/to/props_grid.csv]     default: logs/props_grid.csv
"""
import io, os, re, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import props_model as M

def norm(s):
    """lowercase letters only, generational suffix dropped: snap counts list 'Chris Godwin'
    where the roster says 'Chris Godwin Jr.'. Works on already-collapsed grid names too."""
    s = re.sub(r"[^a-z]", "", str(s).lower())
    return re.sub(r"(jr|sr|iii|ii|iv)$", "", s)
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "props_grid.csv")
g = pd.read_csv(path, low_memory=False)
g["inj"] = g.inj.fillna("").astype(str)
season, week = int(g.season.iloc[0]), int(g.week.iloc[0])
r = M._get(f"{M.NV}/snap_counts/snap_counts_{season}.csv")
sc = pd.read_csv(io.StringIO(r.text)) if r is not None and r.status_code == 200 else pd.DataFrame()
sc = sc[(sc.week == week) & (sc.game_type == "REG")] if len(sc) else sc
r = M._get(f"{M.NV}/stats_player/stats_player_week_{season}.csv")
st = pd.read_csv(io.StringIO(r.text), low_memory=False) if r is not None and r.status_code == 200 else pd.DataFrame()
st = st[(st.week == week) & (st.season_type == "REG")] if len(st) else st
if sc.empty and st.empty:
    print(f"no nflverse data for {season} week {week} yet"); sys.exit(0)
snaps = {(norm(r_.player), r_.team): float(r_.offense_snaps) for r_ in sc.itertuples()} if len(sc) else {}
stats = {r_.player_id: (int(r_.targets), int(r_.receptions), float(r_.receiving_yards)) for r_ in st.itertuples()} if len(st) else {}
removed = set(M.REMOVE_STATUSES)
g["snaps"] = [snaps.get((norm(n), t)) for n, t in zip(g.name, g.team)]
g["stat"] = g.pid.map(stats)
played = g.snaps.fillna(0) > 0
flagged_played = g[g.inj.astype(str).isin(removed) & played]
unflagged_dnp = g[(g.inj.astype(str) == "") & (g.game_status != "scheduled") & (~played) & g.snaps.notna()] if "game_status" in g.columns else g[(g.inj.astype(str) == "") & (~played) & g.snaps.notna()]
no_snap_row = g[(g.inj.astype(str) == "") & g.snaps.isna()]
print(f"{season} week {week}: {len(g)} grid rows, snap counts for {int(g.snaps.notna().sum())}")
print(f"\nflagged {'/'.join(sorted(removed))} but PLAYED ({len(flagged_played)}):")
for r_ in flagged_played.itertuples():
    print(f"  {r_.team} {r_.name:<22} {r_.inj:<12} snaps {r_.snaps:.0f}  stats {r_.stat}")
print(f"\nno flag but ZERO offensive snaps ({len(unflagged_dnp)}):")
for r_ in unflagged_dnp.itertuples():
    print(f"  {r_.team} {r_.name:<22} proj_targets {r_.proj_targets:.2f}")
print(f"\nno flag and no snap-count row at all (inactive or name mismatch) ({len(no_snap_row)}):")
for r_ in no_snap_row.itertuples():
    print(f"  {r_.team} {r_.name:<22} proj_targets {r_.proj_targets:.2f}")

# ---- raw vs anchored medians, weekly series (Phase 1, item 8) ------------------------------
raw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "props_grid_raw.csv")
if len(st) and os.path.exists(raw_path):
    raw = pd.read_csv(raw_path, low_memory=False)
    yds = {r_.player_id: float(r_.receiving_yards) for r_ in st.itertuples()}
    pub = g[g.pid.isin(yds) & (g.proj_targets > 0)].copy(); pub["act"] = pub.pid.map(yds)
    rw = raw.set_index("pid").reindex(pub.pid)
    lined = pub[pub.get("anchored", pd.Series(False, index=pub.index)).astype(bool)] if "anchored" in pub.columns else pub.iloc[0:0]
    if len(lined):
        a = lined.act.values
        mae_anc = float(np.mean(np.abs(lined.med_yds.values - a)))
        mae_raw = float(np.mean(np.abs(rw.loc[lined.pid, "med_yds"].values - a)))
        mae_line = float(np.mean(np.abs(lined.market_line.values - a)))
        print(f"\nlined players {len(lined)}: MAE raw median {mae_raw:.1f} | anchored median {mae_anc:.1f} | market line {mae_line:.1f}")
        rp = os.path.join(os.path.dirname(raw_path), "props_anchor_results.csv")
        pd.DataFrame([dict(season=season, week=week, n=len(lined), mae_raw=round(mae_raw, 2), mae_anchored=round(mae_anc, 2), mae_line=round(mae_line, 2))]) \
            .to_csv(rp, mode="a", index=False, header=not os.path.exists(rp))
        print(f"appended -> logs/props_anchor_results.csv")
    else:
        print("\nno anchored players in this grid (no odds snapshot that week); raw-vs-anchored series not updated")
