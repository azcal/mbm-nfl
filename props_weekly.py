"""
props_weekly.py: Sunday-morning receiving projections for this week's games (mbm-nfl).

Pulls nflverse player weeks, injury reports and snap counts for 2019 to now, builds the
model's states through the last completed week, and projects every receiver on the slate
that kicks off in the next 7 days. Writes logs/props_grid.csv: one row per player with the
median, the mean, P(receiving yards > X) for X on the model grid and P(receptions >= k).
The props ledger in mbm-data joins this to the archived prop prices.

Spec (props_spec.json) was tuned on 2020-22 and scored once on 2023-25:
receiving-yards over/under claims within 2.6 points of actual in every probability band,
actual over the model median 0.514, receptions within 2.1 points in every band.
"""
import datetime as dt, io, json, os, sys
import numpy as np, pandas as pd, requests
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import props_model as M

spec = json.load(open(os.path.join(HERE, "props_spec.json")))
NOW = dt.datetime.now(dt.timezone.utc)
g = pd.read_csv(io.StringIO(requests.get("https://github.com/nflverse/nfldata/raw/master/data/games.csv", timeout=120).text))
g = g[g.game_type == "REG"].copy(); g["kick"] = pd.to_datetime(g.gameday + " " + g.gametime.fillna("13:00"), errors="coerce").dt.tz_localize("US/Eastern").dt.tz_convert("UTC")
slate = g[(g.kick > NOW - dt.timedelta(hours=1)) & (g.kick <= NOW + dt.timedelta(days=7)) & g.result.isna()]
if slate.empty:
    print("no games in the next 7 days"); sys.exit(0)
season, week = int(slate.season.iloc[0]), int(slate.week.iloc[0])
up = pd.concat([pd.DataFrame(dict(season=season, week=week, team=slate.home_team, opponent_team=slate.away_team)),
                pd.DataFrame(dict(season=season, week=week, team=slate.away_team, opponent_team=slate.home_team))], ignore_index=True)
print(f"{season} week {week}: {len(slate)} games, {len(up)} teams")
M.RNG = np.random.default_rng(int(NOW.strftime("%Y%m%d")))
roster = M.load_roster(season, live=True)          # current team, status ACT: offseason moves, cuts and rookies
print(f"roster: {'none, falling back to last-seen teams' if roster is None else str(len(roster)) + ' active receivers'}")
df = M.run(range(2019, season + 1), kappa=spec["kappa"], alpha=spec["alpha"], H=spec["H"], upcoming=up, live=True, vol_mult=spec.get("volume_mult"), roster=roster)
df = df[(df.season == season) & (df.week == week)].copy()
out = df[["season", "week", "team", "opp", "pid", "name", "pos", "proj_targets", "share", "cr", "ypc", "med_yds", "mean_yds", "med_rec"]].copy()
for x, col in zip(M.YGRID, [f"p_over_{int(x)}" for x in M.YGRID]):
    out[col] = [round(p[i], 4) for p, i in zip(df.p_over, [list(M.YGRID).index(x)] * len(df))]
for k in M.RGRID:
    out[f"p_rec_{k}"] = [round(p[k - 1], 4) for p in df.p_rec]
out["ts"] = NOW.strftime("%Y-%m-%dT%H:%M:%SZ")
os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
out.to_csv(os.path.join(HERE, "logs", "props_grid.csv"), index=False)
print(f"props grid: {len(out)} players -> logs/props_grid.csv")
top = out.sort_values("med_yds", ascending=False)
lines = [f"{season} week {week}: {len(out)} receivers projected. median yards | P(over 50 / 75 / 100) | median rec | P(rec >= 5 / 7)"]
for r in top.head(40).itertuples():
    lines.append(f"  {r.team:<3} {r.name:<22} {r.pos:<2} {r.med_yds:>6.1f} | {r.p_over_50:.2f} / {r.p_over_75:.2f} / {r.p_over_100:.2f} | {r.med_rec:>4.1f} | {r.p_rec_5:.2f} / {r.p_rec_7:.2f}")
lines.append("full grid (every line 10 to 120, every reception count) in logs/props_grid.csv; fair decimal odds = 1 / probability")
print("\n".join(lines))
try:
    import notify
    notify.send(f"NFL receiving projections, week {week}", lines, "bottom-up, no market input; calibrated on 2023-25 within 3 points per band", getattr(notify, "GREY", None))
except Exception as ex:
    print("discord post skipped:", type(ex).__name__)
