"""
sim_nfl.py: joint game simulation for mbm-nfl. One set of dice per game.

    python sim_nfl.py            # this week's games: logs/sim_totals.csv, logs/sim_players.csv
    python sim_nfl.py --test     # unit tests only

PAPER ONLY until the bars in sim_backtest.py are met. Nothing here feeds the board, picks.csv,
the props grid or the Unabated file.

Why it exists. The sides model prices the winner and the margin; the props scripts price
players one at a time. Nothing priced a total as a distribution, and nothing tied a player's
volume to how the game went. This file plays each game N times so that every number comes
off the same simulated games.

What it does NOT do is rate the teams. All of the team strength comes from the sides model:

  centre    home and away expected points = (model_total +/- sp_margin) / 2, both read from
            logs/sim_inputs.json, which daily_nfl.py writes for every game it boards.
  drives    each team gets about the same number of possessions; each drive ends in a
            touchdown, a field goal or nothing, at league rates scaled to the team's expected
            points. Extra points, two-point tries, returns and overtime are rolled so the
            scores are real football scores. Team-specific pace and touchdown mix (decayed,
            opponent-adjusted, from play-by-play drives) were tested on 2023 and added nothing,
            so they are off; team_states() and game_shape() remain for the backtest.
  margin    the simulated games are then re-weighted so the home margin follows the sides
            model's own distribution (spread.py: fitted to history, carries 3 and 7). A plain
            drive sim lands on 3 half as often as real games do; the weights fix that without
            touching the board's numbers. A tilt on the total keeps its mean on model_total
            and its spread at what real totals show around that number (a plain drive sim
            is about two points too wide).
  players   carries and pass attempts in each simulated game move with that game's final
            margin, then yards are drawn around the means in logs/rushpass_grid.csv.

Constants in K were measured on 2016-2023 regular seasons (sim_backtest.py --fit prints them).
"""
import argparse, json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
LOGS = os.path.join(HERE, "logs")
DATA = os.path.join(HERE, "data")
NV = "https://github.com/nflverse/nflverse-data/releases/download"
VERSION = 1
N_SIMS = 20000
JMAX = 70

# Measured on 2016-2023 regular seasons, frozen. See sim_backtest.py --fit.
K = dict(
    lg_drives=10.78, drive_sd=1.60,           # possessions per team per game (drives with a run or pass)
    drive_diff=(0.30, 0.40, 0.30),            # away drives minus home drives: -1, 0, +1
    lg_mix=0.598,                             # touchdowns / (touchdowns + field goals), per scoring drive
    p_two=0.090, p_xp=0.941, p_two_good=0.489,
    other_pts=1.11,                           # points per team per game not from its own drives
    ot_margin=((3, 0.55), (6, 0.25), (7, 0.12), (1, 0.08)),
    hl_games=10.0, season_carry=0.6, pace_tau=6.0, mix_tau=40.0,
    pace_weight=0.0, mix_weight=0.0,          # team pace and TD mix: tested on 2023, no gain, left off
    total_sd=13.35,                           # spread of real totals around model_total, training seasons
    w_cap=25.0,
    # players: script slopes per point of FINAL margin, and yardage noise
    rush_slope=0.0108, pass_slope=0.0024, script_cap=28.0,
    car_kappa=20.0, bust_p=0.03, bust_mult=0.2, rush_sd=5.92, rush_skew=0.87, att_sd=7.76, pass_sd=9.89, pass_skew=0.43,
)
VTD = 6 + (1 - K["p_two"]) * K["p_xp"] + K["p_two"] * K["p_two_good"] * 2      # points per touchdown

TOTAL_LINES = np.arange(30.0, 66.0, 0.5)
TEAM_LINES = np.arange(9.5, 40.0, 1.0)
RUSH_LINES = [24.5, 34.5, 44.5, 49.5, 54.5, 59.5, 64.5, 69.5, 74.5, 79.5, 89.5, 99.5, 109.5, 124.5, 149.5]
PASS_LINES = [149.5, 174.5, 199.5, 209.5, 219.5, 224.5, 229.5, 239.5, 249.5, 259.5, 274.5, 299.5, 324.5, 349.5]


# ------------------------------------------------------------------ drive data
PBP_COLS = ["game_id", "season", "season_type", "week", "posteam", "defteam", "fixed_drive",
            "fixed_drive_result", "play_type"]
TEAM_NORM = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}


def drive_table(pbp):
    """One row per team per regular-season game: drives (with at least one run or pass),
    touchdown drives, field-goal drives."""
    d = pbp[pbp.posteam.notna() & pbp.fixed_drive.notna() & (pbp.season_type == "REG")]
    dr = d.groupby(["game_id", "season", "week", "posteam", "defteam", "fixed_drive"]).agg(
        res=("fixed_drive_result", "first"),
        plays=("play_type", lambda s: int(s.isin(["run", "pass"]).sum()))).reset_index()
    dr = dr[dr.plays >= 1]
    dr["td"] = (dr.res == "Touchdown").astype(int); dr["fg"] = (dr.res == "Field goal").astype(int)
    t = dr.groupby(["game_id", "season", "week", "posteam", "defteam"]).agg(
        drives=("res", "size"), td=("td", "sum"), fg=("fg", "sum")).reset_index()
    for c in ("posteam", "defteam"):
        t[c] = t[c].replace(TEAM_NORM)
    return t


def team_states(T, season, before_week=None):
    """Per team, from games before (season, before_week): pace deviation (drives per game
    above the league) and touchdown mix on offence and allowed on defence, decayed by game
    (half-life K['hl_games'], earlier seasons times K['season_carry']) and shrunk to league."""
    t = T[(T.season < season) | ((T.season == season) & ((before_week is None) | (T.week < before_week)))]
    t = t[t.season >= season - 2].sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    out = {}
    if not len(t):
        return out, K["lg_drives"], K["lg_mix"]
    cur = t[t.season == season]
    lg = cur if cur.game_id.nunique() >= 48 else t
    lg_d = float(lg.drives.mean()); lg_m = float(lg.td.sum() / max(lg.td.sum() + lg.fg.sum(), 1))
    for key, role in (("posteam", "off"), ("defteam", "def")):
        age = t.groupby(key).cumcount(ascending=False).to_numpy()
        w = (0.5 ** (age / K["hl_games"])) * np.where(t.season.to_numpy() < season, K["season_carry"], 1.0)
        a = pd.DataFrame({"team": t[key].values, "w": w, "wd": w * t.drives.values,
                          "wt": w * t.td.values, "wf": w * t.fg.values}).groupby("team").sum()
        for tm, r in a.iterrows():
            s = out.setdefault(tm, {})
            s[f"pace_{role}"] = (r.wd + lg_d * K["pace_tau"]) / (r.w + K["pace_tau"]) - lg_d
            s[f"mix_{role}"] = (r.wt + lg_m * K["mix_tau"]) / (r.wt + r.wf + K["mix_tau"]) - lg_m
    return out, lg_d, lg_m


def game_shape(states, lg_d, lg_m, home, away):
    """(expected drives per team, home touchdown mix, away touchdown mix) for one game."""
    h, a = states.get(home, {}), states.get(away, {})
    # A team's own drive count and the count it allows are the same game clock, so the
    # two deviations are averaged per team and the two teams' effects add.
    dev = lambda s: 0.5 * (s.get("pace_off", 0.0) + s.get("pace_def", 0.0))
    d = lg_d + K["pace_weight"] * (dev(h) + dev(a))
    mh = lg_m + K["mix_weight"] * (h.get("mix_off", 0.0) + a.get("mix_def", 0.0))
    ma = lg_m + K["mix_weight"] * (a.get("mix_off", 0.0) + h.get("mix_def", 0.0))
    return float(np.clip(d, 8.5, 13.0)), float(np.clip(mh, 0.35, 0.80)), float(np.clip(ma, 0.35, 0.80))


# ------------------------------------------------------------------ the dice
def _team_points(rng, mu, mix, drives, d_mean):
    s = max(mu - K["other_pts"], 3.0) / d_mean / (mix * VTD + (1 - mix) * 3.0)   # P(scoring drive)
    s = min(s, 0.85)
    p_td, p_fg = mix * s, (1 - mix) * s
    td = rng.binomial(drives, p_td)
    fg = rng.binomial(drives - td, p_fg / (1 - p_td))
    two = rng.binomial(td, K["p_two"])
    pts = 6 * td + rng.binomial(td - two, K["p_xp"]) + 2 * rng.binomial(two, K["p_two_good"]) + 3 * fg
    return pts + 7 * rng.poisson(K["other_pts"] / 7.0, len(drives)), td, fg


def simulate_scores(mu_h, mu_a, mix_h, mix_a, d_mean, ph, n=N_SIMS, seed=0):
    """n games. Returns home points, away points, home drives, away drives. Ties after
    regulation go to overtime: the winner is drawn at the pregame probability and the margin
    from K['ot_margin']."""
    rng = np.random.default_rng(seed)
    hd = np.clip(np.rint(rng.normal(d_mean, K["drive_sd"], n)), 6, 16).astype(int)
    ad = np.clip(hd + rng.choice([-1, 0, 1], n, p=K["drive_diff"]), 6, 16).astype(int)
    hp, _, _ = _team_points(rng, mu_h, mix_h, hd, d_mean)
    ap, _, _ = _team_points(rng, mu_a, mix_a, ad, d_mean)
    tie = hp == ap
    k = int(tie.sum())
    if k:
        m = rng.choice([x for x, _ in K["ot_margin"]], k, p=[p for _, p in K["ot_margin"]])
        home_w = rng.random(k) < ph
        # a one-point overtime win is touchdown plus two against touchdown plus one
        hp[tie] += np.where(home_w, np.where(m == 1, 8, m), np.where(m == 1, 7, 0))
        ap[tie] += np.where(home_w, np.where(m == 1, 7, 0), np.where(m == 1, 8, m))
    return hp, ap, hd, ad


def margin_target(ph, Ph, Pa):
    """The sides model's margin distribution as {margin: probability}, margin = home - away."""
    j = np.arange(1, JMAX + 1)
    t = {int(v): float(ph * p) for v, p in zip(j, Ph)}
    t.update({int(-v): float((1 - ph) * p) for v, p in zip(j, Pa)})
    return t


def _tilt(w, t, mean, sd=None):
    """Exponential tilt of the weights on the total, exp(a*z + b*z*z), so the weighted total
    has the given mean and, when sd is given, that spread. Solved on the handful of distinct
    totals, by bisection: the mean rises with a and the spread rises with b."""
    ut, inv = np.unique(t, return_inverse=True)
    mass = np.bincount(inv, weights=w)
    z = (ut - mean) / 10.0

    def solve_a(b):
        lo, hi = -6.0, 6.0
        for _ in range(36):
            a = 0.5 * (lo + hi)
            e = a * z + b * z * z
            q = mass * np.exp(e - e.max()); q = q / q.sum()
            lo, hi = (a, hi) if (q * ut).sum() < mean else (lo, a)
        return a, q

    if sd is None:
        a, q = solve_a(0.0); b = 0.0
    else:
        lo, hi = -3.0, 1.0
        for _ in range(30):
            b = 0.5 * (lo + hi)
            a, q = solve_a(b)
            cur = np.sqrt((q * (ut - (q * ut).sum()) ** 2).sum())
            lo, hi = (b, hi) if cur < sd else (lo, b)
    e = a * z + b * z * z
    out = w * np.exp(e - e.max())[inv]
    return out / out.sum()


def rake(hp, ap, target, total_mean=None, total_sd=None, iters=4):
    """Weights that make the simulated margins follow `target` and, if given, hold the total's
    mean at total_mean and its spread at total_sd. Returns (weights summing to 1, effective
    sample size)."""
    m = hp - ap; t = (hp + ap).astype(float)
    w = np.ones(len(m)) / len(m)
    tv = np.array([target.get(int(x), 0.0) for x in m])
    idx = pd.factorize(m)[0]
    for it in range(iters):
        if total_mean is not None:
            w = _tilt(w, t, total_mean, total_sd)
        # margins last, so the published spread numbers are matched exactly
        mass = np.bincount(idx, weights=w)[idx]
        w = w * np.clip(tv / np.maximum(mass, 1e-12), 0.0, K["w_cap"]); w = w / w.sum()
        if total_mean is None:
            break
    return w, float(1.0 / np.sum(w ** 2))


def over_push(x, w, line):
    """(P over, P push) for an array of simulated values at one line."""
    return float(w[x > line].sum()), float(w[x == line].sum())


def american(p):
    if not (0 < p < 1):
        return np.nan
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def sim_game(inp, states, lg_d, lg_m, n=N_SIMS, seed=0):
    """Simulate one game from a sim_inputs row. Returns a dict of arrays and the weights."""
    M, T, ph = float(inp["sp_margin"]), float(inp["model_total"]), float(inp["p_home"])
    d, mh, ma = game_shape(states, lg_d, lg_m, inp["home"], inp["away"])
    hp, ap, hd, ad = simulate_scores((T + M) / 2, (T - M) / 2, mh, ma, d, ph, n, seed)
    w, ess = rake(hp, ap, margin_target(ph, np.asarray(inp["Ph"]), np.asarray(inp["Pa"])), total_mean=T, total_sd=K["total_sd"])
    return dict(hp=hp, ap=ap, hd=hd, ad=ad, w=w, ess=ess, drives=d, mix_h=mh, mix_a=ma)


# ------------------------------------------------------------------ players
def _skewed(rng, n, skew):
    """Standardised draw (mean 0, sd 1) with the given skewness, from a shifted gamma."""
    if skew <= 0.05:
        return rng.normal(0, 1, n)
    k = 4.0 / skew ** 2
    return (rng.gamma(k, 1.0, n) - k) / np.sqrt(k)


def player_draws(g, row, side, seed=0):
    """Rushing or passing yards for one rushpass_grid row in every simulated game.
    The grid's mean was built at the game's EXPECTED margin, so the script effect here is
    applied to the difference between each simulated margin and that expected margin."""
    rng = np.random.default_rng(seed)
    n = len(g["hp"])
    m = (g["hp"] - g["ap"]) * (1 if side == "home" else -1)
    em = float((g["w"] * m).sum())
    dm = np.clip(m - em, -K["script_cap"], K["script_cap"])
    out = {}
    if pd.notna(row.get("proj_carries")) and row["proj_carries"] > 0 and pd.notna(row.get("mean_rush_yds")):
        mu_c = row["proj_carries"] * np.clip(1 + K["rush_slope"] * dm, 0.4, 1.8)
        # A share of games the role collapses (early injury, benched, a committee the usage
        # history did not show). The other games are scaled up so the mean carries hold.
        bust = rng.random(n) < K["bust_p"]
        mu_c = mu_c * np.where(bust, K["bust_mult"], (1 - K["bust_p"] * K["bust_mult"]) / (1 - K["bust_p"]))
        lam = rng.gamma(K["car_kappa"], mu_c / K["car_kappa"])             # over-dispersed count
        c = rng.poisson(lam)
        ypc = row["mean_rush_yds"] / row["proj_carries"]
        y = c * ypc + np.sqrt(np.maximum(c, 0)) * K["rush_sd"] * _skewed(rng, n, K["rush_skew"])
        out["rush"] = np.where(c > 0, np.rint(y), 0.0)
    if pd.notna(row.get("proj_att")) and row["proj_att"] > 0 and pd.notna(row.get("mean_pass_yds")):
        mu_a = row["proj_att"] * np.clip(1 - K["pass_slope"] * dm, 0.5, 1.6)
        a = np.maximum(np.rint(rng.normal(mu_a, K["att_sd"])), 1)
        ypa = row["mean_pass_yds"] / row["proj_att"]
        y = a * ypa + np.sqrt(a) * K["pass_sd"] * _skewed(rng, n, K["pass_skew"])
        out["pass"] = np.maximum(np.rint(y), 0.0)
    return out


# ------------------------------------------------------------------ weekly run
SHOW_TOTALS = (-6.0, -3.0, 0.0, 3.0, 6.0)      # lines shown on Discord, relative to the sim's own total


def _fmt_odds(a):
    return "n/a" if a != a else (f"+{int(a)}" if a > 0 else f"{int(a)}")


def discord_lines(tot, pl, today):
    """Today's and tomorrow's games only, a header per day, one block per game."""
    lines = ["PAPER ONLY. Fair prices from the joint sim; not a bet list. Totals have shown no edge against the closing line.", ""]
    keep = [today.strftime("%Y-%m-%d"), (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")]
    shown = 0
    for day in keep:
        dg = tot[tot.date == day]
        if not len(dg):
            continue
        lines.append(pd.Timestamp(day).strftime("%a %b %-d"))
        for gid, g in dg.groupby("game_id", sort=False):
            r0 = g.iloc[0]; t = g[g.market == "total"]
            centre = np.floor(r0.sim_mean) + 0.5
            parts = []
            for off in SHOW_TOTALS:
                q = t[t.line == centre + off]
                if len(q):
                    parts.append(f"o{centre + off:g} {_fmt_odds(q.fair_over.iloc[0])}")
            tt = g[g.market == "team_total"].drop_duplicates("team")
            teams = ", ".join(f"{x.team} {x.sim_mean:.1f}" for x in tt.itertuples())
            lines.append(f"  {r0.away} @ {r0.home}: total {r0.sim_mean:.1f} ({teams})")
            lines.append("    " + "  ".join(parts))
            if len(pl):
                for x in pl[pl.game_id == gid].sort_values("sim_mean", ascending=False).itertuples():
                    cols = RUSH_SHOW if x.stat == "rush_yds" else PASS_SHOW
                    lad = "  ".join(f"o{L:g} {100 * getattr(x, 'p_over_' + str(L).replace('.', '_')):.0f}%" for L in cols)
                    lines.append(f"    {x.name} {'rush' if x.stat == 'rush_yds' else 'pass'}: median {x.sim_median:.0f}  {lad}")
            shown += 1
        lines.append("")
    return lines if shown else []


RUSH_SHOW = (44.5, 59.5, 74.5, 89.5, 109.5)
PASS_SHOW = (199.5, 224.5, 249.5, 274.5, 299.5)


def run():
    path = os.path.join(LOGS, "sim_inputs.json")
    if not os.path.exists(path):
        print("no logs/sim_inputs.json yet: the daily board has to run once with this version"); return
    inputs = json.load(open(path))
    now = pd.Timestamp.now(tz="UTC")
    age_h = (now - pd.Timestamp(inputs.get("ts"))).total_seconds() / 3600
    today = now.tz_convert("America/Edmonton").normalize().tz_localize(None)
    games = [x for x in inputs.get("games", []) if x.get("model_total") is not None and pd.Timestamp(x["date"]) >= today]
    if age_h > 30:
        print(f"MBM-CHECK WARN stale | NFL sim: board inputs are {age_h:.0f}h old; not simulating"); return
    if not games:
        print("sim: no upcoming boarded games"); return
    states, lg_d, lg_m = {}, K["lg_drives"], K["lg_mix"]
    print(f"sim v{VERSION}: {len(games)} game(s), board inputs {age_h:.1f}h old, {N_SIMS} sims each")
    rp = os.path.join(LOGS, "rushpass_grid.csv")
    grid = pd.read_csv(rp) if os.path.exists(rp) else pd.DataFrame()
    ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    tot_rows, pl_rows = [], []
    for gi, inp in enumerate(games):
        g = sim_game(inp, states, lg_d, lg_m, seed=1000 + gi)
        total = g["hp"] + g["ap"]; w = g["w"]
        base = dict(ts=ts, game_id=inp["game_id"], date=inp["date"], week=inp["week"], home=inp["home"], away=inp["away"],
                    p_home=round(float(inp["p_home"]), 4), sp_margin=round(float(inp["sp_margin"]), 2),
                    model_total=round(float(inp["model_total"]), 2), ess=int(g["ess"]))
        for market, arr, lines in (("total", total, TOTAL_LINES), (inp["home"], g["hp"], TEAM_LINES), (inp["away"], g["ap"], TEAM_LINES)):
            mean = float((w * arr).sum())
            for L in lines:
                po, pp = over_push(arr, w, L)
                if not (0.03 <= po <= 0.97):
                    continue
                fo = po / max(1 - pp, 1e-9)
                tot_rows.append({**base, "market": "total" if market == "total" else "team_total", "team": "" if market == "total" else market,
                                 "line": L, "p_over": round(po, 4), "p_push": round(pp, 4), "p_under": round(1 - po - pp, 4),
                                 "fair_over": american(fo), "fair_under": american(1 - fo), "sim_mean": round(mean, 2)})
        wk = grid[(grid.week.astype(int) == int(inp["week"])) & (grid.season.astype(int) == int(inp["season"]))] if len(grid) else grid
        for side, tm in (("home", inp["home"]), ("away", inp["away"])):
            sub = wk[wk.team.replace(TEAM_NORM) == tm] if len(wk) else wk
            for pi, r in enumerate(sub.to_dict("records")):
                dr = player_draws(g, r, side, seed=5000 + 100 * gi + pi)
                for stat, lines, mean_col in (("rush", RUSH_LINES, "mean_rush_yds"), ("pass", PASS_LINES, "mean_pass_yds")):
                    if stat not in dr or (stat == "rush" and (r["proj_carries"] < 8 or bool(r["starter"]))):
                        continue        # the ladders were only scored for backs projected for 8+ carries and for starting QBs' passing
                    y = dr[stat]
                    row = dict(ts=ts, game_id=inp["game_id"], date=inp["date"], week=inp["week"], team=tm, gsis_id=r["gsis_id"],
                               name=r["name"], pos=r["pos"], stat=f"{stat}_yds", grid_mean=r[mean_col],
                               sim_mean=round(float((w * y).sum()), 1), sim_median=float(_wquantile(y, w, 0.5)))
                    for L in lines:
                        row["p_over_" + str(L).replace(".", "_")] = round(float(w[y > L].sum()), 4)
                    pl_rows.append(row)
    tot = pd.DataFrame(tot_rows); pl = pd.DataFrame(pl_rows)
    tot.to_csv(os.path.join(LOGS, "sim_totals.csv"), index=False)
    print(f"wrote logs/sim_totals.csv ({len(tot)} rows)")
    if len(pl):
        pl.to_csv(os.path.join(LOGS, "sim_players.csv"), index=False)
        print(f"wrote logs/sim_players.csv ({len(pl)} rows)")
    else:
        print("sim: no rushing/passing grid for these weeks yet, player ladders skipped")
    lines = discord_lines(tot, pl, today)
    for ln in lines:
        print(ln)
    if lines and os.environ.get("SIM_POST") == "1":
        import notify
        notify.send("NFL joint sim (paper): totals, team totals, rush and pass ladders", lines,
                    "sim_nfl.py v%d. Margins match the board. Backtest: sim_backtest.py" % VERSION, getattr(notify, "GREY", None))


def _wquantile(x, w, q):
    o = np.argsort(x); c = np.cumsum(w[o])
    return x[o][min(np.searchsorted(c, q * c[-1]), len(x) - 1)]


def unit_tests():
    # the raked margins follow the target, and the mean total holds
    j = np.arange(1, JMAX + 1)
    pm = np.exp(-0.5 * ((j - 9) / 8.0) ** 2); pm[2] *= 3; pm[6] *= 2; pm = pm / pm.sum()
    hp, ap, _, _ = simulate_scores(24.5, 21.5, 0.6, 0.58, 10.6, 0.58, n=40000, seed=3)
    assert not np.any(hp == ap), "overtime left a tie"
    tgt = margin_target(0.58, pm, pm)
    w, ess = rake(hp, ap, tgt, total_mean=46.0)
    m = hp - ap
    assert abs(w.sum() - 1) < 1e-9 and ess > 2000, ess
    assert abs(w[m > 0].sum() - 0.58) < 0.01, w[m > 0].sum()
    assert abs(w[m == 3].sum() - tgt[3]) < 0.01, (w[m == 3].sum(), tgt[3])
    assert abs((w * (hp + ap)).sum() - 46.0) < 0.15, (w * (hp + ap)).sum()
    # expected points come back out of the drive roll
    hp2, ap2, _, _ = simulate_scores(27.0, 18.0, 0.6, 0.6, 10.6, 0.7, n=60000, seed=5)
    assert abs(hp2.mean() - 27.0) < 0.6 and abs(ap2.mean() - 18.0) < 0.6, (hp2.mean(), ap2.mean())
    # over + push + under = 1, and the american conversion is symmetric
    po, pp = over_push(hp + ap, w, 44.0)
    assert 0 < po < 1 and 0 < pp < 0.1
    assert american(0.6) == -150 and american(0.4) == 150
    print("sim_nfl unit tests passed")


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser(); ap_.add_argument("--test", action="store_true")
    a = ap_.parse_args()
    unit_tests() if a.test else run()
