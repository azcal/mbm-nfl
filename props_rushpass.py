"""
props_rushpass.py: bottom-up mean rushing yards and passing yards for this week's games (mbm-nfl).

    python props_rushpass.py           # after props_weekly.py, same run
    python props_rushpass.py --test    # unit tests only

UNTESTED. No backtest has been run. Mark's call on Nov 20, 2026 was to ship it for that Sunday's
Unabated file and score it as the season goes. The receiving model in props_model.py is the
tested one, and nothing here touches it, props_grid.csv or the board.

Built from nflverse weekly player stats only, with no market input. props_unabated.py uploads
the results to Unabated as plain means, and Unabated simulates its own spread around them.
  team volume    decayed carries and pass attempts per game (half-life 6 team games, last
                 season's games x0.6), shrunk to the league average by 2 games, then moved by
                 game script from the sides model's own margin in logs/predictions.csv:
                 +0.77% carries and -0.32% pass attempts per point of expected margin (the
                 within team-season slopes on 2024-26 team games, n=1386, se 0.15% and 0.13%)
  starting QB    nflverse depth chart minus anyone ruled out (ESPN, then the nflverse report),
                 else the QB with the most attempts in the team's last game
  carry share    decayed share of team carries on the current team (half-life 4), shrunk by 30
                 team carries to a depth-chart prior (RB1 .55, RB2 .21, RB3 .05, QB1 .13,
                 everyone else .01). Ruled-out players drop out and the remaining non-QB shares
                 are rescaled to fill the team's non-QB carries, so a team keeps its volume.
  efficiency     yards per carry and per pass attempt, decayed per player (half-life 8 games),
                 shrunk to position priors, times the opponent's decayed rate allowed at half
                 weight
  means          rush yds = team carries x share x ypc; pass yds = team attempts x 0.97 x ypa
Writes logs/rushpass_grid.csv.
"""
import argparse, io, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import props_model as M
import props_weekly as W

OUT_TAGS = ("Out", "Doubtful", "IR", "Injured Reserve", "PUP", "Suspended", "Suspension")
POSITIONS = ("QB", "RB", "FB", "WR", "TE")
HL_TEAM, HL_SHARE, HL_EFF, SEASON_CARRY = 6.0, 4.0, 8.0, 0.6
PRIOR_GAMES = 2.0                       # team volume shrink, in games
SHARE_TAU = 30.0                        # carry-share shrink, in team carries (about one game)
SHARE_PRIOR = {("RB", 1): 0.55, ("RB", 2): 0.21, ("RB", 3): 0.05, ("QB", 1): 0.13, ("FB", 1): 0.02}
SHARE_PRIOR_DEFAULT = 0.01
YPC_PRIOR = {"QB": 4.3, "RB": 4.3, "FB": 3.0, "WR": 5.6, "TE": 2.3}   # 2025-26 position averages
YPC_TAU = 60.0                          # carries
YPA_PRIOR, YPA_TAU = 7.0, 150.0         # gross passing yards per attempt; league 7.09 in 2024-26
OPP_WEIGHT, OPP_TAU_CARRIES, OPP_TAU_ATT = 0.5, 150.0, 200.0
GS_RUSH, GS_PASS, GS_CAP = 0.0077, 0.0032, 17.0
QB_ATT_SHARE = 0.97                     # the passer with the most attempts throws 97.3% of them
NONQB_FILL = 0.99                       # non-QB shares fill this much of the non-QB carries
SHARE_CAP = 0.80
MIN_CARRIES = 1.5                       # below this a non-QB gets no rushing number
GRID_COLS = ["season", "week", "team", "opp", "kickoff_utc", "game_status", "gsis_id", "name", "pos", "depth_rank", "inj",
             "starter", "starter_source", "margin", "team_carries", "carry_share", "proj_carries", "ypc", "opp_ypc_mult",
             "mean_rush_yds", "team_att", "proj_att", "ypa", "opp_ypa_mult", "mean_pass_yds", "rec_yds_simple"]


def recency(df, key, hl, season):
    """Weight per row, rows in time order: 0.5 ** (age / hl), age counted back from the newest
    row of each key (newest = 0); rows from earlier seasons times SEASON_CARRY."""
    age = df.groupby(key).cumcount(ascending=False).to_numpy()
    return (0.5 ** (age / hl)) * np.where(df.season.to_numpy() < season, SEASON_CARRY, 1.0)


def shrink(num, den, prior, tau):
    return (num + prior * tau) / (den + tau)


def team_games(st):
    """One row per team per game: carries, rushing yards, pass attempts, passing yards."""
    tg = st.groupby(["season", "week", "game_id", "team"], as_index=False).agg(
        opp=("opponent_team", "first"), carries=("carries", "sum"), ry=("rushing_yards", "sum"),
        att=("attempts", "sum"), py=("passing_yards", "sum"))
    return tg.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)


def team_volume(tg, season):
    """Decayed carries and attempts per game per team, shrunk to the league average."""
    tg = tg.copy(); tg["w"] = recency(tg, "team", HL_TEAM, season)
    cur = tg[tg.season == season]
    lg = cur if len(cur) >= 64 else tg
    l_car, l_att = float(lg.carries.mean()), float(lg.att.mean())
    tg["wc"] = tg.w * tg.carries; tg["wa"] = tg.w * tg.att
    v = tg.groupby("team")[["w", "wc", "wa"]].sum()
    v["team_carries"] = shrink(v.wc, v.w, l_car, PRIOR_GAMES)
    v["team_att"] = shrink(v.wa, v.w, l_att, PRIOR_GAMES)
    return v[["team_carries", "team_att"]], l_car, l_att


def defense_rates(tg, season):
    """Each defence's decayed yards per carry and per attempt allowed, as a half-weight
    multiplier on the offence's own rate."""
    dg = tg.rename(columns={"opp": "defense"}).sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    dg["w"] = recency(dg, "defense", HL_TEAM, season)
    l_ypc = float(dg.ry.sum() / max(dg.carries.sum(), 1)); l_ypa = float(dg.py.sum() / max(dg.att.sum(), 1))
    for c in ("carries", "ry", "att", "py"):
        dg["w_" + c] = dg.w * dg[c]
    d = dg.groupby("defense")[["w_carries", "w_ry", "w_att", "w_py"]].sum()
    d["ypc_allowed"] = shrink(d.w_ry, d.w_carries, l_ypc, OPP_TAU_CARRIES)
    d["ypa_allowed"] = shrink(d.w_py, d.w_att, l_ypa, OPP_TAU_ATT)
    d["opp_ypc_mult"] = 1 + OPP_WEIGHT * (d.ypc_allowed / l_ypc - 1)
    d["opp_ypa_mult"] = 1 + OPP_WEIGHT * (d.ypa_allowed / l_ypa - 1)
    return d[["opp_ypc_mult", "opp_ypa_mult"]]


def player_rates(st, season):
    """Per player: decayed ypc and ypa with their raw weighted sums (all teams: it's the player)."""
    rows = {}
    ru = st[st.carries > 0].sort_values(["season", "week"]).reset_index(drop=True)
    ru["w"] = recency(ru, "player_id", HL_EFF, season)
    ru["wc"] = ru.w * ru.carries; ru["wy"] = ru.w * ru.rushing_yards
    a = ru.groupby("player_id")[["wc", "wy"]].sum()
    pa = st[st.attempts > 0].sort_values(["season", "week"]).reset_index(drop=True)
    pa["w"] = recency(pa, "player_id", HL_EFF, season)
    pa["wa"] = pa.w * pa.attempts; pa["wp"] = pa.w * pa.passing_yards
    b = pa.groupby("player_id")[["wa", "wp"]].sum()
    for pid, r in a.iterrows():
        rows.setdefault(pid, {}).update(w_carries=float(r.wc), w_rush_yds=float(r.wy))
    for pid, r in b.iterrows():
        rows.setdefault(pid, {}).update(w_att=float(r.wa), w_pass_yds=float(r.wp))
    return rows


def team_shares(st, tg, season):
    """Per (player, team): decayed carries and team carries over the games he played for that
    team (half-life 4 team games), plus decayed receiving yards per game there."""
    tg = tg.copy(); tg["ws"] = recency(tg, "team", HL_SHARE, season)
    pg = st.merge(tg[["game_id", "team", "carries", "ws"]].rename(columns={"carries": "team_c"}), on=["game_id", "team"], how="inner")
    pg["wc"] = pg.ws * pg.carries; pg["wC"] = pg.ws * pg.team_c; pg["wr"] = pg.ws * pg.receiving_yards
    return pg.groupby(["player_id", "team"])[["wc", "wC", "wr", "ws"]].sum()


def game_script(pred_path, season, week):
    """{team: expected margin} from the sides model's latest prediction of each game, clamped."""
    if not os.path.exists(pred_path):
        return {}
    p = pd.read_csv(pred_path, low_memory=False)
    p = p[(p.season == season) & (p.week == week)].drop_duplicates("game_id", keep="last")
    out = {}
    for r in p.itertuples():
        m = float(r.sp_margin) if pd.notna(r.sp_margin) else 0.0
        m = max(-GS_CAP, min(GS_CAP, m))
        out[W.norm_team(r.home)] = m; out[W.norm_team(r.away)] = -m
    return out


def load_stats(season, live=True):
    frames = [d for d in (M._read("stats", y, live) for y in (season - 1, season)) if d is not None and len(d)]
    st = pd.concat(frames, ignore_index=True)
    st = st[st.season_type == "REG"].copy()
    st["team"] = st.team.map(W.norm_team); st["opponent_team"] = st.opponent_team.map(W.norm_team)
    for c in ("carries", "rushing_yards", "attempts", "passing_yards", "receiving_yards"):
        st[c] = pd.to_numeric(st[c], errors="coerce").fillna(0.0)
    return st


def load_roster(season):
    """Every active QB, RB, FB, WR and TE on the latest weekly roster, with his current team."""
    r = M._get(f"{M.NV}/weekly_rosters/roster_weekly_{season}.parquet")
    if r is None or r.status_code != 200 or len(r.content) < 100:
        sys.exit("could not download the nflverse weekly roster")
    ro = pd.read_parquet(io.BytesIO(r.content))
    ro = ro[(ro.week == ro.week.max()) & ro.position.isin(POSITIONS) & (ro.status == "ACT")].copy()
    ro["team"] = ro.team.map(W.norm_team)
    return ro[["gsis_id", "full_name", "position", "team"]].drop_duplicates("gsis_id").reset_index(drop=True)


def load_depth(season):
    """Latest depth chart per team: best pos_rank per player."""
    r = M._get(f"{M.NV}/depth_charts/depth_charts_{season}.parquet")
    if r is None or r.status_code != 200 or len(r.content) < 100:
        print("depth chart unavailable: starters fall back to last game's passer, shares to the default prior")
        return pd.DataFrame(columns=["team", "gsis_id", "pos_abb", "pos_rank"])
    dc = pd.read_parquet(io.BytesIO(r.content))
    dc["dt"] = pd.to_datetime(dc.dt, utc=True, errors="coerce")
    dc = dc[dc.dt == dc.groupby("team").dt.transform("max")].copy()
    dc["team"] = dc.team.map(W.norm_team); dc["pos_rank"] = pd.to_numeric(dc.pos_rank, errors="coerce")
    dc = dc[dc.pos_abb.isin(POSITIONS)].sort_values("pos_rank")
    return dc.drop_duplicates(["team", "gsis_id", "pos_abb"])[["team", "gsis_id", "pos_abb", "pos_rank"]]


def injury_status(season, week, roster):
    """{gsis_id: status}: the nflverse report for the week, then ESPN by name and team on top."""
    status = {}
    inj = M._read("injuries", season, True)
    if inj is not None and len(inj):
        for r in inj[inj.week == week].itertuples():
            if pd.notna(r.report_status):
                status[r.gsis_id] = str(r.report_status)
    espn, _ = W.espn_injuries()
    for r in roster.itertuples():
        s = espn.get((W.norm_name(r.full_name), r.team))
        if s:
            status[r.gsis_id] = s
    return status


def pick_starter(team, roster, depth, status, st):
    """The team's expected starting QB: depth chart order minus anyone ruled out or off the
    active roster; failing that, the active QB with the most attempts in the team's last game."""
    active = set(roster[(roster.team == team) & (roster.position == "QB")].gsis_id)
    ok = lambda g: g in active and status.get(g) not in OUT_TAGS
    for g in depth[(depth.team == team) & (depth.pos_abb == "QB")].sort_values("pos_rank").gsis_id:
        if ok(g):
            return g, "depth_chart"
    q = st[(st.team == team) & (st.attempts > 0)].sort_values(["season", "week", "attempts"], ascending=[False, False, False])
    for g in q.player_id:
        if ok(g):
            return g, "last_game"
    return None, "none"


def project(st, roster, depth, status, slate, season, week, margins):
    """One row per projected player in the slate. st must hold only games before the week."""
    tg = team_games(st)
    vol, _, _ = team_volume(tg, season)
    dfn = defense_rates(tg, season)
    rates = player_rates(st, season)
    sh = team_shares(st, tg, season)
    rank = {(r.gsis_id, r.pos_abb): r.pos_rank for r in depth.itertuples()}
    rows = []
    for g in slate.itertuples():
        tm, opp = g.team, g.opponent_team
        if tm not in vol.index:
            continue
        m = margins.get(tm, 0.0)
        t_car = float(vol.at[tm, "team_carries"]) * (1 + GS_RUSH * m)
        t_att = float(vol.at[tm, "team_att"]) * (1 - GS_PASS * m)
        ypc_m = float(dfn.at[opp, "opp_ypc_mult"]) if opp in dfn.index else 1.0
        ypa_m = float(dfn.at[opp, "opp_ypa_mult"]) if opp in dfn.index else 1.0
        qb, how = pick_starter(tm, roster, depth, status, st)
        cand = []
        for r in roster[roster.team == tm].itertuples():
            pos = r.position
            if pos == "QB" and r.gsis_id != qb:
                continue
            rk = rank.get((r.gsis_id, pos))
            has_rank = rk is not None and rk == rk          # NaN-safe
            prior = SHARE_PRIOR[("QB", 1)] if pos == "QB" else SHARE_PRIOR.get((pos, int(rk)) if has_rank else (pos, 0), SHARE_PRIOR_DEFAULT)
            h = sh.loc[(r.gsis_id, tm)] if (r.gsis_id, tm) in sh.index else None
            share = shrink(float(h.wc), float(h.wC), prior, SHARE_TAU) if h is not None else prior
            rec = float(h.wr) / (float(h.ws) + 1.0) if h is not None else 0.0
            cand.append(dict(gsis_id=r.gsis_id, name=r.full_name, pos=pos, depth_rank=rk, inj=status.get(r.gsis_id, ""),
                             share_hist=share, rec_yds_simple=rec, starter=(r.gsis_id == qb)))
        if not cand:
            continue
        c = pd.DataFrame(cand)
        out = c.inj.isin(OUT_TAGS)
        qb_share = float(c.loc[c.starter, "share_hist"].sum())
        non = (c.pos != "QB") & ~out
        tot = float(c.loc[non, "share_hist"].sum())
        c["share"] = 0.0
        c.loc[c.starter, "share"] = qb_share
        if tot > 0:
            c.loc[non, "share"] = c.loc[non, "share_hist"] * (NONQB_FILL * (1 - qb_share) / tot)
        c["share"] = c.share.clip(upper=SHARE_CAP)
        for r in c[~out].itertuples():
            pr = rates.get(r.gsis_id, {})
            ypc = shrink(pr.get("w_rush_yds", 0.0), pr.get("w_carries", 0.0), YPC_PRIOR.get(r.pos, 4.3), YPC_TAU)
            carries = t_car * r.share
            row = dict(season=season, week=week, team=tm, opp=opp, kickoff_utc=g.kickoff_utc, game_status=g.game_status,
                       gsis_id=r.gsis_id, name=r.name, pos=r.pos, depth_rank=r.depth_rank, inj=r.inj, starter=bool(r.starter),
                       starter_source=how if r.starter else "", margin=round(m, 2), team_carries=round(t_car, 2),
                       carry_share=round(r.share, 4), proj_carries=round(carries, 2), ypc=round(ypc, 3),
                       opp_ypc_mult=round(ypc_m, 3), mean_rush_yds=round(carries * ypc * ypc_m, 2),
                       team_att=np.nan, proj_att=np.nan, ypa=np.nan, opp_ypa_mult=np.nan, mean_pass_yds=np.nan,
                       rec_yds_simple=round(r.rec_yds_simple, 2))
            if r.starter:
                ypa = shrink(pr.get("w_pass_yds", 0.0), pr.get("w_att", 0.0), YPA_PRIOR, YPA_TAU)
                att = t_att * QB_ATT_SHARE
                row.update(team_att=round(t_att, 2), proj_att=round(att, 2), ypa=round(ypa, 3), opp_ypa_mult=round(ypa_m, 3),
                           mean_pass_yds=round(att * ypa * ypa_m, 2))
            elif carries < MIN_CARRIES:
                continue
            rows.append(row)
    return pd.DataFrame(rows, columns=GRID_COLS)


def unit_tests():
    # recency: newest row of each key weighs 1, one half-life back weighs 0.5, last season x carry
    d = pd.DataFrame(dict(team=["A"] * 3 + ["B"], season=[2025, 2026, 2026, 2026]))
    w = recency(d, "team", 1.0, 2026)
    assert np.allclose(w, [0.25 * SEASON_CARRY, 0.5, 1.0, 1.0]), w
    assert abs(shrink(10.0, 2.0, 4.0, 2.0) - 4.5) < 1e-9
    # a toy league: team X runs 30 times a game, its RB1 takes 20; the RB1 is ruled out this week
    # and the backup inherits the non-QB carries; the QB keeps his own share
    games = []
    for wk in range(1, 5):
        for tm, opp in (("X", "Y"), ("Y", "X")):
            gid = f"g{wk}"
            games += [dict(season=2026, week=wk, game_id=gid, team=tm, opponent_team=opp, player_id=f"{tm}rb1", position="RB",
                           carries=20, rushing_yards=90, attempts=0, passing_yards=0, receiving_yards=10),
                      dict(season=2026, week=wk, game_id=gid, team=tm, opponent_team=opp, player_id=f"{tm}rb2", position="RB",
                           carries=6, rushing_yards=24, attempts=0, passing_yards=0, receiving_yards=5),
                      dict(season=2026, week=wk, game_id=gid, team=tm, opponent_team=opp, player_id=f"{tm}qb", position="QB",
                           carries=4, rushing_yards=20, attempts=32, passing_yards=224, receiving_yards=0)]
    st = pd.DataFrame(games)
    ro = pd.DataFrame(dict(gsis_id=["Xrb1", "Xrb2", "Xqb", "Yrb1", "Yrb2", "Yqb"], full_name=list("abcdef"),
                           position=["RB", "RB", "QB"] * 2, team=["X"] * 3 + ["Y"] * 3))
    dp = pd.DataFrame(dict(team=["X", "X", "X"], gsis_id=["Xqb", "Xrb1", "Xrb2"], pos_abb=["QB", "RB", "RB"], pos_rank=[1, 1, 2]))
    sl = pd.DataFrame(dict(team=["X", "Y"], opponent_team=["Y", "X"], kickoff_utc=["k", "k"], game_status=["scheduled"] * 2))
    p = project(st, ro, dp, {"Xrb1": "Out"}, sl, 2026, 5, {"X": 0.0, "Y": 0.0}).set_index("gsis_id")
    assert "Xrb1" not in p.index and p.at["Xqb", "starter"] and abs(p.at["Xqb", "team_carries"] - 30) < 1e-6
    nonqb = 30 * NONQB_FILL * (1 - p.at["Xqb", "carry_share"])
    assert abs(p.at["Xrb2", "proj_carries"] - min(nonqb, 30 * SHARE_CAP)) < 0.02, p.at["Xrb2", "proj_carries"]
    assert abs(p.at["Xqb", "proj_att"] - 32 * QB_ATT_SHARE) < 1e-6 and abs(p.at["Yqb", "proj_carries"] - p.at["Xqb", "proj_carries"]) < 1e-6
    assert abs(p.at["Yrb1", "proj_carries"] + p.at["Yrb2", "proj_carries"] - 30 * NONQB_FILL * (1 - p.at["Yqb", "carry_share"])) < 0.02
    # game script moves volume the right way
    q = project(st, ro, dp, {}, sl, 2026, 5, {"X": 7.0, "Y": -7.0}).set_index("gsis_id")
    assert q.at["Xqb", "team_carries"] > 30 > q.at["Yqb", "team_carries"] and q.at["Xqb", "team_att"] < 32 < q.at["Yqb", "team_att"]
    return True


def main():
    unit_tests(); print("unit tests: PASS")
    now = pd.Timestamp.now(tz="UTC")
    g = W.schedule()
    season = int(g[g.kickoff_utc <= now + pd.Timedelta(days=7)].season.max()) if (g.kickoff_utc <= now + pd.Timedelta(days=7)).any() else int(g.season.max())
    gs = g[g.season == season]
    week = int(os.environ["PROPS_WEEK"]) if os.environ.get("PROPS_WEEK") else W.target_week(gs, now)
    slate, w = W.slate_for(gs, week, now)
    W.assert_slate(slate, w)
    if os.environ.get("PROPS_WINDOW_H"):
        # the Thursday run prices only the games in its window; the Saturday-night run does the rest
        horizon = now + pd.Timedelta(hours=float(os.environ["PROPS_WINDOW_H"]))
        slate = slate[pd.to_datetime(slate.kickoff_utc, utc=True) <= horizon]
        print(f"window {os.environ['PROPS_WINDOW_H']}h: {slate.game_id.nunique()} game(s)")
    st = load_stats(season)
    st = st[(st.season < season) | (st.week < week)]
    roster = load_roster(season)
    depth = load_depth(season)
    status = injury_status(season, week, roster)
    margins = game_script(os.path.join(HERE, "logs", "predictions.csv"), season, week)
    print(f"{season} week {week}: {len(w)} games; stats through week {int(st[st.season == season].week.max())}; "
          f"{len(roster)} active QB/RB/FB/WR/TE; {sum(1 for s in status.values() if s in OUT_TAGS)} ruled out; "
          f"model margins for {len(margins)} teams")
    out = project(st, roster, depth, status, slate, season, week, margins)
    out["ts"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    qbs = out[out.starter.astype(bool)]
    missing = sorted(set(slate.team) - set(qbs.team))
    if missing:
        print(f"WARNING no starting QB found for {missing}")
    fallback = qbs[qbs.starter_source != "depth_chart"]
    if len(fallback):
        print("starter from the last game, not the depth chart: " + ", ".join(f"{r.team} {r.name}" for r in fallback.itertuples()))
    path = os.path.join(HERE, "logs", "rushpass_grid.csv"); os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    live = out[out.game_status == "scheduled"]
    print(f"rush/pass grid: {len(out)} players ({int(out.starter.sum())} starting QBs) -> logs/rushpass_grid.csv  [UNTESTED]")
    print("starting QBs (scheduled games): mean pass yds | mean rush yds")
    for r in live[live.starter.astype(bool)].sort_values("mean_pass_yds", ascending=False).itertuples():
        print(f"  {r.team:<3} {r.name:<24} {r.mean_pass_yds:6.1f} | {r.mean_rush_yds:5.1f}{'  (' + r.inj + ')' if r.inj else ''}")
    print("top rushers (scheduled games): carries x ypc = mean rush yds")
    for r in live[~live.starter.astype(bool)].sort_values("mean_rush_yds", ascending=False).head(24).itertuples():
        print(f"  {r.team:<3} {r.name:<24} {r.pos:<2} {r.proj_carries:5.1f} x {r.ypc:4.2f} = {r.mean_rush_yds:5.1f}{'  (' + r.inj + ')' if r.inj else ''}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--test", action="store_true"); a = ap.parse_args()
    if a.test:
        unit_tests(); print("unit tests: PASS")
    else:
        main()
