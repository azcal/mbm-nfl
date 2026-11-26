"""
props_model.py: bottom-up NFL receiving props (receptions, receiving yards).

Per player-game, before kickoff:
  team pass volume     decayed team targets per game (hl 6 games), season carry, opponent adjust
  target share         decayed share of team targets (hl H games), reset on team change, shrunk to a
                       position prior; Out/Doubtful on the week's injury report removed and the
                       remaining shares renormalised (proportional redistribution)
  catch rate, ypc      decayed, shrunk to position priors
  simulation           attempts ~ N(proj, 6); targets ~ BetaBinomial(att, share, kappa);
                       receptions ~ Binomial(targets, cr); yards ~ sum of Gamma(alpha, ypc/alpha)
Outputs P(yards > X) on a grid, P(receptions >= k), median, mean.

Walk-forward: states only ever see completed games; the injury report used for week w is the
one filed for week w. Tune (kappa, alpha, H) on 2020-2022, score 2023-2025 once.
"""
import re
import numpy as np, pandas as pd, os, math, json, sys
from collections import defaultdict

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
POS = {"WR", "TE", "RB", "FB"}
PRIOR_CR = {"WR": 0.63, "TE": 0.70, "RB": 0.76, "FB": 0.72}
PRIOR_YPC = {"WR": 12.8, "TE": 10.6, "RB": 7.8, "FB": 6.5}
PRIOR_SHARE = {"WR": 0.02, "TE": 0.02, "RB": 0.02, "FB": 0.005}
POOL = 0.96          # share of team targets that goes to tracked, active players
PRIOR_K = 6          # player prior: w = n_games_last_season / (n_games_last_season + PRIOR_K)
K_UPDATE = 5         # in-season: w_data = n_games_this_team_this_season / (n_games + K_UPDATE)
DEPTH_PRIOR = {("WR", 1): 0.21, ("WR", 2): 0.15, ("WR", 3): 0.10, ("WR", 4): 0.04,
               ("TE", 1): 0.13, ("TE", 2): 0.04, ("RB", 1): 0.09, ("RB", 2): 0.03, ("FB", 1): 0.02}
SEASON_CARRY = 0.6   # share memory kept across the offseason
DEPTH_EVIDENCE = 3.0 # below this many weighted games of history, the depth-chart prior applies; above it the player's own record does
REMOVE_STATUSES = ("Out", "Doubtful", "Injured Reserve", "IR", "PUP", "Suspended", "Suspension")
RENORMALIZE_SHARES = True   # when a player is removed, the remaining players' shares are rescaled so team volume is conserved
GS_BETA = 0.0        # game-script hook; tested on 2024-25 with the sides model margins, no gain, left off
YGRID = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 120], float)
RGRID = np.arange(1, 11)
RNG = np.random.default_rng(7)


class Dec:
    __slots__ = ("d", "sw", "sx")
    def __init__(self, hl): self.d = 0.5 ** (1 / hl); self.sw = 0.0; self.sx = 0.0
    def add(self, x, w=1.0): self.sw = self.sw * self.d + w; self.sx = self.sx * self.d + w * x
    def brk(self, k): self.sw *= k; self.sx *= k
    def mean(self, prior, tau): return (self.sx + prior * tau) / (self.sw + tau)


NV = "https://github.com/nflverse/nflverse-data/releases/download"


def _get(url, tries=3):
    """One download with retries; GitHub release downloads reset connections now and then."""
    import requests, time
    for i in range(tries):
        try:
            return requests.get(url, timeout=120)
        except requests.exceptions.RequestException as ex:
            print(f"  download failed ({type(ex).__name__}), retry {i + 1}/{tries}: {url.split('/')[-1]}")
            time.sleep(5 * (i + 1))
    return None


def _read(kind, y, live):
    """kind in stats/injuries/snap_counts. Live mode downloads from nflverse; a missing or
    empty season (2026 before it has data) is skipped."""
    import io, requests
    url = {"stats": f"{NV}/stats_player/stats_player_week_{y}.parquet",
           "injuries": f"{NV}/injuries/injuries_{y}.parquet",
           "snap_counts": f"{NV}/snap_counts/snap_counts_{y}.parquet"}[kind]
    if live:
        r = _get(url)
        if r is None or r.status_code != 200 or len(r.content) < 100:
            return None
        return pd.read_parquet(io.BytesIO(r.content))
    p = f"{DATA}/{'stats_player_week' if kind == 'stats' else kind}_{y}.parquet"
    return pd.read_parquet(p) if os.path.exists(p) else None


def load(seasons, live=False):
    frames = {"stats": [], "injuries": [], "snap_counts": []}
    for y in seasons:
        for k in frames:
            d = _read(k, y, live)
            if d is not None and len(d):
                frames[k].append(d)
    st = pd.concat(frames["stats"], ignore_index=True)
    st = st[(st.season_type == "REG") & st.position.isin(POS)].copy()
    inj = pd.concat(frames["injuries"], ignore_index=True) if frames["injuries"] else pd.DataFrame(columns=["season", "week", "gsis_id", "report_status"])
    inj = inj[inj.report_status.isin(["Out", "Doubtful"])]
    out_set = set(zip(inj.season, inj.week, inj.gsis_id))
    if frames["snap_counts"]:
        sc = pd.concat(frames["snap_counts"], ignore_index=True)
        sc = sc[(sc.game_type == "REG") & (sc.offense_snaps > 0)]
        played_names = set(zip(sc.season, sc.week, sc.team, sc.player.str.lower().str.replace(r"[^a-z]", "", regex=True)))
    else:
        played_names = set()
    return st, out_set, played_names


def simulate(att_mu, share, cr, ypc, kappa, alpha, n=1500):
    att = np.clip(np.round(RNG.normal(att_mu, 6.0, n)), 12, 60).astype(int)
    p = RNG.beta(max(share, 1e-4) * kappa, max(1 - share, 1e-4) * kappa, n)
    tg = RNG.binomial(att, p)
    rc = RNG.binomial(tg, cr)
    # sum of k Gamma(alpha, scale) draws is Gamma(k*alpha, scale): one vectorised draw
    y = np.where(rc > 0, RNG.gamma(np.maximum(alpha * rc, 1e-6), ypc / alpha), 0.0)
    return tg, rc, y


def load_roster(season, live=True):
    """nflverse weekly roster for the season: who is on which team, with status. Used only
    for an upcoming slate, so offseason moves, cuts and rookies are reflected."""
    import io, requests
    url = f"{NV}/weekly_rosters/roster_weekly_{season}.parquet"
    if live:
        r = requests.get(url, timeout=120)
        if r.status_code != 200 or len(r.content) < 100:
            return None
        ro = pd.read_parquet(io.BytesIO(r.content))
    else:
        p = f"{DATA}/roster_weekly_{season}.parquet"
        if not os.path.exists(p):
            return None
        ro = pd.read_parquet(p)
    ro = ro[ro.position.isin(POS) & (ro.status == "ACT")]
    ro = ro[["season", "week", "team", "gsis_id", "position", "full_name"]].copy()
    # latest depth chart: pos_rank per player (1 = starter)
    try:
        if live:
            r = _get(f"{NV}/depth_charts/depth_charts_{season}.parquet")
            dc = pd.read_parquet(io.BytesIO(r.content)) if r is not None and r.status_code == 200 and len(r.content) > 100 else None
        else:
            p = f"{DATA}/depth_charts_{season}.parquet"; dc = pd.read_parquet(p) if os.path.exists(p) else None
        if dc is not None and len(dc):
            dc["dt"] = pd.to_datetime(dc.dt); dc = dc[dc.dt == dc.dt.max()]
            dc = dc[dc.pos_abb.isin(POS)][["gsis_id", "pos_rank"]].drop_duplicates("gsis_id")
            dc["pos_rank"] = pd.to_numeric(dc.pos_rank, errors="coerce")
            ro = ro.merge(dc, on="gsis_id", how="left")
    except Exception:
        pass
    if "pos_rank" not in ro.columns:
        ro["pos_rank"] = np.nan
    return ro


def run(seasons, kappa=30.0, alpha=1.6, H=4.0, min_proj_targets=3.0, verbose=False, upcoming=None, live=False, vol_mult=None, roster=None, script=None, inj_status=None,
        replay=None):
    """script: {(season, week, team): expected margin for that team} from the sides model;
    trailing teams throw more. inj_status: {(season, week, gsis_id): 'Questionable'|'Doubtful'|'Out'}."""
    """upcoming: DataFrame(season, week, team, opponent_team) for a slate that has not been
    played; predictions are made for it from states through the last completed week and no
    update runs on it. vol_mult: {"WR|lo": 1.12, ...} volume multipliers from the spec."""
    st, out_set, played_names = load(seasons, live=live)
    # PLAYER PRIOR (Phase 2, item 9): last season's target share per player, averaged over the
    # games he played, and how many games that was. A player who changed teams keeps this.
    tt = st.groupby(["season", "week", "team"]).targets.sum().rename("team_tg")
    ps = st[st.targets.notna()].merge(tt, on=["season", "week", "team"], how="left")
    ps = ps[ps.team_tg > 0]; ps["sh"] = ps.targets / ps.team_tg
    last_share = ps.groupby(["season", "player_id"]).agg(sh=("sh", "mean"), n=("sh", "size"))
    LAST = {(int(k[0]) + 1, k[1]): (float(v.sh), int(v.n)) for k, v in last_share.iterrows()}   # keyed to the NEXT season
    games_on_team = {}     # (pid, team, season) -> games played there this season, for K_UPDATE
    if upcoming is not None and len(upcoming):
        dummy = upcoming.copy(); dummy["season_type"] = "REG"; dummy["player_id"] = None; dummy["player_display_name"] = ""
        dummy["position"] = "WR"; dummy["targets"] = 0; dummy["receptions"] = 0; dummy["receiving_yards"] = 0.0
        st = pd.concat([st[~((st.season == dummy.season.iloc[0]) & (st.week == dummy.week.iloc[0]))], dummy], ignore_index=True)
        stop_at = (int(dummy.season.iloc[0]), int(dummy.week.iloc[0]))
    else:
        stop_at = None
    st["nm"] = st.player_display_name.str.lower().str.replace(r"[^a-z]", "", regex=True)
    name_of = {}
    st = st.sort_values(["season", "week"])
    team_att = defaultdict(lambda: Dec(6.0)); opp_allowed = defaultdict(lambda: Dec(8.0))
    share = {}; cr = {}; ypc = {}; last_team = {}; last_seen = {}; pos_of = {}
    season_of = {}
    rows = []
    for (season, week), wk in st.groupby(["season", "week"], sort=True):
        team_tot = wk.groupby("team").targets.sum()
        lg_mean = team_tot.mean() if len(team_tot) else 33.0
        # ---------- predictions for this week (states are pregame) ----------
        for team, g in wk.groupby("team"):
            opp = g.opponent_team.iloc[0]
            if season_of.get(team) != season:
                team_att[team].brk(0.5); season_of[team] = season
            ta = team_att[team].mean(33.0, 3.0)
            oa = opp_allowed[opp].mean(33.0, 3.0)
            att_mu = ta * (oa / 33.0) ** 0.5
            if script is not None and (season, week, team) in script:
                att_mu *= float(np.clip(1 - GS_BETA * script[(season, week, team)], 0.88, 1.12))
            # active roster: players with a share state on this team, seen in one of the team's
            # last 3 games, not Out/Doubtful this week
            prior_of = {}; prior_src = {}
            if "disp_of" not in dir():
                disp_of = {}
            ro = None
            if roster is not None and stop_at is not None and (int(season), int(week)) == stop_at:
                # upcoming slate: the current roster decides who is on the team. Players with no
                # history (rookies, first-timers) get the position prior; players who moved
                # teams bring their share state, shrunk toward the prior.
                wk_avail = roster[roster.week <= week]
                ro = wk_avail[wk_avail.week == wk_avail.week.max()] if len(wk_avail) else roster[roster.week == roster.week.min()]
                ro = ro[ro.team == team]
            elif replay is not None and (int(season), int(week), team) in replay:
                # backtest only: {(season, week, team): roster rows as they stood before that
                # week}, so past weeks are projected through the same roster path the live
                # run uses instead of the history-only path below.
                ro = replay[(int(season), int(week), team)]
            if ro is not None:
                cands = []; removed = []
                for rr in ro.itertuples():
                    pid = rr.gsis_id
                    if pid not in share:
                        share[pid] = Dec(H); cr[pid] = Dec(30.0); ypc[pid] = Dec(30.0)
                    pos_of[pid] = rr.position; name_of[pid] = re.sub(r"[^a-z]", "", str(rr.full_name).lower()); last_team[pid] = team
                    disp_of[pid] = str(rr.full_name)
                    rk = rr.pos_rank if hasattr(rr, "pos_rank") and rr.pos_rank == rr.pos_rank else None
                    slot = DEPTH_PRIOR.get((rr.position, int(min(rk, 4 if rr.position == "WR" else 2))), PRIOR_SHARE[rr.position]) if rk is not None else PRIOR_SHARE[rr.position]
                    ls = LAST.get((int(season), pid))
                    w_prior = (ls[1] / (ls[1] + PRIOR_K)) if ls else 0.0
                    prior_of[pid] = w_prior * (ls[0] if ls else 0.0) + (1 - w_prior) * slot
                    prior_src[pid] = f"w_last={w_prior:.2f} last={ls[0]:.3f} ({ls[1]} gp)" if ls else "slot only (no last season)"
                    # STATUS STEP: Out/Doubtful/IR/PUP/Suspended are removed from the active set. Their
                    # rows are still written with proj_targets = 0 and status_applied = True.
                    st_ = (inj_status or {}).get((season, week, pid), "")
                    if (season, week, pid) in out_set or st_ in REMOVE_STATUSES:
                        removed.append(pid); continue
                    cands.append(pid)
            else:
                cands = [pid for pid, tm in last_team.items()
                         if tm == team and (season, week, pid) not in out_set
                         and (last_seen.get(pid, (0, 0)) >= (season, week - 2)
                              or (week <= 2 and last_seen.get(pid, (0, 0))[0] == season - 1 and last_seen[pid][1] >= 15))]
                for pid in cands:
                    ls = LAST.get((int(season), pid)); w_prior = (ls[1] / (ls[1] + PRIOR_K)) if ls else 0.0
                    prior_of[pid] = w_prior * (ls[0] if ls else 0.0) + (1 - w_prior) * PRIOR_SHARE[pos_of[pid]]
            shares = {}; blend_of = {}
            for pid in cands:
                # UPDATE RULE (Phase 2, item 10): one shrinkage rule for everyone.
                n_g = games_on_team.get((pid, team, int(season)), 0)
                w_data = n_g / (n_g + K_UPDATE)
                obs = (share[pid].sx / share[pid].sw) if share[pid].sw > 0 else prior_of.get(pid, PRIOR_SHARE[pos_of[pid]])
                pr = prior_of.get(pid, PRIOR_SHARE[pos_of[pid]])
                s_ = w_data * obs + (1 - w_data) * pr
                shares[pid] = s_; blend_of[pid] = (pr, obs, w_data, s_)
            if roster is not None and not RENORMALIZE_SHARES and "removed" in dir():
                # toggle off: removed players' shares stay in the denominator, so their targets vanish
                # from the team rather than flowing to the remaining players
                tot = max(sum(shares.values()) + sum(prior_of.get(p, PRIOR_SHARE[pos_of[p]]) for p in removed), 1e-6)
            else:
                tot = max(sum(shares.values()), 1e-6)
            for pid, s_ in shares.items():
                sh = s_ / tot * POOL
                if vol_mult:
                    b = "lo" if att_mu * sh < 7 else "hi"
                    sh = min(sh * float(vol_mult.get(f"{pos_of[pid]}|{b}", 1.0)), 0.6)
                pt = att_mu * sh
                if pt < min_proj_targets:
                    continue
                c = cr[pid].mean(PRIOR_CR[pos_of[pid]], 20.0); y_ = ypc[pid].mean(PRIOR_YPC[pos_of[pid]], 15.0)
                tg, rc, yds = simulate(att_mu, sh, c, y_, kappa, alpha)
                act = g[g.player_id == pid]
                on_field = len(act) > 0 or (season, week, team, name_of.get(pid, "")) in played_names
                rows.append(dict(season=season, week=week, team=team, opp=opp, pid=pid, pos=pos_of[pid], name=name_of.get(pid, ""), display_name=disp_of.get(pid, ""),
                                 inj=(inj_status or {}).get((season, week, pid), ""), status_applied=False, depth_prior=prior_of.get(pid),
                                 prior_share=round(blend_of[pid][0], 4), obs_share=round(blend_of[pid][1], 4), w_data=round(blend_of[pid][2], 3),
                                 blend_share=round(blend_of[pid][3], 4), final_share=round(sh, 4), n_games_team=games_on_team.get((pid, team, int(season)), 0),
                                 prior_src=prior_src.get(pid, ""), att_mu=round(att_mu, 2),
                                 proj_targets=round(pt, 2), share=round(sh, 3), cr=round(c, 3), ypc=round(y_, 2),
                                 med_yds=float(np.median(yds)), mean_yds=float(yds.mean()), med_rec=float(np.median(rc)),
                                 p_over=[float(np.mean(yds > x)) for x in YGRID],
                                 p_rec=[float(np.mean(rc >= k)) for k in RGRID],
                                 played=on_field,
                                 act_tg=int(act.targets.iloc[0]) if len(act) else (0 if on_field else None),
                                 act_rec=int(act.receptions.iloc[0]) if len(act) else (0 if on_field else None),
                                 act_yds=float(act.receiving_yards.iloc[0]) if len(act) else (0.0 if on_field else None)))
        if stop_at is not None and (int(season), int(week)) == stop_at:
            # rows for players removed by status, so the grid shows them with zeros
            if roster is not None:
                wk_avail = roster[roster.week <= week]
                ro_all = wk_avail[wk_avail.week == wk_avail.week.max()] if len(wk_avail) else roster[roster.week == roster.week.min()]
                for team, g in wk.groupby("team"):
                    opp = g.opponent_team.iloc[0]
                    for rr in ro_all[ro_all.team == team].itertuples():
                        pid = rr.gsis_id; st_ = (inj_status or {}).get((season, week, pid), "")
                        if (season, week, pid) in out_set or st_ in REMOVE_STATUSES:
                            rows.append(dict(season=season, week=week, team=team, opp=opp, pid=pid, pos=rr.position,
                                             name=re.sub(r"[^a-z]", "", str(rr.full_name).lower()), inj=st_ or "Out", status_applied=True,
                                             depth_prior=None, proj_targets=0.0, share=0.0, cr=0.0, ypc=0.0, med_yds=0.0, mean_yds=0.0, med_rec=0.0,
                                             played=False, act_tg=None, act_rec=None, act_yds=None,
                                             p_over=[0.0] * len(YGRID), p_rec=[0.0] * len(RGRID)))
            break
        # ---------- updates after the week ----------
        for team, g in wk.groupby("team"):
            opp = g.opponent_team.iloc[0]; tt = float(team_tot[team])
            team_att[team].add(tt); opp_allowed[opp].add(tt)
            for r in g.itertuples():
                pid = r.player_id
                if pid not in share or last_team.get(pid) != team:
                    share[pid] = Dec(H); cr[pid] = Dec(30.0); ypc[pid] = Dec(30.0)
                    if pid in share and last_team.get(pid) != team and last_team.get(pid) is not None:
                        pass
                pos_of[pid] = r.position; last_team[pid] = team; last_seen[pid] = (season, week); name_of[pid] = r.nm
                games_on_team[(pid, team, int(season))] = games_on_team.get((pid, team, int(season)), 0) + 1
                share[pid].add(r.targets / tt if tt > 0 else 0.0)
                if r.targets > 0:
                    cr[pid].add(r.receptions / r.targets, w=r.targets)
                if r.receptions > 0:
                    ypc[pid].add(r.receiving_yards / r.receptions, w=r.receptions)
        if week == 1:      # new season: soften every player's share memory
            for pid in share: share[pid].brk(SEASON_CARRY)
    df = pd.DataFrame(rows)
    return df


def calibration(df, label=""):
    d = df[df.played].copy()
    lines = []
    P = np.array(d.p_over.tolist()); Y = d.act_yds.values[:, None] > YGRID[None, :]
    p = P.ravel(); y = Y.ravel().astype(float)
    ll = -np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
    lines.append(f"{label} receiving yards: {len(d)} player-games, {len(p)} over/under claims, log loss {ll:.4f}")
    lines.append(f"  {'band':>10} {'n':>6} {'said':>6} {'actual':>7} {'gap':>7}")
    for lo, hi in ((0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)):
        m = (p >= lo) & (p < hi)
        if m.sum() < 50: continue
        lines.append(f"  {lo:.1f}-{min(hi,1):.1f}   {m.sum():>6} {p[m].mean():>6.3f} {y[m].mean():>7.3f} {y[m].mean()-p[m].mean():>+7.3f}")
    over_med = np.mean(d.act_yds.values > d.med_yds.values); push = np.mean(d.act_yds.values == d.med_yds.values)
    lines.append(f"  actual over the model median: {over_med:.3f} (target 0.500; exact {push:.3f})")
    R = np.array(d.p_rec.tolist()); YR = d.act_rec.values[:, None] >= RGRID[None, :]
    pr = R.ravel(); yr = YR.ravel().astype(float)
    llr = -np.mean(yr * np.log(np.clip(pr, 1e-6, 1)) + (1 - yr) * np.log(np.clip(1 - pr, 1e-6, 1)))
    lines.append(f"{label} receptions: log loss {llr:.4f}")
    for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
        m = (pr >= lo) & (pr < hi)
        if m.sum() < 50: continue
        lines.append(f"  {lo:.1f}-{min(hi,1):.1f}   {m.sum():>6} {pr[m].mean():>6.3f} {yr[m].mean():>7.3f} {yr[m].mean()-pr[m].mean():>+7.3f}")
    return "\n".join(lines), ll, llr


if __name__ == "__main__":
    df = run(range(2019, 2023), verbose=True)
    print(calibration(df[df.season >= 2020], "tune 2020-22")[0])
