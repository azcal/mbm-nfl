"""
props_unabated.py: write Unabated's NFL projections upload from the published props grid.

    python props_unabated.py            # after props_weekly.py, same run
    python props_unabated.py --test     # unit tests only

Unabated treats every uploaded number as a MEAN and simulates from it: Poisson on catches,
a per-catch yardage shape we don't control. So each uploaded value is solved so that
Unabated's fair price at the posted line equals the model's price at that line:
  catches = Poisson mean with P(catches > rec_line) = model rec_p_over
  yards   = mean with P(yards > yds_line) = model yds_p_over, given those catches and the
            configured per-catch Gamma shape (config unabated_yds_shape, per position)
Uploading a median or a market line prices wrong; the grid means stay in the export log.

Rushing and passing (UNTESTED, added Nov 20, 2026): props_rushpass.py writes bottom-up means to
logs/rushpass_grid.csv and they go in as plain means, with no line solving and no market input:
rushing_yards, passing_yards (starting QB), passing_and_rushing_yards (QB) and
rushing_and_receiving_yards (everyone else). A missing or stale file leaves those columns blank
and never stops the receiving export. Their own log is export_log_rushpass.csv.

Inputs: logs/props_grid.csv (published, anchored), templates/projections-nfl.csv (Unabated's
template, read fresh, never edited), logs/props_board.csv (cross-check).
Outputs: logs/unabated/projections-nfl.csv, export_log.csv, export_meta.json. Written to temp
paths and moved into place only after every check passes; on failure the previous export is
deleted so a stale file can't be uploaded by mistake.

The template has to be current. Unabated's ids, names and teams change every week (trades,
signings, cuts), and an upload built on an old download carries players on old teams and ids
Unabated has since dropped. Refresh it by uploading a new download to templates/ under any
projections-nfl*.csv name: the next run swaps it in (see promote_template). Every run checks
the template against the grid's active receivers and says on Discord who doesn't line up.
"""
import argparse, datetime as dt, hashlib, json, math, os, re, shutil, sys, tempfile, unicodedata
import numpy as np, pandas as pd
from scipy import optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs"); OUT = os.path.join(LOGS, "unabated")
TEMPLATE_DIR = os.path.join(HERE, "templates")
TEMPLATE = os.path.join(TEMPLATE_DIR, "projections-nfl.csv")
# Unabated's download is projections-nfl.csv, or projections-nfl (14).csv / projections-nfl_14.csv
# once the downloads folder already holds copies
TEMPLATE_DROP = re.compile(r"^projections-nfl.*\.csv$", re.I)
TEMPLATE_NEED = ["id", "name", "position", "team", "receptions", "receiving_yards"]
CONFIG = os.path.join(HERE, "unabated_config.json")
DEFAULT_CFG = {"unabated_yds_shape": {"WR": 0.75, "TE": 0.75, "RB": 0.75}, "team_map": {"LA": "LAR"}, "alias": {},
               "positions": ["WR", "TE", "RB"], "roundtrip_tol_pts": 0.5, "board_tol_pts": 0.15,
               # below this many uploaded catches the per-catch compound puts a quarter or more of its
               # mass at zero yards and the solved yards mean overshoots Unabated's shape badly
               # (Thornton 30.0 on a 13.5 line read +18% on their screen). Catches still upload;
               # yards stay blank so Unabated falls back to its own number. Raise once section 7 is fit.
               "min_catches_for_yards": 2.5}
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
OUT_TAGS = ("Out", "Doubtful", "IR", "PUP", "Suspended")
EXPORTED = ["filled", "duplicate_filled", "filled_team_differs", "low_volume_yds_blank"]
# UNTESTED rushing/passing means from props_rushpass.py, uploaded as plain means (Nov 20, 2026)
RUSHPASS_COLS = ["rushing_yards", "passing_yards", "rushing_and_receiving_yards", "passing_and_rushing_yards"]
FILL_COLS = ["receptions", "receiving_yards"] + RUSHPASS_COLS
RUSH_RANGE, PASS_RANGE = (0.5, 200.0), (50.0, 450.0)


# --- conversion math (reference code from the spec, unchanged) --------------------------

def rec_upload_mean(line, p_over):
    """Poisson mean whose P(catches > line) equals the model's P(over) at that line."""
    need = math.floor(line) + 1
    return optimize.brentq(lambda lam: stats.poisson.sf(need - 1, lam) - p_over, 1e-6, 40.0)


def yds_sf(y, lam, shape, mean_yds):
    """P(yards > y) for Poisson(lam) catches, each catch Gamma(shape) yards, total mean = mean_yds."""
    theta = mean_yds / (lam * shape)
    n = np.arange(1, 80)
    return float(np.sum(stats.poisson.pmf(n, lam) * stats.gamma.sf(y, a=n * shape, scale=theta)))


def yds_upload_mean(line, p_over, lam, shape):
    """Mean yards that gives the model's P(over line), given the uploaded catches. None if unsolvable."""
    try:
        return optimize.brentq(lambda m: yds_sf(line, lam, shape, m) - p_over, 0.01, 400.0)
    except (ValueError, RuntimeError):
        return None


def yds_median(lam, shape, mean_yds):
    """Median yards of the same distribution. Needs lam > ln 2 so the median sits above zero."""
    return optimize.brentq(lambda y: yds_sf(y, lam, shape, mean_yds) - 0.5, 1e-6, 500.0)


def fit_shape(reads):
    """reads: [(catches, mean_yds, unabated_median), ...]. Returns the shape that best matches Unabated's medians."""
    loss = lambda lk: sum((yds_median(c, math.exp(lk), m) - med) ** 2 for c, m, med in reads)
    res = optimize.minimize_scalar(loss, bounds=(math.log(0.1), math.log(20.0)), method="bounded")
    return math.exp(res.x)


def unit_tests():
    assert round(rec_upload_mean(3.5, 0.659), 2) == 4.51
    assert round(rec_upload_mean(4.5, 0.479), 2) == 4.56
    assert round(yds_upload_mean(59.5, 0.504, 4.56, 0.75), 1) == 69.5
    assert yds_upload_mean(10.5, 0.80, 1.41, 0.75) is None
    assert round(yds_median(4.56, 0.75, 69.5), 1) == 60.0
    assert abs(fit_shape([(4.56, 69.5, 60.0), (1.41, 22.0, 11.5)]) - 0.75) < 0.02
    # template matching: exact, moved team (unique name), ambiguous, missing
    t = pd.DataFrame(dict(name=["Keenan Allen", "Mike Williams", "Mike Williams", "Trebor Peña", " Deshaun Fenwick"],
                          team=["SF", "LAC", "NYJ", "JAX", "NE"], position=["WR", "WR", "WR", "WR", "RB"]))
    tkeys = pd.Series([f"{match_key(n)}|{tm}|{p}" for n, tm, p in zip(t.name, t.team, t.position)]).to_frame("k").groupby("k").indices
    tnames = t.groupby([t.name.map(match_key), t.position]).indices
    assert find_rows("Keenan Allen", "SF", "WR", tkeys, tnames)[1] == ""
    assert find_rows("Keenan Allen", "IND", "WR", tkeys, tnames)[1] == "team_differs"
    assert list(find_rows("Keenan Allen", "IND", "WR", tkeys, tnames)[0]) == [0]
    assert find_rows("Mike Williams", "PIT", "WR", tkeys, tnames) == (None, "team_mismatch")
    assert find_rows("Mike Williams", "NYJ", "WR", tkeys, tnames)[1] == ""
    assert find_rows("Trebor Pena", "JAX", "WR", tkeys, tnames)[1] == ""          # accents fold
    assert find_rows("Deshaun Fenwick", "NE", "RB", tkeys, tnames)[1] == ""       # stray leading space
    assert find_rows("Keenan Allen", "SF", "TE", tkeys, tnames) == (None, "unmatched")
    # newest download wins, whatever the browser named it
    assert pick_newest(["projections-nfl_13.csv", "projections-nfl (14).csv", "projections-nfl (2).csv"]) == "projections-nfl (14).csv"
    assert TEMPLATE_DROP.match("projections-nfl (14).csv") and TEMPLATE_DROP.match("Projections-NFL_13.csv")
    assert not TEMPLATE_DROP.match("projections-nba.csv")
    # a filled file is refused as a template, a blank one passes
    d = tempfile.mkdtemp()
    hdr = "id,name,first_name,last_name,position,team,receptions,receiving_yards\n"
    open(os.path.join(d, "blank.csv"), "w").write(hdr + "1,A B,A,B,WR,SF,,\n")
    open(os.path.join(d, "filled.csv"), "w").write(hdr + "1,A B,A,B,WR,SF,4.1,50.2\n")
    assert template_problem(os.path.join(d, "blank.csv")) == ""
    assert "already hold numbers" in template_problem(os.path.join(d, "filled.csv"))
    shutil.rmtree(d, ignore_errors=True)
    # rushing/passing cells: QB gets pass, rush and pass+rush; RB gets rush and rush+rec (the
    # receiving part from the receiving grid when he is in it); a started game and a stale file fill nothing
    global LOGS
    keep, LOGS = LOGS, tempfile.mkdtemp()
    try:
        now = dt.datetime(2026, 11, 22, 12, tzinfo=dt.timezone.utc)
        rp = pd.DataFrame(dict(name=["Keenan Allen", "Trebor Pena", "Mike Williams"], team=["SF", "JAX", "LAC"], pos=["WR", "WR", "WR"],
                               gsis_id=["a", "b", "c"], inj=["", "", ""], starter=[True, False, False],
                               kickoff_utc=["2026-11-22T18:00:00+00:00", "2026-11-22T18:00:00+00:00", "2026-11-22T10:00:00+00:00"],
                               game_status=["scheduled", "scheduled", "scheduled"], mean_rush_yds=[20.04, 12.0, 30.0],
                               mean_pass_yds=[250.0, np.nan, np.nan], rec_yds_simple=[0.0, 7.0, 0.0], ts=["2026-11-22T11:00:00Z"] * 3))
        rp.to_csv(os.path.join(LOGS, "rushpass_grid.csv"), index=False)
        g = pd.DataFrame(dict(pid=["b"], mean_yds=[40.0]))
        t2 = t.assign(id=["1", "2", "3", "4", "5"])
        ex, lg, _ = rushpass_fills(g, t2, tkeys, tnames, {}, now)
        assert ex[0] == {"rushing_yards": "20.0", "passing_yards": "250.0", "passing_and_rushing_yards": "270.0"}, ex
        assert ex[3] == {"rushing_yards": "12.0", "rushing_and_receiving_yards": "52.0"}, ex
        assert 1 not in ex and 2 not in ex and set(lg.status) == {"filled", "skipped_started"}
        ex, _, note = rushpass_fills(g, t2, tkeys, tnames, {}, now + dt.timedelta(days=2))
        assert ex == {} and "left blank" in note
    finally:
        shutil.rmtree(LOGS, ignore_errors=True); LOGS = keep
    # a blank display name reads back as NaN and must fall back to the grid key
    Row = pd.DataFrame(dict(name=["marvinharrisonjr"], display_name=[np.nan])).itertuples().__next__()
    assert grid_name(Row, {}) == "marvinharrisonjr"
    return True


# --- names -------------------------------------------------------------------------------

def norm_name(s):
    """Fold accents, lowercase, drop periods and apostrophes, hyphens to spaces, drop suffixes,
    collapse whitespace. The grid's `name` is already a collapsed key (no spaces), so the match
    key removes spaces on both sides and strips a trailing suffix from collapsed names too."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = s.replace(".", "").replace("'", "").replace("\u2019", "").replace("-", " ")
    s = SUFFIX.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def match_key(s):
    k = re.sub(r"\s+", "", norm_name(s))
    return re.sub(r"(jr|sr|iii|ii|iv)$", "", k) if len(k) > 6 else k


def find_rows(nm, team, pos, tkeys, tnames):
    """Template rows for one grid player: (row indices or None, note). Exact name + team +
    position first (note ''). Failing that, the same name and position on exactly one other
    team is the same player listed under another club, so his Unabated id is still right and he
    gets filled (note 'team_differs'). Two or more other teams is ambiguous ('team_mismatch',
    unfilled); no such name at that position is 'unmatched'."""
    idx = tkeys.get(f"{match_key(nm)}|{team}|{pos}")
    if idx is not None:
        return idx, ""
    other = tnames.get((match_key(nm), pos))
    if other is None:
        return None, "unmatched"
    if len(other) == 1:
        return other, "team_differs"
    return None, "team_mismatch"


# --- template ------------------------------------------------------------------------------

def template_problem(path):
    """Why a file can't be the upload template, or '' when it can."""
    try:
        t = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as ex:
        return f"not a readable CSV ({type(ex).__name__})"
    miss = [c for c in TEMPLATE_NEED if c not in t.columns]
    if miss:
        return f"no {miss} column(s), so it is not Unabated's NFL projections template"
    if not len(t):
        return "no player rows"
    if t.id.duplicated().any():
        return "the same player id appears twice"
    proj = list(t.columns[list(t.columns).index("team") + 1:])
    n = int((t[proj].apply(lambda s: s.str.strip()) != "").sum().sum())
    if n:
        return (f"{n} projection cell(s) already hold numbers; this looks like a filled upload, not the blank "
                "template from Unabated's download button")
    return ""


def pick_newest(files):
    """Of several downloads, the highest copy number: projections-nfl (14).csv beats _13."""
    num = lambda f: max([int(x) for x in re.findall(r"\d+", f)] or [-1])
    return sorted(files, key=lambda f: (num(f), f))[-1]


def promote_template():
    """A download uploaded to templates/ under any projections-nfl*.csv name replaces
    templates/projections-nfl.csv, so nobody has to rename it. The workflow commits the swap.
    Returns the promoted file name, or ''. A file that isn't a blank template stops the export."""
    if not os.path.isdir(TEMPLATE_DIR):
        return ""
    drops = [f for f in os.listdir(TEMPLATE_DIR) if TEMPLATE_DROP.match(f) and f != os.path.basename(TEMPLATE)]
    if not drops:
        return ""
    newest = pick_newest(drops)
    problem = template_problem(os.path.join(TEMPLATE_DIR, newest))
    if problem:
        fail(f"templates/{newest} was not swapped in: {problem}")
    for f in drops:
        if f != newest:
            os.remove(os.path.join(TEMPLATE_DIR, f))
    shutil.move(os.path.join(TEMPLATE_DIR, newest), TEMPLATE)
    print(f"template: {newest} -> templates/projections-nfl.csv")
    return newest


def roster_check(grid, t, tkeys, tnames, team_map, alias, positions):
    """Active receivers in the grid that the template doesn't list on the same team. More than a
    couple means the download is out of date. Returns ['Name (TEAM, template OTHER)', ...]."""
    out = []
    for r in grid.itertuples():
        if str(r.pos) not in positions or str(r.inj) in OUT_TAGS or str(r.status_applied).lower() in ("true", "1"):
            continue
        nm = grid_name(r, alias)
        _, note = find_rows(nm, team_map.get(r.team, r.team), r.pos, tkeys, tnames)
        if note:
            other = tnames.get((match_key(nm), r.pos))
            where = "/".join(sorted({t.team.iloc[i] for i in other})) if other is not None else "missing"
            out.append(f"{nm} ({r.team}, template {where})")
    return out


def rushpass_fills(grid, t, tkeys, tnames, team_map, now, alias=None):
    """Template cells for the UNTESTED rushing and passing means in logs/rushpass_grid.csv:
    rushing_yards for every projected ballcarrier, passing_yards and passing_and_rushing_yards
    for each starting QB, rushing_and_receiving_yards for everyone else (receiving part from the
    published receiving grid when the player is in it, else the rush/pass grid's own decayed
    receiving yards per game). Plain means, no solving: Unabated simulates around them.
    Never stops the receiving export: a missing or stale file leaves these columns blank.
    Returns ({template row: {column: text}}, log frame, one-line note)."""
    p = os.path.join(LOGS, "rushpass_grid.csv")
    if not os.path.exists(p):
        return {}, pd.DataFrame(), "rushing/passing: no logs/rushpass_grid.csv, columns left blank"
    rp = pd.read_csv(p, low_memory=False)
    if not len(rp) or "ts" not in rp.columns or (now - pd.to_datetime(rp.ts.iloc[0], utc=True)) > pd.Timedelta(hours=20):
        return {}, pd.DataFrame(), "rushing/passing: logs/rushpass_grid.csv is empty or not from this run, columns left blank"
    rec_mean = dict(zip(grid.pid, grid.mean_yds)) if {"pid", "mean_yds"} <= set(grid.columns) else {}
    alias = alias or {}
    out, rows = {}, []
    for r in rp.itertuples():
        starter = str(r.starter).lower() in ("true", "1")
        rec = dict(name=r.name, team=r.team, pos=r.pos, starter=starter, kickoff_utc=r.kickoff_utc, template_id="", status="",
                   **{c: "" for c in RUSHPASS_COLS})
        if str(r.inj) in OUT_TAGS:
            rec["status"] = "skipped_out"
        elif pd.to_datetime(r.kickoff_utc, utc=True) <= now or str(r.game_status) in ("in_progress", "final"):
            rec["status"] = "skipped_started"
        else:
            idx, note = find_rows(alias.get(r.name, r.name), team_map.get(r.team, r.team), "RB" if r.pos == "FB" else r.pos, tkeys, tnames)
            if idx is None:
                rec["status"] = note
            else:
                rush = float(r.mean_rush_yds) if pd.notna(r.mean_rush_yds) else 0.0
                vals = {}
                if RUSH_RANGE[0] <= rush <= RUSH_RANGE[1]:
                    vals["rushing_yards"] = round(rush, 1)
                if starter:
                    pas = float(r.mean_pass_yds) if pd.notna(r.mean_pass_yds) else 0.0
                    if PASS_RANGE[0] <= pas <= PASS_RANGE[1]:
                        vals["passing_yards"] = round(pas, 1)
                        vals["passing_and_rushing_yards"] = round(pas + max(rush, 0.0), 1)
                elif "rushing_yards" in vals:
                    ry = rec_mean.get(r.gsis_id, r.rec_yds_simple)
                    ry = float(ry) if pd.notna(ry) else 0.0
                    vals["rushing_and_receiving_yards"] = round(rush + max(ry, 0.0), 1)
                if not vals:
                    rec["status"] = "out_of_range"
                else:
                    for i in idx:
                        out.setdefault(i, {}).update({c: str(v) for c, v in vals.items()})
                    rec.update({c: v for c, v in vals.items()}, template_id=";".join(str(t.id.iloc[i]) for i in idx),
                               status="filled_team_differs" if note else "filled")
        rows.append(rec)
    log = pd.DataFrame(rows)
    got = log[log.status.astype(str).str.startswith("filled")] if len(log) else log
    n = {c: int((got[c].astype(str) != "").sum()) for c in RUSHPASS_COLS} if len(got) else {c: 0 for c in RUSHPASS_COLS}
    note = (f"rushing/passing (UNTESTED model means): {n['rushing_yards']} rushing, {n['passing_yards']} passing, "
            f"{n['rushing_and_receiving_yards']} rush+rec, {n['passing_and_rushing_yards']} pass+rush")
    return out, log, note


def grid_name(r, alias):
    """The grid's display name when it has one (a blank reads back as NaN), else its key."""
    disp = getattr(r, "display_name", "")
    nm = disp.strip() if isinstance(disp, str) and disp.strip() else str(r.name)
    return alias.get(nm, nm)


# --- export ------------------------------------------------------------------------------

def load_cfg():
    cfg = dict(DEFAULT_CFG)
    if os.path.exists(CONFIG):
        cfg.update(json.load(open(CONFIG)))
    return cfg


def post(lines):
    try:
        import notify
        notify.send("Unabated export", [lines] if isinstance(lines, str) else list(lines), None, getattr(notify, "GREY", None))
    except Exception:
        pass


def fail(msg, keep_previous=False):
    print(f"EXPORT FAILED: {msg}")
    post(f"FAILED: {msg}")
    if not keep_previous:
        for f in ("projections-nfl.csv", "export_log.csv", "export_log_rushpass.csv", "export_meta.json"):
            p = os.path.join(OUT, f)
            if os.path.exists(p):
                os.remove(p)
    sys.exit(1)


def main():
    cfg = load_cfg(); shapes = cfg["unabated_yds_shape"]
    os.makedirs(OUT, exist_ok=True)
    unit_tests(); print("unit tests: PASS")
    now = dt.datetime.now(dt.timezone.utc)
    gp = os.path.join(LOGS, "props_grid.csv")
    if not os.path.exists(gp):
        fail("logs/props_grid.csv missing")
    grid = pd.read_csv(gp, low_memory=False)
    need_cols = ["name", "team", "pos", "kickoff_utc", "inj", "play_prob", "anchored", "rec_anchored", "yds_line", "yds_p_over",
                 "rec_line", "rec_p_over", "mean_rec", "mean_yds", "ts", "game_status"]
    missing = [c for c in need_cols if c not in grid.columns]
    if missing:
        fail(f"props_grid.csv lacks {missing}; rebuild the grid with the current props_market.py")
    grid_ts = pd.to_datetime(grid.ts.iloc[0], utc=True)
    if (now - grid_ts) > pd.Timedelta(hours=20):
        fail(f"grid is from {grid_ts.isoformat()}, not this run")
    promoted = promote_template()
    if not os.path.exists(TEMPLATE):
        fail("templates/projections-nfl.csv missing: download it from Unabated's Upload Projections button and commit it")
    problem = template_problem(TEMPLATE)
    if problem:
        fail(f"templates/projections-nfl.csv: {problem}")
    raw = open(TEMPLATE, "rb").read()
    t = pd.read_csv(TEMPLATE, dtype=str, keep_default_na=False)
    # lossless read/write proof
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
    t.to_csv(tmp, index=False, lineterminator="\n", encoding="utf-8")
    if open(tmp, "rb").read() != raw:
        fail("template round trip is not byte-identical, so the writer would alter cells outside the projection columns. "
             "Usually the template was opened and saved in Excel after download; upload it straight from the downloads folder",
             keep_previous=True)
    t["_key"] = [f"{match_key(n)}|{tm}|{p}" for n, tm, p in zip(t.name, t.team, t.position)]
    tkeys = t.groupby("_key").indices
    tnames = t.groupby([t.name.map(match_key), t.position]).indices
    team_map = cfg.get("team_map", {}); alias = cfg.get("alias", {})
    template_teams = set(t.team.unique())
    bad_team = sorted({team_map.get(x, x) for x in grid.team.unique()} - template_teams)
    if bad_team:
        fail(f"grid team code(s) not in the template: {bad_team}; add to team_map in unabated_config.json")
    stale = roster_check(grid, t, tkeys, tnames, team_map, alias, cfg["positions"])
    rows, filled = [], {}
    for r in grid.itertuples():
        rec = dict(name=r.name, team=r.team, pos=r.pos, kickoff_utc=r.kickoff_utc, inj=r.inj, play_prob=r.play_prob,
                   rec_line=r.rec_line, yds_line=r.yds_line, model_rec_p_over=r.rec_p_over, model_yds_p_over=r.yds_p_over,
                   grid_mean_rec=r.mean_rec, grid_mean_yds=r.mean_yds, template_id="", template_team="", upload_rec="",
                   upload_yds="", rebuilt_rec_p_over="", rebuilt_yds_p_over="", err_rec="", err_yds="", status="")
        status = ""
        if str(r.pos) not in cfg["positions"]:
            status = "skipped_position"
        elif str(r.inj) in OUT_TAGS or bool(r.status_applied):
            status = "skipped_out"
        elif not (bool(r.anchored) and bool(r.rec_anchored)):
            status = "skipped_unanchored"
        elif pd.to_datetime(r.kickoff_utc, utc=True) <= now or str(r.game_status) in ("in_progress", "final"):
            status = "skipped_started"
        if status:
            rec["status"] = status; rows.append(rec); continue
        # the grid's P(over) already assumes he plays (the model removes DNPs; Questionables are
        # projected as active), so no play_prob division is needed; recorded for audit
        p_rec, p_yds = float(r.rec_p_over), float(r.yds_p_over)
        try:
            catches = round(rec_upload_mean(float(r.rec_line), p_rec), 2)
        except Exception as ex:
            rec["status"] = "solver_fail"; rec["err_rec"] = type(ex).__name__; rows.append(rec); continue
        shape = float(shapes.get(str(r.pos), 0.75))
        low_volume = catches < float(cfg.get("min_catches_for_yards", 0))
        yards = None if low_volume else yds_upload_mean(float(r.yds_line), p_yds, catches, shape)
        rec["upload_rec"] = catches
        rec["rebuilt_rec_p_over"] = round(float(stats.poisson.sf(math.floor(float(r.rec_line)), catches)), 4)
        rec["err_rec"] = round(abs(rec["rebuilt_rec_p_over"] - p_rec) * 100, 3)
        if low_volume:
            rec["status"] = "low_volume_yds_blank"; rec["err_yds"] = "catches below min_catches_for_yards"
        elif yards is None:
            rec["status"] = "solver_fail"; rec["err_yds"] = "unsolvable at this shape"
        else:
            yards = round(yards, 1); rec["upload_yds"] = yards
            rec["rebuilt_yds_p_over"] = round(yds_sf(float(r.yds_line), catches, shape, yards), 4)
            rec["err_yds"] = round(abs(rec["rebuilt_yds_p_over"] - p_yds) * 100, 3)
        # template match
        tteam = team_map.get(r.team, r.team)
        idx, note = find_rows(grid_name(r, alias), tteam, r.pos, tkeys, tnames)
        if idx is None:
            rec["status"] = note
            rows.append(rec); continue
        if note == "team_differs":
            rec["template_team"] = t.team.iloc[idx[0]]
        if rec["status"] not in ("solver_fail", "low_volume_yds_blank"):
            rec["status"] = "filled_team_differs" if note else ("filled" if len(idx) == 1 else "duplicate_filled")
        rec["template_id"] = ";".join(str(t.id.iloc[i]) for i in idx)
        for i in idx:
            filled[i] = (str(catches), "" if yards is None else str(yards))
        rows.append(rec)
    log = pd.DataFrame(rows)
    counts = log.status.value_counts().to_dict()
    extra, rp_log, rp_note = rushpass_fills(grid, t, tkeys, tnames, team_map, now, alias)
    # write the upload
    out = t.drop(columns=["_key"]).copy()
    for i, (c, y) in filled.items():
        out.at[i, "receptions"] = c; out.at[i, "receiving_yards"] = y
    for i, vals in extra.items():
        for c, v in vals.items():
            out.at[i, c] = v
    tmp_up = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
    out.to_csv(tmp_up, index=False, lineterminator="\n", encoding="utf-8")
    # checks
    chk = pd.read_csv(tmp_up, dtype=str, keep_default_na=False)
    if list(chk.columns) != list(t.drop(columns=["_key"]).columns) or len(chk) != len(t) or list(chk.id) != list(t.id):
        fail("structure check: header, row count or id order differs from the template")
    diff_cols = [c for c in chk.columns if not (chk[c] == t[c]).all()]
    if set(diff_cols) - set(FILL_COLS):
        fail(f"structure check: cells changed outside {FILL_COLS}: {diff_cols}")
    exported = log[log.status.isin(EXPORTED)]
    max_err_rec = float(pd.to_numeric(exported.err_rec, errors="coerce").max()) if len(exported) else 0.0
    max_err_yds = float(pd.to_numeric(exported.err_yds, errors="coerce").max()) if len(exported) else 0.0
    if max(max_err_rec, max_err_yds) > cfg["roundtrip_tol_pts"]:
        fail(f"round trip: max error rec {max_err_rec:.2f} / yds {max_err_yds:.2f} points, tolerance {cfg['roundtrip_tol_pts']}")
    ur = pd.to_numeric(exported.upload_rec, errors="coerce"); uy = pd.to_numeric(exported.upload_yds, errors="coerce")
    if len(exported) and (((ur < 0) | (ur > 15)).any() or ((uy < 0) | (uy > 250)).any() or (((uy / ur) < 2) | ((uy / ur) > 35)).any()):
        fail("sanity check: catches outside 0..15, yards outside 0..250, or yards per catch outside 2..35")
    board_ok, board_max = "no board", 0.0
    bp = os.path.join(LOGS, "props_board.csv")
    if os.path.exists(bp):
        b = pd.read_csv(bp); b = b[b.side == "Over"]
        diffs = []
        for r in exported.itertuples():
            for mk, line, p in (("receptions", r.rec_line, r.model_rec_p_over), ("rec_yds", r.yds_line, r.model_yds_p_over)):
                m = b[(b.player == r.name) & (b.market == mk) & (np.isclose(b.line, float(line)))]
                if len(m):
                    diffs.append(abs(float(m.model_pct.iloc[0]) - float(p) * 100))
        board_max = max(diffs) if diffs else 0.0; board_ok = f"{len(diffs)} rows compared"
        if board_max > cfg["board_tol_pts"]:
            fail(f"board agreement: max difference {board_max:.2f} points, tolerance {cfg['board_tol_pts']}")
    if len(exported) and (pd.to_datetime(exported.kickoff_utc, utc=True) <= now).any():
        fail("freshness: an exported player's kickoff has passed")
    # move into place
    shutil.move(tmp_up, os.path.join(OUT, "projections-nfl.csv"))
    log.to_csv(os.path.join(OUT, "export_log.csv"), index=False)
    rp_log.to_csv(os.path.join(OUT, "export_log_rushpass.csv"), index=False)
    unmatched = int(counts.get("unmatched", 0) + counts.get("team_mismatch", 0))
    meta = dict(grid_build_ts=str(grid.ts.iloc[0]), export_ts=now.strftime("%Y-%m-%dT%H:%M:%SZ"), counts=counts,
                max_roundtrip_err_pts=dict(receptions=round(max_err_rec, 3), receiving_yards=round(max_err_yds, 3)),
                board_check=board_ok, board_max_diff_pts=round(board_max, 3), shapes=shapes, n_exported=int(len(exported)),
                template_rows=int(len(t)), template_sha256=hashlib.sha256(raw).hexdigest()[:16], template_promoted=promoted,
                template_roster_mismatches=stale, template_stale_warning=bool(stale),
                rushpass=dict(note=rp_note, untested=True, counts=rp_log.status.value_counts().to_dict() if len(rp_log) else {}))
    json.dump(meta, open(os.path.join(OUT, "export_meta.json"), "w"), indent=1)
    for st in ("unmatched", "team_mismatch", "filled_team_differs", "duplicate_filled", "solver_fail", "low_volume_yds_blank"):
        rows_ = log[log.status == st]
        if len(rows_):
            print(f"WARNING {st}: " + ", ".join(f"{r.name} ({r.team} {r.pos})" for r in rows_.itertuples()))
    if stale:
        print(f"WARNING template: {len(stale)} active receiver(s) not on the same team in the template: " + "; ".join(stale))
    line = (f"export: {len(exported)} players -> logs/unabated/projections-nfl.csv | counts {counts} | "
            f"max round-trip err rec {max_err_rec:.2f} yds {max_err_yds:.2f} pts | board check {board_ok}, max diff {board_max:.2f}")
    print(line)
    print(rp_note)
    if len(rp_log):
        miss = rp_log[rp_log.status.isin(["unmatched", "team_mismatch", "out_of_range"])]
        if len(miss):
            print("WARNING rushing/passing not filled: " + ", ".join(f"{r.name} ({r.team} {r.pos}, {r.status})" for r in miss.itertuples()))
    msg = [f"logs/unabated/projections-nfl.csv: {len(exported)} players exported, {unmatched} unmatched, "
           f"max round-trip error {max(max_err_rec, max_err_yds):.2f} pts", rp_note]
    if promoted:
        msg.append(f"new Unabated template in use: {promoted} ({len(t)} players)")
    if stale:
        msg.append(f"template out of date? {len(stale)} receiver(s) not on the same team in it: " + "; ".join(stale[:8])
                   + ("; ..." if len(stale) > 8 else "") + ". Download a fresh template from Unabated and upload it to the templates folder.")
    post(msg)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--test", action="store_true"); a = ap.parse_args()
    if a.test:
        unit_tests(); print("unit tests: PASS")
    else:
        main()
