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

Inputs: logs/props_grid.csv (published, anchored), templates/projections-nfl.csv (Unabated's
template, read fresh, never edited), logs/props_board.csv (cross-check).
Outputs: logs/unabated/projections-nfl.csv, export_log.csv, export_meta.json. Written to temp
paths and moved into place only after every check passes; on failure the previous export is
deleted so a stale file can't be uploaded by mistake.
"""
import argparse, datetime as dt, json, math, os, re, shutil, sys, tempfile, unicodedata
import numpy as np, pandas as pd
from scipy import optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs"); OUT = os.path.join(LOGS, "unabated")
TEMPLATE = os.path.join(HERE, "templates", "projections-nfl.csv")
CONFIG = os.path.join(HERE, "unabated_config.json")
DEFAULT_CFG = {"unabated_yds_shape": {"WR": 0.75, "TE": 0.75, "RB": 0.75}, "team_map": {"LA": "LAR"}, "alias": {},
               "positions": ["WR", "TE", "RB"], "roundtrip_tol_pts": 0.5, "board_tol_pts": 0.15, "stale_template_pct": 5.0}
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


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


# --- export ------------------------------------------------------------------------------

def load_cfg():
    cfg = dict(DEFAULT_CFG)
    if os.path.exists(CONFIG):
        cfg.update(json.load(open(CONFIG)))
    return cfg


def post(line):
    try:
        import notify
        notify.send("Unabated export", [line], None, getattr(notify, "GREY", None))
    except Exception:
        pass


def fail(msg, keep_previous=False):
    print(f"EXPORT FAILED: {msg}")
    post(f"FAILED: {msg}")
    if not keep_previous:
        for f in ("projections-nfl.csv", "export_log.csv", "export_meta.json"):
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
    if not os.path.exists(TEMPLATE):
        fail("templates/projections-nfl.csv missing: download it from Unabated's Upload Projections button and commit it")
    raw = open(TEMPLATE, "rb").read()
    t = pd.read_csv(TEMPLATE, dtype=str, keep_default_na=False)
    # lossless read/write proof
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
    t.to_csv(tmp, index=False, lineterminator="\n", encoding="utf-8")
    if open(tmp, "rb").read() != raw:
        fail("template round trip is not byte-identical; the writer would alter cells outside receptions/receiving_yards", keep_previous=True)
    t["_key"] = [f"{match_key(n)}|{tm}|{p}" for n, tm, p in zip(t.name, t.team, t.position)]
    tkeys = t.groupby("_key").indices
    tnames = t.groupby([t.name.map(match_key), t.position]).indices
    team_map = cfg.get("team_map", {}); alias = cfg.get("alias", {})
    template_teams = set(t.team.unique())
    bad_team = sorted({team_map.get(x, x) for x in grid.team.unique()} - template_teams)
    if bad_team:
        fail(f"grid team code(s) not in the template: {bad_team}; add to team_map in unabated_config.json")
    rows, filled = [], {}
    for r in grid.itertuples():
        rec = dict(name=r.name, team=r.team, pos=r.pos, kickoff_utc=r.kickoff_utc, inj=r.inj, play_prob=r.play_prob,
                   rec_line=r.rec_line, yds_line=r.yds_line, model_rec_p_over=r.rec_p_over, model_yds_p_over=r.yds_p_over,
                   grid_mean_rec=r.mean_rec, grid_mean_yds=r.mean_yds, template_id="", upload_rec="", upload_yds="",
                   rebuilt_rec_p_over="", rebuilt_yds_p_over="", err_rec="", err_yds="", status="")
        status = ""
        if str(r.pos) not in cfg["positions"]:
            status = "skipped_position"
        elif str(r.inj) in ("Out", "Doubtful", "IR", "PUP", "Suspended") or bool(r.status_applied):
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
        yards = yds_upload_mean(float(r.yds_line), p_yds, catches, shape)
        rec["upload_rec"] = catches
        rec["rebuilt_rec_p_over"] = round(float(stats.poisson.sf(math.floor(float(r.rec_line)), catches)), 4)
        rec["err_rec"] = round(abs(rec["rebuilt_rec_p_over"] - p_rec) * 100, 3)
        if yards is None:
            rec["status"] = "solver_fail"; rec["err_yds"] = "unsolvable at this shape"
        else:
            yards = round(yards, 1); rec["upload_yds"] = yards
            rec["rebuilt_yds_p_over"] = round(yds_sf(float(r.yds_line), catches, shape, yards), 4)
            rec["err_yds"] = round(abs(rec["rebuilt_yds_p_over"] - p_yds) * 100, 3)
        # template match
        tteam = team_map.get(r.team, r.team)
        disp = str(getattr(r, "display_name", "") or "")
        nm = alias.get(disp or r.name, disp or r.name)
        key = f"{match_key(nm)}|{tteam}|{r.pos}"
        idx = tkeys.get(key)
        if idx is None:
            other = tnames.get((match_key(nm), r.pos))
            rec["status"] = "team_mismatch" if other is not None else "unmatched"
            rows.append(rec); continue
        if rec["status"] != "solver_fail":
            rec["status"] = "filled" if len(idx) == 1 else "duplicate_filled"
        rec["template_id"] = ";".join(str(t.id.iloc[i]) for i in idx)
        for i in idx:
            filled[i] = (str(catches), "" if yards is None else str(yards))
        rows.append(rec)
    log = pd.DataFrame(rows)
    counts = log.status.value_counts().to_dict()
    # write the upload
    out = t.drop(columns=["_key"]).copy()
    for i, (c, y) in filled.items():
        out.at[i, "receptions"] = c; out.at[i, "receiving_yards"] = y
    tmp_up = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
    out.to_csv(tmp_up, index=False, lineterminator="\n", encoding="utf-8")
    # checks
    chk = pd.read_csv(tmp_up, dtype=str, keep_default_na=False)
    if list(chk.columns) != list(t.drop(columns=["_key"]).columns) or len(chk) != len(t) or list(chk.id) != list(t.id):
        fail("structure check: header, row count or id order differs from the template")
    diff_cols = [c for c in chk.columns if not (chk[c] == t[c]).all()]
    if set(diff_cols) - {"receptions", "receiving_yards"}:
        fail(f"structure check: cells changed outside receptions/receiving_yards: {diff_cols}")
    exported = log[log.status.isin(["filled", "duplicate_filled"])]
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
    n_elig = int(len(log[~log.status.str.startswith("skipped")]))
    unmatched = int(counts.get("unmatched", 0))
    meta = dict(grid_build_ts=str(grid.ts.iloc[0]), export_ts=now.strftime("%Y-%m-%dT%H:%M:%SZ"), counts=counts,
                max_roundtrip_err_pts=dict(receptions=round(max_err_rec, 3), receiving_yards=round(max_err_yds, 3)),
                board_check=board_ok, board_max_diff_pts=round(board_max, 3), shapes=shapes, n_exported=int(len(exported)),
                template_stale_warning=bool(n_elig and unmatched / n_elig > cfg["stale_template_pct"] / 100))
    json.dump(meta, open(os.path.join(OUT, "export_meta.json"), "w"), indent=1)
    for st in ("unmatched", "team_mismatch", "duplicate_filled", "solver_fail"):
        rows_ = log[log.status == st]
        if len(rows_):
            print(f"WARNING {st}: " + ", ".join(f"{r.name} ({r.team} {r.pos})" for r in rows_.itertuples()))
    if meta["template_stale_warning"]:
        print("WARNING: unmatched players exceed 5% of eligible; the template is probably stale, download a fresh one from Unabated")
    line = (f"export: {len(exported)} players -> logs/unabated/projections-nfl.csv | counts {counts} | "
            f"max round-trip err rec {max_err_rec:.2f} yds {max_err_yds:.2f} pts | board check {board_ok}, max diff {board_max:.2f}")
    print(line)
    post(f"logs/unabated/projections-nfl.csv: {len(exported)} players exported, {unmatched} unmatched, "
         f"max round-trip error {max(max_err_rec, max_err_yds):.2f} pts")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--test", action="store_true"); a = ap.parse_args()
    if a.test:
        unit_tests(); print("unit tests: PASS")
    else:
        main()
