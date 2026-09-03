"""
paper_open.py: the pre-registered NCAAF opener rule, paper only, 2026 season.

Rule (written 2026-09-02, do not edit mid-season):
  Take the model's side against the Monday-morning spread when the claimed cover
  probability is 0.65 or better. Flat one unit, graded at -110.
  Pass at season end if hit rate >= 0.55 on at least 100 triggers. Fail: retired, no
  second version against the same season.

How it works: the daily runner already prices every game against the market line it sees
and writes pick_line / pick_prob / pick_claim into logs/predictions.csv, then overwrites
them on the next run. This script runs 40 minutes after the Monday runner, copies that
Monday snapshot into logs/paper_open.csv (append-only, one row per game per week), grades
last week's rows from the margins the runner has since filled in, and posts the tally.

Every game with a Monday line is logged, not just triggers, so the file doubles as the
calibration curve of claimed-cover-at-the-opener. Only rows with trigger=1 count toward
the rule.
"""
import datetime as dt, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "logs", "predictions.csv")
OUT = os.path.join(HERE, "logs", "paper_open.csv")
TRIGGER = float(os.environ.get("PAPER_THRESHOLD", "0.65"))
WEEKDAY = int(os.environ.get("PAPER_WEEKDAY", "0"))          # 0 = Monday (NCAAF), 1 = Tuesday (NFL)
LABEL = os.environ.get("PAPER_LABEL", "NCAAF paper: Monday-line rule")
VALIDATED = os.environ.get("PAPER_VALIDATED", "1") == "1"    # NFL threshold is provisional, not backtested
BAR_HIT, BAR_N = 0.55, 100
TODAY = dt.date.today()
COLS = ["snap_date", "date", "week", "season", "game_id", "away", "home", "pick_team", "pick_side",
        "pick_line", "pick_prob", "pick_claim", "book_line", "book_count", "trigger",
        "result", "pnl", "graded"]


def load_out():
    out = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=COLS)
    for c in ("result", "pnl", "graded"):
        out[c] = out[c].astype(object).where(out[c].notna(), "")
    return out


def snapshot(log, out):
    if "date" not in log.columns and "gameday" in log.columns:
        log = log.rename(columns={"gameday": "date"})
    for c in ("week", "season", "book_line", "book_count"):
        if c not in log.columns:
            log[c] = ""
    up = log[log.pick_team.notna() & (log.pick_team.astype(str) != "") & log.result.isna()].copy()
    up["date"] = pd.to_datetime(up.date).dt.date
    up = up[(up.date >= TODAY) & (up.date <= TODAY + dt.timedelta(days=8))]
    up = up[~up.game_id.isin(out.game_id)]
    if up.empty:
        print("no new Monday rows (already logged, or the runner has not priced this week yet)")
        return out
    rows = pd.DataFrame({
        "snap_date": str(TODAY), "date": up.date.astype(str), "week": up.week, "season": up.season,
        "game_id": up.game_id, "away": up.away, "home": up.home, "pick_team": up.pick_team,
        "pick_side": up.pick_side, "pick_line": pd.to_numeric(up.pick_line, errors="coerce"),
        "pick_prob": pd.to_numeric(up.pick_prob, errors="coerce").round(4), "pick_claim": up.pick_claim,
        "book_line": up.get("book_line"), "book_count": up.get("book_count"),
        # only a Monday snapshot counts toward the rule; a mid-week manual run still logs the
        # claims (for the calibration curve) but never as triggers
        "trigger": ((pd.to_numeric(up.pick_prob, errors="coerce") >= TRIGGER) & (up.pick_claim == "covers")
                    & (TODAY.weekday() == WEEKDAY)).astype(int),
        "result": "", "pnl": "", "graded": ""})
    print(f"logged {len(rows)} games for week {rows.week.iloc[0]}, {int(rows.trigger.sum())} trigger(s)")
    return pd.concat([out, rows], ignore_index=True)


def grade(log, out):
    fin = log.set_index("game_id").result
    n = 0
    for i, r in out.iterrows():
        if str(r.result) != "" or r.game_id not in fin.index or pd.isna(fin[r.game_id]):
            continue
        margin = float(fin[r.game_id])
        mp = margin if r.pick_side == "home" else -margin
        pl = float(r.pick_line) if pd.notna(r.pick_line) else 0.0
        base = mp + pl if r.pick_claim == "covers" else mp
        res = "win" if base > 0 else "loss" if base < 0 else "push"
        out.loc[i, ["result", "pnl", "graded"]] = [res, 0.909 if res == "win" else -1.0 if res == "loss" else 0.0, str(TODAY)]
        n += 1
    print(f"graded {n} row(s)")
    return out


def tally(out):
    d = out[out.result.isin(["win", "loss"])].copy()
    d["pnl"] = pd.to_numeric(d.pnl); d["pick_prob"] = pd.to_numeric(d.pick_prob)
    lines = []
    t = d[d.trigger == 1]
    if len(t):
        hit = (t.result == "win").mean()
        lines.append(f"rule (claimed >= {TRIGGER}): {len(t)} bets, {(t.result=='win').sum()}-{(t.result=='loss').sum()}, "
                     f"hit {hit:.3f}, ROI {t.pnl.mean():+.3f}  | bar {BAR_HIT} on {BAR_N}: "
                     f"{'PASS so far' if hit >= BAR_HIT else 'short'}, {len(t)}/{BAR_N} bets")
    else:
        lines.append("rule: no graded triggers yet")
    if len(d):
        lines.append("all Monday claims, by band:")
        for lo, hi in ((.5, .55), (.55, .6), (.6, .65), (.65, .7), (.7, 1.01)):
            b = d[(d.pick_prob >= lo) & (d.pick_prob < hi)]
            if len(b) >= 5:
                lines.append(f"  {lo:.2f}-{min(hi,1):.2f}  n={len(b):>3}  said {b.pick_prob.mean():.3f}  covered {(b.result=='win').mean():.3f}")
    return lines


def main():
    if not os.path.exists(LOG):
        sys.exit("logs/predictions.csv not found; run from the repo root")
    log = pd.read_csv(LOG, low_memory=False)
    if "date" not in log.columns and "gameday" in log.columns:
        log = log.rename(columns={"gameday": "date"})
    missing = [c for c in ("pick_team", "pick_side", "pick_line", "pick_prob", "pick_claim", "result", "game_id") if c not in log.columns]
    if missing:
        sys.exit(f"predictions.csv is missing {missing}; this board's runner does not log market-anchored picks")
    out = load_out()
    before = len(out)
    out = grade(log, out)
    out = snapshot(log, out)
    out = out[COLS]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    new = out.iloc[before:]
    body = []
    if len(new):
        trig = new[new.trigger == 1]
        body.append(f"this week: {len(new)} games priced at {TODAY.strftime('%A')} lines, {len(trig)} trigger(s)")
        for _, r in trig.sort_values("pick_prob", ascending=False).iterrows():
            body.append(f"  {r.date}  {r.pick_team} {float(r.pick_line):+g}  claimed {float(r.pick_prob):.3f}   ({r.away} @ {r.home})")
    body += tally(out)
    print("\n".join(body))
    if "--dry" in sys.argv:
        return
    try:
        import notify
        foot = "paper only, 2026; rule fixed 2026-09-02" if VALIDATED else "paper only; threshold PROVISIONAL, no backtest behind it yet"
        notify.send(LABEL, body, foot, getattr(notify, "GREY", None))
    except Exception as ex:
        print("discord post skipped:", type(ex).__name__)


if __name__ == "__main__":
    main()
