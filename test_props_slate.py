"""Regression test for the slate: python test_props_slate.py. Week 1 2026 must match the fixture."""
import pandas as pd, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import props_weekly as W
FIX = "NE@SEA, SF@LA, CHI@CAR, TB@CIN, NO@DET, BUF@HOU, BAL@IND, CLE@JAX, ATL@PIT, NYJ@TEN, ARI@LAC, MIA@LV, GB@MIN, WAS@PHI, DAL@NYG, DEN@KC"
want = {}
for pair in FIX.split(", "):
    a, h = pair.split("@"); want[a] = h; want[h] = a
g = W.schedule(2026)
up, w = W.slate_for(g, 1, pd.Timestamp.now(tz="UTC"))
W.assert_slate(up, w)
got = dict(zip(up.team, up.opponent_team))
assert got == want, {t: (got.get(t), want[t]) for t in want if got.get(t) != want[t]}
assert len(got) == 32
print("PASS: 2026 week 1 slate matches the fixture for all 32 teams")
