"""python tests/test_props_odds.py

A 100% model probability (the under for a player ruled out, or a line past the ladder's top
rung) made american() divide by zero and killed the Nov 12 and Nov 15, 2026 runs, so week 10
got no grid, no board and no Unabated file."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import props_market as PM

assert PM.american(2.5) == "+150"
assert PM.american(2.0) == "+100"
assert PM.american(1.909) == "-110"
assert PM.american(1.5) == "-200"
assert PM.american(1 / 0.999) == "-99900"
for p in (1.0, 0.99999):                        # the board computes fair = american(1 / model)
    assert PM.american(1 / max(p, 1e-6)) == "-100000", p
assert PM.american(1 / 1e-6) == "+99999900"     # the other end already worked
print("PASS: odds conversion handles 0% and 100% model probabilities")
