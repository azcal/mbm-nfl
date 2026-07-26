# MBM NFL daily engine (rebuild v1)

Standalone predictive model, no market inputs. Three heads, all floor-validated:
winner P(home), model margin, model total. Ties push. Frozen through 2025;
2026 is the forward test. Holdouts: 2024 LL 0.6021 (70.2% hit), 2025 LL 0.6427 (63.7%).

QB layer is live: expected starters come from nflverse depth charts minus injury-report
Outs, falling back to each team's last actual starter. A benched or injured QB1
reprices the board automatically.

CALIBRATION GATE: predictions print from Week 1, value lines unlock Week 5.
Runs daily 13:00 UTC plus a Sunday pre-slate refresh. Tripwire: last-150 decided
games vs base rate, pings Telegram if it fires.
