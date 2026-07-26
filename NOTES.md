# NFL rebuild v1

Data: nflverse (games file + play-by-play EPA aggregates), 2017 burn-in, 2018-2025 modeled.
No standalone prior model existed (Drive shows NFL ran through the market-anchored
all-sport cell), so the floors carry the comparison.

Winner head (frozen holdouts, ties excluded):
  2024 n=285: LL 0.6021 / Brier 0.2072 / hit 70.2%  (Elo 0.6312, base 0.6887)
  2025 n=284: LL 0.6427 / Brier 0.2259 / hit 63.7%  (Elo 0.6589, base 0.6909)
  combined n=569: LL 0.6224

Reference heads (both beat their floors, printed on the board without thresholds):
  margin MAE: 9.91 / 10.22 vs Elo-margin 10.44 / 10.49 vs constant 11.17 / 11.04
  total  MAE: 9.74 / 10.55 vs constant 10.12 / 10.92

Design: opponent-adjusted decayed EPA/play (fast+slow), Elo (playoff K x1.4),
QB layer = rolling EPA/dropback per listed starter with backup detection,
rest / short-week / divisional / dome. Tuned on 2023 only (K=14, HA=55, carry=0.75, C=0.05).
2026 season = live forward test from Week 1. Runner ships with the calibration-period
flag: predictions print from Week 1, value lines unlock at Week 5.
