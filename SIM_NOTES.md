# Joint sim (sim_nfl.py), v1, 2026-11-26. PAPER ONLY.

Started from Rob Pizzola's "I Ran 10,000 Simulations of One NFL Game" workbook and script. The
idea kept from it is one set of dice per game, so totals, team totals and player yards all come
off the same simulated games. Its team ratings (season points for and against) and its margin
distribution are not used: the sides model supplies the winner probability, the margin
distribution and the total, and the simulated games are re-weighted to match them.

## What was measured (2023 = tuning year, 2024 and 2025 = holdouts, heads trained on earlier seasons only)

| Check | Result | Bar | Verdict |
|---|---|---|---|
| Plain drive sim, games decided by exactly 3 | 8.3% vs 14.3% real | n/a | the video's own finding, reproduced |
| Re-weighted sim, games decided by exactly 3 | 14.8% vs 14.3% real | n/a | fixed by the sides model's margins |
| Plain drive sim, spread of totals | 15.3 points vs 13.0 real | n/a | too wide; corrected to the training seasons' 13.4 |
| Totals, CRPS 2024-25 | 7.268 vs floor 7.263 | at or below floor | MISS by 0.07% |
| Totals, reliability | worst bucket 0.0504 off (stated 36.9%, actual 31.8%, n=644) | within 0.05 | MISS, just |
| Team totals, CRPS 2024-25 | 5.185 vs floor 5.184 | at or below floor | MISS by 0.02% |
| Team totals, reliability | every bucket with n>=100 within 0.02 | within 0.05 | pass |
| Total at exact numbers (37, 40, 41, 43, 44, 45, 47, 48, 51) | log loss 0.1511 vs floor 0.1520 | n/a | sim slightly better |
| Sim vs the closing total | log loss 0.697 vs 0.693 for a coin flip; the 60-70% bucket hit 51% (n=101) | n/a | NO EDGE against the close |
| Passing yards ladder (starting QBs) | every bucket within 0.03 | within 0.05 | pass |
| Rushing yards ladder (backs projected 8+ carries) | 60-90% buckets 0.05 to 0.07 too high | within 0.05 | FAIL at low lines |
| Rushing ladder, lines 59.5 and up | within 0.02 at every line | n/a | usable range |
| Rushing means (props_rushpass.py), MAE | 28.1 vs 30.6 for one number for everyone | n/a | real signal |
| Passing means (props_rushpass.py), MAE | 57.6 vs 58.2 for one number for everyone | n/a | almost none |

The floor for totals is the simplest honest alternative: the model total plus the training
seasons' own misses. The sim ties it and does not beat it. What the sim adds over that floor is
real scores (whole-number pushes), team totals, and player yards that move with the game.

Things tried and left off because the tuning year showed no gain: team-specific pace and
touchdown mix from play-by-play drives; pace and points-per-drive features in the total head.
A role-collapse mixture for carries was fitted on 2023 and kept (3% of games at 20% of the
workload), but it did not close the rushing gap at low lines.

Player ladders were scored GIVEN the player suited up and the starting QB was known. Live, the
depth chart and injury report decide that, and that risk is not in the numbers above.

## Where it runs
- daily_nfl.py writes logs/sim_inputs.json for every boarded game.
- daily workflow: sim_nfl.py writes logs/sim_totals.csv (no post).
- props workflow (Sunday and Thursday): after props_rushpass.py, sim_nfl.py also writes
  logs/sim_players.csv and posts one grey Discord embed for today's and tomorrow's games.

## Reproduce
    python sim_backtest.py --pbp DIR --fit
    python sim_backtest.py --pbp DIR
    python sim_backtest.py --pbp DIR --players --stats DIR2
DIR holds nflverse play_by_play_YYYY.parquet (2016 on), DIR2 holds stats_player_week_YYYY.parquet (2022 on).

## Not built yet
Same-game combo list (correlated two-leg prices per game). The simulated games already carry
the joint results; the list and its own test are the next step.
