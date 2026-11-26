# MBM NFL daily engine (rebuild v1)

Standalone predictive model, no market inputs. Three heads, all floor-validated:
winner P(home), model margin, model total. Ties push. Frozen through 2025;
2026 is the forward test. Holdouts: 2024 LL 0.6021 (70.2% hit), 2025 LL 0.6427 (63.7%).
Those two numbers know each game's actual starting QB, which the live board does not.
Re-run 2026-11-26 with the starter each team used the game before (the live fallback rule):
2024 LL 0.6083, 2025 LL 0.6467, against Elo at 0.6312 and 0.6589. Read the model as about
+0.023 and +0.012 over Elo, not +0.030 and +0.017. Retraining on pregame starters was
tested the same day and did not help (+0.0038 log loss pooled over 2024 to 2026, interval
-0.0007 to +0.0082), so the fit is unchanged. Live 2026 through week 12: 0.6083, Elo 0.6485.

QB layer is live: expected starters come from nflverse depth charts minus injury-report
Outs, falling back to each team's last actual starter. A benched or injured QB1
reprices the board automatically.

CALIBRATION GATE: predictions print from Week 1, value lines unlock Week 5.
Runs daily 13:00 UTC plus a Sunday pre-slate refresh. Tripwire: last-150 decided
games vs base rate, pings Telegram if it fires.
