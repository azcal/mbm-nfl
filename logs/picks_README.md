# picks.csv

One row per published pick from the MustBeMoose NFL model. Rebuilt on every run.

This is a **forecast**, not a bet recommendation. No price is compared anywhere in this
file, so whether a probability beats the odds at your book is entirely your call.

## Reading a pick

`pick` is the whole answer in one field, written the way a book writes it: `ATL +3`,
`SEA -3.5`. If you read one column, read that one.

| column | meaning |
|---|---|
| `pick_team` | the team being picked |
| `pick_side` | whether that team is `home` or `away` in this game |
| `pick_line` | the handicap **on `pick_team`**, taken from the book line. Negative lays points, positive takes them. Not a home-side number; `pick_side` tells you which end it belongs to. |
| `book_line` | the same number as a sportsbook posts it, on the **home** team. Median across `book_count` books. Empty when no market line was available. |
| `book_count` | how many books post that exact number |
| `book_total` | how many books priced the game at all. `3` of `9` is a market that cannot agree with itself; `3` of `3` is just a thin one. |
| `pick_claim` | what is asserted: `covers` or `wins outright` |
| `pick_prob` | probability of **that claim**, with any push excluded |
| `pick_push` | chance the game lands exactly on `pick_line`. Non-zero only on whole numbers, where it reaches 9-10% at 3. |
| `pick_rule` | `market` when anchored to a book line, `median` when it fell back to the model's own |
| `pick_result` | `win`, `loss`, `push`, or empty until the game is played |
| `cal` | true while the model is inside its calibration window, weeks 1-4 |

## Grading

`covers` rows score as `margin_for_pick + pick_line`, where `margin_for_pick` is
`final_margin` when `pick_side` is `home` and `-final_margin` when it is `away`. Above zero
wins, below loses, exactly zero pushes.

`wins outright` rows score on the game result. `pick_line` is still populated on those rows
as the model's own line, but it is context, not the claim.

## Where the line comes from

The published line is **the market's**, not the model's. Every run pulls current NFL spreads,
takes the median across books, and reports the model's probability that one side covers that
exact number.

Books hang their numbers on key margins by themselves, so this lands on 3, 7 and their hooks
without the board forcing it, and it gives a different number per game rather than crowding
the slate onto a handful of lines.

## What the probability means

`pick_prob` is calibrated, not a confidence score. A set of rows reading 0.60 should win
about 60% of the time. Measured over 569 graded games the model runs slightly conservative:
it said 0.651 and delivered 0.670 on outright winners, and 0.539 against 0.543 actual across
its spread lines.
