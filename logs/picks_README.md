# picks.csv

One row per published pick from the MustBeMoose NFL model. Rebuilt on every run.

This is a **forecast**, not a bet recommendation. No market price is compared anywhere in
this file, and nothing here says whether a pick is worth backing at the number your book
is posting. That call is yours.

## Reading a pick

`pick` is the whole answer in one field, written the way a book writes it: `PIT +3.5`,
`SEA -3.5`. If you only read one column, read that one.

The parts of it, if you need them separately:

| column | meaning |
|---|---|
| `pick_team` | the team being picked |
| `pick_side` | whether that team is `home` or `away` in this game |
| `pick_line` | the handicap **on `pick_team`**. Negative lays points, positive takes them. Not a home-side number: `pick_side` tells you which end of the game it belongs to. |
| `pick_claim` | what is being asserted: `covers` or `wins outright` |
| `pick_prob` | probability of **that claim**, not of anything else |
| `pick_rule` | `dog_band` produced a `covers` row, `median` produced a `wins outright` row |
| `pick_result` | `win`, `loss`, `push`, or empty when the game has not been played |
| `cal` | true while the model is inside its calibration window (weeks 1-4) |

Two claim types, graded differently, because they assert different things:

- `covers`  -> `pick_team` stays within `pick_line`. Score it as
  `margin_for_pick + pick_line`, where `margin_for_pick` is `final_margin` when
  `pick_side` is `home` and `-final_margin` when it is `away`. Above zero wins, below
  loses, exactly zero pushes.
- `wins outright` -> `pick_team` wins the game. `pick_line` is still populated on these
  rows as the model's own line, but it is context, not the claim.

## What the probability means

`pick_prob` is calibrated, not a confidence score. A set of rows reading 0.60 should win
about 60% of the time. Measured over 569 graded games the model runs slightly conservative:
it said 0.651 and delivered 0.670 on outright winners, and said 0.539 against 0.543 actual
across its spread lines. Nothing here is a bet recommendation, and no market price is
compared anywhere in this file.

## Other columns

| column | meaning |
|---|---|
| `date` | kickoff date |
| `game_id` | nflverse id, stable, use it as the key |
| `away` / `home` | the fixture |
| `model_line` | the model's own line on the favourite. On a `covers` row this is the view the pick was derived from, so you can see the gap between them. |
| `final_margin` | home score minus away score, empty until played |

## The two rules

`median` rows publish the model's own line and claim the favourite wins outright. `dog_band`
rows fire when the model makes a team a favourite by 0.5 to 2.5 and instead publish the
underdog at +3.5, because a model favourite that small wins by four or more only about 42%
of the time. The band is under active test and may change.
