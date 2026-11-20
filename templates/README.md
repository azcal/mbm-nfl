# Unabated template

`projections-nfl.csv` is Unabated's blank NFL projections template. `props_unabated.py` fills
the `receptions` and `receiving_yards` cells, plus the untested `rushing_yards`,
`passing_yards`, `rushing_and_receiving_yards` and `passing_and_rushing_yards` means from
`props_rushpass.py`, and writes the upload to `logs/unabated/projections-nfl.csv`. Every other
byte stays as Unabated wrote it.

Unabated's player list changes every week (trades, signings, cuts), so refresh this file about
once a week, ideally before the Saturday-night run:

1. Download the blank NFL template from Unabated's Upload Projections window (the same
   `projections-nfl.csv` file as always).
2. On GitHub, open this `templates` folder, click **Add file**, then **Upload files**.
3. Drag the download in as it is. A name like `projections-nfl (14).csv` or
   `projections-nfl_14.csv` is fine; there is no need to rename it.
4. Click **Commit changes**.

The next run swaps the new file in as `projections-nfl.csv`. A filled file (one with numbers
in it, like the upload itself) is refused, so the blank template can't be overwritten by mistake.
If the Discord export post says receivers aren't on the same team in the template, it is out
of date: download a new one.
