"""An empty price file (quota exhausted) is never the week's snapshot."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import props_market as PM

d = tempfile.mkdtemp()
head = ",".join(PM.SNAP_COLS) + "\n"
good = os.path.join(d, "props_2026_w12_20261125T000000Z.csv"); empty = os.path.join(d, "props_2026_w12_20261126T195708Z.csv")
open(empty, "w").write(head)
assert PM.latest_snapshot(2026, 12, d) is None, "a header-only file was treated as a snapshot"
open(good, "w").write(head + "2026-11-25T00:00:00+00:00,e,2026-11-29T18:00:00Z,H,A,dk,,player_reception_yds,Over,P,50.5,1.9,-111\n")
assert PM.latest_snapshot(2026, 12, d) == good, "the older good pull should win over a newer empty one"
print("PASS: empty price files are skipped")
