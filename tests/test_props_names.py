"""python tests/test_props_names.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, props_market as PM
cases = {"Ja'Marr Chase": "jamarrchase", "Tre' Harris": "treharris", "Amon-Ra St. Brown": "amonrastbrown",
         "Kenneth Walker III": "kennethwalker", "Chris Godwin Jr.": "chrisgodwin", "Ja\u2019Marr Chase": "jamarrchase",
         "Amon-Ra St Brown": "amonrastbrown", "Kenneth Walker": "kennethwalker", "José Ramírez": "joseramirez"}
for k, v in cases.items():
    assert PM.norm_name(k) == v, (k, PM.norm_name(k), v)
roster = pd.DataFrame(dict(full_name=["Ja'Marr Chase", "Kenneth Walker III", "Chris Godwin Jr.", "Mike Evans", "Mike Evans"],
                           gsis_id=["a", "b", "c", "d", "e"], team=["CIN", "KC", "TB", "TB", "SEA"]))
by_name, by_team = PM.name_map(roster)
assert PM.match_player("Ja'Marr Chase", ("CIN", "CLE"), by_name, by_team) == "a"
assert PM.match_player("Kenneth Walker", ("DEN", "KC"), by_name, by_team) == "b"
assert PM.match_player("Chris Godwin", ("TB", "CIN"), by_name, by_team) == "c"
assert PM.match_player("Mike Evans", ("TB", "CIN"), by_name, by_team) == "d"     # team-qualified
assert PM.match_player("Mike Evans", ("NYG", "DAL"), by_name, by_team) is None   # ambiguous, no team match
print("PASS: name normalisation and matching")
