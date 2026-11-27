"""
props_weather.py: kickoff-window weather for each game, and the "poor weather" tag (mbm-nfl).

    python props_weather.py            # print this week's games with their forecast and tag
    python props_weather.py --test     # unit tests only

THE RULE (set 2026-11-26, a judgment rule: no prop history with weather exists to test it)
  A game has POOR weather when it is played outdoors (or under an open roof) and, over the
  four hours from kickoff, any of these holds:
      average wind of 15 mph or more, or gusts of 30 mph or more
      a 60% or higher chance of precipitation in any hour
      any snowfall in the forecast
  In poor weather the paper ledgers take no OVERS on receiving yards, receptions or passing
  yards. Unders stand. Rushing is not restricted.

Why wind leads: outdoor games with 10+ mph wind went under the closing total 55.2% of the time
from 2006 to 2025 (n=1,244). Rain and cold alone showed nothing in the total model.

Forecasts come from Open-Meteo (free, no key). Roof and site come from data/games.csv: a dome, a
closed roof or a roof that opens is indoors; a neutral-site game is skipped, because the home team's stadium
is not where it is played. If the feed is down the tag is "unknown" and nothing is excluded.
"""
import argparse, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WIND_MPH, GUST_MPH, PRECIP_PCT, WINDOW_H = 15.0, 30.0, 60.0, 4
# Stadiums with a roof that opens. The roof state is not known before the game, and it is shut
# in exactly the weather this rule is about, so these count as indoors.
RETRACTABLE = {"ARI", "ATL", "DAL", "HOU", "IND"}
API = "https://api.open-meteo.com/v1/forecast"

# home stadium coordinates by the team codes the props grid uses
STADIUMS = {
    "ARI": (33.5276, -112.2626), "ATL": (33.7554, -84.4008), "BAL": (39.2780, -76.6227), "BUF": (42.7738, -78.7870),
    "CAR": (35.2258, -80.8528), "CHI": (41.8623, -87.6167), "CIN": (39.0955, -84.5160), "CLE": (41.5061, -81.6995),
    "DAL": (32.7473, -97.0945), "DEN": (39.7439, -105.0201), "DET": (42.3400, -83.0456), "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639), "JAX": (30.3239, -81.6373), "KC": (39.0489, -94.4839),
    "LA": (33.9535, -118.3392), "LAC": (33.9535, -118.3392), "LV": (36.0909, -115.1833), "MIA": (25.9580, -80.2389),
    "MIN": (44.9737, -93.2575), "NE": (42.0909, -71.2643), "NO": (29.9511, -90.0812), "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745), "PHI": (39.9008, -75.1675), "PIT": (40.4468, -80.0158), "SEA": (47.5952, -122.3316),
    "SF": (37.4030, -121.9700), "TB": (27.9759, -82.5033), "TEN": (36.1665, -86.7713), "WAS": (38.9076, -76.8645),
}
GAMES_ALIAS = {"LAR": "LA", "OAK": "LV", "SD": "LAC", "STL": "LA"}


def is_poor(w):
    """(poor?, reason) from a weather dict with wind_mph, precip_pct, snow_cm."""
    why = []
    if w.get("wind_mph") is not None and w["wind_mph"] >= WIND_MPH:
        why.append(f"wind {w['wind_mph']:.0f} mph")
    elif w.get("gust_mph") is not None and w["gust_mph"] >= GUST_MPH:
        why.append(f"gusts {w['gust_mph']:.0f} mph")
    if w.get("precip_pct") is not None and w["precip_pct"] >= PRECIP_PCT:
        why.append(f"{w['precip_pct']:.0f}% chance of precipitation")
    if w.get("snow_cm") is not None and w["snow_cm"] > 0:
        why.append(f"snow {w['snow_cm']:.1f} cm")
    return bool(why), ", ".join(why)


def window(hourly, kickoff_utc):
    """Summarise an Open-Meteo hourly block over the WINDOW_H hours from kickoff."""
    t = pd.to_datetime(pd.Series(hourly["time"]), utc=True)
    k = pd.Timestamp(kickoff_utc)
    k = k.tz_localize("UTC") if k.tzinfo is None else k.tz_convert("UTC")
    m = ((t >= k.floor("h")) & (t < k.floor("h") + pd.Timedelta(hours=WINDOW_H))).to_numpy()
    if not m.any():
        return None
    col = lambda name: pd.to_numeric(pd.Series(hourly.get(name, [None] * len(t))), errors="coerce")[m]
    return dict(wind_mph=round(float(col("wind_speed_10m").mean()), 1), gust_mph=round(float(col("wind_gusts_10m").max()), 1),
                precip_pct=round(float(col("precipitation_probability").max()), 0), precip_mm=round(float(col("precipitation").sum()), 1),
                snow_cm=round(float(col("snowfall").sum()), 1), temp_c=round(float(col("temperature_2m").mean()), 1))


def fetch_many(points, start, end, tries=5):
    """Hourly blocks for several (lat, lon) points in ONE request, in the order given.
    The free feed answers 429 or 503 when it is busy, so the call is retried with a longer
    wait each time; one request for the whole week keeps it well inside the feed's limits."""
    import requests, time
    params = dict(latitude=",".join(f"{p[0]:.4f}" for p in points), longitude=",".join(f"{p[1]:.4f}" for p in points),
                  timezone="UTC", wind_speed_unit="mph", start_date=start, end_date=end,
                  hourly="temperature_2m,precipitation_probability,precipitation,snowfall,wind_speed_10m,wind_gusts_10m")
    last = None
    for i in range(tries):
        try:
            r = requests.get(API, params=params, timeout=30)
            if r.status_code == 200:
                d = r.json()
                d = d if isinstance(d, list) else [d]
                if len(d) == len(points):
                    return [x["hourly"] for x in d]
                last = f"{len(d)} answers for {len(points)} points"
            else:
                last = f"HTTP {r.status_code}"
        except Exception as ex:
            last = type(ex).__name__
        if i < tries - 1:
            time.sleep(15 * (i + 1))
    raise RuntimeError(f"forecast feed failed after {tries} tries ({last})")


def roofs(season, week):
    """{home team: (roof, location)} for one week, from data/games.csv."""
    g = pd.read_csv(os.path.join(HERE, "data", "games.csv"), usecols=["season", "week", "home_team", "roof", "location"])
    g = g[(g.season == season) & (g.week == week)]
    return {GAMES_ALIAS.get(h, h): (str(r).lower(), str(l)) for h, r, l in zip(g.home_team, g.roof, g.location)}


def site(home, roof_info):
    """None if the game needs a forecast, else the tag that settles it without one."""
    roof, loc = roof_info if roof_info else ("outdoors", "Home")
    if loc != "Home":
        return dict(status="neutral", reason="", detail="neutral site, weather not checked")
    if roof in ("dome", "closed") or home in RETRACTABLE:
        return dict(status="indoors", reason="", detail="retractable roof" if home in RETRACTABLE and roof not in ("dome", "closed") else roof)
    if home not in STADIUMS:
        return dict(status="unknown", reason="", detail="no stadium on file")
    return None


def tag_from(w):
    if w is None:
        return dict(status="unknown", reason="", detail="kickoff is outside the forecast range")
    poor, why = is_poor(w)
    detail = f"{w['temp_c']:.0f}C, wind {w['wind_mph']:.0f} mph (gusts {w['gust_mph']:.0f}), precip {w['precip_pct']:.0f}%" + (f", snow {w['snow_cm']:.1f} cm" if w["snow_cm"] > 0 else "")
    return dict(status="poor" if poor else "ok", reason=why, detail=detail)


def week_tags(raw, fetcher=fetch_many, roof_map=None):
    """{team: tag dict} for every team in a props grid frame (columns team, opp, home_away,
    kickoff_utc, season, week). status is poor, ok, indoors, neutral or unknown. If the feed
    cannot be reached every outdoor game is "unknown" and nothing is excluded."""
    out, need = {}, []
    if not len(raw):
        return out
    rf = roof_map if roof_map is not None else roofs(int(raw.season.iloc[0]), int(raw.week.iloc[0]))
    games = {}
    for (team, opp, ha, kick), _ in raw.groupby(["team", "opp", "home_away", "kickoff_utc"]):
        home, away = (team, opp) if ha == "home" else (opp, team)
        games[home] = (away, kick)
    for home, (away, kick) in games.items():
        t = site(home, rf.get(home))
        if t is None:
            need.append(home)
        out[home] = out[away] = t
    if need:
        ks = [pd.Timestamp(games[h][1]) for h in need]
        ks = [k.tz_localize("UTC") if k.tzinfo is None else k.tz_convert("UTC") for k in ks]
        try:
            blocks = fetcher([STADIUMS[h] for h in need], min(ks).strftime("%Y-%m-%d"), (max(ks) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
            for h, k, blk in zip(need, ks, blocks):
                out[h] = out[games[h][0]] = tag_from(window(blk, k))
        except Exception as ex:
            for h in need:
                out[h] = out[games[h][0]] = dict(status="unknown", reason="", detail=f"forecast unavailable ({ex})")
    return out


def unit_tests():
    hourly = dict(time=[f"2026-11-29T{h:02d}:00" for h in range(24)], temperature_2m=[5] * 24, precipitation_probability=[10] * 18 + [70] + [10] * 5,
                  precipitation=[0] * 24, snowfall=[0] * 24, wind_speed_10m=[8] * 18 + [20, 20, 20, 20] + [8] * 2, wind_gusts_10m=[15] * 18 + [31] * 4 + [15] * 2)
    w = window(hourly, "2026-11-29T18:00:00+00:00")
    assert w["wind_mph"] == 20.0 and w["gust_mph"] == 31.0 and w["precip_pct"] == 70.0 and w["snow_cm"] == 0.0, w
    assert window(hourly, "2026-11-29T10:05:00+00:00")["wind_mph"] == 8.0
    assert window(hourly, "2026-12-05T18:00:00+00:00") is None
    assert is_poor(dict(wind_mph=15.0, precip_pct=0, snow_cm=0))[0] and not is_poor(dict(wind_mph=14.9, precip_pct=59, snow_cm=0))[0]
    assert is_poor(dict(wind_mph=3, precip_pct=60, snow_cm=0))[0] and is_poor(dict(wind_mph=3, precip_pct=10, snow_cm=0.2))[0]
    hrs = [f"2026-11-{d}T{h:02d}:00" for d in (29, 30) for h in range(24)]
    blk = lambda wind, gust, pp: dict(time=hrs, temperature_2m=[5] * 48, precipitation_probability=[pp] * 48, precipitation=[0] * 48,
                                      snowfall=[0] * 48, wind_speed_10m=[wind] * 48, wind_gusts_10m=[gust] * 48)
    raw = pd.DataFrame(dict(season=2026, week=12, team=["CLE", "LV", "MIA", "NYJ", "MIN", "ATL", "HOU", "BAL", "JAX", "TEN"],
                            opp=["LV", "CLE", "NYJ", "MIA", "ATL", "MIN", "BAL", "HOU", "TEN", "JAX"], home_away=["home", "away"] * 5,
                            kickoff_utc=["2026-11-30T01:20:00+00:00"] * 2 + ["2026-11-29T18:00:00+00:00"] * 8))
    rm = {"CLE": ("outdoors", "Home"), "MIA": ("outdoors", "Home"), "MIN": ("dome", "Home"), "HOU": ("nan", "Home"), "JAX": ("outdoors", "Neutral")}
    seen = {}
    def fake(points, start, end):
        seen["n"] = len(points); seen["dates"] = (start, end)
        return [blk(19, 33, 10) if abs(p[0] - 41.5061) < 1e-6 else blk(5, 9, 10) for p in points]
    t = week_tags(raw, fetcher=fake, roof_map=rm)
    assert seen["n"] == 2 and seen["dates"] == ("2026-11-29", "2026-12-01"), seen          # one request, outdoor games only
    assert t["CLE"]["status"] == "poor" and t["LV"]["status"] == "poor" and t["MIA"]["status"] == "ok" and t["NYJ"]["status"] == "ok"
    assert t["MIN"]["status"] == "indoors" and t["HOU"]["status"] == "indoors" and t["BAL"]["detail"] == "retractable roof"   # a roof that opens counts as indoors
    assert t["JAX"]["status"] == "neutral" and t["TEN"]["status"] == "neutral"
    def down(points, start, end): raise RuntimeError("feed down")
    d = week_tags(raw, fetcher=down, roof_map=rm)
    assert d["CLE"]["status"] == "unknown" and d["MIN"]["status"] == "indoors"                # a dead feed excludes nothing
    assert is_poor(dict(wind_mph=14.0, gust_mph=35.0, precip_pct=40, snow_cm=0)) == (True, "gusts 35 mph")
    assert set(STADIUMS) >= {"LA", "LAC", "LV", "JAX", "WAS", "GB", "NO", "NE", "SF", "TB", "KC"} and len(STADIUMS) == 32
    print("props_weather unit tests passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.test:
        unit_tests(); sys.exit(0)
    raw = pd.read_csv(os.path.join(HERE, "logs", "props_grid_raw.csv"), low_memory=False)
    raw = raw[raw.home_away == "home"].drop_duplicates("team")
    tags = week_tags(raw)
    for r in raw.sort_values("kickoff_utc").itertuples():
        t = tags[r.team]
        print(f"{r.opp:>3} @ {r.team:<3} {str(r.kickoff_utc)[:16]}  {t['status']:8s} {t['detail']}" + (f"  <- {t['reason']}" if t["reason"] else ""))
