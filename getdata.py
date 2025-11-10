from functools import lru_cache
from io import StringIO
from typing import Optional, Literal
import soccerdata as sd
import pandas as pd
import requests

#Football-Data jut provides comman-separated value (CSV) files for various football leagues straight in the HTML
# page. This code fetches and caches those CSV files as pandas DataFrames.

#fbref.com provides data in HTML tables. This code fetches and caches specific tables from FBref pages as pandas DataFrames.

# ---------- Football-Data.co.uk ----------

FOOTBALL_DATA_BASE_URL = "https://www.football-data.co.uk/mmz4281"


def build_football_data_url(season: str, league_code: str) -> str:
    """
    Build a football-data.co.uk CSV URL.

    season: e.g. "2425" for 2024–25
    league_code: e.g. "E0" (Premier League), "E1" (Championship), etc.
    """
    return f"{FOOTBALL_DATA_BASE_URL}/{season}/{league_code}.csv"


@lru_cache(maxsize=128)
def get_football_data(season: str, league_code: str) -> pd.DataFrame:
    """
    Download and cache a football-data.co.uk league CSV in memory.

    The first call for a (season, league_code) pair hits the network.
    Subsequent calls in the same process return the cached DataFrame.
    """
    url = build_football_data_url(season, league_code)
    # You can let pandas read directly from the URL, but using requests
    # makes it easier to handle errors / timeouts if you want.
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()

    return pd.read_csv(StringIO(resp.text))


# ---------- FBref ----------

# What kind of data you want out of FBref
FBrefScope = Literal[
    "team_season",   # aggregated team stats per season
    "player_season", # aggregated player stats per season
    "schedule",      # fixtures/results
    "team_match",    # team match-by-match stats
    "player_match",  # player match-by-match stats
]

@lru_cache(maxsize=32)
def get_fbref_table(
    league: str = "ENG-Premier League",
    season: int | str = 2024,
    scope: FBrefScope = "team_season",
    stat_type: str = "standard",
) -> pd.DataFrame:
    """
    Fetch FBref data via soccerdata and cache the resulting DataFrame in memory.

    Parameters
    ----------
    league : str
        FBref / SoccerData league identifier, e.g. "ENG-Premier League".
    season : int | str
        Season identifier (various formats work, e.g. 2024, "2024-2025", "24-25").
    scope : FBrefScope
        What kind of table to return:
        - "team_season"   -> fbref.read_team_season_stats(...)
        - "player_season" -> fbref.read_player_season_stats(...)
        - "schedule"      -> fbref.read_schedule()
        - "team_match"    -> fbref.read_team_match_stats(...)
        - "player_match"  -> fbref.read_player_match_stats(...)
    stat_type : str
        Stat category for *_season_stats / *_match_stats, e.g. "standard",
        "shooting", "passing", "possession", "keeper", etc.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with FBref data.
    """
    # Create the FBref scraper instance
    fbref = sd.FBref(leagues=league, seasons=season)

    if scope == "team_season":
        df = fbref.read_team_season_stats(stat_type=stat_type)
    elif scope == "player_season":
        df = fbref.read_player_season_stats(stat_type=stat_type)
    elif scope == "schedule":
        df = fbref.read_schedule()
    elif scope == "team_match":
        df = fbref.read_team_match_stats(stat_type=stat_type)
    elif scope == "player_match":
        df = fbref.read_player_match_stats(stat_type=stat_type)
    else:
        raise ValueError(f"Unknown FBref scope: {scope}")

    return df

# Example usage:
if __name__ == "__main__":
    # Fetch Premier League data for the all seasons 2011-2025 from football-data.co.uk
    all_seasons = []
    for season in range(2011, 2026):  # inclusive of 2025
        season_str = f"{str(season)[-2:]}{str(season + 1)[-2:]}"  # e.g. "2425"
        print(f"Fetching Football-Data for {season}-{season + 1}...")
        df_football_data = get_football_data(season_str, "E0")
        all_seasons.append(df_football_data)

    df_football_data = pd.concat(all_seasons, join='outer', ignore_index=False)        
    df_football_data.to_csv(f"sup data/all_seasons_data.csv", index=False)

    # Fetch all seasons from FBref and save csv
    all_seasons = []
    for season in range(2015, 2026):  # inclusive of 2025
        print(f"Fetching data for {season} season...")
        try:
            df = get_fbref_table(
                league="ENG-Premier League",
                season=season,
                scope="player_season",
                stat_type="standard",
            )
            df["season"] = season  # add season column
            all_seasons.append(df)
        except Exception as e:
            print(f"⚠️  Skipping {season} due to error: {e}")

    # Combine all into one DataFrame
    current = pd.concat(all_seasons, ignore_index=False)
    current.to_csv("sup data/fbref_player_season_stats_2015-2025.csv", index=False)

    all_seasons = []
    for season in range(2015, 2026):  # inclusive of 2025
        print(f"Fetching data for {season} season...")
        try:
            df = get_fbref_table(
                league="ENG-Premier League",
                season=season,
                scope="schedule"
            )
            df["season"] = season  # add season column
            all_seasons.append(df)
        except Exception as e:
            print(f"⚠️  Skipping {season} due to error: {e}")
    # Combine all into one DataFrame
    team_season = pd.concat(all_seasons, ignore_index=False)
    team_season.to_csv("sup data/fbref_team_season_schedule_with_xg.csv", index=False)



