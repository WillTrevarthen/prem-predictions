from math import inf
import pandas as pd
import glob
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import unicodedata
from collections import deque
import pickle

TEAM_MAPPING = {
    'Arsenal FC': 'Arsenal',
    'Aston Villa': 'Aston Villa',
    'AFC Bournemouth': 'Bournemouth',
    'Brentford FC': 'Brentford',
    'Brighton & Hove Albion': 'Brighton',
    'Burnley FC': 'Burnley',
    'Cardiff City': 'Cardiff',
    'Chelsea FC': 'Chelsea',
    'Crystal Palace': 'Crystal Palace',
    'Everton FC': 'Everton',
    'Fulham FC': 'Fulham',
    'Huddersfield Town': 'Huddersfield',
    'Hull City': 'Hull',
    'Ipswich Town': 'Ipswich',
    'Leeds United': 'Leeds',
    'Leicester City': 'Leicester',
    'Liverpool FC': 'Liverpool',
    'Manchester City': 'Man City',
    'Manchester United': 'Man United',
    'Middlesbrough FC': 'Middlesbrough',
    'Newcastle United': 'Newcastle',
    'Norwich City': 'Norwich',
    'Southampton FC': 'Southampton',
    'Swansea City': 'Swansea',
    'Stoke City': 'Stoke',
    'Sunderland AFC': 'Sunderland',
    'Sheffield United': 'Sheffield United',
    'Tottenham Hotspur': 'Tottenham',
    'Watford FC': 'Watford',
    'West Bromwich Albion': 'West Brom',
    'West Ham United': 'West Ham',
    'Wolverhampton Wanderers': 'Wolves',
    'Nottingham Forest': "Nott'm Forest",
    'Luton Town': 'Luton',
    'Ipswich Town': 'Ipswich',
    'Fulham FC': 'Fulham',
    'Spurs': 'Tottenham',
    'Man Utd': 'Man United'
}

def collate_data():
    # Define the folder path containing the CSV files
    folder_path = 'prem-predictions/data'

    # Use glob to find all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    # Read and concatenate all CSVs into one DataFrame
    df = pd.concat((pd.read_csv(file)
                   for file in csv_files), ignore_index=True)

    return df

def previous_season_ht_leads(df):
    # Indicator columns for leading at half-time home/away
    df['home_leading_ht'] = (df['HTR'] == 'H').astype(int)
    df['away_leading_ht'] = (df['HTR'] == 'A').astype(int)

    # Aggregate by team, season — total times led at half time at home and away
    home_ht_leads = (
        df.groupby(['HomeTeam', 'season_start'])['home_leading_ht']
          .sum()
          .reset_index()
          .rename(columns={'HomeTeam': 'team', 'home_leading_ht': 'prev_season_home_ht_leads', 'season_start': 'season_start'})
    )

    away_ht_leads = (
        df.groupby(['AwayTeam', 'season_start'])['away_leading_ht']
          .sum()
          .reset_index()
          .rename(columns={'AwayTeam': 'team', 'away_leading_ht': 'prev_season_away_ht_leads', 'season_start': 'season_start'})
    )

    # Merge home and away counts per team/season
    prev_ht_leads = pd.merge(home_ht_leads, away_ht_leads, on=[
                             'team', 'season_start'], how='outer').fillna(0)

    # Shift season_start forward by 1 to represent previous season's stats applying to next season
    prev_ht_leads['season_start'] += 1

    # Merge these previous season totals back onto main df for home and away teams
    df = df.merge(
        prev_ht_leads.rename(columns={'team': 'HomeTeam'}),
        on=['HomeTeam', 'season_start'],
        how='left'
    )

    df = df.merge(
        prev_ht_leads.rename(columns={'team': 'AwayTeam'}),
        on=['AwayTeam', 'season_start'],
        how='left',
        suffixes=('_home', '_away')
    )

    # Fill NaNs for teams without previous season data (e.g. promoted teams)
    df['prev_season_home_ht_leads_home'] = df['prev_season_home_ht_leads_home'].fillna(
        0)
    df['prev_season_away_ht_leads_home'] = df['prev_season_away_ht_leads_home'].fillna(
        0)
    df['prev_season_home_ht_leads_away'] = df['prev_season_home_ht_leads_away'].fillna(
        0)
    df['prev_season_away_ht_leads_away'] = df['prev_season_away_ht_leads_away'].fillna(
        0)

    return df

def previous_season_total_wins(df):
    # Create indicator columns for home wins and away wins
    df['home_win'] = (df['FTR'] == 'H').astype(int)
    df['away_win'] = (df['FTR'] == 'A').astype(int)

    # Aggregate total home wins per home team and season
    home_wins = (
        df.groupby(['HomeTeam', 'season_start'])['home_win']
          .sum()
          .reset_index()
          .rename(columns={
              'HomeTeam': 'team',
              'home_win': 'prev_season_home_wins',
              'season_start': 'season_start'
          })
    )

    # Aggregate total away wins per away team and season
    away_wins = (
        df.groupby(['AwayTeam', 'season_start'])['away_win']
          .sum()
          .reset_index()
          .rename(columns={
              'AwayTeam': 'team',
              'away_win': 'prev_season_away_wins',
              'season_start': 'season_start'
          })
    )

    # Shift the season forward by 1 so these counts apply to the *next* season
    home_wins['season_start'] += 1
    away_wins['season_start'] += 1

    # Merge previous season home wins back onto current season data for HomeTeam
    df = df.merge(
        home_wins.rename(columns={'team': 'HomeTeam'}),
        on=['HomeTeam', 'season_start'],
        how='left'
    )

    # Merge previous season away wins back onto current season data for AwayTeam
    df = df.merge(
        away_wins.rename(columns={'team': 'AwayTeam'}),
        on=['AwayTeam', 'season_start'],
        how='left'
    )

    # Fill NaNs with 0 for teams without previous season data (e.g. promoted teams)
    df['prev_season_home_wins'] = df['prev_season_home_wins'].fillna(0)
    df['prev_season_away_wins'] = df['prev_season_away_wins'].fillna(0)

    return df

def previous_season_avg(df, team_col, value_col, season_col):
    """
    For each row, assign the average of `value_col` from the team's most recent 
    *previous* Premier League season. 
    If the team has never appeared before, defaults to 0.
    """
    # Compute per-team averages per season
    team_season_avg = (
        df.groupby([team_col, season_col])[value_col]
        .mean()
        .reset_index()
        .sort_values([team_col, season_col])
    )

    # For each team, carry forward the most recent average into later seasons
    team_season_avg[f"prev_season_avg_{value_col}"] = (
        team_season_avg.groupby(team_col)[value_col].shift()
    )

    # Backfill with the last known value (so if a team skips some seasons,
    # they keep their last PL season’s value)
    team_season_avg[f"prev_season_avg_{value_col}"] = (
        team_season_avg.groupby(
            team_col)[f"prev_season_avg_{value_col}"].ffill()
    )

    # Replace NaNs with 0 (brand new promoted teams)
    team_season_avg[f"prev_season_avg_{value_col}"] = team_season_avg[
        f"prev_season_avg_{value_col}"
    ].fillna(0)

    # Merge back into original df
    df = df.merge(
        team_season_avg[[team_col, season_col,
                         f"prev_season_avg_{value_col}"]],
        on=[team_col, season_col],
        how="left"
    )

    return df

def all_previous_seasons_avg(df, team_col, value_col, season_col):
    """
    For each row, assign the average of `value_col` over all of the team's
    previous seasons (excluding current). If no previous seasons, return 0.
    """
    # Compute per-team averages per season
    team_season_avg = (
        df.groupby([team_col, season_col])[value_col]
        .mean()
        .reset_index()
        .sort_values([team_col, season_col])
    )

    # Compute cumulative mean of previous seasons per team
    def cum_prev_mean(x):
        return x.expanding().mean().shift().fillna(0)

    team_season_avg[f"all_prev_seasons_avg_{value_col}"] = (
        team_season_avg.groupby(team_col, group_keys=False)[value_col]
        .apply(cum_prev_mean)
    )

    # Merge back into original df
    df = df.merge(
        team_season_avg[[team_col, season_col,
                         f"all_prev_seasons_avg_{value_col}"]],
        on=[team_col, season_col],
        how="left"
    )

    return df

def last_n_seasons_avg(df, team_col, value_col, season_col, n=5):
    """
    For each row, assign the average of `value_col` over the team's
    previous `n` seasons (excluding the current season). If no previous seasons, return 0.

    Parameters
    ----------
    df : DataFrame
        Must contain [team_col, season_col, value_col].
    team_col : str
        Column containing team names.
    value_col : str
        Column with the stat to average.
    season_col : str
        Column with season identifier.
    n : int
        Number of previous seasons to include in the average.
    """
    # Compute per-team averages per season
    team_season_avg = (
        df.groupby([team_col, season_col])[value_col]
        .mean()
        .reset_index()
        .sort_values([team_col, season_col])
    )

    # Compute rolling mean over previous n seasons (exclude current season)
    def rolling_prev_n(x):
        return x.shift().rolling(window=n, min_periods=1).mean().fillna(0)

    team_season_avg[f"last_n_seasons_avg_{value_col}"] = (
        team_season_avg.groupby(team_col, group_keys=False)[value_col]
        .apply(rolling_prev_n)
    )

    # Merge back into original df
    df = df.merge(
        team_season_avg[[team_col, season_col,
                         f"last_n_seasons_avg_{value_col}"]],
        on=[team_col, season_col],
        how="left"
    )

    return df

def impute_xg(df):
    """
    Impute Home xG and Away xG for seasons 2015 & 2016 
    using the mean values from seasons 2017–2019.
    
    Args:
        df (pd.DataFrame): must contain ['Season', 'Home xG', 'Away xG']
    
    Returns:
        pd.DataFrame: with imputed values
    """
    # Calculate mean xG for 2017–2019
    ref_years = [2017, 2018, 2019]
    home_mean = df.loc[df['season_start'].isin(ref_years), 'home_xg'].mean()
    away_mean = df.loc[df['season_start'].isin(ref_years), 'away_xg'].mean()
    
    # Impute for 2015 & 2016
    mask = df['season_start'].isin([2015, 2016])
    df.loc[mask, 'home_xg'] = df.loc[mask, 'home_xg'].fillna(home_mean)
    df.loc[mask, 'away_xg'] = df.loc[mask, 'away_xg'].fillna(away_mean)
    
    return df

def previous_season_total_points(df):
    # Map full time result (FTR) to points for home and away teams
    # Home team points
    df['home_points'] = df['FTR'].map({'H': 3, 'D': 1, 'A': 0})
    # Away team points
    df['away_points'] = df['FTR'].map({'A': 3, 'D': 1, 'H': 0})

    # Aggregate total home points per home team and season
    home_points = (
        df.groupby(['HomeTeam', 'season_start'])['home_points']
          .sum()
          .reset_index()
          .rename(columns={
              'HomeTeam': 'team',
              'home_points': 'prev_season_home_points',
              'season_start': 'season_start'
          })
    )

    # Aggregate total away points per away team and season
    away_points = (
        df.groupby(['AwayTeam', 'season_start'])['away_points']
          .sum()
          .reset_index()
          .rename(columns={
              'AwayTeam': 'team',
              'away_points': 'prev_season_away_points',
              'season_start': 'season_start'
          })
    )

    # Shift season forward by 1 to apply to next season
    home_points['season_start'] += 1
    away_points['season_start'] += 1

    # Merge previous season home points onto current season data
    df = df.merge(
        home_points.rename(columns={'team': 'HomeTeam'}),
        on=['HomeTeam', 'season_start'],
        how='left'
    )

    # Merge previous season away points onto current season data
    df = df.merge(
        away_points.rename(columns={'team': 'AwayTeam'}),
        on=['AwayTeam', 'season_start'],
        how='left'
    )

    # Fill NaNs with 0 for new/promoted teams without previous season data
    df['prev_season_home_points'] = df['prev_season_home_points'].fillna(0)
    df['prev_season_away_points'] = df['prev_season_away_points'].fillna(0)

    # Optionally drop the helper columns
    df.drop(columns=['home_points', 'away_points'], inplace=True)

    return df

def previous_season_total_goals(df):
    # Sum home goals per home team per season
    home_goals = (
        df.groupby(['HomeTeam', 'season_start'])[
            'FTHG']  # Full Time Home Goals
          .sum()
          .reset_index()
          .rename(columns={
              'HomeTeam': 'team',
              'FTHG': 'prev_season_total_home_goals',
              'season_start': 'season_start'
          })
    )

    # Sum away goals per away team per season
    away_goals = (
        df.groupby(['AwayTeam', 'season_start'])[
            'FTAG']  # Full Time Away Goals
          .sum()
          .reset_index()
          .rename(columns={
              'AwayTeam': 'team',
              'FTAG': 'prev_season_total_away_goals',
              'season_start': 'season_start'
          })
    )

    # Shift season forward by 1 to apply to the next season
    home_goals['season_start'] += 1
    away_goals['season_start'] += 1

    # Merge previous season total home goals for home team
    df = df.merge(
        home_goals.rename(columns={'team': 'HomeTeam'}),
        on=['HomeTeam', 'season_start'],
        how='left'
    )

    # Merge previous season total away goals for away team
    df = df.merge(
        away_goals.rename(columns={'team': 'AwayTeam'}),
        on=['AwayTeam', 'season_start'],
        how='left'
    )

    # Fill NaN with 0 for new teams
    df['prev_season_total_home_goals'] = df['prev_season_total_home_goals'].fillna(
        0)
    df['prev_season_total_away_goals'] = df['prev_season_total_away_goals'].fillna(
        0)

    return df

def fix_two_digit_year(d):
    try:
        return pd.to_datetime(d, dayfirst=True)
    except:
        return pd.to_datetime(d, dayfirst=True, format='%d/%m/%y')

def add_cumulative_points(df):
    """
    Adds cumulative points per team within the current season up to the game.
    """
    # Map FTR to points
    df['home_points_current'] = df['FTR'].map({'H': 3, 'D': 1, 'A': 0})
    df['away_points_current'] = df['FTR'].map({'A': 3, 'D': 1, 'H': 0})

    # Sort by season and date
    df = df.sort_values(by=['season_start', 'Date']).reset_index(drop=True)

    # Initialize cumulative points
    df['home_points_cum'] = 0
    df['away_points_cum'] = 0

    # Track cumulative points for each team
    home_points_dict = {}
    away_points_dict = {}

    for idx, row in df.iterrows():
        season = row['season_start']

        home_team = row['home_team_encoded']
        away_team = row['away_team_encoded']

        # Get cumulative points so far (default 0)
        df.at[idx, 'home_points_cum'] = home_points_dict.get(
            (season, home_team), 0)
        df.at[idx, 'away_points_cum'] = away_points_dict.get(
            (season, away_team), 0)

        # Update cumulative points after this game
        home_points_dict[(season, home_team)] = home_points_dict.get(
            (season, home_team), 0) + row['home_points_current']
        away_points_dict[(season, away_team)] = away_points_dict.get(
            (season, away_team), 0) + row['away_points_current']

    # Drop helper columns
    df.drop(columns=['home_points_current',
            'away_points_current'], inplace=True)

    return df

def season_so_far_avg(df, team_col, value_col, season_col, match_col):
    """
    For each match, compute the average of `value_col` for the given team
    in the current season, up to but not including the current game.
    If no previous matches, returns 0.

    Parameters
    ----------
    df : DataFrame
        Must contain [team_col, season_col, match_col, value_col].
    team_col : str
        Column containing team names (e.g. "HomeTeam" or "AwayTeam").
    value_col : str
        Column containing the stat to average (e.g. "FTHG" or "FTAG").
    season_col : str
        Column with season identifier (e.g. "season_start").
    match_col : str
        Column that gives the order of matches within a season (e.g. "Date" or "MatchID").
    """
    df = df.sort_values([team_col, season_col, match_col])

    # Compute cumulative sums & counts within season
    df[f"cum_sum_{value_col}"] = df.groupby([team_col, season_col])[
        value_col].cumsum()
    df[f"cum_count_{value_col}"] = df.groupby(
        [team_col, season_col]).cumcount()

    # Average of previous games (shifted to exclude current)
    df[f"season_so_far_avg_{value_col}"] = (
        (df[f"cum_sum_{value_col}"] - df[value_col]) /
        df[f"cum_count_{value_col}"]
    ).fillna(0)

    return df.drop(columns=[f"cum_sum_{value_col}", f"cum_count_{value_col}"])

def last5_avg(df, team_col, value_col, season_col, match_col):
    """
    For each match, compute the average of `value_col` for the given team
    in the current season, using only the *last 5 games* before this one.
    If fewer than 5 games played, average over however many exist.
    If no previous games, returns 0.

    Parameters
    ----------
    df : DataFrame
        Must contain [team_col, season_col, match_col, value_col].
    team_col : str
        Column containing team names (e.g. "HomeTeam" or "AwayTeam").
    value_col : str
        Column containing the stat to average (e.g. "FTHG" or "FTAG").
    season_col : str
        Column with season identifier (e.g. "season_start").
    match_col : str
        Column that gives the order of matches within a season (e.g. "Date" or "MatchID").
    """
    df = df.sort_values([team_col, season_col, match_col])

    # Rolling average of last 5 games, shifted to exclude current
    df[f"last5_avg_{value_col}"] = (
        df.groupby([team_col, season_col])[value_col]
        .apply(lambda s: s.shift(1).rolling(window=5, min_periods=1).mean())
        .reset_index(level=[0, 1], drop=True)
    ).fillna(0)

    return df

def add_win_streaks(df, home_col="HomeTeam", away_col="AwayTeam", result_col="FTR", date_col="Date"):
    """
    Adds win streak features for each match:
      - Home team's *home* win streak
      - Away team's *away* win streak
      - Home team's *total* win streak (all matches)
      - Away team's *total* win streak (all matches)

    Assumes `result_col` is encoded like:
        'H' = home win, 'A' = away win, 'D' = draw

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns for home team, away team, result, and date.
    home_col : str
        Column name for home team.
    away_col : str
        Column name for away team.
    result_col : str
        Column name for full-time result.
    date_col : str
        Column used to order matches chronologically.

    Returns
    -------
    df : pd.DataFrame
        Same as input but with streak features added.
    """

    # Sort chronologically
    df = df.sort_values(date_col).reset_index(drop=True)

    # Containers to track streaks
    home_home_streak = {}
    away_away_streak = {}
    team_total_streak = {}

    # Output columns
    home_home_streaks = []
    away_away_streaks = []
    home_total_streaks = []
    away_total_streaks = []

    for _, row in df.iterrows():
        home = row[home_col]
        away = row[away_col]
        result = row[result_col]

        # Get streaks BEFORE this match
        home_home_streaks.append(home_home_streak.get(home, 0))
        away_away_streaks.append(away_away_streak.get(away, 0))
        home_total_streaks.append(team_total_streak.get(home, 0))
        away_total_streaks.append(team_total_streak.get(away, 0))

        # Update streaks AFTER the match
        # Home win
        if result == "H":
            home_home_streak[home] = home_home_streak.get(home, 0) + 1
            away_away_streak[away] = 0
            team_total_streak[home] = team_total_streak.get(home, 0) + 1
            team_total_streak[away] = 0
        # Away win
        elif result == "A":
            home_home_streak[home] = 0
            away_away_streak[away] = away_away_streak.get(away, 0) + 1
            team_total_streak[home] = 0
            team_total_streak[away] = team_total_streak.get(away, 0) + 1
        # Draw
        else:  # result == "D"
            home_home_streak[home] = 0
            away_away_streak[away] = 0
            team_total_streak[home] = 0
            team_total_streak[away] = 0

    # Assign new columns
    df["home_home_win_streak"] = home_home_streaks
    df["away_away_win_streak"] = away_away_streaks
    df["home_total_win_streak"] = home_total_streaks
    df["away_total_win_streak"] = away_total_streaks

    return df

def add_losing_streaks(df, home_col="HomeTeam", away_col="AwayTeam", result_col="FTR", date_col="Date"):
    """
    Adds losing streak features for each match:
      - Home team's *home* losing streak
      - Away team's *away* losing streak
      - Home team's *total* losing streak (all matches)
      - Away team's *total* losing streak (all matches)

    Assumes `result_col` is encoded like:
        'H' = home win, 'A' = away win, 'D' = draw

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns for home team, away team, result, and date.
    home_col : str
        Column name for home team.
    away_col : str
        Column name for away team.
    result_col : str
        Column name for full-time result.
    date_col : str
        Column used to order matches chronologically.

    Returns
    -------
    df : pd.DataFrame
        Same as input but with losing streak features added.
    """

    # Sort chronologically
    df = df.sort_values(date_col).reset_index(drop=True)

    # Trackers
    home_home_losing = {}
    away_away_losing = {}
    team_total_losing = {}

    # Outputs
    home_home_streaks = []
    away_away_streaks = []
    home_total_streaks = []
    away_total_streaks = []

    for _, row in df.iterrows():
        home = row[home_col]
        away = row[away_col]
        result = row[result_col]

        # Record streaks BEFORE the match
        home_home_streaks.append(home_home_losing.get(home, 0))
        away_away_streaks.append(away_away_losing.get(away, 0))
        home_total_streaks.append(team_total_losing.get(home, 0))
        away_total_streaks.append(team_total_losing.get(away, 0))

        # Update AFTER the match
        if result == "H":  # home win => away loses
            home_home_losing[home] = 0
            away_away_losing[away] = away_away_losing.get(away, 0) + 1
            team_total_losing[home] = 0
            team_total_losing[away] = team_total_losing.get(away, 0) + 1
        elif result == "A":  # away win => home loses
            home_home_losing[home] = home_home_losing.get(home, 0) + 1
            away_away_losing[away] = 0
            team_total_losing[home] = team_total_losing.get(home, 0) + 1
            team_total_losing[away] = 0
        else:  # draw => reset both
            home_home_losing[home] = 0
            away_away_losing[away] = 0
            team_total_losing[home] = 0
            team_total_losing[away] = 0

    # Add to df
    df["home_home_losing_streak"] = home_home_streaks
    df["away_away_losing_streak"] = away_away_streaks
    df["home_total_losing_streak"] = home_total_streaks
    df["away_total_losing_streak"] = away_total_streaks

    return df

def add_goal_difference(df, season_col="season_start", home_col="HomeTeam", away_col="AwayTeam", 
                        home_goals_col="FTHG", away_goals_col="FTAG", date_col="Date"):
    """
    Adds cumulative goal difference features up to (but not including) each match:
      - Home team's goal difference so far in the *current season*
      - Away team's goal difference so far in the *current season*
    """

    # Sort by season then date
    df = df.sort_values([season_col, date_col]).reset_index(drop=True)

    # Preallocate arrays the same length as df
    home_gd_so_far = [0] * len(df)
    away_gd_so_far = [0] * len(df)

    # Process season by season
    for season, season_idx in df.groupby(season_col).groups.items():
        team_goal_diff = {}

        for idx in season_idx:  # iterate over row indices in this season
            row = df.loc[idx]
            home = row[home_col]
            away = row[away_col]
            home_goals = row[home_goals_col]
            away_goals = row[away_goals_col]

            # Record GD before this match
            home_gd_so_far[idx] = team_goal_diff.get(home, 0)
            away_gd_so_far[idx] = team_goal_diff.get(away, 0)

            # Update after the match
            team_goal_diff[home] = team_goal_diff.get(home, 0) + (home_goals - away_goals)
            team_goal_diff[away] = team_goal_diff.get(away, 0) + (away_goals - home_goals)

    # Add new columns
    df["home_goal_diff_so_far"] = home_gd_so_far
    df["away_goal_diff_so_far"] = away_gd_so_far

    return df

def add_head_to_head_lastN(df, N=8, season_col="season_start",
                           home_col="HomeTeam", away_col="AwayTeam",
                           home_goals_col="FTHG", away_goals_col="FTAG", date_col="Date"):
    """
    Adds rolling last-N head-to-head features (combined home & away), 
    but only from *previous seasons* (excluding current season matches).

    Features (before each match):
      - Wins for home team in last N vs away
      - Wins for away team in last N vs home
      - Draws in last N

    Parameters
    ----------
    df : pd.DataFrame
        Match data (must include season, teams, goals, and date).
    N : int
        Number of previous encounters to consider.
    season_col, home_col, away_col, home_goals_col, away_goals_col, date_col : str
        Column names.

    Returns
    -------
    pd.DataFrame
        With new H2H rolling features.
    """

    # Sort chronologically (by season then date)
    df = df.sort_values([season_col, date_col]).reset_index(drop=True)

    # Tracker: dict of frozenset({teamA, teamB}) -> deque of past results
    h2h_record = {}

    # Outputs
    home_wins_lastN = []
    away_wins_lastN = []
    draws_lastN = []

    # Track which season we're in
    current_season = None

    for _, row in df.iterrows():
        home = row[home_col]
        away = row[away_col]
        home_goals = row[home_goals_col]
        away_goals = row[away_goals_col]
        season = row[season_col]

        matchup = frozenset([home, away])

        # Reset (exclude current season matches from features)
        if season != current_season:
            current_season = season  # season changed

        # Get or init rolling history
        history = h2h_record.get(matchup, deque(maxlen=N))

        # --- Compute features BEFORE match ---
        # Only consider matches in history (i.e. strictly from past seasons)
        hw = sum(1 for h, a, s in history if h > a and home in (home, away) and s < season)
        aw = sum(1 for h, a, s in history if a > h and away in (home, away) and s < season)
        dr = sum(1 for h, a, s in history if h == a and s < season)

        home_wins_lastN.append(hw)
        away_wins_lastN.append(aw)
        draws_lastN.append(dr)

        # --- Update history with this match (so it'll count for future seasons) ---
        history.append((home_goals, away_goals, season))
        h2h_record[matchup] = history

    # Add new features
    df["h2h_home_wins_lastN"] = home_wins_lastN
    df["h2h_away_wins_lastN"] = away_wins_lastN
    df["h2h_draws_lastN"] = draws_lastN

    return df

def add_gameweek(df):
    # Convert Date to datetime if it's not already
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Group by season and create gameweek numbers
    df['gameweek'] = df.groupby('season_start').cumcount() // 10 + 1
    
    return df

def main():

    # Loading and prepping data for merging
    df = pd.read_csv("sup data/merged_data.csv")
    # #[[TODO]] select necessary columns properly
    # df = df.iloc[:, :24]

    # df['Date'] = df['Date'].apply(fix_two_digit_year)

    # df['HomeTeam'] = df['HomeTeam'].map(TEAM_MAPPING).fillna(df['HomeTeam'])
    # df['AwayTeam'] = df['AwayTeam'].map(TEAM_MAPPING).fillna(df['AwayTeam'])
    
    # xg_data = pd.read_csv('sup data/clean_xg.csv')

    # xg_data = xg_data[[
    #     'date', 'time', 'home_team', 'away_team', 'ftr', 'home_goals', 'away_goals', 'home_xg', 'away_xg']]
    
    # #[[TODO]] rename columns properly for merging to df

    # xg_data['date'] = pd.to_datetime(xg_data['date'], dayfirst=True, errors='coerce')

    # df= df.merge(
    #     xg_data,
    #     left_on=['Date', 'HomeTeam', 'AwayTeam'],
    #     right_on=['date', 'home_team', 'away_team'],
    #     how='left'
    # )
    # #[[TODO]] Drop redundant columns
    # df = df.drop(columns=['Home', 'Away','Wk'])



    # df['Date'] = df['Date'].apply(fix_two_digit_year)
    # df['month'] = df['Date'].dt.month
    # df['year'] = df['Date'].dt.year
    # df['day'] = df['Date'].dt.dayofweek

    # df['season_start'] = np.where(
    #     df['month'] > 6,
    #     df['year'],
    #     (df['year'] - 1)
    # )

    df = add_gameweek(df)

    df = impute_xg(df)
    df.to_csv('test.csv')
    df = df.copy()

    #[[TODO]]
    target_cols = cols = ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']

    h_cols = [col for col in target_cols if col.lower().startswith('h')]
    a_cols = [col for col in target_cols if col.lower().startswith('a')]

    # Apply the rolling average for each target column
    for col in h_cols:
        df = previous_season_avg(
            df,
            team_col='HomeTeam',
            value_col=col,
            season_col='season_start'
        )

    for col in a_cols:
        df = previous_season_avg(
            df,
            team_col='AwayTeam',
            value_col=col,
            season_col='season_start'
        )

    for col in h_cols:
        df = last_n_seasons_avg(
            df,
            team_col='HomeTeam',
            value_col=col,
            season_col='season_start'
        )

    for col in a_cols:
        df = last_n_seasons_avg(
            df,
            team_col='AwayTeam',
            value_col=col,
            season_col='season_start'
        )

    df = season_so_far_avg(df, team_col="HomeTeam", value_col="FTHG",
                           season_col="season_start", match_col="Date")

    df = season_so_far_avg(df, team_col="AwayTeam", value_col="FTAG",
                           season_col="season_start", match_col="Date")

    df = season_so_far_avg(df, team_col="HomeTeam", value_col="home_xg",
                           season_col="season_start", match_col="Date")

    df = season_so_far_avg(df, team_col="AwayTeam", value_col="away_xg",
                           season_col="season_start", match_col="Date")

    df = last5_avg(df, team_col="HomeTeam", value_col="FTHG",
                   season_col="season_start", match_col="Date")

    df = last5_avg(df, team_col="AwayTeam", value_col="FTAG",
                   season_col="season_start", match_col="Date")

    df = last5_avg(df, team_col="HomeTeam", value_col="home_xg",
                   season_col="season_start", match_col="Date")

    df = last5_avg(df, team_col="AwayTeam", value_col="away_xg",
                   season_col="season_start", match_col="Date")

    df = add_win_streaks(df)
    df = add_losing_streaks(df)
    # df = add_head_to_head_lastN(df)


    df['is_3_ko'] = np.where(
        df['Time'] == '15:00',
        1,
        0
    )

    df['is_early_ko'] = np.where(
        df['Time'] == '12:30',
        1,
        0
    )

    # team_list = df['HomeTeam'].unique()
    # pd.Series(team_list, name='Team').to_csv('team_list.csv', index=False)
    df = df.copy()
    le = LabelEncoder()
    le_result = LabelEncoder()
    df['home_team_encoded'] = le.fit_transform(df['HomeTeam'])
    df['away_team_encoded'] = le.transform(df['AwayTeam'])

    df['ft_result_encoded'] = le_result.fit_transform(df['FTR'])

    team_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    result_mapping = dict(
        zip(le_result.classes_, le_result.transform(le_result.classes_)))

    df = previous_season_ht_leads(df)
    df = previous_season_total_wins(df)
    df = previous_season_total_points(df)
    df = previous_season_total_goals(df)
    df = add_cumulative_points(df)

    # transfers = pd.read_csv('sup data/transfer_data_cleaned.csv')
    # # Merge home team data
    # df = df.merge(
    #     transfers, how='left', left_on=['HomeTeam', 'season_start'], right_on=['Club', 'season_start'],
    #     suffixes=('', '_home')
    # )

    # # Rename columns for clarity
    # df = df.rename(columns={
    #     'Expenditure': 'home_Expenditure',
    #     'Arrivals': 'home_Arrivals',
    #     'Income': 'home_Income',
    #     'Departures': 'home_Departures',
    #     'Balance': 'home_Balance'
    # })

    # # Merge away team data
    # df = df.merge(
    #     transfers, how='left', left_on=['AwayTeam', 'season_start'], right_on=['Club', 'season_start'],
    #     suffixes=('', '_away')
    # )

    # # Rename away columns
    # df = df.rename(columns={
    #     'Expenditure': 'away_Expenditure',
    #     'Arrivals': 'away_Arrivals',
    #     'Income': 'away_Income',
    #     'Departures': 'away_Departures',
    #     'Balance': 'away_Balance'
    # })

    # # Drop extra columns
    # df = df.drop(columns=['Club', 'Club_away', 'Home xG', 'Away xG'])

    # Define the columns to scale
    df.to_csv('prescaling_before.csv', index=False)
    exclude = [
        'Div', 'Date', 'Time', 'HomeTeam', 'AwayTeam', 'Referee', 'gameweek',
        'home_team_encoded', 'away_team_encoded', 'ft_result_encoded',
        'home_leading_ht', 'away_leading_ht',
        'home_win', 'away_win', 'is_3_ko', 'is_early_ko',
        'day',
        'month',
        'year',
        'HomeTeam',
        'AwayTeam',
        'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR',
        'home_win',
        'away_win',
        'season_start'
        
    ]

    columns_to_scale = [c for c in df.select_dtypes(include=['number']).columns if c not in exclude]


    df.to_csv('prescaling.csv', index=False)
    # Initialize StandardScaler
    scaler = StandardScaler()

    # Fit and transform the selected columns
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    df['data_type'] = np.where(df['FTR'].notna(), 'Training', 'Inference')

    df = df[~df['season_start'].isin([2010, 2011, 2012, 2013, 2014])]

    df = df.copy()
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df.drop(
        [   'day',
            'month',
            'year',
            'Div',
            # 'Date',
            'Time',
            'Referee',
            'HomeTeam',
            'AwayTeam',
            'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR',
            'home_win',
            'away_win',
            'home_leading_ht', 'away_leading_ht',
            'Club_away',
            'home_xg',
            'away_xg'
        ], axis=1, inplace=True
    )

    # df.to_csv('test.csv', index=False)
    inference_df = df[df['data_type'] == 'Inference']
    training_df = df[df['data_type'] == 'Training']

    inference_df.to_csv('inference.csv', index=False)
    training_df.to_csv('training.csv', index=False)
    # print(df.head(-5))

    # Save encoders + model + scaler all together
    with open("pipeline.pkl", "wb") as f:
        pickle.dump({
            "team_encoder": le,
            "result_encoder": le_result,
            "scaler": scaler
        }, f)


if __name__ == '__main__':
    main()
