from math import inf
import pandas as pd
import glob
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import unicodedata


def collate_data():
    # Define the folder path containing the CSV files
    folder_path = 'premier league/data'

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

    # Compute previous season averages
    prev_avg = (
        df.groupby([team_col, season_col])[value_col]
        .mean()
        .reset_index()
        .rename(columns={value_col: f"prev_season_avg_{value_col}"})
    )
    prev_avg[season_col] += 1  # shift forward so we can map to next season

    # Merge previous season averages onto main dataframe
    df = df.merge(
        prev_avg,
        on=[team_col, season_col],
        how='left'
    )

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


def main():
    df = collate_data()
    df = df.iloc[:, :24]

    df = df.iloc[370:386]
    print(df['Date'])
    # format="%d/%m/%Y"

    def fix_two_digit_year(d):
        try:
            return pd.to_datetime(d, dayfirst=True)
        except:
            return pd.to_datetime(d, dayfirst=True, format='%d/%m/%y')

    df['Date'] = df['Date'].apply(fix_two_digit_year)
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['day'] = df['Date'].dt.dayofweek

    print(df[['Date', 'month']])


if __name__ == '__main__':
    main()
