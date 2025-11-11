import pandas as pd
from dateutil import parser 
import numpy as np

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

columns = [
    "Div",
    "Date",
    "Time",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HTHG",
    "HTAG",
    "HTR",
    "Referee",
    "HS",
    "AS",
    "HST",
    "AST",
    "HC",
    "AC",
    "HF",
    "AF",
    "HY",
    "AY",
    "HR",
    "AR"
]

FORMATS = ['%d/%m/%Y', '%Y-%m-%d', '%d-%b-%Y', '%d %B %Y', '%Y/%m/%d']
def parse_with_formats(x):
    for fmt in FORMATS:
        try:
            return pd.to_datetime(x, format=fmt)
        except Exception:
            pass
    # last resort
    try:
        return parser.parse(x, dayfirst=True)
    except Exception:
        return pd.NaT



def fix_two_digit_year(d):
    try:
        return pd.to_datetime(d, dayfirst=True)
    except:
        return pd.to_datetime(d, dayfirst=True, format='%d/%m/%y')



if __name__ == "__main__":

    # Loading and prepping data for merging
    df = pd.read_csv("sup data/all_seasons_data.csv")
    #[[TODO]] select necessary columns properly
    df = df[columns]

    df['Date'] = df['Date'].apply(fix_two_digit_year)

    df['HomeTeam'] = df['HomeTeam'].map(TEAM_MAPPING).fillna(df['HomeTeam'])
    df['AwayTeam'] = df['AwayTeam'].map(TEAM_MAPPING).fillna(df['AwayTeam'])

    # Merging xg data
    xg_data = pd.read_csv('sup data/clean_xg.csv', parse_dates=['date'], dayfirst=True)

    xg_data = xg_data[[
        'date', 'time', 'home_team', 'away_team', 'home_xg', 'away_xg']]
    
    #[[TODO]] rename columns properly for merging to df

    xg_data = xg_data.rename(columns={'date': 'Date',
                                      'time': 'Time',
                                      'home_team': 'HomeTeam',
                                      'away_team': 'AwayTeam'})


    xg_data['Date'] = xg_data['Date'].map(lambda x: parse_with_formats(x) if pd.notna(x) else pd.NaT)

    df= df.merge(
        xg_data,
        left_on=['Date', 'Time', 'HomeTeam', 'AwayTeam'],
        right_on=['Date', 'Time', 'HomeTeam', 'AwayTeam'],
        how='outer'
    )

    df = df.dropna(how='all')
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['day'] = df['Date'].dt.dayofweek

    df['season_start'] = np.where(
        df['month'] > 6,
        df['year'],
        (df['year'] - 1)
    )


    # Merging transfer data
    transfer_data = pd.read_csv('sup data/transfer_data_cleaned.csv')
    
    df = df.merge(
        transfer_data, how='left', left_on=['HomeTeam', 'season_start'], right_on=['Club', 'season_start'],
        suffixes=('', '_home')
    )
    
    # Rename columns for clarity
    df = df.rename(columns={
        'Expenditure': 'home_Expenditure',
        'Arrivals': 'home_Arrivals',
        'Income': 'home_Income',
        'Departures': 'home_Departures',
        'Balance': 'home_Balance'
    })

    # Merge away team data
    df = df.merge(
        transfer_data, how='left', left_on=['AwayTeam', 'season_start'], right_on=['Club', 'season_start'],
        suffixes=('', '_away')
    )

    # Rename away columns
    df = df.rename(columns={
        'Expenditure': 'away_Expenditure',
        'Arrivals': 'away_Arrivals',
        'Income': 'away_Income',
        'Departures': 'away_Departures',
        'Balance': 'away_Balance'
    })


    # Merging Player data
    player_data = pd.read_csv('sup data/fbref_player_season_stats_2015-2025.csv')

    df = df.merge(
        player_data, how='left', left_on=['HomeTeam', 'season_start'], right_on=['Club', 'season_start'],
        suffixes=('', '_home')
    )
    

    # Merge away team data
    df = df.merge(
        player_data, how='left', left_on=['AwayTeam', 'season_start'], right_on=['Club', 'season_start'],
        suffixes=('', '_away')
    )


    #[[TODO]] Drop redundant columns
    df = df.drop(columns=['Home', 'Away','Wk'])

    pd.to_csv(df, 'sup data/merged_data.csv', index=False)
