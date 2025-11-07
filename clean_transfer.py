import pandas as pd
import re

def parse_transfermarkt_value(val):
    if pd.isna(val) or val in ['-', '–', '']:
        return 0.0
    val = str(val).replace('‚Ç¨', '').replace(
        '€', '').strip()  # remove garbled € symbol
    try:
        if val.endswith('m'):
            return float(val[:-1].replace(',', '').strip()) * 1e6
        elif val.endswith('k'):
            return float(val[:-1].replace(',', '').strip()) * 1e3
        else:
            return float(val.replace(',', '').strip())
    except ValueError:
        # fallback for unexpected strings
        return 0.0

if __name__ == '__main__':
    df = pd.read_csv('sup data/team_transfers.csv')
    for col in ['Expenditure', 'Income', 'Balance']:
        df[col] = df[col].apply(parse_transfermarkt_value)

    print(df[['Expenditure', 'Income', 'Balance']].head())

    team_mapping = {
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

    df['Club'] = df['Club'].map(team_mapping).fillna(df['Club'])
    df['Expenditure'] = df['Expenditure'].astype(float)
    df['Income'] = df['Income'].astype(float)
    df['Balance'] = df['Balance'].astype(float)

    # Year to inflate to (latest year)
    latest_year = 2025

    # Apply yearly 9% increase
    df['Expenditure'] = df.apply(lambda row: row['Expenditure']
                                * (1.09 ** (latest_year - row['season_start'])), axis=1)
    df['Income'] = df.apply(lambda row: row['Income'] *
                            (1.09 ** (latest_year - row['season_start'])), axis=1)
    df['Balance'] = df.apply(lambda row: row['Balance'] *
                            (1.09 ** (latest_year - row['season_start'])), axis=1)

    df_clean = df[['Club', 'Expenditure', 'Arrivals',
                'Income', 'Departures', 'Balance', 'season_start']]

    df_clean.to_csv('sup data/transfer_data_cleaned.csv', index=False)

    print(df_clean.head())
