import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('sup data/xgpermatch.csv')

    df = df[df['Home xG'] != 'xG']
    df = df.reset_index(drop=True)

    print(df.head())

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
        'Leeds Utd': 'Leeds',
        'Leicester City': 'Leicester',
        'Liverpool FC': 'Liverpool',
        'Manchester City': 'Man City',
        'Manchester United': 'Man United',
        'Manchester Utd': 'Man United',
        'Middlesbrough FC': 'Middlesbrough',
        'Newcastle United': 'Newcastle',
        'Newcastle Utd': 'Newcastle',
        'Norwich City': 'Norwich',
        'Southampton FC': 'Southampton',
        'Swansea City': 'Swansea',
        'Stoke City': 'Stoke',
        'Sunderland AFC': 'Sunderland',
        'Sheffield Utd': 'Sheffield United',
        'Tottenham Hotspur': 'Tottenham',
        'Watford FC': 'Watford',
        'West Bromwich Albion': 'West Brom',
        'West Ham United': 'West Ham',
        'West Ham Utd': 'West Ham',
        'Wolverhampton Wanderers': 'Wolves',
        'Nottingham Forest': "Nott'm Forest",
        "Nott'ham Forest": "Nott'm Forest",
        'Luton Town': 'Luton',
        'Ipswich Town': 'Ipswich',
        'Fulham FC': 'Fulham',
        'Spurs': 'Tottenham',
        'Man Utd': 'Man United'
    }

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    df['Home'] = df['Home'].map(team_mapping).fillna(df['Home'])
    df['Away'] = df['Away'].map(team_mapping).fillna(df['Away'])
    df.to_csv('sup data/clean_xg.csv', index=False)
