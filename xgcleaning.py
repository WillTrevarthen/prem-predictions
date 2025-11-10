import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('sup data/fbref_team_season_schedule_with_xg.csv')
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

    df['date'] = pd.to_datetime(df['date'])

    df['home_team'] = df['home_team'].map(team_mapping).fillna(df['home_team'])
    df['away_team'] = df['away_team'].map(team_mapping).fillna(df['away_team'])

    df["score"] = df["score"].astype(str)

    df["score"] = (
        df["score"]
        .str.replace("‚Äì", "-", regex=False) 
        .str.replace("–", "-", regex=False)   
    )

    df[["home_goals", "away_goals"]] = (
        df["score"].str.extract(r"(\d+)-(\d+)").astype('Int64')
    )

    
    df.to_csv('sup data/clean_xg.csv', index=False)
