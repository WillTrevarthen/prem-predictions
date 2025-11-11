import numpy as np
import pandas as pd
# from getdata import get_fbref_table               #import the function to get FBref data       
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

def minute_weighted_group_aggregate(
    df: pd.DataFrame,
    group_cols=('season', 'team', 'RoleGroup'),
    sum_stats=('MP', 'Min', '90s'),
    weighted_stats=('PrgC_per90', 'PrgP_per90', 'PrgR_per90', 'Gls', 'Ast', 'G+A', 'G-PK', 'G+A-PK', 'xG', 'xAG', 'xG+xAG', 'npxG', 'npxG+xAG'),
) -> pd.DataFrame:
    """
    Compute minutes-weighted averages of selected stats within role groups.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least 'Min' and the given stats.
    group_cols : tuple or list
        Columns to group by (e.g. ['season', 'team', 'RoleGroup']).
    sum_stats : tuple or list
        Columns whose values should simply be summed (like MP, Min, 90s).
    weighted_stats : tuple or list
        Columns whose values should be minutes-weighted averaged.
    
    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with one row per group.
    """
    def agg_func(x):
        data = {}
        # Sum stats like MP, Min, 90s
        for stat in sum_stats:
            data[stat] = x[stat].sum()
        
        total_min = data.get('Min', x['Min'].sum())
        
        # Weighted average for rate stats
        for stat in weighted_stats:
            if total_min > 0:
                data[stat] = (x[stat] * x['Min']).sum() / total_min
            else:
                data[stat] = 0.0
        
        return pd.Series(data)
    
    grouped = df.groupby(list(group_cols), dropna=False).apply(agg_func).reset_index()
    return grouped


def flatten_col(col):
    if isinstance(col, tuple):
        top, sub = col
        if sub != '':
            return sub          # use second level: 'Min', 'Gls', 'xG', ...
        else:
            return top          # use first level when second is empty: 'nation', 'RoleGroup'
    else:
        return col              # already a simple string


def simplify_pos(pos: str) -> str: #function to simplify player positions into main categories              
    if not pos or pd.isna(pos):
        return "Unknown"

    pos = pos.upper().replace(" ", "")
    roles = pos.split(",")

    # Define role priority (defensive first)
    priority = ["GK", "DF", "MF", "FW"]    #list of roles in order of priority

    for r in priority:
        if r in roles:
            return {"GK": "GK", "DF": "DEF", "MF": "MID", "FW": "ATT"}[r]

    return "Unknown"


if __name__ == '__main__':


    current = pd.read_csv('sup data/fbref_player_season_stats_2015-2025.csv')

    current["RoleGroup"] = current["pos"].apply(simplify_pos)

    playing_time_chosen_stats = ['MP', 'Min', '90s']
    progression_chosen_stats = ['PrgC', 'PrgP', 'PrgR']

    # Get all second-level column names under "Per 90 Minutes"
    per90_stats = current.filter(like='Per 90 Minutes').columns.tolist()
    per90_stats = current.loc[0, per90_stats].tolist()



    chosen_stats = (
        playing_time_chosen_stats +
        progression_chosen_stats +
        per90_stats
    )

    fixed_cols = 4
    new_cols = list(current.columns)
    new_cols[fixed_cols:] = current.iloc[0, fixed_cols:].astype(str).tolist()
    current.columns = new_cols
    current = current.drop(current.index[0]).reset_index(drop=True)


    # Start from your stats subset
    current_subset = current[chosen_stats].copy()

    current_subset = current_subset.reset_index()


    current_subset.columns = [flatten_col(c) for c in current_subset.columns]

    current_subset['PrgC_per90'] = np.where(
        current_subset['90s'] > 0,
        current_subset['PrgC'] / current_subset['90s'],
        0
    )

    current_subset['PrgP_per90'] = np.where(
        current_subset['90s'] > 0,
        current_subset['PrgP'] / current_subset['90s'],
        0
    )

    current_subset['PrgR_per90'] = np.where(
        current_subset['90s'] > 0,
        current_subset['PrgR'] / current_subset['90s'],
        0
    )

    grouped_stats = minute_weighted_group_aggregate(current_subset)
    grouped_stats = grouped_stats[grouped_stats['RoleGroup']!='GK']
    grouped_stats = grouped_stats.drop(columns=['Min', 'MP', '90s'])

    id_cols = ['season', 'team']
    value_cols = [c for c in grouped_stats.columns if c not in id_cols + ['RoleGroup']]

    wide = (
        grouped_stats
        .set_index(id_cols + ['RoleGroup'])[value_cols]
        .unstack('RoleGroup')
    )

    wide.columns = [f"{stat}_{role}" for stat, role in wide.columns]
    grouped_stats_wide = wide.reset_index()

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

    grouped_stats_wide['team'] = grouped_stats_wide['team'].map(team_mapping).fillna(grouped_stats_wide['team'])

    grouped_stats_wide = grouped_stats_wide.replace([pd.NA, pd.NaT, float('inf'), float('-inf')], np.nan)

    imputer = SimpleImputer(strategy='median')
    grouped_stats_wide.iloc[:, 2:] = imputer.fit_transform(grouped_stats_wide.iloc[:, 2:])

    pca = PCA(n_components=15, random_state=42)
    features = grouped_stats_wide.drop(columns=['season', 'team'])
    pca_components = pca.fit_transform(features)
    pca_df = pd.DataFrame(pca_components, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
    final_df = pd.concat([grouped_stats_wide[['season', 'team']].reset_index(drop=True), pca_df], axis=1)
    final_df.to_csv('sup data/cleaned_player_stats.csv', index=False)








