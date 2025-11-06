import numpy as np
import pandas as pd
import pickle


def simulate_season_predictions(df, model, feature_cols, result_encoder=None,
                                date_col=None, training_df=None):
    """
    Simulate a season chronologically and predict each game, updating **home/away cumulative points separately**.
    Optionally seed cumulative points from an existing training_df.
    """

    df = df.copy()

    # --- Sort by season and date if provided ---
    if date_col and date_col in df.columns:
        order_col = pd.to_datetime(
            df[date_col], dayfirst=True, errors='coerce')
        df = df.assign(_order=order_col).sort_values(['season_start', '_order']).drop(
            columns=['_order']).reset_index(drop=True)
    else:
        df = df.sort_values(['season_start']).reset_index(drop=True)

    # Initialize storage for cumulative points BEFORE each game
    home_points_before = []
    away_points_before = []
    preds_int = []

    # Running tables: separate home/away cumulative points
    home_points_dict = {}  # {(season, team_id): cumulative home points}
    away_points_dict = {}  # {(season, team_id): cumulative away points}

    # --- Seed cumulative points from training data ---
    if training_df is not None:
        for _, row in training_df.iterrows():
            season = row['season_start']
            h_key = (season, row['home_team_encoded'])
            a_key = (season, row['away_team_encoded'])

            # Use actual FTR from training set
            h_pts = home_points_dict.get(h_key, 0)
            a_pts = away_points_dict.get(a_key, 0)

            if result_encoder.inverse_transform([row['ft_result_encoded']])[0] == 'H':
                home_points_dict[h_key] = h_pts + 3
                away_points_dict[a_key] = a_pts + 0
            elif result_encoder.inverse_transform([row['ft_result_encoded']])[0] == 'D':
                home_points_dict[h_key] = h_pts + 1
                away_points_dict[a_key] = a_pts + 1
            elif result_encoder.inverse_transform([row['ft_result_encoded']])[0] == 'A':
                home_points_dict[h_key] = h_pts + 0
                away_points_dict[a_key] = a_pts + 3

    needs_home_cum = 'home_points_cum' in feature_cols
    needs_away_cum = 'away_points_cum' in feature_cols

    # --- Predict each inference row ---
    for _, row in df.iterrows():
        season = row['season_start']
        h_key = (season, row['home_team_encoded'])
        a_key = (season, row['away_team_encoded'])

        # cumulative points BEFORE this game
        h_pts = home_points_dict.get(h_key, 0)
        a_pts = away_points_dict.get(a_key, 0)

        home_points_before.append(h_pts)
        away_points_before.append(a_pts)

        # Build feature vector
        feat_vals = []
        for col in feature_cols:
            if col == 'home_points_cum' and needs_home_cum:
                feat_vals.append(h_pts)
            elif col == 'away_points_cum' and needs_away_cum:
                feat_vals.append(a_pts)
            else:
                feat_vals.append(row[col] if col in df.columns else np.nan)

        X = np.array(feat_vals, dtype=float).reshape(1, -1)

        # Predict
        pred_int = model.predict(X)[0]
        preds_int.append(pred_int)

        # Decode label
        if result_encoder is not None:
            pred_label = result_encoder.inverse_transform([pred_int])[0]
        else:
            mapping = {0: 'H', 1: 'D', 2: 'A'}
            pred_label = mapping[pred_int]

        # Update cumulative points
        if pred_label == 'H':
            home_points_dict[h_key] = h_pts + 3
            away_points_dict[a_key] = a_pts + 0
        elif pred_label == 'D':
            home_points_dict[h_key] = h_pts + 1
            away_points_dict[a_key] = a_pts + 1
        else:  # 'A'
            home_points_dict[h_key] = h_pts + 0
            away_points_dict[a_key] = a_pts + 3

    # Attach cumulative points BEFORE the match and predicted results
    df['home_points_cum'] = home_points_before
    df['away_points_cum'] = away_points_before
    df['predicted_FTR_int'] = preds_int
    if result_encoder is not None:
        df['predicted_FTR'] = result_encoder.inverse_transform(
            df['predicted_FTR_int'])

    return df


def main():

    # Load processed inference
    inference_df = pd.read_csv('inference2.csv')
    inference_df = inference_df.drop(
        columns=['data_type', 'ft_result_encoded'])
    # Load model + encoders
    with open("rf_model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    with open("xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open("pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)

    result_encoder = pipeline["result_encoder"]
    team_encoder = pipeline['team_encoder']
    # Build features list directly from training
    train = pd.read_csv('training.csv')
    feature_cols = [c for c in train.columns if c not in [
        'ft_result_encoded', 'data_type', 'Date']]

    # Simulate (inference2.csv usually has no Date; pass date_col=None to keep given order)
    pred_df = simulate_season_predictions(
        inference_df, rf_model, feature_cols, result_encoder=result_encoder, date_col='Date', training_df=train
    )

    pred_df['HomeTeam'] = team_encoder.inverse_transform(
        pred_df['home_team_encoded'])
    pred_df['AwayTeam'] = team_encoder.inverse_transform(
        pred_df['away_team_encoded'])

    # Get predicted probabilities for each class
    probs = rf_model.predict_proba(inference_df[feature_cols])

    pred_df['prob_home_win'] = probs[:, 0]
    pred_df['prob_draw'] = probs[:, 1]
    pred_df['prob_away_win'] = probs[:, 2]
    # Optionally, the confidence in the predicted class
    pred_df['predicted_confidence'] = pred_df[['prob_home_win','prob_draw','prob_away_win']].max(axis=1)

    pred_df_clean = pred_df[['Date', 'HomeTeam', 'AwayTeam', 'predicted_FTR', 'predicted_confidence']]

    pred_df_clean.to_csv('results/GW2 results/season_predictions.csv', index=False)
    print(pred_df[['home_team_encoded', 'away_team_encoded',
          'home_points_cum', 'away_points_cum', 'predicted_FTR']].head())

    # Assuming your DataFrame is called pred_df
    pred_df['home_points'] = pred_df['predicted_FTR'].map(
        {'H': 3, 'D': 1, 'A': 0})
    pred_df['away_points'] = pred_df['predicted_FTR'].map(
        {'A': 3, 'D': 1, 'H': 0})

    # --- Home points ---
    home_pts_sum = (
        pred_df.groupby('HomeTeam')['home_points']
        .sum()
        .reset_index()
        .rename(columns={'HomeTeam': 'team', 'home_points': 'home_points_total'})
    )

    home_pts_first_cum = (
        pred_df.groupby('HomeTeam')['home_points_cum']
        .first()
        .reset_index()
        .rename(columns={'HomeTeam': 'team', 'home_points_cum': 'home_points_cum_first'})
    )

    home_pts = home_pts_sum.merge(home_pts_first_cum, on='team', how='left')
    home_pts['home_points_total'] += home_pts['home_points_cum_first']
    home_pts = home_pts.drop(columns=['home_points_cum_first'])

    # --- Away points ---
    away_pts_sum = (
        pred_df.groupby('AwayTeam')['away_points']
        .sum()
        .reset_index()
        .rename(columns={'AwayTeam': 'team', 'away_points': 'away_points_total'})
    )

    away_pts_first_cum = (
        pred_df.groupby('AwayTeam')['away_points_cum']
        .first()
        .reset_index()
        .rename(columns={'AwayTeam': 'team', 'away_points_cum': 'away_points_cum_first'})
    )

    away_pts = away_pts_sum.merge(away_pts_first_cum, on='team', how='left')
    away_pts['away_points_total'] += away_pts['away_points_cum_first']
    away_pts = away_pts.drop(columns=['away_points_cum_first'])

    # Merge home and away points
    table = pd.merge(home_pts, away_pts, on='team', how='outer').fillna(0)

    # Total points
    table['total_points'] = table['home_points_total'] + \
        table['away_points_total']

    # Sort descending by total points
    table = table.sort_values(
        'total_points', ascending=False).reset_index(drop=True)
    table.to_csv('results/GW2 results/predicted_table.csv', index=False)


if __name__ == '__main__':
    main()
