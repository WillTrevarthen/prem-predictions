import pandas as pd
import numpy as np
import pickle


def simulate_season_predictions(
    df,
    model,
    feature_cols,
    result_encoder=None,
    date_col=None,
    training_df=None,
    h2h_N=5,
):
    """
    Simulate a season chronologically and predict each game, updating:

      - home/away cumulative points (home_points_cum / away_points_cum)
      - home/away home-only win/losing streaks
      - home/away total win/losing streaks
      - H2H last-N counts

    State (points, streaks, H2H) is seeded from `training_df`:
      - only for the seasons present in `df`
      - only for matches BEFORE the first inference match in that season.
    """

    df = df.copy()

    # --- Sort inference block by season + date ---
    if date_col and date_col in df.columns:
        order_col = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
        df = (
            df.assign(_order=order_col)
              .sort_values(["season_start", "_order"])
              .drop(columns=["_order"])
              .reset_index(drop=True)
        )
    else:
        df = df.sort_values(["season_start"]).reset_index(drop=True)

    # --- State dictionaries ---
    home_points_dict, away_points_dict = {}, {}
    home_home_win_dict, home_home_loss_dict = {}, {}
    away_away_win_dict, away_away_loss_dict = {}, {}
    total_win_streak_dict, total_loss_streak_dict = {}, {}
    # h2h_history = {}

    # --- Helper: update dicts after a result (true or predicted) ---
    def update_dicts(season, h_team, a_team, result):
        h_key, a_key = (season, h_team), (season, a_team)

        # Points
        h_pts = home_points_dict.get(h_key, 0)
        a_pts = away_points_dict.get(a_key, 0)
        if result == "H":
            home_points_dict[h_key] = h_pts + 3
            away_points_dict[a_key] = a_pts
        elif result == "A":
            home_points_dict[h_key] = h_pts
            away_points_dict[a_key] = a_pts + 3
        elif result == "D":
            home_points_dict[h_key] = h_pts + 1
            away_points_dict[a_key] = a_pts + 1

        # Streaks
        h_home_win = home_home_win_dict.get(h_key, 0)
        h_home_loss = home_home_loss_dict.get(h_key, 0)
        a_away_win = away_away_win_dict.get(a_key, 0)
        a_away_loss = away_away_loss_dict.get(a_key, 0)
        h_total_win = total_win_streak_dict.get(h_key, 0)
        h_total_loss = total_loss_streak_dict.get(h_key, 0)
        a_total_win = total_win_streak_dict.get(a_key, 0)
        a_total_loss = total_loss_streak_dict.get(a_key, 0)

        if result == "H":
            home_home_win_dict[h_key] = h_home_win + 1
            home_home_loss_dict[h_key] = 0

            away_away_loss_dict[a_key] = a_away_loss + 1
            away_away_win_dict[a_key] = 0

            total_win_streak_dict[h_key] = h_total_win + 1
            total_loss_streak_dict[h_key] = 0

            total_loss_streak_dict[a_key] = a_total_loss + 1
            total_win_streak_dict[a_key] = 0

        elif result == "A":
            away_away_win_dict[a_key] = a_away_win + 1
            away_away_loss_dict[a_key] = 0

            home_home_loss_dict[h_key] = h_home_loss + 1
            home_home_win_dict[h_key] = 0

            total_win_streak_dict[a_key] = a_total_win + 1
            total_loss_streak_dict[a_key] = 0

            total_loss_streak_dict[h_key] = h_total_loss + 1
            total_win_streak_dict[h_key] = 0

        else:  # Draw
            home_home_win_dict[h_key] = 0
            home_home_loss_dict[h_key] = 0
            away_away_win_dict[a_key] = 0
            away_away_loss_dict[a_key] = 0
            total_win_streak_dict[h_key] = 0
            total_loss_streak_dict[h_key] = 0
            total_win_streak_dict[a_key] = 0
            total_loss_streak_dict[a_key] = 0

        # H2H history (orientation-specific)
        # h2h_key = (season, h_team, a_team)
        # h2h_history.setdefault(h2h_key, []).append(result)

    # --- Seed state from training_df (true results only) ---
    if training_df is not None and result_encoder is not None:
        past_df = training_df.copy()

        # Restrict to the same seasons as inference
        seasons_in_inf = df["season_start"].unique()
        past_df = past_df[past_df["season_start"].isin(seasons_in_inf)].copy()

        if date_col and date_col in past_df.columns and date_col in df.columns:
            past_df[date_col] = pd.to_datetime(
                past_df[date_col], dayfirst=True, errors="coerce"
            )
            df_dates = pd.to_datetime(
                df[date_col], dayfirst=True, errors="coerce"
            )

            # earliest inference date per season
            first_inf_by_season = (
                df.assign(_date=df_dates)
                  .groupby("season_start")["_date"]
                  .min()
                  .to_dict()
            )

            # keep only past games BEFORE earliest inference match in that season
            def is_before_inference(r):
                season = r["season_start"]
                cutoff = first_inf_by_season.get(season, None)
                if cutoff is None or pd.isna(r[date_col]):
                    return False
                return r[date_col] < cutoff

            past_df = past_df[past_df.apply(is_before_inference, axis=1)]

            past_df = past_df.sort_values(["season_start", date_col])
        else:
            past_df = past_df.sort_values(["season_start"])

        # Replay past matches
        if "ft_result_encoded" in past_df.columns:
            for _, row in past_df.iterrows():
                if pd.isna(row.get("ft_result_encoded")):
                    continue
                season = row["season_start"]
                h_team = row["home_team_encoded"]
                a_team = row["away_team_encoded"]
                result = result_encoder.inverse_transform(
                    [int(row["ft_result_encoded"])]
                )[0]
                update_dicts(season, h_team, a_team, result)

        print(f"Seeded state from {len(past_df)} past games (matching seasons + before inference)")

    # --- Storage for dynamic features + predictions ---
    home_points_before = []
    away_points_before = []
    home_home_win_streaks = []
    home_home_loss_streaks = []
    away_away_win_streaks = []
    away_away_loss_streaks = []
    home_total_win_streaks = []
    home_total_loss_streaks = []
    away_total_win_streaks = []
    away_total_loss_streaks = []
    # h2h_home_wins_lastN = []
    # h2h_away_wins_lastN = []
    # h2h_draws_lastN = []
    preds_int = []
    prob_home_list = []
    prob_draw_list = []
    prob_away_list = []

    # --- Main simulation loop over inference fixtures ---
    for _, row in df.iterrows():
        season = row["season_start"]
        h_team = row["home_team_encoded"]
        a_team = row["away_team_encoded"]
        h_key, a_key = (season, h_team), (season, a_team)

        # Current state BEFORE this game
        h_pts = home_points_dict.get(h_key, 0)
        a_pts = away_points_dict.get(a_key, 0)
        h_home_win = home_home_win_dict.get(h_key, 0)
        h_home_loss = home_home_loss_dict.get(h_key, 0)
        a_away_win = away_away_win_dict.get(a_key, 0)
        a_away_loss = away_away_loss_dict.get(a_key, 0)
        h_total_win = total_win_streak_dict.get(h_key, 0)
        h_total_loss = total_loss_streak_dict.get(h_key, 0)
        a_total_win = total_win_streak_dict.get(a_key, 0)
        a_total_loss = total_loss_streak_dict.get(a_key, 0)

        # H2H BEFORE this game
        # h2h_key = (season, h_team, a_team)
        # past_results = h2h_history.get(h2h_key, [])
        # recent = past_results[-h2h_N:]
        # hwins = sum(1 for r in recent if r == "H")
        # awins = sum(1 for r in recent if r == "A")
        # draws = sum(1 for r in recent if r == "D")

        # Record state (for output)
        home_points_before.append(h_pts)
        away_points_before.append(a_pts)
        home_home_win_streaks.append(h_home_win)
        home_home_loss_streaks.append(h_home_loss)
        away_away_win_streaks.append(a_away_win)
        away_away_loss_streaks.append(a_away_loss)
        home_total_win_streaks.append(h_total_win)
        home_total_loss_streaks.append(h_total_loss)
        away_total_win_streaks.append(a_total_win)
        away_total_loss_streaks.append(a_total_loss)
        # h2h_home_wins_lastN.append(hwins)
        # h2h_away_wins_lastN.append(awins)
        # h2h_draws_lastN.append(draws)

        # Build feature vector for this match
        feat_vals = []
        for col in feature_cols:
            if col == "home_points_cum":
                feat_vals.append(h_pts)
            elif col == "away_points_cum":
                feat_vals.append(a_pts)
            elif col == "home_home_win_streak":
                feat_vals.append(h_home_win)
            elif col == "home_home_losing_streak":
                feat_vals.append(h_home_loss)
            elif col == "away_away_win_streak":
                feat_vals.append(a_away_win)
            elif col == "away_away_losing_streak":
                feat_vals.append(a_away_loss)
            elif col == "home_total_win_streak":
                feat_vals.append(h_total_win)
            elif col == "home_total_losing_streak":
                feat_vals.append(h_total_loss)
            elif col == "away_total_win_streak":
                feat_vals.append(a_total_win)
            elif col == "away_total_losing_streak":
                feat_vals.append(a_total_loss)
            # elif col == "h2h_home_wins_lastN":
            #     feat_vals.append(hwins)
            # elif col == "h2h_away_wins_lastN":
            #     feat_vals.append(awins)
            # elif col == "h2h_draws_lastN":
            #     feat_vals.append(draws)
            else:
                feat_vals.append(row[col] if col in df.columns else np.nan)

        X = np.array(feat_vals, dtype=float).reshape(1, -1)

        # Predict probabilities & class
        probs = model.predict_proba(X)[0]
        prob_home, prob_draw, prob_away = probs[0], probs[1], probs[2]
        pred_int = int(np.argmax(probs))

        prob_home_list.append(prob_home)
        prob_draw_list.append(prob_draw)
        prob_away_list.append(prob_away)
        preds_int.append(pred_int)

        # Decode label
        if result_encoder is not None:
            pred_label = result_encoder.inverse_transform([pred_int])[0]
        else:
            mapping = {0: "H", 1: "D", 2: "A"}
            pred_label = mapping[pred_int]

        # Update state with predicted result
        update_dicts(season, h_team, a_team, pred_label)

    # --- Attach outputs back to df ---
    df["home_points_cum"] = home_points_before
    df["away_points_cum"] = away_points_before
    df["predicted_FTR_int"] = preds_int
    if result_encoder is not None:
        df["predicted_FTR"] = result_encoder.inverse_transform(df["predicted_FTR_int"])

    df["home_home_win_streak"] = home_home_win_streaks
    df["home_home_losing_streak"] = home_home_loss_streaks
    df["away_away_win_streak"] = away_away_win_streaks
    df["away_away_losing_streak"] = away_away_loss_streaks
    df["home_total_win_streak"] = home_total_win_streaks
    df["home_total_losing_streak"] = home_total_loss_streaks
    df["away_total_win_streak"] = away_total_win_streaks
    df["away_total_losing_streak"] = away_total_loss_streaks
    # df["h2h_home_wins_lastN"] = h2h_home_wins_lastN
    # df["h2h_away_wins_lastN"] = h2h_away_wins_lastN
    # df["h2h_draws_lastN"] = h2h_draws_lastN

    df["prob_home_win"] = prob_home_list
    df["prob_draw"] = prob_draw_list
    df["prob_away_win"] = prob_away_list
    df["predicted_confidence"] = df[
        ["prob_home_win", "prob_draw", "prob_away_win"]
    ].max(axis=1)

    return df


def compute_base_table_from_training(train, result_encoder, team_encoder, seasons_to_use):
    """
    Compute the REAL league table from training data (true results only),
    restricted to the seasons in `seasons_to_use`.
    Returns a DataFrame with columns: ['team', 'base_points'].
    """

    df = train.copy()
    df = df[df["season_start"].isin(seasons_to_use)].copy()

    # Decode labels and team names
    df["FTR"] = result_encoder.inverse_transform(df["ft_result_encoded"].astype(int))
    df["HomeTeam"] = team_encoder.inverse_transform(df["home_team_encoded"])
    df["AwayTeam"] = team_encoder.inverse_transform(df["away_team_encoded"])

    # True points from training period
    df["home_points_true"] = df["FTR"].map({"H": 3, "D": 1, "A": 0})
    df["away_points_true"] = df["FTR"].map({"A": 3, "D": 1, "H": 0})

    home = (
        df.groupby("HomeTeam")["home_points_true"]
        .sum()
        .reset_index()
        .rename(
            columns={"HomeTeam": "team", "home_points_true": "home_points_true_total"}
        )
    )

    away = (
        df.groupby("AwayTeam")["away_points_true"]
        .sum()
        .reset_index()
        .rename(
            columns={"AwayTeam": "team", "away_points_true": "away_points_true_total"}
        )
    )

    base = home.merge(away, on="team", how="outer").fillna(0)
    base["base_points"] = base["home_points_true_total"] + base["away_points_true_total"]
    base = base[["team", "base_points"]]

    return base


def pretty_print_table(table):
    """
    Print a nicely formatted league table showing team and points
    """
    display_table = table[["team", "total_points"]].copy()
    display_table.index = range(1, len(display_table) + 1)

    print("\nPredicted League Table:")
    print("=" * 40)
    print(f"{'Pos':4} {'Team':25} {'Pts':>5}")
    print("-" * 40)

    for pos, row in display_table.iterrows():
        print(f"{pos:3}. {row['team']:25} {int(row['total_points']):>5}")

    print("=" * 40)


def main():
    # --- Load inference fixtures ---
    inference_df = pd.read_csv("inference.csv")
    inference_df = inference_df.drop(columns=["data_type", "ft_result_encoded"])

    # --- Load model + encoders ---
    with open("rf_model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    with open("xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open("pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)

    result_encoder = pipeline["result_encoder"]
    team_encoder = pipeline["team_encoder"]

    # --- Load training data ---
    train = pd.read_csv("training.csv")

    # Feature columns from training
    feature_cols = [
        c for c in train.columns if c not in ["ft_result_encoded", "data_type", "Date"]
    ]

    # --- Simulate remaining fixtures (inference) ---
    pred_df = simulate_season_predictions(
        inference_df,
        xgb_model,
        feature_cols,
        result_encoder=result_encoder,
        date_col="Date",
        training_df=train,
        h2h_N=5,
    )

    # Decode team names
    pred_df["HomeTeam"] = team_encoder.inverse_transform(pred_df["home_team_encoded"])
    pred_df["AwayTeam"] = team_encoder.inverse_transform(pred_df["away_team_encoded"])

    # Clean export of match predictions + streaks/H2H
    pred_df_clean = pred_df[
        [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "predicted_FTR",
            "predicted_confidence",
            "home_home_win_streak",
            "home_home_losing_streak",
            "away_away_win_streak",
            "away_away_losing_streak",
            "home_total_win_streak",
            "home_total_losing_streak",
            "away_total_win_streak",
            "away_total_losing_streak",
            # "h2h_home_wins_lastN",
            # "h2h_away_wins_lastN",
            # "h2h_draws_lastN",
            "home_points_cum",
            "away_points_cum",
            "prob_home_win",
            "prob_draw",
            "prob_away_win",
        ]
    ]

    pred_df_clean.to_csv(
        "results/GW12 results/season_predictions.csv", index=False
    )

    # --- Compute incremental points from predicted fixtures only ---
    pred_df["home_points"] = pred_df["predicted_FTR"].map({"H": 3, "D": 1, "A": 0})
    pred_df["away_points"] = pred_df["predicted_FTR"].map({"A": 3, "D": 1, "H": 0})

    home_pred = (
        pred_df.groupby("HomeTeam")["home_points"]
        .sum()
        .reset_index()
        .rename(columns={"HomeTeam": "team", "home_points": "home_points_pred"})
    )

    away_pred = (
        pred_df.groupby("AwayTeam")["away_points"]
        .sum()
        .reset_index()
        .rename(columns={"AwayTeam": "team", "away_points": "away_points_pred"})
    )

    incremental = home_pred.merge(away_pred, on="team", how="outer").fillna(0)
    incremental["predicted_points"] = (
        incremental["home_points_pred"] + incremental["away_points_pred"]
    )
    incremental = incremental[["team", "predicted_points"]]

    # --- Base table from true results in training.csv (matching seasons only) ---
    seasons_in_inf = inference_df["season_start"].unique()
    base_table = compute_base_table_from_training(
        train,
        result_encoder=result_encoder,
        team_encoder=team_encoder,
        seasons_to_use=seasons_in_inf,
    )

    # --- Final table: base_points + predicted_points ---
    table = base_table.merge(incremental, on="team", how="outer").fillna(0)
    table["total_points"] = table["base_points"] + table["predicted_points"]
    table = table.sort_values("total_points", ascending=False).reset_index(drop=True)

    table.to_csv(
        "results/GW12 results/predicted_table.csv",
        index=False,
    )

    pretty_print_table(table)


if __name__ == "__main__":
    main()
