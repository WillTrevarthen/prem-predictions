import pandas as pd
import numpy as np
import pickle

def simulate_season_predictions(
    df: pd.DataFrame,
    model,
    feature_cols,
    result_encoder=None,
    date_col=None,
    training_df: pd.DataFrame | None = None,
    nll_threshold: float | None = None,   # set to your 90th percentile NLL threshold
    epsilon: float = 1e-12,               # numerical safety for logs
):
    """
    Simulate a season chronologically and predict each game, updating:

      - home/away cumulative points (home_points_cum / away_points_cum)
      - home/away home-only win/losing streaks
      - home/away total win/losing streaks
      - (optionally) seed state from training_df's TRUE past results

    Adds:
      - prob_home_win / prob_draw / prob_away_win
      - per-class nonconformity scores (NLL): nll_home / nll_draw / nll_away
      - conformal prediction set columns:
          * prediction_set_labels (list[str])
          * prediction_set_probs  (list[float], normalized within the set)
          * prediction_set_map    (dict[label->normalized prob])
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

    # --- Helper: update dicts after a result (true or predicted) ---
    def update_dicts(season, h_team, a_team, result_label: str):
        """result_label in {'H','D','A'}"""
        h_key, a_key = (season, h_team), (season, a_team)

        # Points
        h_pts = home_points_dict.get(h_key, 0)
        a_pts = away_points_dict.get(a_key, 0)
        if result_label == "H":
            home_points_dict[h_key] = h_pts + 3
            away_points_dict[a_key] = a_pts
        elif result_label == "A":
            home_points_dict[h_key] = h_pts
            away_points_dict[a_key] = a_pts + 3
        else:  # "D"
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

        if result_label == "H":
            home_home_win_dict[h_key] = h_home_win + 1
            home_home_loss_dict[h_key] = 0
            away_away_loss_dict[a_key] = a_away_loss + 1
            away_away_win_dict[a_key] = 0
            total_win_streak_dict[h_key] = h_total_win + 1
            total_loss_streak_dict[h_key] = 0
            total_loss_streak_dict[a_key] = a_total_loss + 1
            total_win_streak_dict[a_key] = 0

        elif result_label == "A":
            away_away_win_dict[a_key] = a_away_win + 1
            away_away_loss_dict[a_key] = 0
            home_home_loss_dict[h_key] = h_home_loss + 1
            home_home_win_dict[h_key] = 0
            total_win_streak_dict[a_key] = a_total_win + 1
            total_loss_streak_dict[a_key] = 0
            total_loss_streak_dict[h_key] = h_total_loss + 1
            total_win_streak_dict[h_key] = 0

        else:  # "D"
            home_home_win_dict[h_key] = 0
            home_home_loss_dict[h_key] = 0
            away_away_win_dict[a_key] = 0
            away_away_loss_dict[a_key] = 0
            total_win_streak_dict[h_key] = 0
            total_loss_streak_dict[h_key] = 0
            total_win_streak_dict[a_key] = 0
            total_loss_streak_dict[a_key] = 0

    # --- Seed state from training_df (TRUE results before inference) ---
    if training_df is not None and result_encoder is not None:
        past_df = training_df.copy()
        seasons_in_inf = df["season_start"].unique()
        past_df = past_df[past_df["season_start"].isin(seasons_in_inf)].copy()

        if date_col and date_col in past_df.columns and date_col in df.columns:
            past_df[date_col] = pd.to_datetime(past_df[date_col], dayfirst=True, errors="coerce")
            df_dates = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
            first_inf_by_season = (
                df.assign(_date=df_dates).groupby("season_start")["_date"].min().to_dict()
            )

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

        # Replay past matches to seed points/streaks
        if "ft_result_encoded" in past_df.columns:
            for _, r in past_df.iterrows():
                if pd.isna(r.get("ft_result_encoded")):
                    continue
                season = r["season_start"]
                h_team = r["home_team_encoded"]
                a_team = r["away_team_encoded"]
                true_label = result_encoder.inverse_transform([int(r["ft_result_encoded"])])[0]
                update_dicts(season, h_team, a_team, true_label)

        print(f"Seeded state from {len(past_df)} past games (matching seasons + before inference)")

    # --- Storage for dynamic features + predictions ---
    home_points_before, away_points_before = [], []
    home_home_win_streaks, home_home_loss_streaks = [], []
    away_away_win_streaks, away_away_loss_streaks = [], []
    home_total_win_streaks, home_total_loss_streaks = [], []
    away_total_win_streaks, away_total_loss_streaks = [], []
    preds_int = []
    prob_home_list, prob_draw_list, prob_away_list = [], [], []

    # NEW: per-class NLLs + prediction set storage
    nll_home_list, nll_draw_list, nll_away_list = [], [], []
    pred_set_labels_list, pred_set_probs_list, pred_set_map_list = [], [], []

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
            else:
                feat_vals.append(row[col] if col in df.columns else np.nan)

        X = np.array(feat_vals, dtype=float).reshape(1, -1)

        # ---------- Predict probabilities & decode class labels robustly ----------
        probs = model.predict_proba(X)[0]                        # shape (n_classes,)
        model_classes = getattr(model, "classes_", np.arange(len(probs)))

        if result_encoder is not None:
            decoded_labels = result_encoder.inverse_transform(model_classes)
        else:
            fallback = {0: "H", 1: "D", 2: "A"}
            decoded_labels = np.array([fallback[int(c)] for c in model_classes])

        # map label -> prob
        label_prob_map = {lbl: float(p) for lbl, p in zip(decoded_labels, probs)}
        prob_home = label_prob_map.get("H", 0.0)
        prob_draw = label_prob_map.get("D", 0.0)
        prob_away = label_prob_map.get("A", 0.0)

        prob_home_list.append(prob_home)
        prob_draw_list.append(prob_draw)
        prob_away_list.append(prob_away)

        # predicted class (encoded + string)
        argmax_idx = int(np.argmax(probs))
        pred_label = decoded_labels[argmax_idx]
        preds_int.append(int(model_classes[argmax_idx]))
        # ------------------------------------------------------------------------

        # ---------- per-class NLL and conformal prediction set ----------
        eps = max(epsilon, 1e-12)
        nlls = -np.log(np.clip(probs, eps, 1.0))                 # aligned with decoded_labels
        nll_by_label = {lbl: float(nll) for lbl, nll in zip(decoded_labels, nlls)}
        nll_home_list.append(nll_by_label.get("H", np.nan))
        nll_draw_list.append(nll_by_label.get("D", np.nan))
        nll_away_list.append(nll_by_label.get("A", np.nan))

        if nll_threshold is not None:
            mask = nlls <= nll_threshold
            if not np.any(mask):
                mask = np.zeros_like(mask, dtype=bool)
                mask[argmax_idx] = True
        else:
            mask = np.zeros_like(probs, dtype=bool)
            mask[argmax_idx] = True

        set_idxs = np.where(mask)[0]
        set_labels = [decoded_labels[i] for i in set_idxs]
        set_probs_raw = probs[set_idxs]
        denom = set_probs_raw.sum()
        set_probs_norm = (set_probs_raw / denom) if denom > 0 else np.ones_like(set_probs_raw) / max(len(set_probs_raw), 1)

        pred_set_labels_list.append(set_labels)
        pred_set_probs_list.append(set_probs_norm.tolist())
        pred_set_map_list.append({lbl: float(p) for lbl, p in zip(set_labels, set_probs_norm)})
        # ------------------------------------------------------------------------

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

    df["prob_home_win"] = prob_home_list
    df["prob_draw"] = prob_draw_list
    df["prob_away_win"] = prob_away_list
    df["predicted_confidence"] = df[["prob_home_win", "prob_draw", "prob_away_win"]].max(axis=1)

    # per-class NLLs + conformal set
    df["nll_home"] = nll_home_list
    df["nll_draw"] = nll_draw_list
    df["nll_away"] = nll_away_list
    df["prediction_set_labels"] = pred_set_labels_list
    df["prediction_set_probs"] = pred_set_probs_list
    df["prediction_set_map"] = pred_set_map_list

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
    with open("calibrated_xgb_model.pkl", "rb") as f:
        calibrated_xgb = pickle.load(f)
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

    with open("xgb_nonconformity_threshold.pkl", "rb") as f:
        nll_threshold = pickle.load(f)

    # --- Simulate remaining fixtures (inference) ---
    pred_df = simulate_season_predictions(
        inference_df,
        xgb_model, #xgb_model
        feature_cols,
        result_encoder=result_encoder,
        date_col="Date",
        training_df=train,
        nll_threshold=nll_threshold,   # <<< pass the loaded threshold
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
            "nll_home",
            "nll_draw",
            "nll_away",
            "prediction_set_labels",           # list of labels included by threshold
            "prediction_set_probs",             # list of normalized probs (same order)
            "prediction_set_map"
        ]
    ]

    pred_df_clean["no_cp_prob_set"] = pred_df_clean.apply(
        lambda r: {
            "H": float(r["prob_home_win"]),
            "D": float(r["prob_draw"]),
            "A": float(r["prob_away_win"]),
        },
        axis=1,
    )

    # (Optional) normalize defensively in case of tiny numeric drift
    def _norm_map(d):
        s = sum(d.values())
        return {k: (v / s if s > 0 else 1.0/len(d)) for k, v in d.items()}
    pred_df_clean["no_cp_prob_set"] = pred_df_clean["no_cp_prob_set"].apply(_norm_map)

    pred_df_clean['predicted_confidence'] = np.round(pred_df_clean['predicted_confidence'],2)

    pred_df_clean.to_csv(
        "results/GW12 test/season_predictions.csv", index=False
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
        "results/GW12 test/predicted_table.csv",
        index=False,
    )

    pretty_print_table(table)


if __name__ == "__main__":
    main()
