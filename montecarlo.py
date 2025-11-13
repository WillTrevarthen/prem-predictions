import numpy as np
import pandas as pd
from ast import literal_eval
import json

def run_monte_carlo_conformal(
    fixtures_df: pd.DataFrame,
    n_sims: int = 10_000,
    seed: int = 42,
    starting_points: dict | None = None,   # optional {team_id: total_points_before}
    predset_col: str = "prediction_set_map",
    home_col: str | None = None,
    away_col: str | None = None,
    # Preferred total-points-before columns (from simulate_season_predictions)
    home_total_before_col: str = "home_total_points_before",
    away_total_before_col: str = "away_total_points_before",
    # Fallback venue-only cumulatives (used only if totals not available)
    home_pts_before_col: str = "home_points_cum",
    away_pts_before_col: str = "away_points_cum",
):
    rng = np.random.default_rng(seed)
    df = fixtures_df.copy()

    # --- auto-detect home/away columns if not provided ---
    if home_col is None or away_col is None:
        candidates_home = ["home_team_encoded", "home_team_id", "home_id", "HomeTeamId", "home_team", "HomeTeam"]
        candidates_away = ["away_team_encoded", "away_team_id", "away_id", "AwayTeamId", "away_team", "AwayTeam"]
        home_col = next((c for c in candidates_home if c in df.columns), home_col)
        away_col = next((c for c in candidates_away if c in df.columns), away_col)

    missing = [c for c in [home_col, away_col, predset_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}. Have: {list(df.columns)}")

    # --- parse/validate prediction_set_map -> dict[label]->prob ---
    def ensure_predset_dict(x):
        if isinstance(x, dict):
            d = x
        elif isinstance(x, str):
            s = x.strip()
            try:
                d = literal_eval(s)
            except Exception:
                d = json.loads(s.replace("'", '"'))
        else:
            raise ValueError(f"{predset_col} cell has unsupported type: {type(x)}")

        valid = {"H", "D", "A"}
        d = {str(k): float(v) for k, v in d.items() if str(k) in valid}
        if not d:
            raise ValueError(f"Empty/invalid prediction set after parsing: {x!r}")

        vals = np.array(list(d.values()), dtype=float)
        vals[~np.isfinite(vals)] = 0.0
        vals = np.clip(vals, 0.0, None)
        ssum = vals.sum()
        if ssum <= 0:
            topk = next(iter(d.keys()))
            d = {topk: 1.0}
        else:
            d = dict(zip(d.keys(), (vals / ssum).tolist()))
        return d

    df[predset_col] = df[predset_col].apply(ensure_predset_dict)

    # --- teams & arrays ---
    teams = np.unique(np.concatenate([df[home_col].to_numpy(), df[away_col].to_numpy()]))
    teams = np.sort(teams)
    team_to_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    home_idx = df[home_col].map(team_to_idx).to_numpy()
    away_idx = df[away_col].map(team_to_idx).to_numpy()
    M = len(df)

    # --- starting points ---
    if starting_points is None:
        derived = {}

        # Preferred: totals produced by simulate_season_predictions
        if (home_total_before_col in df.columns) and (away_total_before_col in df.columns):
            # Take the first time each team appears as home and as away, then sum
            first_home_totals = (
                df.groupby(home_col, sort=False)[home_total_before_col]
                  .first()
                  .astype(float)
            )
            first_away_totals = (
                df.groupby(away_col, sort=False)[away_total_before_col]
                  .first()
                  .astype(float)
            )
            summed = first_home_totals.add(first_away_totals, fill_value=0.0)
            derived = summed.reindex(teams, fill_value=0.0).to_dict()

        # Fallback: venue-only cumulatives; sum first-home + first-away cumulatives
        elif (home_pts_before_col in df.columns) and (away_pts_before_col in df.columns):
            first_home = (
                df.groupby(home_col, sort=False)[home_pts_before_col]
                  .first()
                  .astype(float)
            )
            first_away = (
                df.groupby(away_col, sort=False)[away_pts_before_col]
                  .first()
                  .astype(float)
            )
            summed = first_home.add(first_away, fill_value=0.0)
            derived = summed.reindex(teams, fill_value=0.0).to_dict()

        else:
            derived = {t: 0.0 for t in teams}

        starting_points = derived

    # --- initial points matrix ---
    pts = np.zeros((n_sims, n_teams), dtype=float)
    if starting_points:
        for t, p in starting_points.items():
            if t in team_to_idx:
                pts[:, team_to_idx[t]] = float(p)

    # --- simulate each match across all sims ---
    code = {'H': 0, 'D': 1, 'A': 2}
    sim_idx = np.arange(n_sims)

    for m in range(M):
        ps_map = df.iloc[m][predset_col]          # dict like {'H':0.7,'D':0.3}
        labels = list(ps_map.keys())
        probs  = np.array([ps_map[k] for k in labels], dtype=float)
        probs = probs / probs.sum()
        outcome_codes = np.array([code[lbl] for lbl in labels], dtype=int)

        draws = rng.choice(outcome_codes, size=n_sims, p=probs)

        maskH = draws == 0
        maskD = draws == 1
        maskA = draws == 2

        if maskH.any():
            np.add.at(pts, (sim_idx[maskH], home_idx[m]), 3)
        if maskD.any():
            np.add.at(pts, (sim_idx[maskD], home_idx[m]), 1)
            np.add.at(pts, (sim_idx[maskD], away_idx[m]), 1)
        if maskA.any():
            np.add.at(pts, (sim_idx[maskA], away_idx[m]), 3)

    # --- ranking with tiny random tie-breaker ---
    noise = rng.random(pts.shape) * 1e-6
    order = np.argsort(-(pts + noise), axis=1)

    positions_sims = np.empty_like(order)
    ranks = np.arange(1, n_teams + 1)
    for r in range(n_sims):
        positions_sims[r, order[r]] = ranks

    # --- aggregate ---
    finish_counts = np.zeros((n_teams, n_teams), dtype=int)
    for pos in range(1, n_teams + 1):
        finish_counts[:, pos - 1] = (positions_sims == pos).sum(axis=0)

    proportions = finish_counts / n_sims
    expected_points = pts.mean(axis=0)
    expected_rank = (proportions * np.arange(1, n_teams + 1)).sum(axis=1)

    out = pd.DataFrame({
        "team_id": teams,
        "expected_points": expected_points,
        "expected_rank": expected_rank,
    })
    for pos in range(1, n_teams + 1):
        out[f"pos_{pos}"] = proportions[:, pos - 1]

    return out.sort_values("expected_rank").reset_index(drop=True)




if __name__ == "__main__":

    df = pd.read_csv("results/GW12 test/season_predictions.csv")

    # `pred_df` should be the output of your simulate_season_predictions(...)
    # and must include a `prediction_set_map` column (dict per row).
    dist_df = run_monte_carlo_conformal(
        fixtures_df=df,
        n_sims=500_000,
        seed=7,
        starting_points=None,  # or {team_id: current_points}
        predset_col="no_cp_prob_set",
        home_col="HomeTeam",
        away_col="AwayTeam",
    )

    print(dist_df.head())
    dist_df.to_csv("monte_carlo_standings.csv", index=False)
