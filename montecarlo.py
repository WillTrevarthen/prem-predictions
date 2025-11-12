import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
from ast import literal_eval
import json

def run_monte_carlo_conformal(
    fixtures_df: pd.DataFrame,
    n_sims: int = 10_000,
    seed: int = 42,
    starting_points: dict | None = None,
    predset_col: str = "prediction_set_map",
    home_col: str | None = None,
    away_col: str | None = None,
):
    rng = np.random.default_rng(seed)
    df = fixtures_df.copy()

    # --- auto-detect home/away columns if not provided ---
    if home_col is None or away_col is None:
        candidates_home = ["home_team_encoded", "home_team_id", "home_id", "HomeTeamId", "home_team"]
        candidates_away = ["away_team_encoded", "away_team_id", "away_id", "AwayTeamId", "away_team"]
        home_col = next((c for c in candidates_home if c in df.columns), home_col)
        away_col = next((c for c in candidates_away if c in df.columns), away_col)

    missing = [c for c in [home_col, away_col, predset_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}. Have: {list(df.columns)}")

    # --- parse/validate prediction_set_map -> dict[label]->prob ---
    def ensure_predset_dict(x):
        # already a dict
        if isinstance(x, dict):
            d = x
        # try Python-literal format: "{'H': 1.0, 'D': 0.0}"
        elif isinstance(x, str):
            s = x.strip()
            try:
                d = literal_eval(s)
            except Exception:
                # try JSON (double quotes)
                try:
                    d = json.loads(s.replace("'", '"'))
                except Exception as e:
                    raise ValueError(f"Cannot parse prediction_set_map value: {s!r}") from e
        else:
            raise ValueError(f"prediction_set_map cell has unsupported type: {type(x)}")

        # coerce keys/values and filter to valid labels
        valid = {"H", "D", "A"}
        d = {str(k): float(v) for k, v in d.items() if str(k) in valid}

        if not d:
            raise ValueError(f"Empty/invalid prediction set after parsing: {x!r}")

        # fix negatives/NaNs and re-normalize
        vals = np.array(list(d.values()), dtype=float)
        vals[~np.isfinite(vals)] = 0.0
        vals = np.clip(vals, 0.0, None)
        ssum = vals.sum()
        if ssum <= 0:
            # fallback: put all mass on the most likely key found originally
            topk = next(iter(d.keys()))
            d = {topk: 1.0}
        else:
            vals = vals / ssum
            d = dict(zip(d.keys(), vals.tolist()))
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

    # --- initial points ---
    pts = np.zeros((n_sims, n_teams), dtype=float)
    if starting_points:
        for t, p in starting_points.items():
            if t in team_to_idx:
                pts[:, team_to_idx[t]] = p

    code = {'H': 0, 'D': 1, 'A': 2}
    sim_idx = np.arange(n_sims)

    # --- simulate each match across all sims ---
    for m in range(M):
        ps_map = df.iloc[m][predset_col]          # dict like {'H':0.7,'D':0.3}
        labels = list(ps_map.keys())
        probs  = np.array([ps_map[k] for k in labels], dtype=float)
        probs = probs / probs.sum()                # just in case
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
        n_sims=10_000,
        seed=7,
        starting_points=None,  # or {team_id: current_points}
        predset_col="prediction_set_map",
        home_col="HomeTeam",
        away_col="AwayTeam",
    )

    print(dist_df.head())
    dist_df.to_csv("monte_carlo_standings.csv", index=False)
