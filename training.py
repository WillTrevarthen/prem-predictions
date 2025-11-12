import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import pickle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
import optuna
from optuna.samplers import TPESampler


def main():
    df = pd.read_csv('training.csv')
    
    df = df.sort_values(['season_start', 'Date'])

    df = df.drop(columns=['data_type', 'Date'])

    X = df.drop(columns=['ft_result_encoded'])
    y = df['ft_result_encoded']

    # Define how many most recent matches to use as test set
    n_test = 100

    # Split into training and test sets (most recent n_test matches)
    train_df = df.iloc[:-n_test]
    test_df = df.iloc[-n_test:]

    # Weighting samples: more recent seasons get higher weights

    X_train = train_df.drop(columns=['ft_result_encoded'])
    y_train = train_df['ft_result_encoded']

    X_test = test_df.drop(columns=['ft_result_encoded'])
    y_test = test_df['ft_result_encoded']

    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    
    # Compute class weights
    linear_step = 0.02  # each newer season gets +0.2 higher weight
    seasons = sorted(train_df["season_start"].unique())

    # e.g. [2018→1.0, 2019→1.2, 2020→1.4, …]
    season_weight_map = {season: 1 + i * linear_step for i, season in enumerate(seasons)}

    season_weights = train_df["season_start"].map(season_weight_map)


    # 2) Class weights (per label H/D/A encoded)
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    class_weight_map = {c: w for c, w in zip(classes, class_weights)}

    class_weights_per_row = y_train.map(class_weight_map)

    # 3) Combine: final weight = recency * class
    sample_weights = season_weights * class_weights_per_row

    # (optional but nice) normalise so average weight ≈ 1
    sample_weights = sample_weights / sample_weights.mean()

    def objective(trial):
        booster = trial.suggest_categorical("booster", ["gbtree", "dart", "gblinear"])

        # Common params
        params = {
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "num_class": len(classes),
            "booster": booster,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "random_state": 42,
            "tree_method": "hist",   # fast & stable on CPU; use "gpu_hist" if on GPU
        }

        if booster in ["gbtree", "dart"]:
            params.update({
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            })
        if booster == "dart":
            params.update({
                "rate_drop": trial.suggest_float("rate_drop", 0.0, 0.3),
                "skip_drop": trial.suggest_float("skip_drop", 0.0, 0.3),
            })

        # 3-fold stratified CV with early stopping
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        aucs = []
        for tr_idx, va_idx in skf.split(X_train, y_train):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

            model = XGBClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False,
            )
            proba = model.predict_proba(X_va)  # shape [n_samples, n_classes]
            fold_auc = roc_auc_score(y_va, proba, multi_class="ovo", average="macro")
            aucs.append(fold_auc)
        return float(np.mean(aucs))

    study = optuna.create_study(
        direction="maximize",  # maximize AUC
        sampler=TPESampler(seed=42)
    )
    """
    study.optimize(objective, n_trials=50, show_progress_bar=True) #hyper param optimisation needs some work
    """
    print("Best trial:")
    trial = study.best_trial
    print(f"  AUC: {trial.value:.4f}")
    print("  Params:")
    for k, v in trial.params.items():
        print(f"    {k}: {v}")

    # (optional) visualize tuning history
    try:
        optuna.visualization.plot_optimization_history(study).show()
    except Exception:
        pass

    best_params = trial.params
    best_model = XGBClassifier(**best_params, random_state=42, use_label_encoder=False)
    best_model.fit(X, y)

    # Initialize XGBoost
    xgb = XGBClassifier(
        objective="multi:softmax",
        n_estimators=100, #200
        max_depth=3, #5
        num_class=len(classes),
        min_child_weight=5, #1
        gamma = 0.3, #0.3
        learning_rate=0.01,
        subsample=0.6,
        colsample_bytree=0.8, #0.8
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )

    # Train
    xgb.fit(X_train, y_train, sample_weight=sample_weights)

    # Predict
    y_pred = xgb.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


    # Compute class weights for X and y (full dataset)
    linear_step = 0.02  # each newer season gets +0.2 higher weight
    seasons = sorted(X["season_start"].unique())

    # e.g. [2018→1.0, 2019→1.2, 2020→1.4, …]
    season_weight_map = {season: 1 + i * linear_step for i, season in enumerate(seasons)}

    season_weights = X["season_start"].map(season_weight_map)


    # 2) Class weights (per label H/D/A encoded)
    classes = np.unique(y)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y
    )
    class_weight_map = {c: w for c, w in zip(classes, class_weights)}

    class_weights_per_row = y.map(class_weight_map)

    # 3) Combine: final weight = recency * class
    sample_weights = season_weights * class_weights_per_row

    # (optional but nice) normalise so average weight ≈ 1
    sample_weights = sample_weights / sample_weights.mean()


    xgb.fit(X,y, sample_weight=sample_weights)
    
    #------calibrated-classifier------TEST
    calibrated_xgb = CalibratedClassifierCV(xgb, method='isotonic', cv='prefit')
    calibrated_xgb.fit(X, y, sample_weight=sample_weights)
    with open("calibrated_xgb_model.pkl", "wb") as f:
        pickle.dump(calibrated_xgb, f)
    #------calibrated-classifier------TEST

    # Save model
    with open("xgb_model.pkl", "wb") as f:
        pickle.dump(xgb, f)

    # Load dataset
    # df = pd.read_csv('training.csv')
    # Sort by season and date to maintain chronological order

    # Initialize Random Forest with balanced class weights
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth= None,
        max_features='sqrt',
        min_samples_split=8,      # start here for your imbalance
        min_samples_leaf=3,       # optional, also helps minority class
        class_weight= {0:1, 1:4, 2:1},  
        random_state=42,
        n_jobs=-1
    )

    # Train the model
    rf.fit(X_train, y_train) # eval
    
    # Predict on test set for eval
    y_pred = rf.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    rf.fit(X,y)
    # Save the trained model to a file
    with open("rf_model.pkl", "wb") as f:
        pickle.dump(rf, f)

    # # --- stacking ensemble ---

    # # --- base models ---
    # rf = RandomForestClassifier(
    #     n_estimators=300,
    #     max_depth=None,
    #     max_features='sqrt',
    #     min_samples_split=8,
    #     min_samples_leaf=3,
    #     class_weight={0: 1, 1: 4, 2: 1},
    #     random_state=42,
    #     n_jobs=-1,
    # )

    # xgb_base = XGBClassifier(
    #     objective="multi:softprob",
    #     n_estimators=100,
    #     max_depth=3,
    #     num_class=len(classes),
    #     min_child_weight=5,
    #     gamma=0.3,
    #     learning_rate=0.01,
    #     subsample=0.6,
    #     colsample_bytree=0.8,
    #     random_state=42,
    #     n_jobs=-1,
    #     eval_metric="mlogloss",
    # )

    # estimators = [
    #     ('rf', rf),
    #     ('xgb', xgb_base),
    # ]

    # # meta-model
    # meta = XGBClassifier(
    #     objective="multi:softprob",
    #     n_estimators=100,
    #     max_depth=3,
    #     random_state=42,
    #     n_jobs=-1,
    #     eval_metric="mlogloss",
    # )

    # stack = StackingClassifier(
    #     estimators=estimators,
    #     final_estimator=meta,
    #     n_jobs=-1
    # )

    # # --- fit once with correct weights ---
    # stack.fit(X_train, y_train, sample_weight=sample_weights)

    # # --- evaluate ---
    # y_pred = stack.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy:.4f}\n")
    # print("Classification Report:")
    # print(classification_report(y_test, y_pred))
    # print("Confusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))

    # # --- save the trained stack ---
    # with open("ensemble_model.pkl", "wb") as f:
    #     pickle.dump(stack, f)











    # # # Define the parameter grid
    # param_grid = {
    #     "n_estimators": [100, 200, 300, 500],
    #     "max_depth": [3, 5, 7, 9],
    #     "learning_rate": [0.01, 0.05, 0.1, 0.2],
    #     "subsample": [0.6, 0.8, 1.0],
    #     "colsample_bytree": [0.6, 0.8, 1.0],
    #     "gamma": [0, 0.1, 0.3],
    #     "min_child_weight": [1, 3, 5]
    # }

    # # Initialize classifier
    # xgb = XGBClassifier(
    #     objective="multi:softmax",
    #     num_class=len(classes),  # number of classes
    #     use_label_encoder=False,
    #     eval_metric="mlogloss",
    #     random_state=42,
    #     n_jobs=-1
    # )

    # # Randomized search
    # search = RandomizedSearchCV(
    #     estimator=xgb,
    #     param_distributions=param_grid,
    #     n_iter=50,               # number of random combinations to try
    #     scoring="accuracy",
    #     cv=5,                    # 5-fold cross-validation
    #     verbose=2,
    #     random_state=42,
    #     n_jobs=-1
    # )

    # # Run search
    # search.fit(X_train, y_train, sample_weight=sample_weights)

    # # Best params
    # print("Best parameters:", search.best_params_)
    # print("Best CV accuracy:", search.best_score_)

    # # Evaluate on test set
    # y_pred = search.predict(X_test)

    # print(classification_report(y_test, y_pred))
    # print(confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    main()
