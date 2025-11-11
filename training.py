import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import pickle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier

def main():
    # df = pd.read_csv('training.csv')
    # df = df.drop(columns=['data_type', 'Date'])

    # X = df.drop(columns=['ft_result_encoded'])
    # y = df['ft_result_encoded']

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42, stratify=y
    # )

    # # Compute class weights
    # classes = np.unique(y_train)
    # class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    # weight_dict = {c: w for c, w in zip(classes, class_weights)}

    # # Map weights to each sample
    # sample_weights = y_train.map(weight_dict)

    # # Initialize XGBoost
    # xgb = XGBClassifier(
    #     n_estimators=100, #200
    #     max_depth=3, #5
    #     num_class=len(classes),
    #     min_child_weight=5, #1
    #     gamma = 0.1, #0.3
    #     learning_rate=0.01,
    #     subsample=0.6,
    #     colsample_bytree=0.6, #0.8
    #     random_state=42,
    #     n_jobs=-1,
    #     use_label_encoder=False,
    #     eval_metric="mlogloss"
    # )

    # # Train
    # xgb.fit(X_train, y_train, sample_weight=sample_weights)

    # # Predict
    # y_pred = xgb.predict(X_test)

    # # Evaluate
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy:.4f}\n")
    # print("Classification Report:")
    # print(classification_report(y_test, y_pred))
    # print("Confusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))

    # # Save model
    # with open("xgb_model.pkl", "wb") as f:
    #     pickle.dump(xgb, f)

    # Load dataset
    df = pd.read_csv('training.csv')
    # Sort by season and date to maintain chronological order
    df = df.sort_values(['season_start', 'Date'])


    df = df.drop(columns=['data_type', 'Date'])

    # Define how many most recent matches to use as test set
    n_test = 100

    # Split into training and test sets (most recent n_test matches)
    train_df = df.iloc[:-n_test]
    test_df = df.iloc[-n_test:]

    X = df.drop(columns=['ft_result_encoded'])
    y = df['ft_result_encoded']

    X_train = train_df.drop(columns=['ft_result_encoded'])
    y_train = train_df['ft_result_encoded']

    X_test = test_df.drop(columns=['ft_result_encoded'])
    y_test = test_df['ft_result_encoded']

    # Initialize Random Forest with balanced class weights
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth= None,
        max_features='sqrt',
        min_samples_split=8,      # start here for your imbalance
        min_samples_leaf=3,       # optional, also helps minority class
        class_weight= {0:1, 1:6, 2:1},  
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





    # # Define the parameter grid
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


if __name__ == '__main__':
    main()
