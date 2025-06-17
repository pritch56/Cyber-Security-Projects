# spaceship_titanic_model.py
# Enhanced data processing and modeling pipeline for Spaceship Titanic challenge

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import KNNImputer

def preprocess_data(df):
    # Ensure consistent group IDs from PassengerId
    if 'PassengerId' in df.columns:
        df['Group'] = df['PassengerId'].str.split('_').str[0]
    else:
        df['Group'] = 'Unknown'

    # Fill missing numerical values using KNN
    numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0

    imputer = KNNImputer(n_neighbors=5)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # Binary columns
    df['CryoSleep'] = df['CryoSleep'].map({True: 1, False: 0, np.nan: 0}).astype(int)
    df['VIP'] = df['VIP'].map({True: 1, False: 0, np.nan: 0}).astype(int)

    # Fill missing categorical
    df['HomePlanet'] = df['HomePlanet'].fillna('Unknown')
    df['Destination'] = df['Destination'].fillna('Unknown')

    # Spending-based features
    df['TotalSpending'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
    df['SpendingPerAge'] = df['TotalSpending'] / (df['Age'] + 1)  # Avoid divide-by-zero
    df['HasSpending'] = (df['TotalSpending'] > 0).astype(int)

    # Group size
    df['GroupSize'] = df.groupby('Group')['Group'].transform('count')

    # Cabin decomposition
    df['Cabin'] = df['Cabin'].fillna('Unknown/0/Unknown')
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['CabinDeck'] = cabin_split[0]
    df['CabinNum'] = pd.to_numeric(cabin_split[1], errors='coerce').fillna(0)
    df['CabinSide'] = cabin_split[2]

    # Shared cabin
    df['CabinGroupSize'] = df.groupby('Cabin')['Cabin'].transform('count')
    df['SharedCabin'] = (df['CabinGroupSize'] > 1).astype(int)

    # One-hot encode categorical features
    cat_cols = ['HomePlanet', 'Destination', 'CabinDeck', 'CabinSide', 'Group']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Drop irrelevant columns
    drop_cols = ['PassengerId', 'Name', 'Cabin']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    return df

def main():
    # Load data
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    # Target
    y = train_df['Transported'].astype(int)

    # Preprocess
    X_train_full = preprocess_data(train_df)
    X_test = preprocess_data(test_df)

    # Align test columns
    X_test = X_test.reindex(columns=X_train_full.columns, fill_value=0)

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y, test_size=0.2, random_state=42, stratify=y
    )

    # Base model for tuning
    base_model = XGBClassifier(
        eval_metric='logloss',
        random_state=42
    )

    # Grid search parameters
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 7],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }

    grid_search = GridSearchCV(
        base_model,
        param_grid,
        scoring='accuracy',
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    # Perform grid search
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    # Re-train model with best parameters and early stopping
    best_model = XGBClassifier(
        **best_params,
        eval_metric='logloss',
        random_state=42
    )

    best_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        #early_stopping_rounds=20,
        verbose=True
    )

    # Evaluate
    val_preds = best_model.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val, val_preds))
    print("Classification Report:\n", classification_report(y_val, val_preds, target_names=["Not Transported", "Transported"]))

    # Predict on test
    test_preds = best_model.predict(X_test)
    transported_preds = test_preds.astype(bool)

    # Submission
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Transported': transported_preds
    })

    submission.to_csv('submission.csv', index=False)
    print("Submission file 'submission.csv' created successfully.")

if __name__ == "__main__":
    main()
