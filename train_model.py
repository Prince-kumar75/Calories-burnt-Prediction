import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

def train_and_save_models():
    print("Loading datasets...")
    calories = pd.read_csv('calories.csv')
    exercise = pd.read_csv('exercise.csv')
    
    # Merge datasets on User_ID
    df = pd.merge(exercise, calories, on='User_ID')
    print(f"Dataset shape: {df.shape}")
    print(df.head())
    
    # Feature engineering / encoding
    # Gender: male -> 0, female -> 1 (or 1 for male, 0 for female. Let's make male=0, female=1)
    df['Gender_Code'] = df['Gender'].map({'male': 0, 'female': 1})
    
    # Calculate BMI as a helpful derived feature
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    
    features = ['Gender_Code', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI']
    target = 'Calories'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    models = {
        'xgboost': xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        ),
        'random_forest': RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        ),
        'linear_regression': LinearRegression()
    }
    
    results = {}
    
    os.makedirs('models', exist_ok=True)
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results[name] = {
            'r2_score': round(float(r2), 4),
            'mae': round(float(mae), 3),
            'rmse': round(float(rmse), 3)
        }
        print(f"--> {name}: R2 = {r2:.4f}, MAE = {mae:.3f}, RMSE = {rmse:.3f}")
        
        # Save model
        joblib.dump(model, f'models/{name}_model.pkl')
    
    # Feature importance for XGBoost
    xgb_model = models['xgboost']
    feature_importances = dict(zip(features, [round(float(x), 4) for x in xgb_model.feature_importances_]))
    
    # Save metadata
    metadata = {
        'features': features,
        'model_performance': results,
        'feature_importances': feature_importances,
        'dataset_summary': {
            'total_samples': len(df),
            'avg_calories': round(float(df['Calories'].mean()), 2),
            'max_calories': round(float(df['Calories'].max()), 2),
            'min_calories': round(float(df['Calories'].min()), 2),
            'avg_duration': round(float(df['Duration'].mean()), 2),
            'avg_heart_rate': round(float(df['Heart_Rate'].mean()), 2)
        }
    }
    
    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("Training complete! Metadata and models saved in models/ directory.")

if __name__ == '__main__':
    train_and_save_models()
