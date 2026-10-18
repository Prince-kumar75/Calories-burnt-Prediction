import os
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List

app = FastAPI(
    title="Calories Burnt Prediction AI Dashboard",
    description="Machine Learning based Calories Burnt Prediction System with Modern Fitness Dashboard UI",
    version="1.0.0"
)

# Load trained models and metadata
MODELS_DIR = "models"
MODELS = {}
METADATA = {}

def load_ml_models():
    global MODELS, METADATA
    try:
        if os.path.exists(f"{MODELS_DIR}/metadata.json"):
            with open(f"{MODELS_DIR}/metadata.json", "r") as f:
                METADATA = json.load(f)
        
        for model_name in ["xgboost", "random_forest", "gradient_boosting", "linear_regression"]:
            model_path = f"{MODELS_DIR}/{model_name}_model.pkl"
            if os.path.exists(model_path):
                MODELS[model_name] = joblib.load(model_path)
                print(f"Loaded {model_name} model successfully.")
            else:
                print(f"Warning: {model_path} not found.")
    except Exception as e:
        print(f"Error loading models: {e}")

load_ml_models()

# In-memory history for logged workouts
workout_history = [
    {
        "id": 1,
        "activity_name": "Cardio Masterclass",
        "gender": "female",
        "age": 28,
        "height": 168.0,
        "weight": 58.0,
        "duration": 30.0,
        "heart_rate": 138.0,
        "body_temp": 38.8,
        "calories_burnt": 224.5,
        "burn_rate": 7.48,
        "model_used": "xgboost",
        "timestamp": "Today, 08:30 AM"
    },
    {
        "id": 2,
        "activity_name": "Morning Warmup",
        "gender": "female",
        "age": 28,
        "height": 168.0,
        "weight": 58.0,
        "duration": 15.0,
        "heart_rate": 102.0,
        "body_temp": 37.9,
        "calories_burnt": 68.2,
        "burn_rate": 4.55,
        "model_used": "xgboost",
        "timestamp": "Today, 07:15 AM"
    }
]

# Request Schemas
class PredictionRequest(BaseModel):
    gender: str = Field(default="female", description="'male' or 'female'")
    age: float = Field(default=28.0, ge=10, le=100, description="Age in years")
    height: float = Field(default=168.0, ge=100, le=240, description="Height in cm")
    weight: float = Field(default=58.0, ge=30, le=200, description="Weight in kg")
    duration: float = Field(default=30.0, ge=1, le=180, description="Exercise duration in minutes")
    heart_rate: float = Field(default=135.0, ge=50, le=220, description="Heart rate in beats per minute")
    body_temp: float = Field(default=38.8, ge=35.0, le=43.0, description="Body temperature in Celsius")
    model_name: Optional[str] = Field(default="xgboost", description="xgboost, random_forest, gradient_boosting, linear_regression")
    activity_name: Optional[str] = Field(default="Custom Workout", description="Name of activity")

class LogWorkoutRequest(PredictionRequest):
    calories_burnt: Optional[float] = None

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": list(MODELS.keys()),
        "dataset_samples": METADATA.get("dataset_summary", {}).get("total_samples", 15000)
    }

@app.get("/api/model-info")
def get_model_info():
    return {
        "performance": METADATA.get("model_performance", {}),
        "feature_importances": METADATA.get("feature_importances", {}),
        "dataset_summary": METADATA.get("dataset_summary", {}),
        "available_models": [
            {"id": "xgboost", "name": "XGBoost Regressor", "accuracy": "99.93%", "mae": "1.18 kcal", "badge": "Most Accurate"},
            {"id": "gradient_boosting", "name": "Gradient Boosting", "accuracy": "99.91%", "mae": "1.43 kcal", "badge": "Ultra Smooth"},
            {"id": "random_forest", "name": "Random Forest", "accuracy": "99.79%", "mae": "1.90 kcal", "badge": "Ensemble"},
            {"id": "linear_regression", "name": "Linear Regression", "accuracy": "96.88%", "mae": "8.20 kcal", "badge": "Baseline"}
        ]
    }

@app.post("/api/predict")
def predict_calories(req: PredictionRequest):
    model_key = req.model_name if req.model_name in MODELS else "xgboost"
    model = MODELS.get(model_key)
    
    if not model:
        raise HTTPException(status_code=500, detail=f"Model '{model_key}' is not loaded.")
        
    gender_code = 1 if req.gender.lower() == "female" else 0
    bmi = req.weight / ((req.height / 100) ** 2)
    
    # Feature columns matching training
    # ['Gender_Code', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI']
    input_data = pd.DataFrame([{
        'Gender_Code': gender_code,
        'Age': req.age,
        'Height': req.height,
        'Weight': req.weight,
        'Duration': req.duration,
        'Heart_Rate': req.heart_rate,
        'Body_Temp': req.body_temp,
        'BMI': bmi
    }])
    
    pred_val = float(model.predict(input_data)[0])
    pred_val = max(1.0, round(pred_val, 1)) # non-negative
    
    burn_rate = round(pred_val / max(1.0, req.duration), 2)
    
    # Calculate physiological estimates
    # Estimated METs = (Calories / (Weight * (Duration / 60)))
    mets = round(pred_val / (req.weight * (req.duration / 60.0)), 2) if req.weight > 0 and req.duration > 0 else 5.0
    
    intensity = "Light"
    if mets >= 9.0 or req.heart_rate >= 155:
        intensity = "High / Peak"
    elif mets >= 6.0 or req.heart_rate >= 125:
        intensity = "Vigorous / Moderate-High"
    elif mets >= 3.5 or req.heart_rate >= 100:
        intensity = "Moderate"
        
    # Food energy equivalents
    equivalents = {
        "avocados": round(pred_val / 240, 1), # 1 avocado ~ 240 kcal
        "pizza_slices": round(pred_val / 285, 1), # 1 slice pizza ~ 285 kcal
        "running_km": round(pred_val / 65, 1), # ~65 kcal per km
        "apples": round(pred_val / 95, 1) # 1 apple ~ 95 kcal
    }
    
    # Feature contributions for explainability (normalized)
    fi = METADATA.get("feature_importances", {})
    contributions = [
        {"feature": "Exercise Duration", "value": f"{req.duration} min", "impact_pct": round(fi.get("Duration", 0.69) * 100, 1)},
        {"feature": "Heart Rate", "value": f"{req.heart_rate} bpm", "impact_pct": round(fi.get("Heart_Rate", 0.19) * 100, 1)},
        {"feature": "Body Temperature", "value": f"{req.body_temp} °C", "impact_pct": round(fi.get("Body_Temp", 0.07) * 100, 1)},
        {"feature": "Age", "value": f"{int(req.age)} yrs", "impact_pct": round(fi.get("Age", 0.027) * 100, 1)},
        {"feature": "Gender & Body Mass", "value": f"{req.gender.title()}, {req.weight}kg", "impact_pct": round((fi.get("Gender_Code", 0.016) + fi.get("Weight", 0.006)) * 100, 1)},
    ]
    
    # Compare with other models
    comparison = {}
    for m_name, m_inst in MODELS.items():
        try:
            c_val = max(1.0, round(float(m_inst.predict(input_data)[0]), 1))
            comparison[m_name] = c_val
        except Exception:
            pass

    return {
        "calories_burnt": pred_val,
        "burn_rate_kcal_min": burn_rate,
        "mets": mets,
        "intensity": intensity,
        "bmi": round(bmi, 1),
        "model_used": model_key,
        "comparison": comparison,
        "food_equivalents": equivalents,
        "contributions": contributions
    }

@app.post("/api/log-workout")
def log_workout(req: LogWorkoutRequest):
    global workout_history
    calories = req.calories_burnt
    if calories is None:
        p = predict_calories(req)
        calories = p["calories_burnt"]
        
    entry = {
        "id": len(workout_history) + 1,
        "activity_name": req.activity_name or "Custom Exercise",
        "gender": req.gender,
        "age": req.age,
        "height": req.height,
        "weight": req.weight,
        "duration": req.duration,
        "heart_rate": req.heart_rate,
        "body_temp": req.body_temp,
        "calories_burnt": calories,
        "burn_rate": round(calories / max(1.0, req.duration), 2),
        "model_used": req.model_name or "xgboost",
        "timestamp": "Just now"
    }
    workout_history.insert(0, entry)
    return {"status": "success", "entry": entry, "total_logged": len(workout_history)}

@app.get("/api/workouts")
def get_workouts():
    return {"workouts": workout_history}

@app.get("/api/stats")
def get_dashboard_stats():
    total_today = sum(w["calories_burnt"] for w in workout_history[:5])
    return {
        "user_name": "Sarah Keith",
        "hydration_level": 69,
        "hydration_liters": "1,6 L",
        "hydration_target": "2L Per day",
        "water_today": "292 ml mineral water today",
        "heart_rate_bpm": 88,
        "heart_rate_range": "60-100 beats per minute",
        "calories_today": round(total_today, 1),
        "weekly_stats": [
            {"day": "Mon", "activity": 65, "nutrition": 50, "mindfulness": 35},
            {"day": "Tue", "activity": 45, "nutrition": 60, "mindfulness": 40},
            {"day": "Wed", "activity": 75, "nutrition": 55, "mindfulness": 50},
            {"day": "Thu", "activity": 50, "nutrition": 65, "mindfulness": 45},
            {"day": "Fri", "activity": 85, "nutrition": 70, "mindfulness": 60},
            {"day": "Sat", "activity": 90, "nutrition": 80, "mindfulness": 70},
            {"day": "Sun", "activity": 60, "nutrition": 65, "mindfulness": 55}
        ],
        "activities": [
            {
                "id": "cardio-1",
                "title": "Cardio Masterclass",
                "subtitle": "6 Reps - Advance",
                "progress_pct": 78,
                "status": "In Progress",
                "button_text": "Resume ▶",
                "default_params": {"duration": 30, "heart_rate": 138, "body_temp": 38.8, "activity_name": "Cardio Masterclass"}
            },
            {
                "id": "run-10k",
                "title": "10K City Running",
                "subtitle": "Weekly Challenge - Advance",
                "progress_pct": 0,
                "status": "Not Started",
                "button_text": "Start Now",
                "default_params": {"duration": 50, "heart_rate": 155, "body_temp": 39.5, "activity_name": "10K City Running"}
            },
            {
                "id": "hiit-burn",
                "title": "HIIT Metabolic Burn",
                "subtitle": "12 Intervals - High Intensity",
                "progress_pct": 42,
                "status": "In Progress",
                "button_text": "Resume ▶",
                "default_params": {"duration": 25, "heart_rate": 165, "body_temp": 39.3, "activity_name": "HIIT Metabolic Burn"}
            }
        ]
    }

# Serve static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")
