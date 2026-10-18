# 🏃‍♀️ AI Calories Burnt Prediction System & Fitness Dashboard

A production-ready Machine Learning web application predicting **Calories Burnt** during exercise based on physiological biometrics, paired with an ultra-clean, modern fitness dashboard UI inspired by your design.

---

## 📸 Design & Features

- **Dashboard Layout**: Modeled precisely after the reference design with soft neumorphic surfaces, clean typography, rounded card panels, and ambient lighting.
- **Biometric Health Status**:
  - **Hydration Level**: Segmented circular radial dial (69%, 1.6L, 91 BPM, Well indicator).
  - **Heart Rate**: Dynamic real-time pulse equalizer with animated cyan cardiac waveforms (88 BPM).
  - **Motivational Banner**: Congratulatory activity progress banner with interactive quick links.
- **Interactive ML Predictor Modal**:
  - Fine-tune **Gender**, **Age**, **Height**, **Weight**, **Duration**, **Heart Rate**, and **Body Temperature** via sliders.
  - One-click presets: *Cardio Masterclass*, *10K City Running*, *HIIT Metabolic Burn*, and *Morning Warmup*.
  - Real-time instant prediction calculations as you adjust sliders.
  - Live metric breakdown: **Calories Burnt**, **Burn Rate (kcal/min)**, **METs score**, **Intensity level**, and **Food Equivalents** (Avocados, Pizza slices, km running).
  - **Model Explainability**: Visual feature importance breakdown highlighting key physiological drivers (Exercise Duration 69.2%, Heart Rate 18.8%, Body Temp 7.1%, Age 2.7%, Weight 0.6%).
- **Weekly Statistics**: Stacked bar chart for Mon–Sun comparing Activity, Nutrition, and Mindfulness.
- **Your Activity Section**: Workout cards with circular progress rings and interactive "Resume" / "Start Now" actions.
- **Light & Dark Mode**: Smooth one-click toggle between crisp off-white minimalism and deep slate midnight themes.

---

## 🤖 Machine Learning Model Benchmarks

Trained on **15,000 real-world exercise biometric samples** using cross-validated train-test splits:

| Model | $R^2$ Score (Accuracy) | MAE (Mean Absolute Error) | RMSE | Status |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost Regressor** | **0.9993 (99.93%)** | **1.183 kcal** | **1.649** | 🏆 **Best Performer** |
| **Gradient Boosting** | **0.9991 (99.91%)** | **1.431 kcal** | **1.948** | Ultra-smooth |
| **Random Forest** | **0.9979 (99.79%)** | **1.901 kcal** | **2.921** | Robust Ensemble |
| **Linear Regression** | **0.9688 (96.88%)** | **8.195 kcal** | **11.220** | Baseline |

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Retrain Models (Optional)
The pre-trained models are already saved in the `models/` directory. If you ever want to retrain:
```bash
python train_model.py
```

### 4. Run the Application
Start the FastAPI server:
```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```
Then open your browser at:
**[http://127.0.0.1:8000](http://127.0.0.1:8000)**

Interactive API Swagger documentation is available at:
**[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

---

## 📂 Project Architecture

```
calories prediction/
├── app.py                     # FastAPI application & ML inference endpoints
├── train_model.py             # Model training, evaluation & serialization pipeline
├── calories.csv               # Ground truth calories dataset
├── exercise.csv               # Exercise biometric features dataset
├── requirements.txt           # Project package dependencies
├── models/
│   ├── xgboost_model.pkl      # Serialized XGBoost model (99.93% R²)
│   ├── random_forest_model.pkl# Serialized Random Forest model
│   ├── gradient_boosting_model.pkl
│   ├── linear_regression_model.pkl
│   └── metadata.json          # Metrics, feature importances & stats
└── static/
    ├── index.html             # High-fidelity dashboard interface
    └── assets/
        ├── hero_athlete.png   # Athlete banner visual
        ├── sarah_avatar.png   # Profile avatar
        └── team_avatars.png   # Team avatars
```
