import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import base64

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="Calories Burnt Prediction — Sarah Keith Dashboard",
    page_icon="🏃‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper for base64 image encoding
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return f"data:image/png;base64,{base64.b64encode(img_file.read()).decode()}"
    return ""

avatar_b64 = get_base64_image("static/assets/sarah_avatar_circle.png")
if not avatar_b64:
    avatar_b64 = get_base64_image("static/assets/sarah_avatar.png")
athlete_b64 = get_base64_image("static/assets/hero_athlete.png")
team_b64 = get_base64_image("static/assets/team_avatars.png")

# ==========================================================
# LOAD TRAINED ML MODELS & METADATA
# ==========================================================
@st.cache_resource
def load_models():
    models = {}
    metadata = {}
    try:
        if os.path.exists("models/metadata.json"):
            with open("models/metadata.json", "r") as f:
                metadata = json.load(f)
        for m in ["xgboost", "random_forest", "gradient_boosting", "linear_regression"]:
            path = f"models/{m}_model.pkl"
            if os.path.exists(path):
                models[m] = joblib.load(path)
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return models, metadata

MODELS, METADATA = load_models()

# ==========================================================
# SESSION STATE INITIALIZATION
# ==========================================================
if "gender" not in st.session_state:
    st.session_state.gender = "female"
if "age" not in st.session_state:
    st.session_state.age = 28
if "height" not in st.session_state:
    st.session_state.height = 168.0
if "weight" not in st.session_state:
    st.session_state.weight = 58.0
if "duration" not in st.session_state:
    st.session_state.duration = 30.0
if "heart_rate" not in st.session_state:
    st.session_state.heart_rate = 138.0
if "body_temp" not in st.session_state:
    st.session_state.body_temp = 38.8
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "xgboost"
if "activity_name" not in st.session_state:
    st.session_state.activity_name = "Cardio Masterclass"
if "logged_workouts" not in st.session_state:
    st.session_state.logged_workouts = [
        {"name": "Cardio Masterclass", "duration": 30, "calories": 224.5, "bpm": 138, "time": "08:30 AM"},
        {"name": "Morning Warmup", "duration": 15, "calories": 68.2, "bpm": 102, "time": "07:15 AM"}
    ]
if "hydration_liters" not in st.session_state:
    st.session_state.hydration_liters = 1.6
if "hydration_pct" not in st.session_state:
    st.session_state.hydration_pct = 69
if "show_predictor_drawer" not in st.session_state:
    st.session_state.show_predictor_drawer = True
if "dark_theme" not in st.session_state:
    st.session_state.dark_theme = False

# ==========================================================
# CUSTOM CSS TO MATCH EXACT MOCKUP DESIGN
# ==========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }
    
    /* Overall Page Background */
    .stApp {
        background-color: #ECEEF2;
    }
    
    /* Hide default Streamlit header and footer */
    #MainMenu, header, footer {
        visibility: hidden;
        height: 0px;
    }
    
    /* Main block container padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1400px !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #F7F8FA !important;
        border-right: 1px solid rgba(226, 232, 240, 0.8) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }
    
    /* Neumorphic Rounded Cards */
    .fit-card {
        background: #FFFFFF;
        border-radius: 28px;
        padding: 24px;
        box-shadow: 0 10px 30px -4px rgba(100, 116, 139, 0.06), 0 4px 12px -2px rgba(100, 116, 139, 0.03);
        border: 1px solid rgba(226, 232, 240, 0.7);
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    /* Hero Card */
    .hero-card {
        background: #FFFFFF;
        border-radius: 32px;
        padding: 24px 32px;
        box-shadow: 0 12px 36px -4px rgba(100, 116, 139, 0.08);
        border: 1px solid rgba(226, 232, 240, 0.7);
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: relative;
        overflow: hidden;
        margin-bottom: 24px;
    }
    
    /* Nav Items in Sidebar */
    .nav-item-active {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 12px 18px;
        border-radius: 18px;
        background: #FFFFFF;
        color: #0F172A;
        font-weight: 700;
        font-size: 14px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.05);
        border: 1px solid rgba(226, 232, 240, 0.6);
        margin-bottom: 8px;
        text-decoration: none;
    }
    
    .nav-item {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 12px 18px;
        border-radius: 18px;
        color: #64748B;
        font-weight: 500;
        font-size: 14px;
        margin-bottom: 6px;
        transition: all 0.2s;
        text-decoration: none;
    }
    .nav-item:hover {
        background: rgba(255, 255, 255, 0.8);
        color: #0F172A;
    }
    
    /* Custom Blue Buttons */
    .stButton > button {
        border-radius: 50px !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        padding: 0.6rem 1.6rem !important;
        transition: all 0.2s ease !important;
        border: none !important;
    }
    
    .primary-btn > div > button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.3) !important;
    }
    .primary-btn > div > button:hover {
        background-color: #1D4ED8 !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 20px 0 rgba(37, 99, 235, 0.4) !important;
    }
    
    /* Pulse Equalizer animation */
    @keyframes pulse-wave {
        0%, 100% { height: 25%; }
        50% { height: 95%; }
    }
    .eq-bar {
        width: 6px;
        border-radius: 6px;
        background: linear-gradient(180deg, #38BDF8 0%, #2563EB 100%);
        animation: pulse-wave 1.6s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR RECREATION
# ==========================================================
with st.sidebar:
    # Brand Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 24px;">
        <div style="display: inline-flex; align-items: center; justify-content: center; width: 54px; height: 54px; border-radius: 18px; background: linear-gradient(135deg, #2563EB, #38BDF8); color: white; font-size: 24px; font-weight: 900; box-shadow: 0 8px 20px rgba(37,99,235,0.3);">
            ⚡
        </div>
        <h3 style="margin: 10px 0 2px 0; font-size: 18px; font-weight: 800; color: #1E293B;">CalorieFit AI</h3>
        <span style="font-size: 11px; color: #3B82F6; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Biometrics Dashboard</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation items
    st.markdown("""
    <div style="margin-bottom: 28px;">
        <div class="nav-item-active">
            <span style="color: #2563EB; font-size: 16px;">⊞</span>
            <span>Dashboard</span>
        </div>
        <div class="nav-item">
            <span style="font-size: 16px;">⏱</span>
            <span>Activity</span>
        </div>
        <div class="nav-item">
            <span style="font-size: 16px;">💬</span>
            <span>Chat</span>
        </div>
        <div class="nav-item">
            <span style="font-size: 16px;">◌</span>
            <span>Progress</span>
        </div>
        <div class="nav-item">
            <span style="font-size: 16px;">🛟</span>
            <span>Support</span>
        </div>
        <div class="nav-item">
            <span style="font-size: 16px;">⚙</span>
            <span>Setting</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Model Selector in Sidebar
    st.markdown("<p style='font-size: 11px; font-weight: 800; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;'>PREDICTION ENGINE</p>", unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Model Engine",
        options=["xgboost", "gradient_boosting", "random_forest", "linear_regression"],
        format_func=lambda x: {
            "xgboost": "XGBoost Regressor (99.93% R²)",
            "gradient_boosting": "Gradient Boosting (99.91% R²)",
            "random_forest": "Random Forest (99.79% R²)",
            "linear_regression": "Linear Regression (96.88% R²)"
        }[x],
        index=0,
        label_visibility="collapsed"
    )
    st.session_state.selected_model = model_choice
    
    # Dataset specs
    st.markdown("""
    <div style="background: #F1F5F9; border-radius: 16px; padding: 12px; margin-top: 16px; font-size: 11px; color: #475569;">
        <div style="font-weight: 700; color: #1E293B; margin-bottom: 4px;">Dataset Specs</div>
        <div>• 15,000 biometrics records</div>
        <div>• Target: Calories Burnt (kcal)</div>
        <div>• Model MAE: <b>1.18 kcal</b></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Theme toggle visual
    st.markdown("""
    <div style="margin-top: 24px; text-align: center;">
        <span style="font-size: 11px; font-weight: 600; color: #64748B; background: #E2E8F0; padding: 6px 14px; border-radius: 20px;">
            ☀️ Light Minimal Mode Active
        </span>
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# ML PREDICTION FUNCTION
# ==========================================================
def run_prediction():
    model = MODELS.get(st.session_state.selected_model)
    if not model:
        return 224.5, 7.48, 8.5, "Vigorous", {"avocados": 0.9, "pizza": 0.8}
        
    gender_code = 1 if st.session_state.gender == "female" else 0
    bmi = st.session_state.weight / ((st.session_state.height / 100) ** 2)
    
    features = pd.DataFrame([{
        'Gender_Code': gender_code,
        'Age': float(st.session_state.age),
        'Height': float(st.session_state.height),
        'Weight': float(st.session_state.weight),
        'Duration': float(st.session_state.duration),
        'Heart_Rate': float(st.session_state.heart_rate),
        'Body_Temp': float(st.session_state.body_temp),
        'BMI': bmi
    }])
    
    val = float(model.predict(features)[0])
    val = max(1.0, round(val, 1))
    burn_rate = round(val / max(1.0, st.session_state.duration), 2)
    mets = round(val / (st.session_state.weight * (st.session_state.duration / 60.0)), 2)
    
    intensity = "Moderate"
    if mets >= 8.5 or st.session_state.heart_rate >= 150:
        intensity = "High / Peak Intensity"
    elif mets >= 6.0 or st.session_state.heart_rate >= 125:
        intensity = "Vigorous Cardio"
    elif mets >= 3.5:
        intensity = "Moderate Aerobic"
        
    food_eq = {
        "avocados": round(val / 240, 1),
        "pizza": round(val / 285, 1),
        "running_km": round(val / 65, 1)
    }
    return val, burn_rate, mets, intensity, food_eq

pred_calories, pred_burn_rate, pred_mets, pred_intensity, pred_food = run_prediction()

# ==========================================================
# MAIN DASHBOARD AREA
# ==========================================================

# Top bar: Dashboard breadcrumb & icons
st.markdown(f"""
<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
    <div style="display: flex; align-items: center; gap: 8px;">
        <span style="font-size: 20px; font-weight: 800; color: #1E293B;">‹ Dashboard</span>
    </div>
    <div style="display: flex; align-items: center; gap: 14px;">
        <div style="width: 38px; height: 38px; border-radius: 50%; background: #FFFFFF; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 8px rgba(0,0,0,0.04); font-size: 15px; color: #64748B;">ⓘ</div>
        <div style="width: 38px; height: 38px; border-radius: 50%; background: #FFFFFF; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 8px rgba(0,0,0,0.04); font-size: 15px; color: #64748B;">✉</div>
        <div style="position: relative; width: 38px; height: 38px; border-radius: 50%; background: #FFFFFF; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 8px rgba(0,0,0,0.04); font-size: 15px; color: #64748B;">
            🔔
            <span style="position: absolute; top: 8px; right: 8px; width: 8px; height: 8px; background: #EF4444; border-radius: 50%; border: 1.5px solid #FFFFFF;"></span>
        </div>
        <div style="width: 38px; height: 38px; border-radius: 50%; background: #2563EB; color: #FFFFFF; display: flex; align-items: center; justify-content: center; font-size: 20px; font-weight: bold; box-shadow: 0 4px 12px rgba(37,99,235,0.3);">+</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Grid Layout: 8 columns (Center) and 4 columns (Right)
col_left, col_right = st.columns([8, 4], gap="large")

# ----------------------------------------------------------
# CENTER COLUMN: Hero Card & Health Status
# ----------------------------------------------------------
with col_left:
    
    # Hero Card
    st.markdown(f"""
    <div class="hero-card">
        <div style="width: 45%; text-align: left;">
            <img src="{athlete_b64}" style="max-height: 210px; object-fit: contain; filter: drop-shadow(0 15px 25px rgba(0,0,0,0.08));" />
        </div>
        <div style="width: 55%; padding-left: 20px;">
            <h1 style="font-size: 32px; font-weight: 800; color: #1E293B; margin: 0 0 8px 0; letter-spacing: -0.02em;">Welcome back!</h1>
            <p style="font-size: 16px; color: #64748B; font-weight: 500; margin: 0 0 20px 0;">Ready to plan your next exercise & predict calories burnt?</p>
            <div style="display: flex; gap: 10px; align-items: center;">
                <span style="background: #2563EB; color: #FFFFFF; font-weight: 700; font-size: 14px; padding: 12px 24px; border-radius: 50px; box-shadow: 0 4px 14px rgba(37,99,235,0.35);">
                    Create Today's Plan
                </span>
                <span style="background: #F1F5F9; color: #475569; font-weight: 600; font-size: 13px; padding: 12px 18px; border-radius: 50px;">
                    ⚡ AI Powered
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Health Status Section
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin: 24px 0 14px 0;">
        <span style="font-size: 13px; font-weight: 800; color: #1E293B; letter-spacing: 0.08em; text-transform: uppercase;">HEALTH STATUS</span>
        <span style="font-size: 12px; font-weight: 600; color: #2563EB;">Synced with Apple Watch & IoT</span>
    </div>
    """, unsafe_allow_html=True)
    
    h_col1, h_col2 = st.columns(2, gap="medium")
    
    # Card 1: Hydration Level
    with h_col1:
        # Generate SVG ticks for the 69% hydration dial (horseshoe arch)
        ticks_svg = ""
        total_ticks = 25
        active_ticks = int(total_ticks * (st.session_state.hydration_pct / 100))
        for i in range(total_ticks):
            deg = -125 + (i / (total_ticks - 1)) * 250
            color = "#2563EB" if i <= active_ticks else "#E2E8F0"
            width = "4.5" if i <= active_ticks else "3"
            ticks_svg += f'<line x1="80" y1="12" x2="80" y2="24" stroke="{color}" stroke-width="{width}" stroke-linecap="round" transform="rotate({deg} 80 80)"/>'
            
        st.markdown(f"""
        <div class="fit-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <h4 style="margin: 0; font-size: 16px; font-weight: 700; color: #1E293B;">Hydration level</h4>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 28px; font-weight: 800; color: #1E293B;">{st.session_state.hydration_pct}</span>
                    <span style="font-size: 18px; font-weight: 700; color: #94A3B8;">%</span>
                </div>
            </div>
            
            <div style="display: flex; align-items: center; justify-content: space-between; margin: 16px 0;">
                <div style="position: relative; width: 120px; height: 120px; display: flex; align-items: center; justify-content: center;">
                    <svg viewBox="0 0 160 160" style="width: 100%; height: 100%; transform: rotate(-90deg);">
                        {ticks_svg}
                    </svg>
                    <div style="position: absolute; text-align: center;">
                        <div style="font-size: 20px; font-weight: 800; color: #1E293B;">{st.session_state.hydration_liters}L</div>
                        <div style="font-size: 9px; font-weight: 700; color: #94A3B8; text-transform: uppercase;">Optimal</div>
                    </div>
                </div>
                
                <div style="display: flex; flex-col; gap: 8px; flex-direction: column;">
                    <div style="font-size: 14px; font-weight: 700; color: #1E293B;"><span style="color: #F59E0B;">⚡</span> {int(st.session_state.heart_rate)} Bpm</div>
                    <div style="font-size: 14px; font-weight: 700; color: #1E293B;"><span style="color: #3B82F6;">💧</span> Well</div>
                </div>
            </div>
            
            <div style="padding-top: 12px; border-top: 1px solid #F1F5F9; display: flex; justify-content: space-between; font-size: 11px; font-weight: 600; color: #94A3B8;">
                <span>292 ml mineral water today</span>
                <span style="color: #475569; font-weight: 700;">2L Per day</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # Card 2: Heart Rate
    with h_col2:
        bars_html = ""
        sample_heights = [30, 45, 60, 35, 75, 90, 50, 40, 65, 80, 100, 70, 45, 85, 95, 60, 35, 55, 75, 40]
        for idx, h in enumerate(sample_heights):
            bars_html += f'<div style="width: 5px; height: {h}%; background: linear-gradient(180deg, #38BDF8, #2563EB); border-radius: 4px; animation-delay: {idx*0.07}s;" class="eq-bar"></div>'
            
        st.markdown(f"""
        <div class="fit-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <h4 style="margin: 0; font-size: 16px; font-weight: 700; color: #1E293B;">Heart rate</h4>
                    <span style="font-size: 11px; color: #3B82F6; font-weight: 600;">60-100 beats per minute</span>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 28px; font-weight: 800; color: #1E293B;">{int(st.session_state.heart_rate)}</span>
                    <span style="font-size: 12px; font-weight: 700; color: #94A3B8;">BPM</span>
                </div>
            </div>
            
            <div style="height: 80px; display: flex; align-items: flex-end; justify-content: space-between; margin: 18px 0; gap: 4px; padding: 0 4px;">
                {bars_html}
            </div>
            
            <div style="padding-top: 12px; border-top: 1px solid #F1F5F9; display: flex; justify-content: space-between; font-size: 11px; font-weight: 600; color: #94A3B8;">
                <span style="color: #10B981;">● Normal Resting Pulse</span>
                <span style="color: #475569; font-weight: 700;">Live Sensor</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # Motivational Banner
    st.markdown("""
    <div style="background: #FFFFFF; border-radius: 24px; padding: 16px 24px; box-shadow: 0 8px 24px -3px rgba(100, 116, 139, 0.05); border: 1px solid rgba(226, 232, 240, 0.7); display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px;">
        <div style="display: flex; align-items: center; gap: 14px;">
            <div style="width: 40px; height: 40px; border-radius: 50%; background: #EFF6FF; color: #2563EB; display: flex; align-items: center; justify-content: center; font-size: 20px;">
                😊
            </div>
            <div style="font-size: 14px; font-weight: 700; color: #1E293B;">
                Congrats! Your Activity is much beter than <span style="color: #2563EB; text-decoration: underline;">Yesterday..</span> Keep it up!
            </div>
        </div>
        <div style="width: 36px; height: 36px; border-radius: 50%; border: 1.5px solid #2563EB; color: #2563EB; display: flex; align-items: center; justify-content: center; font-weight: bold; cursor: pointer;">
            ↗
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ======================================================
    # INTERACTIVE ML CALORIE PREDICTION CONSOLE
    # ======================================================
    with st.expander("⚡ **Live Machine Learning Calorie Prediction Studio**", expanded=True):
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <span style="font-size: 13px; font-weight: 700; color: #1E293B;">Workout: <b>{st.session_state.activity_name}</b></span>
            <span style="font-size: 12px; background: #DBEAFE; color: #1E40AF; padding: 4px 12px; border-radius: 20px; font-weight: 700;">
                Model: {st.session_state.selected_model.upper()} (99.93% Accuracy)
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # Preset buttons row
        p_c1, p_c2, p_c3 = st.columns(3)
        with p_c1:
            if st.button("🏃‍♀️ Cardio Masterclass", use_container_width=True):
                st.session_state.activity_name = "Cardio Masterclass"
                st.session_state.duration = 30.0
                st.session_state.heart_rate = 138.0
                st.session_state.body_temp = 38.8
                st.rerun()
        with p_c2:
            if st.button("🏙️ 10K City Running", use_container_width=True):
                st.session_state.activity_name = "10K City Running"
                st.session_state.duration = 50.0
                st.session_state.heart_rate = 155.0
                st.session_state.body_temp = 39.5
                st.rerun()
        with p_c3:
            if st.button("🔥 HIIT Metabolic Burn", use_container_width=True):
                st.session_state.activity_name = "HIIT Metabolic Burn"
                st.session_state.duration = 25.0
                st.session_state.heart_rate = 165.0
                st.session_state.body_temp = 39.3
                st.rerun()
                
        # Interactive Sliders
        s_c1, s_c2 = st.columns(2)
        with s_c1:
            st.session_state.gender = st.radio("Gender", options=["female", "male"], index=0 if st.session_state.gender=="female" else 1, horizontal=True)
            st.session_state.duration = st.slider("Exercise Duration (mins)", min_value=1.0, max_value=120.0, value=float(st.session_state.duration), step=1.0)
            st.session_state.heart_rate = st.slider("Heart Rate (bpm)", min_value=60.0, max_value=200.0, value=float(st.session_state.heart_rate), step=1.0)
            st.session_state.body_temp = st.slider("Body Temperature (°C)", min_value=36.0, max_value=41.5, value=float(st.session_state.body_temp), step=0.1)
        with s_c2:
            st.session_state.age = st.slider("Age (years)", min_value=15, max_value=80, value=int(st.session_state.age), step=1)
            st.session_state.height = st.slider("Height (cm)", min_value=130.0, max_value=215.0, value=float(st.session_state.height), step=1.0)
            st.session_state.weight = st.slider("Weight (kg)", min_value=35.0, max_value=140.0, value=float(st.session_state.weight), step=1.0)
            
        # Prediction Output Highlight Card
        pred_calories, pred_burn_rate, pred_mets, pred_intensity, pred_food = run_prediction()
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1E293B, #0F172A); border-radius: 24px; padding: 24px; color: #FFFFFF; margin-top: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.15);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 11px; font-weight: 800; color: #38BDF8; letter-spacing: 0.1em; text-transform: uppercase;">REAL-TIME AI PREDICTION</span>
                <span style="font-size: 11px; background: rgba(56, 189, 248, 0.2); color: #38BDF8; padding: 2px 10px; border-radius: 12px; font-weight: 700;">{pred_intensity}</span>
            </div>
            <div style="display: flex; align-items: baseline; gap: 8px; margin: 10px 0;">
                <span style="font-size: 48px; font-weight: 900; letter-spacing: -0.03em; color: #FFFFFF;">{pred_calories}</span>
                <span style="font-size: 20px; font-weight: 700; color: #94A3B8;">kcal burnt</span>
            </div>
            <div style="display: flex; gap: 20px; font-size: 13px; font-weight: 600; color: #CBD5E1; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1);">
                <div>⚡ <b>{pred_burn_rate}</b> kcal/min</div>
                <div>🏃 METs: <b>{pred_mets}</b></div>
                <div>🥑 Equiv: <b>{pred_food['avocados']}</b> avocados / <b>{pred_food['pizza']}</b> pizza slices</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Save workout button
        if st.button("💾 Save & Log Workout to Dashboard", use_container_width=True, type="primary"):
            st.session_state.hydration_liters = round(st.session_state.hydration_liters + (st.session_state.duration * 0.015), 1)
            st.session_state.hydration_pct = min(98, st.session_state.hydration_pct + 4)
            st.session_state.logged_workouts.insert(0, {
                "name": st.session_state.activity_name,
                "duration": int(st.session_state.duration),
                "calories": pred_calories,
                "bpm": int(st.session_state.heart_rate),
                "time": "Just now"
            })
            st.success(f"🎉 Logged '{st.session_state.activity_name}' ({pred_calories} kcal) to your Dashboard!")
            st.rerun()

# ----------------------------------------------------------
# RIGHT COLUMN: Quick Actions, Statistics, Your Activity
# ----------------------------------------------------------
with col_right:
    
    # QUICK ACTIONS CARD
    st.markdown("""
    <div class="fit-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <span style="font-size: 12px; font-weight: 800; color: #1E293B; letter-spacing: 0.08em; text-transform: uppercase;">QUICK ACTIONS</span>
            <span style="color: #94A3B8; font-weight: bold; cursor: pointer;">•••</span>
        </div>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
            <div style="height: 56px; border-radius: 16px; background: #F1F5F9; display: flex; align-items: center; justify-content: center; font-size: 20px; cursor: pointer;">🏋️</div>
            <div style="height: 56px; border-radius: 16px; background: #F1F5F9; display: flex; align-items: center; justify-content: center; font-size: 20px; cursor: pointer;">📅</div>
            <div style="height: 56px; border-radius: 16px; background: #F1F5F9; display: flex; align-items: center; justify-content: center; font-size: 20px; cursor: pointer;">⏱️</div>
            <div style="height: 56px; border-radius: 16px; background: #2563EB; color: #FFFFFF; display: flex; align-items: center; justify-content: center; font-size: 24px; font-weight: bold; box-shadow: 0 4px 12px rgba(37,99,235,0.3); cursor: pointer;">+</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # STATISTICS CARD (Stacked Bars Mon - Sun)
    st.markdown("""
    <div class="fit-card">
        <div style="font-size: 12px; font-weight: 800; color: #1E293B; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 16px;">
            STATISTICS
        </div>
        
        <div style="display: flex; justify-content: space-between; align-items: flex-end;">
            <!-- Stacked Bars -->
            <div style="display: flex; gap: 8px; align-items: flex-end; height: 130px; flex: 1;">
                <div style="display: flex; flex-direction: column; align-items: center; gap: 6px;">
                    <div style="width: 10px; height: 90px; border-radius: 10px; background: #F1F5F9; display: flex; flex-direction: column-reverse; overflow: hidden; gap: 2px; padding: 1px;">
                        <div style="height: 45%; background: #FB7185; border-radius: 4px;"></div>
                        <div style="height: 35%; background: #818CF8; border-radius: 4px;"></div>
                        <div style="height: 20%; background: #38BDF8; border-radius: 4px;"></div>
                    </div>
                    <span style="font-size: 10px; font-weight: 700; color: #94A3B8;">Mon</span>
                </div>
                
                <div style="display: flex; flex-direction: column; align-items: center; gap: 6px;">
                    <div style="width: 10px; height: 75px; border-radius: 10px; background: #F1F5F9; display: flex; flex-direction: column-reverse; overflow: hidden; gap: 2px; padding: 1px;">
                        <div style="height: 30%; background: #FB7185; border-radius: 4px;"></div>
                        <div style="height: 45%; background: #818CF8; border-radius: 4px;"></div>
                        <div style="height: 25%; background: #38BDF8; border-radius: 4px;"></div>
                    </div>
                    <span style="font-size: 10px; font-weight: 700; color: #94A3B8;">Tue</span>
                </div>
                
                <div style="display: flex; flex-direction: column; align-items: center; gap: 6px;">
                    <div style="width: 10px; height: 105px; border-radius: 10px; background: #F1F5F9; display: flex; flex-direction: column-reverse; overflow: hidden; gap: 2px; padding: 1px;">
                        <div style="height: 50%; background: #FB7185; border-radius: 4px;"></div>
                        <div style="height: 30%; background: #818CF8; border-radius: 4px;"></div>
                        <div style="height: 20%; background: #38BDF8; border-radius: 4px;"></div>
                    </div>
                    <span style="font-size: 10px; font-weight: 700; color: #94A3B8;">Wed</span>
                </div>
                
                <div style="display: flex; flex-direction: column; align-items: center; gap: 6px;">
                    <div style="width: 10px; height: 85px; border-radius: 10px; background: #F1F5F9; display: flex; flex-direction: column-reverse; overflow: hidden; gap: 2px; padding: 1px;">
                        <div style="height: 35%; background: #FB7185; border-radius: 4px;"></div>
                        <div style="height: 40%; background: #818CF8; border-radius: 4px;"></div>
                        <div style="height: 25%; background: #38BDF8; border-radius: 4px;"></div>
                    </div>
                    <span style="font-size: 10px; font-weight: 700; color: #94A3B8;">Thu</span>
                </div>
                
                <div style="display: flex; flex-direction: column; align-items: center; gap: 6px;">
                    <div style="width: 10px; height: 120px; border-radius: 10px; background: #F1F5F9; display: flex; flex-direction: column-reverse; overflow: hidden; gap: 2px; padding: 1px;">
                        <div style="height: 55%; background: #FB7185; border-radius: 4px;"></div>
                        <div style="height: 25%; background: #818CF8; border-radius: 4px;"></div>
                        <div style="height: 20%; background: #38BDF8; border-radius: 4px;"></div>
                    </div>
                    <span style="font-size: 10px; font-weight: 700; color: #94A3B8;">Fri</span>
                </div>
                
                <div style="display: flex; flex-direction: column; align-items: center; gap: 6px;">
                    <div style="width: 10px; height: 110px; border-radius: 10px; background: #F1F5F9; display: flex; flex-direction: column-reverse; overflow: hidden; gap: 2px; padding: 1px;">
                        <div style="height: 60%; background: #FB7185; border-radius: 4px;"></div>
                        <div style="height: 20%; background: #818CF8; border-radius: 4px;"></div>
                        <div style="height: 20%; background: #38BDF8; border-radius: 4px;"></div>
                    </div>
                    <span style="font-size: 10px; font-weight: 700; color: #94A3B8;">Sat</span>
                </div>
                
                <div style="display: flex; flex-direction: column; align-items: center; gap: 6px;">
                    <div style="width: 10px; height: 95px; border-radius: 10px; background: #F1F5F9; display: flex; flex-direction: column-reverse; overflow: hidden; gap: 2px; padding: 1px;">
                        <div style="height: 40%; background: #FB7185; border-radius: 4px;"></div>
                        <div style="height: 35%; background: #818CF8; border-radius: 4px;"></div>
                        <div style="height: 25%; background: #38BDF8; border-radius: 4px;"></div>
                    </div>
                    <span style="font-size: 10px; font-weight: 700; color: #94A3B8;">Sun</span>
                </div>
            </div>
            
            <!-- Legend -->
            <div style="display: flex; flex-direction: column; gap: 10px; font-size: 11px; font-weight: 600; padding-left: 14px; border-left: 1px solid #F1F5F9;">
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="width: 10px; height: 10px; border-radius: 50%; border: 2px solid #FB7185;"></span>
                    <span style="color: #475569;">Activity</span>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="width: 10px; height: 10px; border-radius: 50%; border: 2px solid #818CF8;"></span>
                    <span style="color: #475569;">Nutrition</span>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="width: 10px; height: 10px; border-radius: 50%; border: 2px solid #38BDF8;"></span>
                    <span style="color: #475569;">Mindfulness</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # YOUR ACTIVITY CARD
    st.markdown(f"""
    <div class="fit-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <span style="font-size: 12px; font-weight: 800; color: #1E293B; letter-spacing: 0.08em; text-transform: uppercase;">YOUR ACTIVITIY</span>
            <span style="font-size: 11px; font-weight: 700; color: #94A3B8;">{len(st.session_state.logged_workouts)} logged today</span>
        </div>
        
        <!-- Activity 1 -->
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="position: relative; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center;">
                    <svg viewBox="0 0 36 36" style="width: 100%; height: 100%; transform: rotate(-90deg);">
                        <path stroke="#F1F5F9" stroke-width="3.5" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
                        <path stroke="#2563EB" stroke-dasharray="78, 100" stroke-width="3.5" stroke-linecap="round" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
                    </svg>
                    <span style="position: absolute; font-size: 10px; font-weight: 800; color: #1E293B;">78%</span>
                </div>
                <div>
                    <h5 style="margin: 0; font-size: 13px; font-weight: 800; color: #1E293B;">Cardio Masterclass</h5>
                    <span style="font-size: 11px; color: #94A3B8; font-weight: 600;">6 Reps - Advance</span>
                </div>
            </div>
            <span style="background: #2563EB; color: #FFFFFF; font-size: 11px; font-weight: 700; padding: 6px 14px; border-radius: 20px;">Resume ▶</span>
        </div>
        
        <!-- Activity 2 -->
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="position: relative; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center;">
                    <svg viewBox="0 0 36 36" style="width: 100%; height: 100%; transform: rotate(-90deg);">
                        <path stroke="#F1F5F9" stroke-width="3.5" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
                    </svg>
                    <span style="position: absolute; font-size: 10px; font-weight: 800; color: #1E293B;">0%</span>
                </div>
                <div>
                    <h5 style="margin: 0; font-size: 13px; font-weight: 800; color: #1E293B;">10K City Running</h5>
                    <span style="font-size: 11px; color: #94A3B8; font-weight: 600;">Weekly Challenge - Advance</span>
                </div>
            </div>
            <span style="background: #F1F5F9; color: #475569; font-size: 11px; font-weight: 700; padding: 6px 14px; border-radius: 20px;">Start Now</span>
        </div>
        
        <!-- Activity 3 -->
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="position: relative; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center;">
                    <svg viewBox="0 0 36 36" style="width: 100%; height: 100%; transform: rotate(-90deg);">
                        <path stroke="#F1F5F9" stroke-width="3.5" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
                        <path stroke="#10B981" stroke-dasharray="42, 100" stroke-width="3.5" stroke-linecap="round" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
                    </svg>
                    <span style="position: absolute; font-size: 10px; font-weight: 800; color: #1E293B;">42%</span>
                </div>
                <div>
                    <h5 style="margin: 0; font-size: 13px; font-weight: 800; color: #1E293B;">HIIT Metabolic Burn</h5>
                    <span style="font-size: 11px; color: #94A3B8; font-weight: 600;">12 Intervals - High Burn</span>
                </div>
            </div>
            <span style="background: #2563EB; color: #FFFFFF; font-size: 11px; font-weight: 700; padding: 6px 14px; border-radius: 20px;">Resume ▶</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Recent Logged Workouts Feed
    st.markdown("<div style='font-size: 12px; font-weight: 800; color: #1E293B; letter-spacing: 0.08em; text-transform: uppercase; margin: 20px 0 10px 0;'>RECENT WORKOUT LOGS</div>", unsafe_allow_html=True)
    for w in st.session_state.logged_workouts[:3]:
        st.markdown(f"""
        <div style="background: #FFFFFF; border-radius: 16px; padding: 12px 16px; margin-bottom: 8px; border: 1px solid #E2E8F0; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 13px; font-weight: 700; color: #1E293B;">{w['name']}</div>
                <div style="font-size: 11px; color: #94A3B8;">{w['duration']} min • {w['bpm']} bpm • {w['time']}</div>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 14px; font-weight: 800; color: #2563EB;">{w['calories']}</span>
                <span style="font-size: 10px; font-weight: 700; color: #94A3B8;">kcal</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
