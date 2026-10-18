import streamlit as st
import streamlit.components.v1 as components
import os
import base64
import re

# Page config
st.set_page_config(
    page_title="Calories Burnt Prediction — Sarah Keith Dashboard",
    page_icon="🏃‍♀️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to hide Streamlit default chrome completely
st.markdown("""
<style>
    /* Hide Streamlit header, footer, decoration, toolbar */
    #MainMenu, header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
    }
    
    /* Hide default Streamlit sidebar */
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    /* Full bleed container */
    .block-container {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
    }
    
    .stApp {
        background-color: #ECEEF2 !important;
    }
    
    iframe {
        border: none !important;
        width: 100% !important;
        height: 100vh !important;
        min-height: 980px !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper for base64 image encoding
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return f"data:image/png;base64,{base64.b64encode(img_file.read()).decode()}"
    return ""

athlete_b64 = get_base64_image("static/assets/hero_athlete.png")

# Read static/index.html and inject base64 images so it works anywhere
html_path = "static/index.html"
if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    # Replace static asset paths with base64 data URIs
    if athlete_b64:
        html_content = html_content.replace('/static/assets/hero_athlete.png', athlete_b64)
        
    components.html(html_content, height=1000, scrolling=True)
else:
    st.error("static/index.html not found.")
