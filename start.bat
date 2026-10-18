@echo off
echo ==========================================================
echo  Starting Calories Burnt Prediction (Streamlit Dashboard)...
echo ==========================================================
echo.
start "" "http://localhost:8501"
python -m streamlit run streamlit_app.py --server.port 8501
pause
