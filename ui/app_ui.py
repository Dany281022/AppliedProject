# app_ui.py

import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="ML Prediction App", page_icon="🎯", layout="centered")
st.title("🎯 ML Prediction App")
st.write("Enter your data below to get a prediction.")

# Sidebar — API health check
try:
    health = requests.get(f"{API_URL}/health", timeout=5)
    if health.status_code == 200:
        st.sidebar.success("✅ API Connected")
    else:
        st.sidebar.error("❌ API Error")
except:
    st.sidebar.error("❌ API Offline")
    st.sidebar.info("Start API: `python -m uvicorn api.main:app --reload`")

st.sidebar.divider()

# Sidebar — About
st.sidebar.markdown("### ℹ️ About This App")
st.sidebar.write("This app predicts outcomes based on your input.")
st.sidebar.markdown("**How to use:**")
st.sidebar.markdown("1. Enter your data\n2. Click 'Get Prediction'\n3. View results")

st.sidebar.divider()

# Sidebar — Model Info
try:
    info = requests.get(f"{API_URL}/info", timeout=5)
    if info.status_code == 200:
        data = info.json()
        st.sidebar.markdown("### 📊 Model Info")
        st.sidebar.write(f"Model: {data['model_type']}")
        st.sidebar.write(f"Features: {len(data['features_expected'])}")
except:
    pass

st.divider()

# Input form
st.markdown("### 📝 Enter Your Data")
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        lag_1 = st.number_input("Lag 1 (previous week value)", min_value=0.0, value=10.0, help="Value from last week")
    with col2:
        lag_2 = st.number_input("Lag 2 (value 2 weeks ago)", min_value=0.0, value=10.0, help="Value from 2 weeks ago")
    lag_52 = st.number_input("Lag 52 (value 1 year ago)", min_value=0.0, value=10.0, help="Value from 1 year ago")
    submitted = st.form_submit_button("🔮 Get Prediction")

# When form is submitted
if submitted:
    with st.spinner("Getting prediction..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"lag_1": lag_1, "lag_2": lag_2, "lag_52": lag_52}
            )
            if response.status_code == 200:
                result = response.json()
                st.success(f"✅ Prediction: **{result['prediction']}**")
                if result.get("confidence"):
                    st.metric("Confidence", f"{result['confidence']:.1%}")
            else:
                st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Make sure the server is running!")

st.divider()
st.caption("Built with ❤️ by Team [Dany] | AIE1014 Capstone Project")
