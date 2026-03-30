# app_ui.py
# Weekly Sales Forecaster — Team Dany | AIE1014 Capstone Project
# This UI connects to a FastAPI backend to predict weekly retail sales
# using a RandomForestRegressor trained on historical lag features.

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

# Use 127.0.0.1 instead of localhost to avoid Windows DNS resolution
# overhead (~2s delay caused by IPv6 fallback on Windows systems)
API_URL = "http://127.0.0.1:8000"

# Page config must be the first Streamlit command —
# calling any other st.* before this raises a StreamlitAPIException
st.set_page_config(page_title="Weekly Sales Forecaster", page_icon="📈", layout="wide")

# ─── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("📈 Weekly Sales Forecaster")
st.sidebar.markdown("**Team Dany | AIE1014**")
st.sidebar.divider()

# Check API health on every page load so the user knows immediately
# whether the backend is reachable before they submit a prediction
try:
    health = requests.get(f"{API_URL}/health", timeout=5)
    if health.status_code == 200:
        st.sidebar.success("✅ API Connected")
    else:
        st.sidebar.error("❌ API Error")
except:
    # Show the exact command needed to start the API — reduces confusion
    st.sidebar.error("❌ API Offline")
    st.sidebar.code("cd api && python app.py", language="bash")

st.sidebar.divider()

# Fetch model metadata dynamically from /info so the sidebar always reflects
# the current deployed model without requiring a UI code change
try:
    info = requests.get(f"{API_URL}/info", timeout=5)
    if info.status_code == 200:
        data = info.json()
        perf = data.get('performance', {})
        st.sidebar.markdown("### 📊 Model Info")
        st.sidebar.write(f"**Model:** {data['model_type']}")
        st.sidebar.write(f"**Features:** {data['num_features']}")
        st.sidebar.write(f"**R2 Score:** {perf.get('r2', 'N/A')}")
        st.sidebar.write(f"**RMSE:** ${perf.get('rmse', 0):,.0f}")
        st.sidebar.write(f"**MAE:** ${perf.get('mae', 0):,.0f}")
except:
    pass

st.sidebar.divider()
st.sidebar.markdown("**Stakeholder:** Retail Business Manager")
st.sidebar.caption("Built with ❤️ by Team Dany")

# ─── Main Title ────────────────────────────────────────────────────────────────

st.title("📈 Weekly Sales Forecaster — Team Dany")
st.write("Predict next week's retail sales using historical lag features and moving averages.")
st.divider()

# ─── Tabs ──────────────────────────────────────────────────────────────────────

# Three tabs: prediction form, model dashboard, prediction history table
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Dashboard", "📋 History"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 📝 Enter Your Sales Data")

    # Auto-fill date fields with today's values so the user does not need
    # to look up the current week number or month manually
    now = datetime.now()

    # st.form batches all inputs into a single submission —
    # prevents the API from being called on every widget interaction
    with st.form("prediction_form"):
        st.markdown("#### Lag Features")
        col1, col2 = st.columns(2)
        with col1:
            lag_1  = st.number_input("Lag 1 — Previous week ($)",  min_value=0.0, value=100.0)
            lag_4  = st.number_input("Lag 4 — 4 weeks ago ($)",    min_value=0.0, value=100.0)
            lag_12 = st.number_input("Lag 12 — 12 weeks ago ($)",  min_value=0.0, value=100.0)
            lag_52 = st.number_input("Lag 52 — 1 year ago ($)",    min_value=0.0, value=100.0)
        with col2:
            lag_2  = st.number_input("Lag 2 — 2 weeks ago ($)",    min_value=0.0, value=100.0)
            lag_8  = st.number_input("Lag 8 — 8 weeks ago ($)",    min_value=0.0, value=100.0)
            lag_26 = st.number_input("Lag 26 — 26 weeks ago ($)",  min_value=0.0, value=100.0)

        st.markdown("#### Moving Averages & Volatility")
        col3, col4, col5 = st.columns(3)
        with col3:
            ma_4  = st.number_input("MA 4 weeks ($)",  min_value=0.0, value=100.0)
        with col4:
            ma_12 = st.number_input("MA 12 weeks ($)", min_value=0.0, value=100.0)
        with col5:
            # std_4 measures sales volatility over the past 4 weeks —
            # higher values indicate less stable recent sales patterns
            std_4 = st.number_input("Std Dev 4 weeks", min_value=0.0, value=10.0)

        st.markdown("#### Date Features (Auto-filled)")
        col6, col7, col8 = st.columns(3)
        with col6:
            weekofyear = st.number_input("Week of Year", min_value=1,    max_value=52,   value=int(now.isocalendar()[1]))
        with col7:
            month      = st.number_input("Month",        min_value=1,    max_value=12,   value=now.month)
        with col8:
            year       = st.number_input("Year",         min_value=2000, max_value=2100, value=now.year)

        submitted = st.form_submit_button("🔮 Get Prediction", use_container_width=True)

    if submitted:
        # Show a spinner during the API call so the user knows the system
        # is processing — important given the network overhead on Windows
        with st.spinner("Generating sales forecast..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={
                        "lag_1": lag_1, "lag_2": lag_2, "lag_4": lag_4,
                        "lag_8": lag_8, "lag_12": lag_12, "lag_26": lag_26,
                        "lag_52": lag_52, "ma_4": ma_4, "ma_12": ma_12,
                        "std_4": std_4, "weekofyear": weekofyear,
                        "month": month, "year": year
                    }
                )
                if response.status_code == 200:
                    result     = response.json()
                    prediction = result['prediction']
                    confidence = result.get('confidence', 'N/A')
                    resp_time  = result.get('response_time_ms', 'N/A')

                    # Format for the stakeholder: currency, units, context
                    formatted = f"${prediction:,.2f}"

                    st.success(f"✅ Predicted Next-Week Sales: **{formatted}**")

                    # Contextual interpretation — only shown when lag_1 is
                    # a realistic sales value (> $1M) to avoid absurd percentages
                    if lag_1 > 1_000_000:
                        pct_change = ((prediction - lag_1) / lag_1) * 100
                        direction  = "above" if pct_change >= 0 else "below"
                        st.info(
                            f"📊 This forecast is **{abs(pct_change):.1f}% {direction}** "
                            f"last week's sales of **${lag_1:,.2f}**. "
                            f"Use this to adjust inventory and staffing for next week."
                        )

                    # Confidence interval from the API
                    st.markdown(f"**Prediction Interval:** {confidence}")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Forecast",      formatted)
                    with col2:
                        st.metric("Week",          f"W{weekofyear}")
                    with col3:
                        st.metric("Model",         "Random Forest")
                    with col4:
                        st.metric("Response Time", f"{resp_time} ms")

                    # Persist prediction in session_state history for Tab 3
                    if "history" not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.append({
                        "Timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Week":               weekofyear,
                        "Month":              month,
                        "Year":               year,
                        "Lag 1":              lag_1,
                        "Lag 52":             lag_52,
                        "Prediction ($)":     round(prediction, 2),
                        "Response Time (ms)": resp_time
                    })

                else:
                    st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")

            # Catch connection errors separately so the message is actionable
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure the server is running!")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Model Performance Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R2 Score", "0.3025", help="Higher is better (max 1.0)")
    with col2:
        st.metric("RMSE", "$2,034