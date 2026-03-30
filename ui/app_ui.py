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

st.set_page_config(page_title="Weekly Sales Forecaster", page_icon="📈", layout="wide")

# ─── Sidebar ───────────────────────────────────────────────
st.sidebar.title("📈 Weekly Sales Forecaster")
st.sidebar.markdown("**Team Dany | AIE1014**")
st.sidebar.divider()

try:
    health = requests.get(f"{API_URL}/health", timeout=5)
    if health.status_code == 200:
        st.sidebar.success("✅ API Connected")
    else:
        st.sidebar.error("❌ API Error")
except:
    st.sidebar.error("❌ API Offline")
    st.sidebar.code("cd api && python app.py", language="bash")

st.sidebar.divider()

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

# ─── Main Title ────────────────────────────────────────────
st.title("📈 Weekly Sales Forecaster — Team Dany")
st.write("Predict next week's retail sales using historical lag features and moving averages.")
st.divider()

# ─── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Dashboard", "📋 History"])

# ══════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 📝 Enter Your Sales Data")

    now = datetime.now()

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
                    formatted  = f"${prediction:,.2f}"

                    st.success(f"✅ Predicted Next-Week Sales: **{formatted}**")

                    if lag_1 > 1_000_000:
                        pct_change = ((prediction - lag_1) / lag_1) * 100
                        direction  = "above" if pct_change >= 0 else "below"
                        st.info(
                            f"📊 This forecast is **{abs(pct_change):.1f}% {direction}** "
                            f"last week's sales of **${lag_1:,.2f}**. "
                            f"Use this to adjust inventory and staffing for next week."
                        )

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

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure the server is running!")

# ══════════════════════════════════════════════════════════
# TAB 2 — DASHBOARD
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Model Performance Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R2 Score", "0.3025",    help="Higher is better (max 1.0)")
    with col2:
        st.metric("RMSE",     "$2,034,160", help="Root Mean Squared Error in dollars")
    with col3:
        st.metric("MAE",      "$1,472,779", help="Mean Absolute Error in dollars")
    with col4:
        st.metric("Features", "13",         help="Number of input features")

    st.divider()

    # Feature importance — real values fetched from API
    st.markdown("#### 🎯 Feature Importance")
    try:
        info_resp = requests.get(f"{API_URL}/info", timeout=5)
        if info_resp.status_code == 200:
            info_data   = info_resp.json()
            features    = info_data['features_expected']
            importances = info_data['feature_importances']  # real values from model

            df_imp = pd.DataFrame({
                "Feature":    features,
                "Importance": importances
            }).sort_values("Importance", ascending=True)

            fig_imp = px.bar(
                df_imp, x="Importance", y="Feature", orientation="h",
                title="Feature Importance — Random Forest",
                color="Importance", color_continuous_scale="blues",
                text="Importance"
            )
            fig_imp.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_imp.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig_imp, use_container_width=True)
    except Exception:
        st.info("Feature importance will appear after API is connected.")

    st.divider()

    st.markdown("#### 📈 Prediction History Chart")
    if "history" in st.session_state and len(st.session_state.history) > 0:
        df_hist  = pd.DataFrame(st.session_state.history)
        fig_hist = px.line(
            df_hist, x="Timestamp", y="Prediction ($)",
            title="Sales Predictions Over Time",
            markers=True, line_shape="spline"
        )
        fig_hist.update_traces(line_color="#1f77b4", marker_size=8)
        fig_hist.update_layout(xaxis_title="Time", yaxis_title="Predicted Sales ($)", height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Make predictions in the Prediction tab to see the chart here.")

    st.divider()

    st.markdown("#### ⚡ Response Time History")
    if "history" in st.session_state and len(st.session_state.history) > 0:
        df_hist = pd.DataFrame(st.session_state.history)
        if "Response Time (ms)" in df_hist.columns:
            fig_rt = px.bar(
                df_hist, x="Timestamp", y="Response Time (ms)",
                title="API Response Time per Prediction (ms)",
                color="Response Time (ms)", color_continuous_scale="reds"
            )
            fig_rt.add_hline(y=1000, line_dash="dash", line_color="red",
                             annotation_text="Target < 1000ms")
            fig_rt.update_layout(height=350)
            st.plotly_chart(fig_rt, use_container_width=True)
    else:
        st.info("Make predictions to see response time chart.")

# ══════════════════════════════════════════════════════════
# TAB 3 — HISTORY
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📋 Prediction History")

    if "history" in st.session_state and len(st.session_state.history) > 0:
        df_hist = pd.DataFrame(st.session_state.history)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", len(df_hist))
        with col2:
            st.metric("Avg Forecast", f"${df_hist['Prediction ($)'].mean():,.2f}")
        with col3:
            st.metric("Max Forecast", f"${df_hist['Prediction ($)'].max():,.2f}")
        with col4:
            st.metric("Min Forecast", f"${df_hist['Prediction ($)'].min():,.2f}")

        st.dataframe(df_hist, use_container_width=True)

        csv = df_hist.to_csv(index=False)
        st.download_button(
            label="📥 Download History as CSV",
            data=csv,
            file_name="predictions_history.csv",
            mime="text/csv"
        )

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions yet. Go to the Prediction tab to get started!")