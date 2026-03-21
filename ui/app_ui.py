# app_ui.py
# AIE1014 — Assignment 03 | Dany Deugoue (6024)
# Weekly Sales Prediction App — Complete UI with all challenges implemented

import streamlit as st
import requests
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
# Centralized API URL — changing port requires editing only one line
API_URL = "http://localhost:8000"

# ── Page setup (must be the FIRST Streamlit command) ───────────────────────────
st.set_page_config(
    page_title="ML Prediction App",
    page_icon="🎯",
    layout="centered"
)

# ── Prediction history — persists across re-runs within same browser session ───
# Challenge 3: session_state resets on browser refresh (new WebSocket connection)
if "history" not in st.session_state:
    st.session_state.history = []

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🎯 ML Prediction App")
st.write("Weekly sales prediction using historical lag features.")
st.divider()

# ── Sidebar: API health check ──────────────────────────────────────────────────
# Runs on every page load to always reflect current API status
with st.sidebar:
    st.header("🔌 System Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        if health.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.error(f"❌ API returned status {health.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("❌ API Offline")
        st.info("Start the API:\n`cd api && python app.py`")
    except requests.exceptions.Timeout:
        # timeout=5 prevents UI hanging indefinitely if API is slow
        st.warning("⚠️ API slow to respond")

    st.divider()

    # ── About section ──────────────────────────────────────────────────────────
    st.header("ℹ️ About")
    st.write("""
    **Model:** RandomForestRegressor

    **Predicts:** Weekly sales figures

    **How to use:**
    1. Fill in the form
    2. Click 'Get Prediction'
    3. Review the result
    """)

    st.divider()

    # ── Challenge 5: Live model info panel — fetched dynamically from /info ────
    # Better than hardcoding: if model changes, UI updates automatically
    # If hardcoded, a model update would silently show wrong information
    st.header("📊 Model Information")
    try:
        info_response = requests.get(f"{API_URL}/info", timeout=5)
        if info_response.status_code == 200:
            info = info_response.json()
            st.write(f"**Type:** {info.get('model_type', 'Unknown')}")
            st.write(f"**Version:** {info.get('version', 'Unknown')}")
            features = info.get('features_expected', [])
            st.write(f"**Features:** {len(features)}")
            with st.expander("View all features"):
                for f in features:
                    st.write(f"• {f}")
    except Exception:
        st.write("Model info unavailable")

# ── API helper function ────────────────────────────────────────────────────────
# Encapsulates all error handling — display code only checks "success" key
def call_predict_api(payload: dict) -> dict:
    """
    Send a prediction request to the API.

    Parameters
    ----------
    payload : dict
        Feature values matching the API's expected schema.

    Returns
    -------
    dict
        {"success": True, "data": <response JSON>}
        or
        {"success": False, "error": <error message string>}
    """
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            return {"success": True, "data": response.json()}

        # API returned an error — extract the message
        try:
            detail = response.json().get("detail", f"HTTP {response.status_code}")
        except Exception:
            detail = f"HTTP {response.status_code}"
        return {"success": False, "error": detail}

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Cannot connect to API. Is it running? "
                     "Start it with: cd api && python app.py"
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "The API took too long to respond. Try again."
        }
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

# ── Input form ─────────────────────────────────────────────────────────────────
st.subheader("📝 Enter Your Data")

# Use st.form() to batch all inputs — without it, Streamlit calls the API
# on every keystroke, hammering the server with unnecessary requests
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        # min_value=0.0 prevents negative sales values (nonsensical for this model)
        lag_1 = st.number_input(
            "Lag 1 (previous week value)",
            min_value=0.0,
            value=10.0,
            help="Sales value from the previous week"
        )
    with col2:
        lag_2 = st.number_input(
            "Lag 2 (value 2 weeks ago)",
            min_value=0.0,
            value=10.0,
            help="Sales value from 2 weeks ago"
        )

    lag_52 = st.number_input(
        "Lag 52 (value 1 year ago)",
        min_value=0.0,
        value=10.0,
        help="Sales value from the same week 1 year ago"
    )

    submitted = st.form_submit_button(
        "🔮 Get Prediction",
        use_container_width=True
    )

# ── Prediction output ──────────────────────────────────────────────────────────
if submitted:
    payload = {
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_52": lag_52,
    }

    with st.spinner("🔮 Getting prediction..."):
        result = call_predict_api(payload)

    if result["success"]:
        data = result["data"]
        st.divider()
        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Predicted Weekly Sales",
                value=f"{data.get('prediction'):,.2f}"
            )
        with col2:
            if "confidence" in data:
                st.metric(
                    label="Confidence",
                    value=f"{data['confidence']:.1%}"
                )

        st.success(f"✅ Predicted weekly sales: **{data.get('prediction'):,.2f}**")

        # Collapsible raw response for debugging
        with st.expander("🔍 View full API response"):
            st.json(data)

        # ── Challenge 3: Append to session history ─────────────────────────────
        st.session_state.history.append({
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_52": lag_52,
            "prediction": data.get("prediction"),
            "timestamp": pd.Timestamp.now().strftime("%H:%M:%S")
        })

    else:
        # st.error() shows friendly message — user never sees a Python traceback
        st.error(f"❌ {result['error']}")

# ── Challenge 3: Prediction history ───────────────────────────────────────────
# session_state persists across re-runs but resets on browser refresh
if st.session_state.history:
    st.divider()
    st.subheader("📋 Prediction History (this session)")
    st.dataframe(pd.DataFrame(st.session_state.history))

# ── Challenge 2: Batch prediction via CSV upload ───────────────────────────────
st.divider()
st.subheader("📂 Batch Prediction (Optional)")

uploaded_file = st.file_uploader(
    "Upload a CSV file with columns: lag_1, lag_2, lag_52",
    type=["csv"]
)

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    st.write(f"Loaded {len(batch_df)} rows")
    st.dataframe(batch_df.head())

    # Validate required columns before sending to API
    required_cols = {"lag_1", "lag_2", "lag_52"}
    if not required_cols.issubset(batch_df.columns):
        st.error(f"❌ CSV must contain columns: {required_cols}")
    elif len(batch_df) > 100:
        st.warning("⚠️ Maximum 100 rows allowed for batch prediction.")
    else:
        if st.button("🔮 Run Batch Prediction"):
            results = []
            progress = st.progress(0)
            for i, row in batch_df.iterrows():
                res = call_predict_api({
                    "lag_1": float(row["lag_1"]),
                    "lag_2": float(row["lag_2"]),
                    "lag_52": float(row["lag_52"])
                })
                results.append(
                    res["data"]["prediction"] if res["success"] else "ERROR"
                )
                progress.progress((i + 1) / len(batch_df))

            batch_df["prediction"] = results
            st.success(f"✅ {len(batch_df)} predictions completed!")
            st.dataframe(batch_df)

st.divider()
st.caption("Built with ❤️ by Dany Deugoue (6024) | AIE1014 Assignement 03_TeamDany")
