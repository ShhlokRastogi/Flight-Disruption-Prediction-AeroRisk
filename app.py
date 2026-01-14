# ============================================================
# ✈️ AERORISK — FLIGHT DISRUPTION PREDICTOR
# PREMIUM UI | MEMORY SAFE | MULTI-MODEL
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gc
from scipy.special import softmax

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="AeroRisk ✈️",
    page_icon="✈️",
    layout="wide"
)

# ------------------------------------------------------------
# CUSTOM STYLES
# ------------------------------------------------------------

st.markdown("""
<style>
body { background-color: #0e1117; }

.section {
    background: #161b22;
    padding: 1.5rem;
    border-radius: 14px;
    margin-bottom: 1.2rem;
}

.title {
    font-size: 1.8rem;
    font-weight: 700;
}

.subtitle {
    color: #9aa4b2;
    font-size: 0.95rem;
}

.metric {
    background: linear-gradient(135deg,#2563eb,#38bdf8);
    padding: 1rem;
    border-radius: 14px;
    color: white;
    text-align: center;
}

.risk-low { color: #22c55e; font-weight: 700; }
.risk-mid { color: #facc15; font-weight: 700; }
.risk-high { color: #ef4444; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# FEATURES (ORDER LOCKED)
# ------------------------------------------------------------

FEATURES = [
    'DayOfWeek','DayofMonth','Month','Distance',
    'CRSDepMin','CRSArrMin','ScheduledElapsedTime',
    'OriginReliability','DestReliability','CarrierReliability',
    'DepTimeOfDay_enc','ArrTimeOfDay_enc'
]

CLASSES = ["Cancelled", "Delayed", "Diverted", "On Time"]

# ------------------------------------------------------------
# LOAD ENCODINGS (CACHED)
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_encodings():
    carrier = pd.read_csv("encodings/carrier_reliability_encoding.csv")
    origin  = pd.read_csv("encodings/origin_reliability_encoding.csv")
    dest    = pd.read_csv("encodings/dest_reliability_encoding.csv")

    return (
        carrier,
        origin,
        dest,
        dict(zip(carrier["UniqueCarrier"], carrier["CarrierReliability"])),
        dict(zip(origin["Origin"], origin["OriginReliability"])),
        dict(zip(dest["Dest"], dest["DestReliability"]))
    )

carrier_df, origin_df, dest_df, carrier_map, origin_map, dest_map = load_encodings()

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def time_of_day(hour):
    if 5 <= hour < 11: return 0
    if 11 <= hour < 17: return 1
    if 17 <= hour < 22: return 2
    return 3

def load_score_free(path, X):
    model = joblib.load(path)
    score = model.predict_proba(X)[0,1]
    del model
    gc.collect()
    return score

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

st.markdown("""
<div class="section">
    <div class="title">✈️ AeroRisk</div>
    <div class="subtitle">
        Flight Disruption Prediction & Risk Scoring System<br>
        Multi-Stage ML Ensembles • Memory-Safe Inference
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# MODEL SELECTION (MEMORY SAFE)
# ------------------------------------------------------------

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "4-Stage Binary + Meta"

model_choice = st.segmented_control(
    "🧠 Select Inference Engine",
    options=[
        "4-Stage Binary + Meta",
        "OvR + Softmax Ensemble"
    ],
    default=st.session_state.selected_model
)

# Free memory if model changed
if model_choice != st.session_state.selected_model:
    gc.collect()
    st.session_state.selected_model = model_choice

# ------------------------------------------------------------
# INPUTS
# ------------------------------------------------------------

st.markdown("### ✍️ Flight Details")

col1, col2, col3 = st.columns(3)

with col1:
    carrier = st.selectbox("✈️ Carrier", sorted(carrier_df["UniqueCarrier"].unique()))
    origin  = st.selectbox("🛫 Origin Airport", sorted(origin_df["Origin"].unique()))

with col2:
    dest = st.selectbox("🛬 Destination Airport", sorted(dest_df["Dest"].unique()))
    distance = st.number_input("📏 Distance (miles)", 50, 5000, 900)

with col3:
    month = st.slider("📅 Month", 1, 12, 6)
    day = st.slider("📆 Day of Month", 1, 31, 15)
    dow = st.slider("🗓 Day of Week", 1, 7, 3)

st.markdown("### ⏱ Schedule")

c4, c5 = st.columns(2)
with c4:
    dep_hour = st.slider("Departure Hour", 0, 23, 9)
with c5:
    arr_hour = st.slider("Arrival Hour", 0, 23, 12)

# ------------------------------------------------------------
# PREPROCESS
# ------------------------------------------------------------

crs_dep = dep_hour * 60
crs_arr = arr_hour * 60
elapsed = crs_arr - crs_dep
if elapsed < 0: elapsed += 1440

X = pd.DataFrame([{
    "DayOfWeek": dow,
    "DayofMonth": day,
    "Month": month,
    "Distance": distance,
    "CRSDepMin": crs_dep,
    "CRSArrMin": crs_arr,
    "ScheduledElapsedTime": elapsed,
    "OriginReliability": origin_map[origin],
    "DestReliability": dest_map[dest],
    "CarrierReliability": carrier_map[carrier],
    "DepTimeOfDay_enc": time_of_day(dep_hour),
    "ArrTimeOfDay_enc": time_of_day(arr_hour),
}])[FEATURES]

# ------------------------------------------------------------
# INFERENCE
# ------------------------------------------------------------

if st.button("🔮 Predict Disruption", use_container_width=True):

    with st.spinner("Running inference..."):

        if model_choice.startswith("4"):
            meta = {}
            meta["p_div"] = load_score_free("models/class_4stage_models/clf_diverted.pkl", X)
            meta["p_ot_del"] = load_score_free("models/class_4stage_models/clf_ot_delayed.pkl", X)
            meta["p_del_can"] = load_score_free("models/class_4stage_models/clf_del_cancelled.pkl", X)
            meta["p_ot_can"] = load_score_free("models/class_4stage_models/clf_ot_cancelled.pkl", X)

            X_meta = pd.DataFrame([{
                "P_Diverted": meta["p_div"],
                "P_Cancelled": (meta["p_del_can"] + meta["p_ot_can"]) / 2,
                "P_Delayed": (1 - meta["p_ot_del"]) * (1 - meta["p_del_can"]),
                "P_OnTime": meta["p_ot_del"] * meta["p_ot_can"]
            }])

            meta_clf = joblib.load("models/meta_classifier/meta_extratrees.pkl")
            probs = meta_clf.predict_proba(X_meta)[0]
            del meta_clf
            gc.collect()

        else:
            scores = {
                c: load_score_free(f"models/OneVRest_models/ovr_extratrees_{c}.joblib", X)
                for c in CLASSES
            }
            probs = softmax([scores[c] for c in CLASSES])

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    pred = CLASSES[np.argmax(probs)]
    risk = 1 - probs[CLASSES.index("On Time")]

    st.markdown("### 🧠 Prediction Result")

    colA, colB = st.columns([1,2])

    with colA:
        st.markdown(
            f"<div class='metric'><div class='title'>{pred}</div><div>Final Outcome</div></div>",
            unsafe_allow_html=True
        )

    with colB:
        for cls, p in zip(CLASSES, probs):
            st.progress(float(p), text=f"{cls}: {p:.3f}")

    risk_class = "risk-low" if risk < 0.3 else "risk-mid" if risk < 0.6 else "risk-high"
    st.markdown(f"### ⚠️ Risk Score: <span class='{risk_class}'>{risk:.3f}</span>", unsafe_allow_html=True)