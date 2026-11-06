"""
Project Vajra-Net | MoD Tender Win-Probability Calculator
BADM Defense Project: Netra-Helmet and Vajra-K9

Run with:
    pip install streamlit scikit-learn pandas numpy
    streamlit run vajra_net_dashboard.py
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Government Tender Success Rate",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; max-width: 1100px;}
        .big-number {font-size: 4rem; font-weight: 700; line-height: 1;}
        .status-pill {
            display: inline-block; padding: 0.3rem 0.9rem; border-radius: 999px;
            font-weight: 600; font-size: 0.95rem; color: white;
        }
        .muted {color: #7a7f87; font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Reference dataset
# NOTE: Figures are ILLUSTRATIVE benchmark values built for academic
# demonstration. Real MoD tender price/score data is not fully public.
# Replace with sourced figures (RFP documents, DAC/MoD press releases,
# CAG reports, company filings) before final submission.
# ------------------------------------------------------------------
FEATURES = ["Unit Price (₹ Lakhs)", "MoD Budget (₹ Crores)", "Trial Score (%)"]
TARGET = "Tender Outcome"

DATA = [
    # System, Developer, Price (L), Budget (Cr), Trial (%), Won
    ("Loitering Munition Mk-I", "TASL", 38, 1850, 84, 1),
    ("Loitering Munition Mk-II", "TASL", 52, 1600, 76, 1),
    ("Loitering Munition Export Variant", "TASL", 61, 1400, 68, 0),
    ("Tactical Radio-Link Node", "TASL", 30, 1000, 70, 0),
    ("Exoskeleton Gen-1", "DRDO", 22, 900, 64, 0),
    ("Exoskeleton Gen-2", "DRDO", 18, 1100, 78, 1),
    ("Exoskeleton Lite", "DRDO", 26, 800, 58, 0),
    ("Hand-held Drone Jammer", "DRDO", 34, 1400, 83, 1),
    ("Soldier Sensor Suite", "DRDO", 28, 1250, 74, 1),
    ("Vest-Integrated Sensor", "DRDO", 36, 700, 66, 0),
    ("QRSAM Seeker Module", "DRDO / BEL", 74, 2600, 88, 1),
    ("QRSAM Launcher Sub-system", "DRDO / BEL", 92, 2400, 81, 1),
    ("QRSAM Radar Sub-system", "DRDO / BEL", 110, 2200, 69, 0),
    ("Anti-Drone System AD-1", "Zen Technologies", 46, 1700, 86, 1),
    ("Anti-Drone System AD-2", "Zen Technologies", 58, 1300, 73, 0),
    ("Compact Counter-UAS", "Zen Technologies", 40, 1500, 79, 1),
    ("Compact Counter-UAS v2", "Zen Technologies", 49, 1200, 62, 0),
    ("Counter-UAS Jammer Gun", "Zen Technologies", 32, 1600, 90, 1),
    ("Recon Drone Type-A", "Indigenous OEM", 44, 1900, 77, 1),
    ("Recon Drone Type-B", "Indigenous OEM", 66, 1100, 60, 0),
    ("Ground Robot UGV-A", "Indigenous OEM", 70, 1350, 72, 0),
    ("Ground Robot UGV-B", "Indigenous OEM", 55, 1950, 85, 1),
    ("Tactical Helmet HUD", "Indigenous OEM", 24, 1300, 75, 1),
    ("Competing Helmet Variant", "Indigenous OEM", 29, 800, 55, 0),
]

df = pd.DataFrame(
    DATA,
    columns=["System", "Developer"] + FEATURES + [TARGET],
)


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
@st.cache_resource
def train_model(data: pd.DataFrame):
    X = data[FEATURES].values
    y = data[TARGET].values
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, random_state=42, max_iter=1000),
    )
    model.fit(X, y)
    return model


model = train_model(df)
train_acc = model.score(df[FEATURES].values, df[TARGET].values)


def win_probability(price, budget, trial):
    x = np.array([[price, budget, trial]])
    return float(model.predict_proba(x)[0, 1])


def status_for(p):
    if p >= 0.70:
        return "High Chance", "#1f9d55"
    if p >= 0.40:
        return "Competitive", "#d98e04"
    return "Low Chance", "#c0392b"


# ------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------
st.sidebar.title("🎛️ Bid Parameters")
st.sidebar.caption("Adjust the inputs to simulate a tender submission.")

price = st.sidebar.slider(
    "Target Bidding Price (₹ Lakhs)", min_value=10, max_value=150, value=45, step=1
)
budget = st.sidebar.slider(
    "MoD Defense Budget Allocation (₹ Crores)",
    min_value=500, max_value=3000, value=1500, step=50,
)
trial = st.sidebar.slider(
    "Field Trial Success Score (%)", min_value=40, max_value=100, value=75, step=1
)

st.sidebar.markdown("---")
st.sidebar.caption("Project Vajra-Net · Netra-Helmet & Vajra-K9")

# ------------------------------------------------------------------
# Main panel
# ------------------------------------------------------------------
st.title("🛡️ Project Vajra-Net")
st.markdown("##### Ministry of Defence Tender Win-Probability Calculator")
st.markdown(
    '<span class="muted">Logistic regression model trained on indigenous Indian '
    "defense systems benchmark data.</span>",
    unsafe_allow_html=True,
)
st.write("")

prob = win_probability(price, budget, trial)
label, color = status_for(prob)

col1, col2 = st.columns([1, 1.4], gap="large")

with col1:
    st.markdown("**Live Win Probability**")
    st.markdown(
        f'<div class="big-number" style="color:{color}">{prob * 100:.1f}%</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(
        f'<span class="status-pill" style="background:{color}">{label}</span>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.progress(min(max(prob, 0.0), 1.0))
    st.caption(
        f"Inputs: ₹{price} L bid · ₹{budget:,} Cr budget · {trial}% trial score"
    )

with col2:
    st.markdown("**Sensitivity: win probability vs. bid price**")
    price_range = np.arange(10, 151, 2)
    curve = pd.DataFrame(
        {
            "Bid Price (₹ Lakhs)": price_range,
            "Win Probability (%)": [
                win_probability(p, budget, trial) * 100 for p in price_range
            ],
        }
    ).set_index("Bid Price (₹ Lakhs)")
    st.line_chart(curve, height=240)
    st.caption("Budget and trial score held at the current slider values.")

st.write("")

# ------------------------------------------------------------------
# Model insight
# ------------------------------------------------------------------
with st.expander("Model details"):
    scaler = model.named_steps["standardscaler"]
    clf = model.named_steps["logisticregression"]
    # Convert standardized coefficients to per-unit coefficients
    raw_coef = clf.coef_[0] / scaler.scale_
    coef_df = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Standardized Coefficient": np.round(clf.coef_[0], 3),
            "Per-Unit Coefficient (log-odds)": np.round(raw_coef, 4),
            "Effect": [
                "Higher value lowers win odds" if c < 0 else "Higher value raises win odds"
                for c in clf.coef_[0]
            ],
        }
    )
    st.dataframe(coef_df, hide_index=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Training records", len(df))
    m2.metric("Training accuracy", f"{train_acc * 100:.1f}%")
    m3.metric("Win rate in data", f"{df[TARGET].mean() * 100:.0f}%")
    st.caption(
        "Accuracy is measured on the training set itself, so treat it as a fit "
        "indicator, not a forecast of real-world performance."
    )

# ------------------------------------------------------------------
# Dataset display
# ------------------------------------------------------------------
st.markdown("### 📊 Reference Benchmark Dataset")
st.caption(
    "Training records used by the model (Won = 1, Lost = 0). Values are "
    "illustrative benchmarks for academic demonstration and should be "
    "replaced with sourced procurement data for final evaluation."
)

display_df = df.copy()
display_df["Status"] = display_df[TARGET].map({1: "Won", 0: "Lost"})
st.dataframe(
    display_df,
    hide_index=True,
    column_config={
        "Unit Price (₹ Lakhs)": st.column_config.NumberColumn(format="₹ %d L"),
        "MoD Budget (₹ Crores)": st.column_config.NumberColumn(format="₹ %d Cr"),
        "Trial Score (%)": st.column_config.ProgressColumn(
            min_value=0, max_value=100, format="%d%%"
        ),
    },
)

st.download_button(
    "Download dataset (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="vajra_net_benchmark_dataset.csv",
    mime="text/csv",
)
