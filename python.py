"""
Project Business Analytics Tender Success Dashboard | MoD Tender Win-Probability Calculator
BADM Defense Project: Netra-Helmet and Vajra-K9

Run with:
    pip install streamlit scikit-learn pandas numpy
    streamlit run python.py
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
    page_title="Business Analytics Tender Success Dashboard",
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
    ("QRSAM Radar Sub-system", "DRDO / BEL", 110,
