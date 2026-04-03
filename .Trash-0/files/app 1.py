import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import torch
import random
from sentence_transformers import SentenceTransformer, util
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from transformers import pipeline

# --- PAGE CONFIG ---
st.set_page_config(page_title="Authenti-Annotator: Blackwell", layout="wide", page_icon="🛡️")

# --- FUNCTIONS ---
@st.cache_resource
def load_forensic_engines():
    # SBERT for Similarity
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
    # GPT-2 for Audit Reasoning
    reasoner = pipeline("text-generation", model="gpt2", device=0 if torch.cuda.is_available() else -1)
    return embedder, reasoner

def get_lexical_diversity(text):
    words = str(text).lower().split()
    return len(set(words)) / len(words) if len(words) > 0 else 0

# --- INITIALIZATION ---
st.title("🛡️ Authenti-Annotator: Blackwell AI Auditor")
st.markdown("### *High-Fidelity RLHF Integrity Suite powered by RTX 5060 Ti*")

with st.spinner("Initializing Blackwell Engines..."):
    embedder, reasoner = load_forensic_engines()

# --- SIDEBAR NAVIGATION ---
st.sidebar.header("🕹️ Control Center")
mode = st.sidebar.radio("Mode Selection", ["Static Audit Report", "Interactive Sandbox"])
uploaded_file = st.sidebar.file_uploader("Upload RLHF Dataset (CSV)", type=["csv"])

# --- DATA LOADING ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"✅ Loaded {len(df)} rows.")
elif mode == "Static Audit Report":
    try:
        df = pd.read_csv("forensic_data_static.csv")
    except Exception as e:
        st.error(f"⚠️ 'forensic_data_static.csv' not found. Error: {e}")
        st.stop()

# --- THE MAIN RESULTS ENGINE ---
if 'df' in locals():
    # 1. PRE-PROCESSING
    df['chosen_diversity'] = df['chosen'].apply(get_lexical_diversity)
    df['rejected_diversity'] = df['rejected'].apply(get_lexical_diversity)
    df['quality_gap'] = df['chosen_diversity'] - df['rejected_diversity']
    
    # Standardize column names
    x_col = 'similarity_score' if 'similarity_score' in df.columns else 'similarity'
    y_col = 'time_seconds' if 'time_seconds' in df.columns else 'len_diff'
    risk_id = 'expert_id' if 'expert_id' in df.columns else 'annotator_id'
    
    # Standardize anomaly flags
    if 'is_anomaly' in df.columns:
        df['is_anomaly_bool'] = df['is_anomaly'].apply(lambda x: 1 if x in [-1, 1] else 0)
    else:
        df['is_anomaly_bool'] = 0
        
    anomalies = df[df['is_anomaly_bool'] == 1]

    # 2. KPI OVERVIEW
    st.header("🛡️ Dataset Health Overview")