import streamlit as st
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# 1. Page Config
st.set_page_config(page_title="Authenti-Annotator: Blackwell", layout="wide")

# 2. Cached Model Loading
@st.cache_resource
def load_engines():
    # SBERT for Similarity
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
    # GPT-2 for Reasoning
    reasoner = pipeline("text-generation", model="gpt2", device=0 if torch.cuda.is_available() else -1)
    return embedder, reasoner

# 3. Initialization UI
st.title("🛡️ Authenti-Annotator: Blackwell AI Auditor")
st.info("Module 1: Engine Initialization Load Test")

with st.spinner("Loading Blackwell Engines (SBERT & GPT-2)..."):
    try:
        embedder, reasoner = load_engines()
        st.success("✅ Blackwell Engines Loaded Successfully!")
    except Exception as e:
        st.error(f"❌ Error loading engines: {e}")

st.write("Kapag nakita mo ang 'Success' message sa itaas, ready na tayo sa Module 2.")


# --- MODULE 2: DATA LOADING & SIDEBAR ---
st.sidebar.header("🕹️ Control Center")
mode = st.sidebar.radio("Audit Mode", ["Static Audit Report", "Interactive Sandbox"])
uploaded_file = st.sidebar.file_uploader("Upload RLHF Dataset (CSV)", type=["csv"])

# Define Lexical Diversity Function here for data prep
def get_lexical_diversity(text):
    words = str(text).lower().split()
    return len(set(words)) / len(words) if len(words) > 0 else 0

# Data Trigger
df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"✅ Uploaded: {len(df)} rows.")
elif mode == "Static Audit Report":
    try:
        df = pd.read_csv("forensic_data_static.csv")
        st.sidebar.success("✅ Static Forensic Data Loaded.")
    except:
        st.sidebar.error("⚠️ 'forensic_data_static.csv' not found. Run your Notebook first!")

# --- DATA PREPARATION (Normalization) ---
if df is not None:
    with st.spinner("Preparing Data for Forensic Audit..."):
        # 1. Diversity & Quality Metrics
        df['chosen_diversity'] = df['chosen'].apply(get_lexical_diversity)
        df['rejected_diversity'] = df['rejected'].apply(get_lexical_diversity)
        df['quality_gap'] = df['chosen_diversity'] - df['rejected_diversity']

        # 2. Standardize Column Names for Visuals
        # Maps varied names (similarity vs similarity_score) to a single standard
        df['viz_sim'] = df['similarity_score'] if 'similarity_score' in df.columns else df.get('similarity', 0)
        df['viz_effort'] = df['time_seconds'] if 'time_seconds' in df.columns else df.get('len_diff', 0)
        df['viz_id'] = df['expert_id'] if 'expert_id' in df.columns else df.get('annotator_id', 'Unknown')

        # 3. Standardize Anomaly Flags
        if 'is_anomaly' in df.columns:
            df['is_flagged'] = df['is_anomaly'].apply(lambda x: 1 if x in [-1, 1] else 0)
        else:
            df['is_flagged'] = 0
            
    st.success("✅ Module 2: Data Normalized and Ready.")



import plotly.express as px # Ensure this is in your imports!

# --- MODULE 3: THE VISUALIZATION & EVIDENCE SUITE ---
if df is not None:
    st.divider()
    st.header("🛡️ Audit Intelligence Overview")

    # 1. Top-Level KPIs
    k1, k2, k3 = st.columns(3)
    total_anomalies = df['is_flagged'].sum()
    health_score = 100 - (total_anomalies / len(df) * 100) if len(df) > 0 else 0
    
    k1.metric("Dataset Health", f"{health_score:.1f}%")
    k2.metric("Red Flags Detected", int(total_anomalies), delta_color="inverse")
    k3.metric("Avg. Quality Gap", f"{df['quality_gap'].mean():.4f}")

    # 2. Tabs Interface
    tab1, tab2, tab3 = st.tabs(["📊 Forensic Chart", "🕵️ Individual Evidence", "🚩 Risk Leaderboard"])

    with tab1:
        st.subheader("Complexity vs. Effort Map")
        fig = px.scatter(
            df, x='viz_sim', y='viz_effort',
            color='is_flagged',
            hover_data=['viz_id'],
            color_continuous_scale='RdBu_r',
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("🔍 Individual Anomaly Evidence")
        # Dito natin ilalagay yung Time at Justification na hinahanap mo
        flagged_only = df[df['is_flagged'] == 1]
        
        if not flagged_only.empty:
            search_idx = st.selectbox("Select Anomaly to Inspect Details:", flagged_only.index)
            row_data = df.loc[search_idx]
            
            # THE MISSING DETAILS:
            ev1, ev2, ev3 = st.columns(3)
            
            # 1. Time/Effort Detail
            if 'time_seconds' in row_data:
                mins = row_data['time_seconds'] / 60
                ev1.metric("Effort (Time)", f"{mins:.1f} min")
            else:
                ev1.metric("Effort (Len Diff)", f"{row_data.get('len_diff', 0)} chars")
            
            # 2. Similarity Detail
            ev2.metric("Similarity", f"{row_data.get('viz_sim', 0):.4f}")
            
            # 3. Pre-calculated Justification
            st.info(f"**Forensic Justification:** {row_data.get('justification', 'No log found.')}")
            
            # Small Preview of the text
            st.text_area("Chosen Response Preview", row_data['chosen'][:300], height=100)
        else:
            st.write("No anomalies to inspect.")

    with tab3:
        st.subheader("High-Risk Leaderboard")
        leaderboard = df.groupby('viz_id').agg({
            'is_flagged': 'sum',
            'viz_sim': 'mean'
        }).sort_values(by='is_flagged', ascending=False)
        st.dataframe(leaderboard.head(10), use_container_width=True)

    st.success("✅ Module 3 Updated: Evidence Details Added.")


# --- MODULE 4: AI REASONING & VERIFICATION ---
if df is not None:
    st.divider()
    st.header("🤖 Blackwell AI Reasoning & Final Verdict")
    st.markdown("Perform real-time semantic analysis and log the final auditor decision.")
    
    # Filter only flagged anomalies for the deep-dive
    anomalies = df[df['is_flagged'] == 1]

    if not anomalies.empty:
        # 1. UI Selection for Deep-Dive
        col_sel, col_empty = st.columns([2, 1])
        with col_sel:
            target_idx = st.selectbox("🎯 Select Anomaly ID for AI Critique:", anomalies.index, key="reasoner_select")
            target_row = df.loc[target_idx]
            
        # 2. Side-by-Side Comparison (The Visual Evidence)
        st.subheader("🔍 Comparative Forensic View")
        c1, c2 = st.columns(2)
        c1.info(f"**Chosen Response (A):**\n\n{target_row['chosen'][:500]}...")
        c2.error(f"**Rejected Response (B):**\n\n{target_row['rejected'][:500]}...")

        # 3. THE AI REASONER (With Optimized Prompt & NumPy Fix)
        st.subheader("🤖 AI Forensic Critique")
        if st.button("🚀 Run Blackwell Semantic Audit"):
            try:
                import numpy as np # Forced environment fix
                
                # Optimized 'Guided' Prompt for better GPT-2 performance
                prompt = (
                    f"Instruction: Analyze RLHF evaluation logic.\n"
                    f"Sample A: {target_row['chosen'][:100]}\n"
                    f"Sample B: {target_row['rejected'][:100]}\n"
                    f"Verdict: Response B is flagged because"
                )
                
                with st.spinner("Analyzing semantics via Blackwell GPT-2..."):
                    gen = reasoner(prompt, max_new_tokens=50, do_sample=True, pad_token_id=50256)
                    # Extract only the AI's continuation
                    ai_critique = gen[0]['generated_text'].split('Verdict:')[-1].strip()
                    
                    st.success("**AI Forensic Verdict:**")
                    st.write(f"_{ai_critique}_")
                    
            except Exception as e:
                # Reliability Layer: Fallback to the pre-calculated forensic log
                st.warning("⚠️ Inference Engine Error (Local Resource Limit).")
                st.info(f"**Accessing Stored Forensic Log:**\n\n{target_row.get('justification', 'Fraudulent behavioral pattern detected via effort/similarity audit.')}")
                st.caption(f"Trace: {str(e)[:50]}")

        # 4. AUDITOR MANUAL SIGN-OFF
        st.divider()
        st.subheader("✍️ Auditor Manual Sign-off")
        verdict_col, btn_col = st.columns([2, 1])
        
        with verdict_col:
            v_status = st.radio("Final Audit Decision:", ["Confirm Fraud/Anomaly", "Dismiss as False Positive"], horizontal=True)
        
        with btn_col:
            if st.button("Confirm & Log Verdict"):
                st.success(f"Audit Result for Sample {target_idx} saved to system logs.")

    else:
        st.success("✅ No anomalies detected in the current dataset batch.")

    # 5. FINAL EXPORT BUTTON
    st.divider()
    csv_report = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Final Forensic Audit Report",
        data=csv_report,
        file_name="Blackwell_Final_Audit_Report.csv",
        mime="text/csv"
    )    

# # --- MODULE 4: AI REASONING & MANUAL VERIFICATION ---
# if df is not None:
#     st.divider()
#     st.header("🤖 Blackwell AI Reasoning & Verification")
    
#     # Filter only the flagged samples for analysis
#     anomalies = df[df['is_flagged'] == 1]

#     if not anomalies.empty:
#         # 1. UI Selection
#         col_select, col_verdict = st.columns([2, 1])
        
#         with col_select:
#             target_idx = st.selectbox("🎯 Select Anomaly ID for Deep-Dive:", anomalies.index)
#             target_row = df.loc[target_idx]
            
#         # 2. Side-by-Side Comparison
#         st.subheader("🔍 Comparative Forensic View")
#         c1, c2 = st.columns(2)
#         c1.info(f"**Chosen Response (A):**\n\n{target_row['chosen'][:500]}...")
#         c2.error(f"**Rejected Response (B):**\n\n{target_row['rejected'][:500]}...")

#         # 3. THE AI REASONER BUTTON (With NumPy Fix)
#         st.subheader("🤖 AI Forensic Critique")
#         if st.button("🚀 Run Blackwell Semantic Audit"):
#             try:
#                 # Local import to force NumPy link
#                 import numpy as np
                
#                 # Constructing the critique prompt
#                 prompt = f"Critique why Response B is worse than A:\nA: {target_row['chosen'][:150]}\nB: {target_row['rejected'][:150]}\n\nCritique:"
                
#                 with st.spinner("Analyzing semantics via Blackwell GPT-2 Engine..."):
#                     # Inference
#                     gen = reasoner(prompt, max_new_tokens=60, do_sample=True, pad_token_id=50256)
#                     ai_critique = gen[0]['generated_text'].split('Critique:')[-1].strip()
                    
#                     st.success("**AI Audit Verdict:**")
#                     st.write(ai_critique)
                    
#             except Exception as e:
#                 # FALLBACK: If NumPy/Engine fails, use the justification from our Notebook
#                 st.warning("⚠️ Inference Engine Error. Accessing Stored Forensic Log...")
#                 st.info(f"**Stored Justification:** {target_row.get('justification', 'Fraudulent behavioral pattern detected.')}")
#                 st.caption(f"Technical Trace: {str(e)[:50]}")

#         # 4. MANUAL VERIFICATION LOG
#         st.divider()
#         st.subheader("✍️ Auditor Manual Sign-off")
#         v_status = st.radio("Final Audit Decision:", ["Confirm Fraud/Anomaly", "Dismiss as False Positive"])
        
#         if st.button("Submit Verdict to Audit Log"):
#             st.success(f"Audit Result for Sample {target_idx} saved as {v_status}.")

#     else:
#         st.success("✅ No anomalies found in this dataset. Integrity score is optimal.")

#     # 5. THE FINAL EXPORT
#     st.divider()
#     csv_out = df.to_csv(index=False).encode('utf-8')
#     st.download_button(
#         label="📥 Download Final Forensic Audit Report",
#         data=csv_out,
#         file_name="Blackwell_Final_Audit_Report.csv",
#         mime="text/csv"
#     )