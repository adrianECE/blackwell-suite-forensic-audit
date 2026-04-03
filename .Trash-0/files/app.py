import numpy as np # IMPORT THIS FIRST
import torch
import streamlit as st
# ... rest of your imports

# --- ADD THIS LOGIC IN YOUR REASONER BLOCK ---
with tab2:
    st.subheader("Blackwell Reasoning Engine")
    anomalies = df[df['is_anomaly'] == 1]
    if not anomalies.empty:
        # Pili tayo ng specific anomaly sample para hindi laging yung una lang
        sample_to_reason = st.selectbox("Select Anomaly to Critique:", anomalies.index)
        target_row = anomalies.loc[sample_to_reason]
        
        prompt = f"Critique why Response B is rejected:\nA: {target_row['chosen'][:200]}\nB: {target_row['rejected'][:200]}\n\nVerdict:"
        
        if st.button("Generate Audit Reason"):
            try:
                # Force numpy to be recognized in the local scope before calling reasoner
                import numpy as np 
                
                with st.spinner("Blackwell Engine is analyzing text..."):
                    gen = reasoner(prompt, max_new_tokens=50, do_sample=True, pad_token_id=50256)
                    st.info(f"**Forensic Verdict for Sample {sample_to_reason}:**\n\n{gen[0]['generated_text'].split('Verdict:')[-1]}")
            except RuntimeError as e:
                st.error(f"Engine Error: {e}")
                st.warning("Hack: Try restarting the app or check if 'pip install numpy' is needed in the terminal.")