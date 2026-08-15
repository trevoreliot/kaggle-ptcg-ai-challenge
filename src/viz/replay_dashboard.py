import streamlit as st
import pandas as pd
import json
import os
import glob
import streamlit.components.v1 as components

# Resolve the root directory regardless of where the script is run from
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

st.set_page_config(page_title="Agent Dashboard", layout="wide")

st.title("PTCG Agent Dashboard")

tab1, tab2, tab3 = st.tabs(["Latest Replay", "Evaluation Diagnostics", "Training Metrics"])

# Tab 1: Latest Replay
with tab1:
    st.header("Latest Interactive Replay")
    replay_path = os.path.join(ROOT_DIR, "assets", "results", "diagnostics", "latest_replay.html")
    if os.path.exists(replay_path):
        with open(replay_path, "r") as f:
            html_data = f.read()
            
        if "waiting_for_turn" not in st.session_state:
            st.session_state["waiting_for_turn"] = 0
            
        col1, col2 = st.columns([3, 1])
        
        with col1:
            components.html(html_data, height=1400, scrolling=True)
            
        with col2:
            st.markdown('<div style="height: 1000px;"></div>', unsafe_allow_html=True)
            
            options_path = os.path.join(ROOT_DIR, "assets", "results", "diagnostics", "human_options.json")
            if os.path.exists(options_path):
                st.session_state["waiting_for_turn"] = 0
                try:
                    import json
                    with open(options_path, "r") as f:
                        opt_data = json.load(f)
                    
                    min_count = opt_data.get('min_count', 1)
                    max_count = opt_data.get('max_count', 1)
                    turn = opt_data.get('turn', 0)
                    
                    if max_count > 1:
                        st.write(f"### Action Required: {opt_data.get('context', 'Choose an action')} (Select {min_count} to {max_count})")
                        selected_indices = []
                        for i, opt_text in enumerate(opt_data.get("options", [])):
                            if st.checkbox(opt_text, key=f"opt_cb_{turn}_{i}"):
                                selected_indices.append(i)
                                
                        if min_count <= len(selected_indices) <= max_count:
                            if st.button("Submit Actions", use_container_width=True):
                                action_path = os.path.join(ROOT_DIR, "assets", "results", "diagnostics", "human_action.txt")
                                with open(action_path, "w") as f:
                                    f.write(",".join(map(str, selected_indices)))
                                os.remove(options_path)
                                st.session_state["waiting_for_turn"] = 60
                                st.rerun()
                    else:
                        st.write(f"### Action Required: {opt_data.get('context', 'Choose an action')}")
                        for i, opt_text in enumerate(opt_data.get("options", [])):
                            if st.button(opt_text, key=f"action_btn_{turn}_{i}", use_container_width=True):
                                action_path = os.path.join(ROOT_DIR, "assets", "results", "diagnostics", "human_action.txt")
                                with open(action_path, "w") as f:
                                    f.write(str(i))
                                os.remove(options_path)
                                st.session_state["waiting_for_turn"] = 60
                                st.rerun()
                except Exception as e:
                    pass
            elif st.session_state["waiting_for_turn"] > 0:
                st.info(f"⏳ Waiting for opponent... ({st.session_state['waiting_for_turn']}s)")
                import time
                time.sleep(1)
                st.session_state["waiting_for_turn"] -= 1
                st.rerun()
    else:
        st.info("No replay found. Run `uv run python main.py --mode evaluate` to generate one.")

# Tab 2: Evaluation Diagnostics
with tab2:
    st.header("Evaluation Diagnostics")
    diag_files = glob.glob(os.path.join(ROOT_DIR, "assets", "results", "diagnostics", "diagnostics_*.json"))
    if diag_files:
        # Display just the filenames for cleaner UI
        file_options = {os.path.basename(f): f for f in diag_files}
        selected_file_name = st.selectbox("Select Diagnostic File", list(file_options.keys()))
        selected_file = file_options[selected_file_name]
        
        with open(selected_file, "r") as f:
            data = json.load(f)
            
        col1, col2, col3, col4 = st.columns(4)
        
        matches = data.get("matches", 0)
        wins = data.get("wins", 0)
        win_rate = (wins / matches * 100) if matches > 0 else 0
        
        col1.metric("Win Rate", f"{win_rate:.1f}%", f"{wins} / {matches} wins")
        col2.metric("Avg Game Length", f"{data.get('avg_length', 0):.1f} steps")
        col3.metric("First Prize Taken %", f"{data.get('first_prize_taken_pct', 0):.1f}%")
        col4.metric("Avg KO's Received", f"{data.get('avg_pokemon_kos_received', 0):.2f}")
        
        st.subheader("Agent Telemetry")
        col5, col6, col7 = st.columns(3)
        col5.metric("Avg Action Entropy", f"{data.get('avg_entropy', 0):.3f}")
        col6.metric("Avg Hand Size", f"{data.get('avg_hand_size', 0):.1f}")
        col7.metric("Total Action Paralysis", data.get("total_action_paralysis", 0))
    else:
        st.info("No diagnostic files found.")

# Tab 3: Training Metrics
with tab3:
    st.header("Training Metrics")
    csv_files = glob.glob(os.path.join(ROOT_DIR, "assets", "results", "rl_training", "*_training_metrics.csv"))
    if csv_files:
        file_options = {os.path.basename(f): f for f in csv_files}
        selected_csv_name = st.selectbox("Select Training Log", list(file_options.keys()))
        selected_csv = file_options[selected_csv_name]
        try:
            df = pd.read_csv(selected_csv)
            if not df.empty:
                st.subheader("Value Loss")
                st.line_chart(df["V_Loss"])
                
                st.subheader("Policy Loss")
                st.line_chart(df["P_Loss"])
                
                if "Reward" in df.columns:
                    st.subheader("Episode Rewards")
                    st.line_chart(df["Reward"])
            else:
                st.info("Training log is empty.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    else:
        st.info("No training metrics found. Training logs will appear here during training.")
