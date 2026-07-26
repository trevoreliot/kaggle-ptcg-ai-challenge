import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(layout="wide", page_title="AI Training Dashboard", page_icon="📈")

st.title("🏆 Pokémon TCG AI Training Dashboard")

# Function to load data
@st.cache_data(ttl=5) # Cache data for 5 seconds to prevent spam when refreshing
def load_data(csv_path):
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["Episode", "Opponent_Deck", "Reward", "Episode_Length", "Policy_Loss", "Value_Loss"])
    
    # Read the data
    df = pd.read_csv(csv_path)
    return df

results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assets", "results", "rl_training")

if not os.path.exists(results_dir):
    st.warning("Results directory not found! Run the training loop first.")
    st.stop()

csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
if not csv_files:
    st.warning("No CSV files found in the results directory! Run the training loop to generate metrics.")
    st.stop()

selected_csv = st.selectbox("Select Training Metrics CSV", csv_files)
csv_path = os.path.join(results_dir, selected_csv)

df = load_data(csv_path)

if df.empty:
    st.warning(f"No training data found in {selected_csv}!")
    st.stop()

# ---- Filters ----
st.sidebar.header("Filters")
min_match = st.sidebar.number_input("Start from Match # (e.g. 100000)", min_value=0, value=0, step=1000)

if min_match > 0 and len(df) > min_match:
    df = df.iloc[min_match:]
elif min_match > 0:
    st.warning(f"Dataset only has {len(df)} matches. Cannot filter from {min_match}.")

if df.empty:
    st.warning("Filtered dataset is empty!")
    st.stop()

# ---- Tabs ----
tab_overview, tab_diagnostics = st.tabs(["📊 Training Overview", "🔬 Matchup Diagnostics"])

with tab_overview:
    # ---- KPIs ----
    col1, col2, col3, col4 = st.columns(4)
    total_episodes = len(df)
    global_winrate = (df["Reward"] == 1).mean() * 100
    avg_len = df["Episode_Length"].mean()
    
    # Latest 1000 stats
    recent_df = df.tail(1000)
    recent_winrate = (recent_df["Reward"] == 1).mean() * 100 if not recent_df.empty else 0.0
    recent_avg_len = recent_df["Episode_Length"].mean() if not recent_df.empty else 0.0
    
    latest_policy_loss = df["Policy_Loss"].iloc[-1] if not df.empty else 0
    latest_value_loss = df["Value_Loss"].iloc[-1] if not df.empty else 0
    
    col1.metric("Total Matches (Filtered)", f"{total_episodes:,}")
    col2.metric("Global Win Rate", f"{global_winrate:.1f}%")
    col3.metric("Avg Episode Length", f"{avg_len:.1f}")
    col4.metric("Latest Losses (P/V)", f"{latest_policy_loss:.2f} / {latest_value_loss:.2f}")
    
    # Row 2 for recent metrics, aligned under the global ones
    r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
    r2_c2.metric("Recent Win Rate (1k)", f"{recent_winrate:.1f}%")
    r2_c3.metric("Recent Avg Length (1k)", f"{recent_avg_len:.1f}")
    
    st.divider()
    
    # ---- Layout Row 1: Performance Analysis ----
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.subheader("⚔️ Matchup Analysis")
        # Group by deck
        matchups = df.groupby("Opponent_Deck").agg(
            Matches=("Reward", "count"),
            Win_Rate=("Reward", lambda x: (x == 1).mean() * 100)
        ).reset_index()
        matchups = matchups.sort_values("Win_Rate", ascending=False)
        
        matchups["Label"] = matchups.apply(lambda row: f"{row['Win_Rate']:.1f}% (n={row['Matches']/1000:.1f}k)" if row['Matches'] >= 1000 else f"{row['Win_Rate']:.1f}% (n={row['Matches']})", axis=1)
        
        fig_bar = px.bar(
            matchups, 
            x="Opponent_Deck", 
            y="Win_Rate", 
            color="Win_Rate",
            color_continuous_scale="RdYlGn",
            title="Win Rate by Opponent Archetype (%)",
            text="Label"
        )
        fig_bar.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig_bar, width='stretch')
    
    with row1_col2:
        st.subheader("📈 Win Rate Trend")
        # Rolling average of reward
        rolling_window = min(1000, max(1, len(df) // 10))
        df["Rolling_Win_Rate"] = (df["Reward"] == 1).rolling(window=rolling_window, min_periods=1).mean() * 100
        
        # Downsample to a maximum of 500 points to prevent Plotly from hanging the browser
        sample_step = max(1, len(df) // 500)
        sampled_df = df.iloc[::sample_step]
        
        fig_trend = px.line(
            sampled_df, 
            x=sampled_df.index, 
            y="Rolling_Win_Rate",
            title=f"Rolling Win Rate (%) [Window={rolling_window} matches]"
        )
        fig_trend.update_layout(yaxis_range=[0, 100], xaxis_title="Total Matches Played")
        st.plotly_chart(fig_trend, width='stretch')
    
    st.divider()
    
    # ---- Layout Row 2: Neural Network Loss Curves ----
    st.subheader("🧠 Neural Network Convergence")
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        # Filter out the 0.0s since we only optimize every N steps
        policy_df = df[df["Policy_Loss"] != 0.0].copy()
        if not policy_df.empty:
            # Smooth the loss for better visibility
            policy_df["Smooth_Loss"] = policy_df["Policy_Loss"].rolling(window=min(50, len(policy_df)), min_periods=1).mean()
            
            # Downsample
            sample_step_policy = max(1, len(policy_df) // 500)
            sampled_policy_df = policy_df.iloc[::sample_step_policy]
            
            fig_policy = px.line(
                sampled_policy_df,
                x=sampled_policy_df.index,
                y=["Policy_Loss", "Smooth_Loss"],
                title="Policy Loss over Time"
            )
            fig_policy.update_layout(xaxis_title="Total Matches Played")
            st.plotly_chart(fig_policy, width='stretch')
        else:
            st.info("No Policy Loss data yet.")
        
    with row2_col2:
        value_df = df[df["Value_Loss"] != 0.0].copy()
        if not value_df.empty:
            value_df["Smooth_Loss"] = value_df["Value_Loss"].rolling(window=min(50, len(value_df)), min_periods=1).mean()
            
            # Downsample
            sample_step_value = max(1, len(value_df) // 500)
            sampled_value_df = value_df.iloc[::sample_step_value]
            
            fig_value = px.line(
                sampled_value_df,
                x=sampled_value_df.index,
                y=["Value_Loss", "Smooth_Loss"],
                title="Value Loss over Time"
            )
            fig_value.update_layout(xaxis_title="Total Matches Played")
            st.plotly_chart(fig_value, width='stretch')
        else:
            st.info("No Value Loss data yet.")

with tab_diagnostics:
    st.subheader("🔬 Matchup Telemetry & Diagnostics")
    st.markdown("Use `--mode evaluate` in your simulator to generate detailed diagnostic metrics.")
    diag_dir = os.path.join(results_dir, "..", "diagnostics")
    if os.path.exists(diag_dir):
        diag_files = [f for f in os.listdir(diag_dir) if f.endswith(".json")]
        if diag_files:
            import json
            for df_name in diag_files:
                with open(os.path.join(diag_dir, df_name), "r") as f:
                    data = json.load(f)
                
                agent_display_name = df_name.replace('diagnostics_', '').replace('.json', '')
                with st.expander(f"Diagnostics vs {agent_display_name}", expanded=True):
                    dc1, dc2, dc3, dc4 = st.columns(4)
                    dc1.metric("Matches Analyzed", data.get("matches", 0))
                    dc2.metric("Wins", data.get("wins", 0))
                    dc3.metric("Avg Episode Length", f"{data.get('avg_length', 0):.1f}")
                    dc4.metric("Avg Hand Size", f"{data.get('avg_hand_size', 0):.1f}")
                    
                    st.divider()
                    dc5, dc6, dc7, dc8 = st.columns(4)
                    dc5.metric("Avg Policy Entropy", f"{data.get('avg_entropy', 0):.2f}")
                    dc6.metric("Total Action Paralysis", data.get("total_action_paralysis", 0))
                    dc7.metric("First Prize Taken %", f"{data.get('first_prize_taken_pct', 0):.1f}%")
                    dc8.metric("Avg Bench/Active KOs Received", f"{data.get('avg_pokemon_kos_received', 0):.1f}")
        else:
            st.info("No diagnostic JSON files found. Run the simulator in evaluate mode.")
    else:
        st.info("Diagnostics directory not found. Run the simulator in evaluate mode.")

if st.button("🔄 Refresh Data"):
    st.rerun()
