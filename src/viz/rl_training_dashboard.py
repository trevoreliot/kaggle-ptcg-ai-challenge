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
    
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        # File has mixed rows due to new reward keys being added mid-run
        import json
        import csv
        
        is_reward_file = csv_path.endswith('_reward_metrics.csv')
        
        if is_reward_file:
            try:
                with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assets", "reward", "reward_shaping.json"), "r") as f:
                    reward_keys = sorted(list(json.load(f).keys()))
                correct_header = ["Episode", "Opponent_Deck", "Reward"]
                for k in reward_keys:
                    correct_header.extend([f"Count_{k}", f"Total_{k}"])
            except:
                correct_header = None
        else:
            correct_header = ["Episode", "Opponent_Deck", "Reward", "Episode_Length", "Policy_Loss", "Value_Loss"]
            
        if correct_header:
            parsed_rows = []
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                next(reader, None) # Skip old header
                for row in reader:
                    if len(row) == len(correct_header):
                        parsed_rows.append(row)
            if parsed_rows:
                df = pd.DataFrame(parsed_rows, columns=correct_header)
                for c in correct_header:
                    if c != "Opponent_Deck":
                        df[c] = pd.to_numeric(df[c], errors='coerce')
                return df
                
        return pd.DataFrame()
        
    return df

results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assets", "results", "rl_training")

if not os.path.exists(results_dir):
    st.warning("Results directory not found! Run the training loop first.")
    st.stop()

csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv') and not f.endswith('_reward_metrics.csv')]
if not csv_files:
    st.warning("No CSV files found in the results directory! Run the training loop to generate metrics.")
    st.stop()

selected_csv = st.selectbox("Select Training Metrics CSV", csv_files)
csv_path = os.path.join(results_dir, selected_csv)

df = load_data(csv_path)

if df.empty:
    st.warning(f"No training data found in {selected_csv}!")
    st.stop()
    
reward_csv_path = csv_path.replace('_training_metrics.csv', '_reward_metrics.csv')
reward_df = load_data(reward_csv_path)

# ---- Filters ----
st.sidebar.header("Filters")
min_match = st.sidebar.number_input("Start from Match # (e.g. 100000)", min_value=0, value=0, step=1000)

if min_match > 0 and len(df) > min_match:
    df = df.iloc[min_match:]
    if not reward_df.empty and len(reward_df) > min_match:
        reward_df = reward_df.iloc[min_match:]
elif min_match > 0:
    st.warning(f"Dataset only has {len(df)} matches. Cannot filter from {min_match}.")

limit_last_1000 = st.sidebar.checkbox("Limit to Last 1000 Matches", value=False)
if limit_last_1000:
    df = df.tail(1000)
    if not reward_df.empty:
        reward_df = reward_df.tail(1000)

if df.empty:
    st.warning("Filtered dataset is empty!")
    st.stop()

# ---- Tabs ----
tab_overview, tab_diagnostics, tab_rewards = st.tabs(["📊 Training Overview", "🔬 Matchup Diagnostics", "💰 Reward Analytics"])

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
    
    non_zero_policy = df[df["Policy_Loss"] != 0.0]
    non_zero_value = df[df["Value_Loss"] != 0.0]
    latest_policy_loss = non_zero_policy["Policy_Loss"].iloc[-1] if not non_zero_policy.empty else 0
    latest_value_loss = non_zero_value["Value_Loss"].iloc[-1] if not non_zero_value.empty else 0
    
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
            title=f"Rolling Win Rate (%) [Window={rolling_window} matches]",
            render_mode="svg"
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
                title="Policy Loss over Time",
                render_mode="svg"
            )
            fig_policy.update_layout(xaxis_title="Total Matches Played")
            st.plotly_chart(fig_policy, width='stretch')
        else:
            st.info("No Policy Loss data yet.")
            
        st.divider()
        st.subheader("💰 Accumulated Match Reward")
        if not reward_df.empty:
            total_cols = [c for c in reward_df.columns if c.startswith("Total_")]
            if total_cols:
                reward_sum_df = reward_df[["Episode"]].copy()
                reward_sum_df["Total_Reward"] = reward_df[total_cols].sum(axis=1)
                reward_sum_df["Smooth_Reward"] = reward_sum_df["Total_Reward"].rolling(window=min(50, len(reward_sum_df)), min_periods=1).mean()
                
                sample_step_rew = max(1, len(reward_sum_df) // 500)
                sampled_rew_df = reward_sum_df.iloc[::sample_step_rew]
                
                fig_rew = px.line(
                    sampled_rew_df,
                    x=sampled_rew_df.index,
                    y=["Total_Reward", "Smooth_Reward"],
                    title="Total Accumulated Reward per Match",
                    render_mode="svg"
                )
                fig_rew.update_layout(xaxis_title="Total Matches Played", yaxis_title="Total Reward")
                st.plotly_chart(fig_rew, width='stretch')
            else:
                st.info("No detailed reward data available yet.")
        
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
                title="Value Loss over Time",
                render_mode="svg"
            )
            fig_value.update_layout(xaxis_title="Total Matches Played")
            st.plotly_chart(fig_value, width='stretch')
        else:
            st.info("No Value Loss data yet.")
            
        st.divider()
        st.subheader("⚔️ Damage Dealt Trend")
        if not reward_df.empty and "Count_r_damage_dealt_per_10" in reward_df.columns:
            damage_df = reward_df[["Episode", "Count_r_damage_dealt_per_10"]].copy()
            damage_df["Total_Damage"] = damage_df["Count_r_damage_dealt_per_10"] * 10
            damage_df["Smooth_Damage"] = damage_df["Total_Damage"].rolling(window=min(50, len(damage_df)), min_periods=1).mean()
            
            sample_step_dmg = max(1, len(damage_df) // 500)
            sampled_dmg_df = damage_df.iloc[::sample_step_dmg]
            
            fig_dmg = px.line(
                sampled_dmg_df,
                x=sampled_dmg_df.index,
                y=["Total_Damage", "Smooth_Damage"],
                title="Damage Dealt per Match",
                render_mode="svg"
            )
            fig_dmg.update_layout(xaxis_title="Total Matches Played", yaxis_title="Total Damage")
            st.plotly_chart(fig_dmg, width='stretch')
        else:
            st.info("No Damage data available yet.")

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

with tab_rewards:
    st.subheader("💰 Reward Analytics")
    if not reward_df.empty:
        # Get count and total columns
        count_cols = [c for c in reward_df.columns if c.startswith("Count_")]
        total_cols = [c for c in reward_df.columns if c.startswith("Total_")]
        
        # We'll group by the Reward column (1 = Win, -1 = Loss) to compare them
        # Let's map it for easier reading
        reward_df_mapped = reward_df.copy()
        reward_df_mapped["Outcome"] = reward_df_mapped["Reward"].map({1.0: "Win", -1.0: "Loss"})
        
        if count_cols and total_cols:
            st.markdown("#### Average Reward Metrics Per Match (Wins vs Losses)")
            st.caption("A combined view of trigger counts and total value accumulated for each reward.")
            
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            
            def absolute_gradient(subset_df, cmap_name):
                cmap = plt.get_cmap(cmap_name)
                styles = pd.DataFrame('', index=subset_df.index, columns=subset_df.columns)
                
                # Calculate global max across the entire subset (Triggers vs Totals are handled separately)
                max_val = subset_df.abs().max().max()
                
                for idx, row in subset_df.iterrows():
                    for c, val in row.items():
                        if pd.isna(val) or max_val == 0:
                            norm = 0
                        else:
                            norm = abs(val) / max_val
                            
                        bg_color = mcolors.to_hex(cmap(norm))
                        rgb = mcolors.to_rgb(bg_color)
                        luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                        text_color = "#000000" if luminance > 0.5 else "#ffffff"
                        styles.loc[idx, c] = f"background-color: {bg_color}; color: {text_color}"
                        
                return styles
            
            count_summary = reward_df_mapped.groupby("Outcome")[count_cols].mean().T
            count_summary.index = count_summary.index.str.replace("Count_", "")
            count_summary = count_summary.drop(index=["r_win", "r_loss"], errors="ignore")
            
            total_summary = reward_df_mapped.groupby("Outcome")[total_cols].mean().T
            total_summary.index = total_summary.index.str.replace("Total_", "")
            total_summary = total_summary.drop(index=["r_win", "r_loss"], errors="ignore")
            
            import pandas as pd
            combined = pd.DataFrame(index=count_summary.index)
            
            subset_count = []
            subset_total = []
            
            if "Win" in count_summary.columns:
                combined["Win_Triggers"] = count_summary["Win"]
                subset_count.append("Win_Triggers")
                combined["Win_Total"] = total_summary["Win"]
                subset_total.append("Win_Total")
                
            if "Loss" in count_summary.columns:
                combined["Loss_Triggers"] = count_summary["Loss"]
                subset_count.append("Loss_Triggers")
                combined["Loss_Total"] = total_summary["Loss"]
                subset_total.append("Loss_Total")
                
            combined = combined.fillna(0).reset_index().rename(columns={"index": "Reward"})
            if "Win_Triggers" in combined.columns:
                combined = combined.sort_values(by="Win_Triggers", ascending=False)
            
            c1, c2, c3 = st.columns([1, 6, 1])
            with c2:
                styled = combined.style.format("{:.2f}", subset=subset_count + subset_total)
                if subset_count:
                    styled = styled.apply(absolute_gradient, cmap_name="Blues", subset=subset_count, axis=None)
                if subset_total:
                    styled = styled.apply(absolute_gradient, cmap_name="Oranges", subset=subset_total, axis=None)
                    
                st.dataframe(styled, height=800, use_container_width=False, hide_index=True)
    else:
        st.info("No reward metrics CSV found for this training session.")

if st.button("🔄 Refresh Data"):
    st.rerun()
