import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(layout="wide", page_title="AI Training Dashboard", page_icon="📈")

st.title("🏆 Pokémon TCG AI Training Dashboard")

# Function to load data
@st.cache_data(ttl=5) # Cache data for 5 seconds to prevent spam when refreshing
def load_data():
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "training_metrics.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["Episode", "Opponent_Deck", "Reward", "Episode_Length", "Policy_Loss", "Value_Loss"])
    
    # Read the data
    df = pd.read_csv(csv_path)
    return df

df = load_data()

if df.empty:
    st.warning("No training data found yet! Run the training loop to generate 'training_metrics.csv'.")
    st.stop()

# ---- KPIs ----
col1, col2, col3, col4 = st.columns(4)
total_episodes = len(df)
global_winrate = df["Reward"].mean() * 100
avg_len = df["Episode_Length"].mean()
latest_policy_loss = df["Policy_Loss"].iloc[-1] if not df.empty else 0
latest_value_loss = df["Value_Loss"].iloc[-1] if not df.empty else 0

col1.metric("Total Matches", f"{total_episodes:,}")
col2.metric("Global Win Rate", f"{global_winrate:.2f}%")
col3.metric("Avg Episode Length", f"{avg_len:.1f} turns")
col4.metric("Latest Losses (P/V)", f"{latest_policy_loss:.2f} / {latest_value_loss:.2f}")

st.divider()

# ---- Layout Row 1: Performance Analysis ----
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("⚔️ Matchup Analysis")
    # Group by deck
    matchups = df.groupby("Opponent_Deck").agg(
        Matches=("Reward", "count"),
        Win_Rate=("Reward", lambda x: x.mean() * 100)
    ).reset_index()
    matchups = matchups.sort_values("Win_Rate", ascending=False)
    
    fig_bar = px.bar(
        matchups, 
        x="Opponent_Deck", 
        y="Win_Rate", 
        color="Win_Rate",
        color_continuous_scale="RdYlGn",
        title="Win Rate by Opponent Archetype (%)",
        text_auto=".1f"
    )
    fig_bar.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig_bar, use_container_width=True)

with row1_col2:
    st.subheader("📈 Win Rate Trend")
    # Rolling average of reward
    rolling_window = min(1000, max(1, len(df) // 10))
    df["Rolling_Win_Rate"] = df["Reward"].rolling(window=rolling_window, min_periods=1).mean() * 100
    
    fig_trend = px.line(
        df, 
        x=df.index, 
        y="Rolling_Win_Rate",
        title=f"Rolling Win Rate (%) [Window={rolling_window} matches]"
    )
    fig_trend.update_layout(yaxis_range=[0, 100], xaxis_title="Total Matches Played")
    st.plotly_chart(fig_trend, use_container_width=True)

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
        fig_policy = px.line(
            policy_df,
            x=policy_df.index,
            y=["Policy_Loss", "Smooth_Loss"],
            title="Policy Loss over Time"
        )
        fig_policy.update_layout(xaxis_title="Total Matches Played")
        st.plotly_chart(fig_policy, use_container_width=True)
    else:
        st.info("No Policy Loss data yet.")
    
with row2_col2:
    value_df = df[df["Value_Loss"] != 0.0].copy()
    if not value_df.empty:
        value_df["Smooth_Loss"] = value_df["Value_Loss"].rolling(window=min(50, len(value_df)), min_periods=1).mean()
        fig_value = px.line(
            value_df,
            x=value_df.index,
            y=["Value_Loss", "Smooth_Loss"],
            title="Value Loss over Time"
        )
        fig_value.update_layout(xaxis_title="Total Matches Played")
        st.plotly_chart(fig_value, use_container_width=True)
    else:
        st.info("No Value Loss data yet.")

if st.button("🔄 Refresh Data"):
    st.rerun()
