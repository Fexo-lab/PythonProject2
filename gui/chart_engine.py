import streamlit as st
import plotly.graph_objects as go
import pandas as pd


def create_trading_chart(df):
    """Creates a clean candlestick chart with EMAs."""
    if df is None or df.empty: return

    fig = go.Figure()
    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Price"
    ))

    # EMAs draw (only if available)
    for col in df.columns:
        if "EMA" in col and "dist" not in col:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(width=1.5)))

    fig.update_layout(
        template="plotly_dark", height=600,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, width='stretch') # Fix for 2026


def display_live_log():
    """Shows the past signals as a table."""
    st.markdown("### 📜 Signal Log")
    if st.session_state.trade_log:
        log_df = pd.DataFrame(st.session_state.trade_log)
        st.table(log_df.head(10))  # Shows the last 10 signals
    else:
        st.info("No signals generated yet.")


def display_stats(df):
    if df is None or 'volatility' not in df.columns:
        st.write("Waiting for indicators...")
        return
    st.markdown("---")
    # Safe access with fallback
    vol = df['volatility'].iloc[-1] if not pd.isna(df['volatility'].iloc[-1]) else 0
    rsi = df['rsi'].iloc[-1] if not pd.isna(df['rsi'].iloc[-1]) else 50
    st.write(f"Volatility: {round(vol, 5)}")
    st.write(f"RSI: {round(rsi, 2)}")


def display_terminal(placeholder):
    # Simple symbol switcher
    sym = st.text_input("Change Symbol (e.g. SI=F):", placeholder)
    return sym if sym else None