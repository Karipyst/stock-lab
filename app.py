import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go


st.set_page_config(
    page_title="Stock Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("株価分析ダッシュボード")

st.caption("株価・出来高・移動平均・RSI・MACDを確認するための初期版です。")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("設定")

symbol = st.sidebar.text_input("銘柄コード", "7203.T")

period = st.sidebar.selectbox(
    "取得期間",
    ["6mo", "1y", "2y", "5y"],
    index=1
)

ma_short = st.sidebar.number_input("短期移動平均", min_value=1, max_value=100, value=5)
ma_mid = st.sidebar.number_input("中期移動平均", min_value=1, max_value=200, value=25)
ma_long = st.sidebar.number_input("長期移動平均", min_value=1, max_value=300, value=75)

# -----------------------------
# Data Fetch
# -----------------------------
@st.cache_data(ttl=3600)
def load_price_data(ticker: str, period_value: str):
    df = yf.download(ticker, period=period_value, auto_adjust=False, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


df = load_price_data(symbol, period)

if df.empty:
    st.error("データを取得できませんでした。銘柄コードを確認してください。例：7203.T, 6758.T, AAPL")
    st.stop()

# -----------------------------
# Indicators
# -----------------------------
df[f"MA{ma_short}"] = df["Close"].rolling(ma_short).mean()
df[f"MA{ma_mid}"] = df["Close"].rolling(ma_mid).mean()
df[f"MA{ma_long}"] = df["Close"].rolling(ma_long).mean()

df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()

macd = ta.trend.MACD(close=df["Close"])
df["MACD"] = macd.macd()
df["MACD_SIGNAL"] = macd.macd_signal()
df["MACD_DIFF"] = macd.macd_diff()

df["Volume_MA20"] = df["Volume"].rolling(20).mean()
df["Volume_Ratio"] = df["Volume"] / df["Volume_MA20"]

latest = df.dropna().iloc[-1]

# -----------------------------
# Summary
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("終値", f"{latest['Close']:.2f}")
col2.metric("RSI", f"{latest['RSI']:.1f}")
col3.metric("出来高倍率", f"{latest['Volume_Ratio']:.2f}倍")
col4.metric("MACD差分", f"{latest['MACD_DIFF']:.2f}")

# -----------------------------
# Price Chart
# -----------------------------
st.subheader("株価チャート")

price_fig = go.Figure()

price_fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="ローソク足"
))

price_fig.add_trace(go.Scatter(
    x=df.index,
    y=df[f"MA{ma_short}"],
    name=f"MA{ma_short}",
    mode="lines"
))

price_fig.add_trace(go.Scatter(
    x=df.index,
    y=df[f"MA{ma_mid}"],
    name=f"MA{ma_mid}",
    mode="lines"
))

price_fig.add_trace(go.Scatter(
    x=df.index,
    y=df[f"MA{ma_long}"],
    name=f"MA{ma_long}",
    mode="lines"
))

price_fig.update_layout(
    height=600,
    xaxis_rangeslider_visible=False,
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(price_fig, use_container_width=True)

# -----------------------------
# Volume
# -----------------------------
st.subheader("出来高")

volume_fig = go.Figure()

volume_fig.add_trace(go.Bar(
    x=df.index,
    y=df["Volume"],
    name="出来高"
))

volume_fig.add_trace(go.Scatter(
    x=df.index,
    y=df["Volume_MA20"],
    name="出来高20日平均",
    mode="lines"
))

volume_fig.update_layout(
    height=300,
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(volume_fig, use_container_width=True)

# -----------------------------
# RSI
# -----------------------------
st.subheader("RSI")

rsi_fig = go.Figure()

rsi_fig.add_trace(go.Scatter(
    x=df.index,
    y=df["RSI"],
    name="RSI",
    mode="lines"
))

rsi_fig.add_hline(y=70, line_dash="dash")
rsi_fig.add_hline(y=30, line_dash="dash")

rsi_fig.update_layout(
    height=300,
    yaxis=dict(range=[0, 100]),
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(rsi_fig, use_container_width=True)

# -----------------------------
# MACD
# -----------------------------
st.subheader("MACD")

macd_fig = go.Figure()

macd_fig.add_trace(go.Scatter(
    x=df.index,
    y=df["MACD"],
    name="MACD",
    mode="lines"
))

macd_fig.add_trace(go.Scatter(
    x=df.index,
    y=df["MACD_SIGNAL"],
    name="Signal",
    mode="lines"
))

macd_fig.add_trace(go.Bar(
    x=df.index,
    y=df["MACD_DIFF"],
    name="MACD Diff"
))

macd_fig.update_layout(
    height=300,
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(macd_fig, use_container_width=True)

# -----------------------------
# Raw Data
# -----------------------------
with st.expander("取得データを表示"):
    st.dataframe(df.tail(100), use_container_width=True)
