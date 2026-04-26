import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Stock Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("株価分析ダッシュボード")
st.caption("株価・出来高・テクニカル指標・買い候補スコアを確認するためのダッシュボードです。")

WATCHLIST_PATH = Path("watchlist.csv")

@st.cache_data(ttl=3600)
def load_watchlist() -> pd.DataFrame:
    if WATCHLIST_PATH.exists():
        df = pd.read_csv(WATCHLIST_PATH)
    else:
        df = pd.DataFrame({
            "symbol": ["7203.T", "6758.T", "7974.T", "8058.T", "8306.T"],
            "name": ["トヨタ自動車", "ソニーグループ", "任天堂", "三菱商事", "三菱UFJ"],
            "memo": ["", "", "", "", ""]
        })

    required_cols = {"symbol", "name"}
    if not required_cols.issubset(df.columns):
        st.error("watchlist.csv には symbol,name 列が必要です。")
        st.stop()

    if "memo" not in df.columns:
        df["memo"] = ""

    return df

@st.cache_data(ttl=3600)
def load_price_data(ticker: str, period_value: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period_value,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA25"] = df["Close"].rolling(25).mean()
    df["MA75"] = df["Close"].rolling(75).mean()

    df["RSI"] = ta.momentum.RSIIndicator(
        close=df["Close"],
        window=14
    ).rsi()

    macd = ta.trend.MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()
    df["MACD_DIFF"] = macd.macd_diff()

    df["Volume_MA20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA20"]

    df["Return_5d"] = df["Close"].pct_change(5) * 100
    df["Return_20d"] = df["Close"].pct_change(20) * 100

    return df

def calculate_buy_score(df: pd.DataFrame) -> dict:
    clean = df.dropna()

    if clean.empty or len(clean) < 75:
        return {
            "score": 0,
            "status": "データ不足",
            "reasons": ["指標計算に必要なデータが不足しています。"],
            "latest": None
        }

    latest = clean.iloc[-1]
    prev = clean.iloc[-2]

    score = 0
    reasons = []

    close = latest["Close"]
    ma5 = latest["MA5"]
    ma25 = latest["MA25"]
    ma75 = latest["MA75"]
    rsi = latest["RSI"]
    macd_diff = latest["MACD_DIFF"]
    prev_macd_diff = prev["MACD_DIFF"]
    volume_ratio = latest["Volume_Ratio"]
    return_5d = latest["Return_5d"]
    return_20d = latest["Return_20d"]

    if close > ma25:
        score += 2
        reasons.append("終値が25日移動平均線を上回っており、中期的には上向きです。")
    else:
        reasons.append("終値が25日移動平均線を下回っており、中期的には弱めです。")

    if ma5 > ma25:
        score += 2
        reasons.append("5日移動平均線が25日移動平均線を上回っており、短期トレンドが改善しています。")

    if ma25 > ma75:
        score += 2
        reasons.append("25日移動平均線が75日移動平均線を上回っており、上昇基調が確認できます。")

    if 40 <= rsi <= 65:
        score += 2
        reasons.append("RSIが40〜65の範囲で、過熱しすぎていない上昇余地のある水準です。")
    elif rsi > 75:
        score -= 2
        reasons.append("RSIが75を超えており、短期的には買われすぎの可能性があります。")
    elif rsi < 30:
        score += 1
        reasons.append("RSIが30未満で売られすぎ水準です。ただし下落中の可能性もあります。")

    if macd_diff > 0:
        score += 2
        reasons.append("MACDがシグナルを上回っており、上昇モメンタムがあります。")

    if prev_macd_diff < 0 and macd_diff > 0:
        score += 2
        reasons.append("MACDが直近で陽転しており、買い転換の候補です。")

    if volume_ratio >= 1.5:
        score += 2
        reasons.append("出来高が20日平均の1.5倍以上で、注目度が高まっています。")
    elif volume_ratio >= 1.2:
        score += 1
        reasons.append("出来高が20日平均をやや上回っています。")

    if return_5d > 0:
        score += 1
        reasons.append("直近5営業日のリターンがプラスです。")

    if return_20d > 0:
        score += 1
        reasons.append("直近20営業日のリターンがプラスです。")

    score = max(0, min(score, 12))

    if score >= 9:
        status = "強い買い候補"
    elif score >= 6:
        status = "買い候補"
    elif score >= 3:
        status = "様子見"
    else:
        status = "弱い / 見送り"

    return {
        "score": score,
        "status": status,
        "reasons": reasons,
        "latest": latest
    }

def analyze_symbol(symbol: str, name: str, period: str) -> dict:
    df = load_price_data(symbol, period)

    if df.empty:
        return {
            "symbol": symbol,
            "name": name,
            "score": 0,
            "status": "取得失敗",
            "close": None,
            "rsi": None,
            "volume_ratio": None,
            "macd_diff": None,
            "return_5d": None,
            "return_20d": None,
            "reasons": ["データを取得できませんでした。"]
        }

    df = add_indicators(df)
    result = calculate_buy_score(df)
    latest = result["latest"]

    if latest is None:
        return {
            "symbol": symbol,
            "name": name,
            "score": result["score"],
            "status": result["status"],
            "close": None,
            "rsi": None,
            "volume_ratio": None,
            "macd_diff": None,
            "return_5d": None,
            "return_20d": None,
            "reasons": result["reasons"]
        }

    return {
        "symbol": symbol,
        "name": name,
        "score": result["score"],
        "status": result["status"],
        "close": latest["Close"],
        "rsi": latest["RSI"],
        "volume_ratio": latest["Volume_Ratio"],
        "macd_diff": latest["MACD_DIFF"],
        "return_5d": latest["Return_5d"],
        "return_20d": latest["Return_20d"],
        "reasons": result["reasons"]
    }

def draw_price_chart(df: pd.DataFrame):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="ローソク足"
    ))

    fig.add_trace(go.Scatter(x=df.index, y=df["MA5"], name="MA5", mode="lines"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA25"], name="MA25", mode="lines"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA75"], name="MA75", mode="lines"))

    fig.update_layout(
        height=560,
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h")
    )

    st.plotly_chart(fig, use_container_width=True)

def draw_volume_chart(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="出来高"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Volume_MA20"], name="出来高20日平均", mode="lines"))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

def draw_rsi_chart(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", mode="lines"))
    fig.add_hline(y=70, line_dash="dash", annotation_text="70: 買われすぎ目安")
    fig.add_hline(y=30, line_dash="dash", annotation_text="30: 売られすぎ目安")
    fig.update_layout(height=280, yaxis=dict(range=[0, 100]), margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

def draw_macd_chart(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", mode="lines"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], name="Signal", mode="lines"))
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_DIFF"], name="MACD差分"))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

def show_beginner_guide():
    st.header("用語・指標の見方")

    st.info(
        "この画面のスコアは投資判断を補助するための簡易指標です。"
        "将来の株価上昇を保証するものではありません。"
        "実際の売買では、業績、決算、ニュース、市場全体の地合いも確認してください。"
    )

    with st.expander("株価・ローソク足"):
        st.markdown("""
### 株価
株が市場で売買されている価格です。

### ローソク足
1本のローソク足は、一定期間の値動きを表します。

- **始値**：その日の最初の価格
- **高値**：その日の一番高い価格
- **安値**：その日の一番安い価格
- **終値**：その日の最後の価格

一般的には、終値が重要視されます。
""")

    with st.expander("出来高"):
        st.markdown("""
### 出来高
その銘柄がどれだけ売買されたかを示します。

### 見方
- 出来高が増える：市場参加者の注目が集まっている
- 株価上昇 + 出来高増加：強い上昇の可能性
- 株価下落 + 出来高増加：強い売り圧力の可能性
- 株価上昇 + 出来高少ない：上昇の信頼度はやや低い

このアプリでは、**出来高倍率 = 今日の出来高 / 20日平均出来高** としています。

| 出来高倍率 | 意味 |
|---:|---|
| 1.0倍未満 | 通常より少ない |
| 1.2倍以上 | やや注目増 |
| 1.5倍以上 | 注目度高め |
| 2.0倍以上 | 大きな材料がある可能性 |
""")

    with st.expander("移動平均線 MA5 / MA25 / MA75"):
        st.markdown("""
### 移動平均線
過去の株価の平均値を線にしたものです。

| 指標 | 意味 |
|---|---|
| MA5 | 約1週間の短期トレンド |
| MA25 | 約1か月の中期トレンド |
| MA75 | 約3か月の長期トレンド |

### 見方
- 株価がMA25より上：中期的には強い
- 株価がMA25より下：中期的には弱い
- MA5 > MA25 > MA75：上昇トレンド
- MA5 < MA25 < MA75：下落トレンド

移動平均線は過去データから作るため、反応が遅れます。
""")

    with st.expander("RSI"):
        st.markdown("""
### RSI
買われすぎ・売られすぎを見る指標です。0〜100の範囲で表示されます。

| RSI | 一般的な見方 |
|---:|---|
| 70以上 | 買われすぎ |
| 50前後 | 中立 |
| 30以下 | 売られすぎ |

### 見方
- RSIが上がる：買いの勢いが強い
- RSIが下がる：売りの勢いが強い
- RSI 70超え：短期的に過熱している可能性
- RSI 30未満：売られすぎだが、下落トレンド中の可能性もある

このアプリでは、**RSI 40〜65** を比較的よい水準としています。
""")

    with st.expander("MACD"):
        st.markdown("""
### MACD
株価の勢い、つまりモメンタムを見る指標です。

| 指標 | 意味 |
|---|---|
| MACD | 短期と中期の勢いの差 |
| Signal | MACDの平均線 |
| MACD差分 | MACD - Signal |

### 見方
- MACDがSignalを上抜け：買いサイン候補
- MACDがSignalを下抜け：売りサイン候補
- MACD差分がプラス：上昇モメンタム
- MACD差分がマイナス：下落モメンタム

MACD単体ではダマシがあるため、移動平均線・出来高・地合いと合わせて見ます。
""")

    with st.expander("買い候補スコア"):
        st.markdown("""
### 買い候補スコア
このアプリ独自の簡易スコアです。

| 要素 | 内容 |
|---|---|
| トレンド | 株価と移動平均線の位置 |
| RSI | 過熱感の有無 |
| MACD | 上昇モメンタム |
| 出来高 | 注目度の上昇 |
| 短期リターン | 直近の値動き |

| スコア | 判定 |
|---:|---|
| 9〜12 | 強い買い候補 |
| 6〜8 | 買い候補 |
| 3〜5 | 様子見 |
| 0〜2 | 弱い / 見送り |

これは売買推奨ではなく、候補を絞るためのスクリーニングです。
""")

watchlist = load_watchlist()

st.sidebar.header("表示設定")

period = st.sidebar.selectbox(
    "取得期間",
    ["6mo", "1y", "2y", "5y"],
    index=1
)

mode = st.sidebar.radio(
    "表示モード",
    ["ランキング", "個別銘柄", "用語説明"],
    index=0
)

st.sidebar.divider()
st.sidebar.caption("日本株は 7203.T のように .T を付けます。")

if mode == "用語説明":
    show_beginner_guide()
    st.stop()

if mode == "ランキング":
    st.header("買い候補ランキング")
    st.caption("watchlist.csv の銘柄を一括分析し、簡易スコア順に表示します。")

    with st.expander("スコアの考え方を表示"):
        st.markdown("""
スコアは以下の条件をもとに作っています。

- 終値が25日移動平均線より上
- 5日移動平均線が25日移動平均線より上
- 25日移動平均線が75日移動平均線より上
- RSIが40〜65
- MACDがSignalより上
- MACDが直近で陽転
- 出来高が20日平均より多い
- 直近5日・20日のリターンがプラス

高スコアほど「テクニカル面では買い候補に近い」という意味です。
""")

    if st.button("ランキングを更新"):
        st.cache_data.clear()
        st.rerun()

    results = []
    progress = st.progress(0)

    for i, row in watchlist.iterrows():
        result = analyze_symbol(row["symbol"], row["name"], period)
        results.append(result)
        progress.progress((i + 1) / len(watchlist))

    progress.empty()

    ranking_df = pd.DataFrame(results)
    display_df = ranking_df.copy()
    display_df = display_df.sort_values(by=["score", "volume_ratio"], ascending=[False, False])

    for col in ["close", "rsi", "volume_ratio", "macd_diff", "return_5d", "return_20d"]:
        display_df[col] = display_df[col].map(lambda x: None if pd.isna(x) else round(x, 2))

    st.dataframe(
        display_df[
            [
                "symbol", "name", "score", "status", "close", "rsi",
                "volume_ratio", "macd_diff", "return_5d", "return_20d"
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "symbol": "銘柄コード",
            "name": "銘柄名",
            "score": "スコア",
            "status": "判定",
            "close": "終値",
            "rsi": "RSI",
            "volume_ratio": "出来高倍率",
            "macd_diff": "MACD差分",
            "return_5d": "5日騰落率%",
            "return_20d": "20日騰落率%"
        }
    )

    st.subheader("上位銘柄の理由")

    top_df = display_df.head(5)
    for _, row in top_df.iterrows():
        with st.expander(f"{row['symbol']} {row['name']} / Score {row['score']} / {row['status']}"):
            matched = ranking_df[ranking_df["symbol"] == row["symbol"]].iloc[0]
            for reason in matched["reasons"]:
                st.write(f"- {reason}")

    st.stop()

if mode == "個別銘柄":
    st.header("個別銘柄分析")

    options = [f"{row['symbol']} | {row['name']}" for _, row in watchlist.iterrows()]
    selected = st.sidebar.selectbox("銘柄を選択", options)
    symbol = selected.split("|")[0].strip()

    selected_row = watchlist[watchlist["symbol"] == symbol].iloc[0]
    name = selected_row["name"]
    memo = selected_row.get("memo", "")

    df = load_price_data(symbol, period)

    if df.empty:
        st.error("データを取得できませんでした。銘柄コードを確認してください。")
        st.stop()

    df = add_indicators(df)
    score_result = calculate_buy_score(df)

    clean = df.dropna()
    latest = clean.iloc[-1]

    st.subheader(f"{symbol} {name}")

    if memo:
        st.caption(memo)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("終値", f"{latest['Close']:.2f}")
    col2.metric("買い候補スコア", f"{score_result['score']} / 12")
    col3.metric("RSI", f"{latest['RSI']:.1f}")
    col4.metric("出来高倍率", f"{latest['Volume_Ratio']:.2f}倍")

    st.info(f"判定：{score_result['status']}")

    with st.expander("この銘柄の判定理由"):
        for reason in score_result["reasons"]:
            st.write(f"- {reason}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["株価", "出来高", "RSI", "MACD", "用語・見方"])

    with tab1:
        draw_price_chart(df)

    with tab2:
        draw_volume_chart(df)

    with tab3:
        draw_rsi_chart(df)

    with tab4:
        draw_macd_chart(df)

    with tab5:
        show_beginner_guide()

    with st.expander("取得データを表示"):
        st.dataframe(df.tail(100), use_container_width=True)
