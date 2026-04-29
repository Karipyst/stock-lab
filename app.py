from pathlib import Path
import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import ta
import yfinance as yf


st.set_page_config(
    page_title="Stock Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("株価分析ダッシュボード")
st.caption("株価・出来高・テクニカル指標・買い候補スコアを確認するためのダッシュボードです。")

WATCHLIST_PATH = Path("watchlist.csv")

HISTORY_PATH = Path("score_history.csv")


@st.cache_data(ttl=300)
def load_score_history(path_text: str = str(HISTORY_PATH)) -> pd.DataFrame:
    """GitHub Actionsなどで保存されたスコア履歴CSVを読み込む。"""
    path = Path(path_text)
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(col).replace("\ufeff", "").strip() for col in df.columns]

    if "run_date" in df.columns:
        df["run_date"] = pd.to_datetime(df["run_date"], errors="coerce")
    numeric_cols = [
        "score", "unit_price", "rsi", "volume_ratio", "macd_diff",
        "return_5d", "return_20d",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def show_score_history(watchlist: pd.DataFrame):
    st.header("スコア履歴")
    st.caption("GitHub Actions などで保存された score_history.csv を表示します。通知機能は使いません。")

    history = load_score_history()
    if history.empty:
        st.info(
            "score_history.csv がまだありません。"
            "GitHub Actions を1回実行するか、ローカルで `python score_history.py` を実行すると作成されます。"
        )
        return

    required_cols = {"run_date", "symbol", "name", "score", "status"}
    missing = required_cols - set(history.columns)
    if missing:
        st.error(f"score_history.csv に必要な列が不足しています: {', '.join(sorted(missing))}")
        return

    history = history.dropna(subset=["run_date", "symbol"]).copy()
    if history.empty:
        st.warning("有効な履歴データがありません。")
        return

    history = history.sort_values(["run_date", "symbol"])
    latest_date = history["run_date"].max()
    latest = history[history["run_date"] == latest_date].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("履歴日数", history["run_date"].dt.date.nunique())
    c2.metric("最新保存日", latest_date.strftime("%Y-%m-%d"))
    c3.metric("最新保存銘柄数", len(latest))

    latest_view = latest.sort_values(["score", "volume_ratio"], ascending=[False, False], na_position="last")
    st.subheader("最新スコア")
    show_cols = [
        col for col in [
            "run_date", "symbol", "name", "theme", "score", "status", "unit_price",
            "rsi", "volume_ratio", "macd_diff", "return_5d", "return_20d",
        ] if col in latest_view.columns
    ]
    st.dataframe(latest_view[show_cols], use_container_width=True, hide_index=True)

    st.subheader("銘柄別の履歴")
    history_symbols = set(history["symbol"])
    options = [f"{row['symbol']} | {row['name']}" for _, row in watchlist.iterrows() if row["symbol"] in history_symbols]
    if not options:
        st.info("現在の銘柄CSVに含まれる銘柄の履歴がありません。")
        return

    selected = st.selectbox("履歴を見る銘柄", options)
    symbol = selected.split("|")[0].strip()
    one = history[history["symbol"] == symbol].sort_values("run_date").copy()

    st.subheader("スコア推移")
    st.caption("選択銘柄に加えて、最新保存日時点で「強い買い候補」の銘柄を同じグラフに重ねて表示できます。")

    latest_strong = latest[latest["status"] == "強い買い候補"].copy()
    latest_strong = latest_strong.sort_values(["score", "volume_ratio"], ascending=[False, False], na_position="last")

    strong_options = []
    strong_label_to_symbol = {}
    for _, row in latest_strong.iterrows():
        label = f"{row['symbol']} | {row['name']} | Score {row['score']}"
        strong_options.append(label)
        strong_label_to_symbol[label] = row["symbol"]

    default_strong_options = strong_options[: min(8, len(strong_options))]
    selected_strong_labels = st.multiselect(
        "同じグラフに重ねる強い買い候補",
        strong_options,
        default=default_strong_options,
        help="最新保存日時点で status が『強い買い候補』の銘柄です。多すぎると見づらくなるため、初期表示は上位8件までにしています。",
    )

    overlay_symbols = [strong_label_to_symbol[label] for label in selected_strong_labels]
    chart_symbols = []
    if symbol not in chart_symbols:
        chart_symbols.append(symbol)
    for overlay_symbol in overlay_symbols:
        if overlay_symbol not in chart_symbols:
            chart_symbols.append(overlay_symbol)

    latest_name_map = latest.drop_duplicates("symbol").set_index("symbol")["name"].to_dict()
    latest_score_map = latest.drop_duplicates("symbol").set_index("symbol")["score"].to_dict()
    latest_status_map = latest.drop_duplicates("symbol").set_index("symbol")["status"].to_dict()

    fig = go.Figure()
    for chart_symbol in chart_symbols:
        trend = history[history["symbol"] == chart_symbol].sort_values("run_date").copy()
        if trend.empty:
            continue
        chart_name = latest_name_map.get(chart_symbol, chart_symbol)
        latest_score = latest_score_map.get(chart_symbol)
        latest_status = latest_status_map.get(chart_symbol, "")
        trace_name = f"{chart_symbol} {chart_name}"
        if pd.notna(latest_score):
            trace_name += f" / 最新{latest_score:.0f}"
        if latest_status:
            trace_name += f" / {latest_status}"

        fig.add_trace(
            go.Scatter(
                x=trend["run_date"],
                y=trend["score"],
                name=trace_name,
                mode="lines+markers",
                line=dict(width=4 if chart_symbol == symbol else 2),
            )
        )

    fig.add_hrect(y0=9, y1=12, opacity=0.08, line_width=0, annotation_text="強い買い候補", annotation_position="top left")
    fig.add_hrect(y0=6, y1=9, opacity=0.05, line_width=0, annotation_text="買い候補", annotation_position="bottom left")
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=24, b=10),
        yaxis=dict(range=[0, 12], title="スコア"),
        xaxis=dict(title="保存日"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    if latest_strong.empty:
        st.info("最新保存日時点で『強い買い候補』の銘柄はありません。")
    else:
        with st.expander("最新の強い買い候補一覧", expanded=False):
            strong_cols = [
                col for col in [
                    "run_date", "symbol", "name", "theme", "score", "status", "unit_price",
                    "rsi", "volume_ratio", "macd_diff", "return_5d", "return_20d",
                ] if col in latest_strong.columns
            ]
            st.dataframe(latest_strong[strong_cols], use_container_width=True, hide_index=True)

    detail_cols = [
        col for col in [
            "run_date", "score", "status", "unit_price", "rsi", "volume_ratio",
            "macd_diff", "return_5d", "return_20d",
        ] if col in one.columns
    ]
    st.dataframe(one[detail_cols].sort_values("run_date", ascending=False), use_container_width=True, hide_index=True)


def find_watchlist_files() -> list[str]:
    """同じ階層にある watchlist*.csv を候補として表示する。"""
    files = sorted(
        Path(".").glob("watchlist*.csv"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )
    return [str(path) for path in files]


def normalize_watchlist(df: pd.DataFrame) -> pd.DataFrame:
    """CSVの行数が増えても使えるように、必須列確認・不足列補完・重複除外を行う。"""
    df = df.copy()
    df.columns = [str(col).replace("\ufeff", "").strip() for col in df.columns]

    if not {"symbol", "name"}.issubset(df.columns):
        st.error("銘柄CSVには symbol,name 列が必要です。")
        st.stop()

    if "theme" not in df.columns:
        df["theme"] = "未分類"
    if "memo" not in df.columns:
        df["memo"] = ""

    for col in ["symbol", "name", "theme", "memo"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df = df[df["symbol"] != ""]
    df = df.drop_duplicates(subset=["symbol"], keep="first")
    df = df.reset_index(drop=True)

    if df.empty:
        st.error("銘柄CSVに有効な銘柄コードがありません。")
        st.stop()

    return df


@st.cache_data(ttl=3600)
def load_watchlist_from_path(path_text: str) -> pd.DataFrame:
    path = Path(path_text)

    if path.exists():
        df = pd.read_csv(path, encoding="utf-8-sig")
    else:
        df = pd.DataFrame(
            {
                "symbol": ["7203.T", "6758.T", "7974.T"],
                "name": ["トヨタ自動車", "ソニーグループ", "任天堂"],
                "theme": ["大型・製造", "大型・テック", "大型・ゲーム"],
                "memo": ["", "", ""],
            }
        )

    return normalize_watchlist(df)


@st.cache_data(ttl=3600)
def load_watchlist_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig")
    return normalize_watchlist(df)


@st.cache_data(ttl=3600)
def load_price_data(ticker: str, period_value: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period_value,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def yen_symbol(symbol: str) -> bool:
    return symbol.endswith(".T")


def format_price(value, symbol: str) -> str:
    if value is None or pd.isna(value):
        return "-"
    if yen_symbol(symbol):
        return f"{value:,.0f}円"
    return f"{value:,.2f}"


def format_unit_amount(value, symbol: str) -> str:
    if value is None or pd.isna(value) or not yen_symbol(symbol):
        return "-"
    return f"{value * 100:,.0f}円"


def price_band_label(value, symbol: str) -> str:
    if value is None or pd.isna(value):
        return "不明"
    if not yen_symbol(symbol):
        return "米国株"
    if value <= 1000:
        return "低単価"
    if value <= 3000:
        return "買いやすい"
    if value <= 10000:
        return "1万円以内"
    return "1万円超"


def ma_labels(ma_short: int, ma_mid: int, ma_long: int) -> dict[str, str]:
    return {
        "short": f"MA{ma_short}",
        "mid": f"MA{ma_mid}",
        "long": f"MA{ma_long}",
    }


def validate_ma_values(ma_short: int, ma_mid: int, ma_long: int) -> bool:
    if ma_short < ma_mid < ma_long:
        return True
    st.warning("移動平均は 短期 < 中期 < 長期 になるように設定してください。例：5 / 25 / 75")
    return False


def add_indicators(
    df: pd.DataFrame,
    ma_short: int,
    ma_mid: int,
    ma_long: int,
) -> pd.DataFrame:
    df = df.copy()
    labels = ma_labels(ma_short, ma_mid, ma_long)

    df[labels["short"]] = df["Close"].rolling(ma_short).mean()
    df[labels["mid"]] = df["Close"].rolling(ma_mid).mean()
    df[labels["long"]] = df["Close"].rolling(ma_long).mean()

    df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()

    macd = ta.trend.MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()
    df["MACD_DIFF"] = macd.macd_diff()

    df["Volume_MA20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA20"]

    df["Return_5d"] = df["Close"].pct_change(5) * 100
    df["Return_20d"] = df["Close"].pct_change(20) * 100

    return df


def calculate_buy_score(
    df: pd.DataFrame,
    ma_short: int,
    ma_mid: int,
    ma_long: int,
) -> dict:
    labels = ma_labels(ma_short, ma_mid, ma_long)
    required = [
        labels["short"],
        labels["mid"],
        labels["long"],
        "RSI",
        "MACD_DIFF",
        "Volume_Ratio",
        "Return_5d",
        "Return_20d",
    ]

    clean = df.dropna(subset=required)

    if clean.empty or len(clean) < 2:
        return {
            "score": 0,
            "status": "データ不足",
            "reasons": ["指標計算に必要なデータが不足しています。期間を長くするか、移動平均の日数を短くしてください。"],
            "latest": None,
        }

    latest = clean.iloc[-1]
    prev = clean.iloc[-2]

    score = 0
    reasons = []

    close = latest["Close"]
    ma_s = latest[labels["short"]]
    ma_m = latest[labels["mid"]]
    ma_l = latest[labels["long"]]
    rsi = latest["RSI"]
    macd_diff = latest["MACD_DIFF"]
    prev_macd_diff = prev["MACD_DIFF"]
    volume_ratio = latest["Volume_Ratio"]
    return_5d = latest["Return_5d"]
    return_20d = latest["Return_20d"]

    if close > ma_m:
        score += 2
        reasons.append(f"終値が{ma_mid}日移動平均線を上回っており、中期的には上向きです。")
    else:
        reasons.append(f"終値が{ma_mid}日移動平均線を下回っており、中期的には弱めです。")

    if ma_s > ma_m:
        score += 2
        reasons.append(f"{ma_short}日移動平均線が{ma_mid}日移動平均線を上回っており、短期トレンドが改善しています。")

    if ma_m > ma_l:
        score += 2
        reasons.append(f"{ma_mid}日移動平均線が{ma_long}日移動平均線を上回っており、上昇基調が確認できます。")

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
        "latest": latest,
    }


def analyze_symbol(
    symbol: str,
    name: str,
    theme: str,
    period: str,
    ma_short: int,
    ma_mid: int,
    ma_long: int,
) -> dict:
    df = load_price_data(symbol, period)

    if df.empty:
        return {
            "symbol": symbol,
            "name": name,
            "theme": theme,
            "score": 0,
            "status": "取得失敗",
            "unit_price": None,
            "price_band": "不明",
            "rsi": None,
            "volume_ratio": None,
            "macd_diff": None,
            "return_5d": None,
            "return_20d": None,
            "reasons": ["データを取得できませんでした。"],
        }

    df = add_indicators(df, ma_short, ma_mid, ma_long)
    result = calculate_buy_score(df, ma_short, ma_mid, ma_long)
    latest = result["latest"]

    if latest is None:
        return {
            "symbol": symbol,
            "name": name,
            "theme": theme,
            "score": result["score"],
            "status": result["status"],
            "unit_price": None,
            "price_band": "不明",
            "rsi": None,
            "volume_ratio": None,
            "macd_diff": None,
            "return_5d": None,
            "return_20d": None,
            "reasons": result["reasons"],
        }

    close = latest["Close"]

    return {
        "symbol": symbol,
        "name": name,
        "theme": theme,
        "score": result["score"],
        "status": result["status"],
        "unit_price": close,
        "price_band": price_band_label(close, symbol),
        "rsi": latest["RSI"],
        "volume_ratio": latest["Volume_Ratio"],
        "macd_diff": latest["MACD_DIFF"],
        "return_5d": latest["Return_5d"],
        "return_20d": latest["Return_20d"],
        "reasons": result["reasons"],
    }


def draw_price_chart(
    df: pd.DataFrame,
    ma_short: int,
    ma_mid: int,
    ma_long: int,
):
    labels = ma_labels(ma_short, ma_mid, ma_long)

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="ローソク足",
        )
    )

    fig.add_trace(go.Scatter(x=df.index, y=df[labels["short"]], name=labels["short"], mode="lines"))
    fig.add_trace(go.Scatter(x=df.index, y=df[labels["mid"]], name=labels["mid"], mode="lines"))
    fig.add_trace(go.Scatter(x=df.index, y=df[labels["long"]], name=labels["long"], mode="lines"))

    fig.update_layout(
        height=440,
        xaxis_rangeslider_visible=False,
        margin=dict(l=12, r=12, t=28, b=12),
        legend=dict(orientation="h"),
    )

    st.plotly_chart(fig, use_container_width=True)


def draw_volume_chart(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="出来高"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Volume_MA20"], name="20日平均", mode="lines"))
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=24, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)


def draw_rsi_chart(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", mode="lines"))
    fig.add_hline(y=70, line_dash="dash", annotation_text="70")
    fig.add_hline(y=30, line_dash="dash", annotation_text="30")
    fig.update_layout(height=220, yaxis=dict(range=[0, 100]), margin=dict(l=10, r=10, t=24, b=10))
    st.plotly_chart(fig, use_container_width=True)


def draw_macd_chart(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", mode="lines"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], name="Signal", mode="lines"))
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_DIFF"], name="差分"))
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=24, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)


def show_beginner_guide():
    st.header("用語・指標の見方")

    st.info(
        "この画面のスコアは投資判断を補助するための簡易指標です。"
        "将来の株価上昇を保証するものではありません。"
        "実際の売買では、業績、決算、ニュース、市場全体の地合いも確認してください。"
    )

    with st.expander("株価単価"):
        st.markdown(
            """
### 株価単価
このアプリでは、**1株あたりの最新終値**を株価単価として表示しています。

日本株の場合、実際の購入は通常100株単位のため、概算の購入金額は以下です。

`株価単価 × 100株`

例：

- 株価単価 800円 → 約8万円
- 株価単価 3,000円 → 約30万円
- 株価単価 10,000円 → 約100万円
"""
        )

    with st.expander("株価・ローソク足"):
        st.markdown(
            """
### 株価
株が市場で売買されている価格です。

### ローソク足
1本のローソク足は、一定期間の値動きを表します。

- **始値**：その日の最初の価格
- **高値**：その日の一番高い価格
- **安値**：その日の一番安い価格
- **終値**：その日の最後の価格
"""
        )

    with st.expander("出来高"):
        st.markdown(
            """
### 出来高
その銘柄がどれだけ売買されたかを示します。

### 見方
- 出来高が増える：市場参加者の注目が集まっている
- 株価上昇 + 出来高増加：強い上昇の可能性
- 株価下落 + 出来高増加：強い売り圧力の可能性
- 株価上昇 + 出来高少ない：上昇の信頼度はやや低い

このアプリでは、**出来高倍率 = 今日の出来高 / 20日平均出来高** としています。
"""
        )

    with st.expander("移動平均線"):
        st.markdown(
            """
### 移動平均線
過去の株価の平均値を線にしたものです。

### 見方
- 株価が中期移動平均より上：中期的には強い
- 株価が中期移動平均より下：中期的には弱い
- 短期 > 中期 > 長期：上昇トレンド
- 短期 < 中期 < 長期：下落トレンド
"""
        )

    with st.expander("RSI"):
        st.markdown(
            """
### RSI
買われすぎ・売られすぎを見る指標です。

| RSI | 一般的な見方 |
|---:|---|
| 70以上 | 買われすぎ |
| 50前後 | 中立 |
| 30以下 | 売られすぎ |
"""
        )

    with st.expander("MACD"):
        st.markdown(
            """
### MACD
株価の勢い、つまりモメンタムを見る指標です。

### 見方
- MACDがSignalを上抜け：買いサイン候補
- MACDがSignalを下抜け：売りサイン候補
- MACD差分がプラス：上昇モメンタム
- MACD差分がマイナス：下落モメンタム
"""
        )


def set_mode(mode: str, symbol: str | None = None):
    st.session_state["mode"] = mode
    if symbol:
        st.session_state["selected_symbol"] = symbol


st.sidebar.header("表示メニュー")

watchlist_files = find_watchlist_files()
if watchlist_files:
    default_index = watchlist_files.index(str(WATCHLIST_PATH)) if str(WATCHLIST_PATH) in watchlist_files else 0
    selected_watchlist_path = st.sidebar.selectbox(
        "銘柄CSV",
        watchlist_files,
        index=default_index,
        format_func=lambda value: Path(value).name,
    )
else:
    selected_watchlist_path = str(WATCHLIST_PATH)
    st.sidebar.warning("watchlist.csv が見つからないため、サンプル銘柄を表示します。")

uploaded_watchlist = st.sidebar.file_uploader(
    "別CSVを一時的に読み込む",
    type=["csv"],
    help="GitHub上のCSVを置き換えずに、手元のCSVを試す場合に使います。",
)

if uploaded_watchlist is not None:
    watchlist = load_watchlist_from_bytes(uploaded_watchlist.getvalue())
    active_watchlist_label = uploaded_watchlist.name
else:
    watchlist = load_watchlist_from_path(selected_watchlist_path)
    active_watchlist_label = Path(selected_watchlist_path).name

st.sidebar.caption(f"読込中：{active_watchlist_label} / {len(watchlist)}銘柄")

if "mode" not in st.session_state:
    st.session_state["mode"] = "ランキング"

if "selected_symbol" not in st.session_state:
    st.session_state["selected_symbol"] = watchlist.iloc[0]["symbol"]

if "detail_period" not in st.session_state:
    st.session_state["detail_period"] = "1y"

if "ma_short" not in st.session_state:
    st.session_state["ma_short"] = 5
if "ma_mid" not in st.session_state:
    st.session_state["ma_mid"] = 25
if "ma_long" not in st.session_state:
    st.session_state["ma_long"] = 75

mode_options = ["ランキング", "スコア履歴", "個別銘柄", "用語説明"]
mode_index = mode_options.index(st.session_state["mode"]) if st.session_state["mode"] in mode_options else 0

selected_mode = st.sidebar.radio("表示モード", mode_options, index=mode_index)

if selected_mode != st.session_state["mode"]:
    st.session_state["mode"] = selected_mode
    st.rerun()

available_themes = ["すべて"] + sorted(watchlist["theme"].dropna().unique().tolist())
selected_theme = st.sidebar.selectbox("テーマ絞り込み", available_themes)
show_under_10000_only = st.sidebar.checkbox("日本株は1株1万円以内だけ表示", value=False)
analysis_limit_label = st.sidebar.selectbox(
    "ランキング分析対象",
    ["先頭100件", "先頭300件", "すべて"],
    index=0,
    help="銘柄数が多いCSVでは、全件を毎回分析すると取得に時間がかかります。",
)

st.sidebar.divider()
st.sidebar.caption("日本株は 7203.T のように .T を付けます。")

filtered_watchlist = watchlist.copy()
if selected_theme != "すべて":
    filtered_watchlist = filtered_watchlist[filtered_watchlist["theme"] == selected_theme]

if filtered_watchlist.empty:
    st.warning("該当する銘柄がありません。")
    st.stop()

with st.sidebar.expander("CSV銘柄一覧", expanded=False):
    st.dataframe(
        filtered_watchlist[["symbol", "name", "theme"]],
        use_container_width=True,
        hide_index=True,
    )


if st.session_state["mode"] == "用語説明":
    show_beginner_guide()
    st.stop()


if st.session_state["mode"] == "スコア履歴":
    show_score_history(watchlist)
    st.stop()


if st.session_state["mode"] == "ランキング":
    st.header("買い候補ランキング")
    st.caption("ランキングは標準設定の 1年 / MA5・MA25・MA75 で判定します。")

    period = "1y"
    ma_short = 5
    ma_mid = 25
    ma_long = 75

    with st.expander("スコアの考え方を表示"):
        st.markdown(
            """
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
"""
        )

    if st.button("ランキングを更新"):
        st.cache_data.clear()
        st.rerun()

    if analysis_limit_label == "先頭100件":
        analysis_watchlist = filtered_watchlist.head(100)
    elif analysis_limit_label == "先頭300件":
        analysis_watchlist = filtered_watchlist.head(300)
    else:
        analysis_watchlist = filtered_watchlist

    if len(analysis_watchlist) < len(filtered_watchlist):
        st.info(
            f"CSV全体は{len(filtered_watchlist)}銘柄です。"
            f"現在は処理負荷を抑えるため、先頭{len(analysis_watchlist)}銘柄をランキング分析しています。"
        )

    results = []
    progress = st.progress(0)

    for _, row in analysis_watchlist.iterrows():
        result = analyze_symbol(row["symbol"], row["name"], row["theme"], period, ma_short, ma_mid, ma_long)
        results.append(result)
        progress.progress(len(results) / len(analysis_watchlist))

    progress.empty()

    ranking_df = pd.DataFrame(results)

    if show_under_10000_only:
        ranking_df = ranking_df[
            (~ranking_df["symbol"].str.endswith(".T"))
            | (ranking_df["unit_price"].notna() & (ranking_df["unit_price"] <= 10000))
        ]

    display_df = ranking_df.copy()
    display_df = display_df.sort_values(by=["score", "volume_ratio"], ascending=[False, False])

    display_df["株価単価"] = display_df.apply(lambda row: format_price(row["unit_price"], row["symbol"]), axis=1)
    display_df["概算単元金額"] = display_df.apply(
        lambda row: format_unit_amount(row["unit_price"], row["symbol"]),
        axis=1,
    )
    display_df["rsi"] = display_df["rsi"].map(lambda x: None if pd.isna(x) else round(x, 1))
    display_df["volume_ratio"] = display_df["volume_ratio"].map(lambda x: None if pd.isna(x) else round(x, 2))
    display_df["macd_diff"] = display_df["macd_diff"].map(lambda x: None if pd.isna(x) else round(x, 2))
    display_df["return_5d"] = display_df["return_5d"].map(lambda x: None if pd.isna(x) else round(x, 2))
    display_df["return_20d"] = display_df["return_20d"].map(lambda x: None if pd.isna(x) else round(x, 2))

    col1, col2, col3 = st.columns(3)
    col1.metric("分析表示銘柄数", len(display_df))
    col2.metric("強い買い候補", int((display_df["score"] >= 9).sum()))
    col3.metric("買い候補以上", int((display_df["score"] >= 6).sum()))

    st.dataframe(
        display_df[
            [
                "symbol",
                "name",
                "theme",
                "score",
                "status",
                "株価単価",
                "price_band",
                "概算単元金額",
                "rsi",
                "volume_ratio",
                "macd_diff",
                "return_5d",
                "return_20d",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "symbol": "銘柄コード",
            "name": "銘柄名",
            "theme": "テーマ",
            "score": "スコア",
            "status": "判定",
            "price_band": "価格帯",
            "rsi": "RSI",
            "volume_ratio": "出来高倍率",
            "macd_diff": "MACD差分",
            "return_5d": "5日騰落率%",
            "return_20d": "20日騰落率%",
        },
    )

    st.subheader("トレンドへ移動")
    for _, row in display_df.iterrows():
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**{row['symbol']} {row['name']}**")
            with c2:
                if st.button("トレンドを見る", key=f"detail_{row['symbol']}"):
                    set_mode("個別銘柄", row["symbol"])
                    st.rerun()

    st.subheader("上位銘柄の理由")
    for _, row in display_df.head(5).iterrows():
        with st.expander(f"{row['symbol']} {row['name']} / Score {row['score']} / {row['status']}"):
            matched = ranking_df[ranking_df["symbol"] == row["symbol"]].iloc[0]
            for reason in matched["reasons"]:
                st.write(f"- {reason}")

    st.stop()


if st.session_state["mode"] == "個別銘柄":
    st.header("個別銘柄分析")

    options = [f"{row['symbol']} | {row['name']} | {row['theme']}" for _, row in filtered_watchlist.iterrows()]
    selected_symbol = st.session_state.get("selected_symbol", filtered_watchlist.iloc[0]["symbol"])
    option_symbols = [option.split("|")[0].strip() for option in options]
    selected_index = option_symbols.index(selected_symbol) if selected_symbol in option_symbols else 0

    selected = st.selectbox("銘柄を選択", options, index=selected_index)
    symbol = selected.split("|")[0].strip()
    st.session_state["selected_symbol"] = symbol

    selected_row = filtered_watchlist[filtered_watchlist["symbol"] == symbol].iloc[0]
    name = selected_row["name"]
    theme = selected_row["theme"]
    memo = selected_row.get("memo", "")

    period = st.session_state["detail_period"]
    ma_short = int(st.session_state["ma_short"])
    ma_mid = int(st.session_state["ma_mid"])
    ma_long = int(st.session_state["ma_long"])

    df = load_price_data(symbol, period)

    if df.empty:
        st.error("データを取得できませんでした。銘柄コードを確認してください。")
        st.stop()

    df = add_indicators(df, ma_short, ma_mid, ma_long)
    score_result = calculate_buy_score(df, ma_short, ma_mid, ma_long)

    labels = ma_labels(ma_short, ma_mid, ma_long)
    required = [
        labels["short"],
        labels["mid"],
        labels["long"],
        "RSI",
        "MACD_DIFF",
        "Volume_Ratio",
        "Return_5d",
        "Return_20d",
    ]
    clean = df.dropna(subset=required)

    if clean.empty:
        st.error("指標計算に必要なデータが不足しています。期間を長くするか、移動平均の日数を短くしてください。")
        st.stop()

    latest = clean.iloc[-1]

    st.subheader(f"{symbol} {name}")
    st.caption(f"テーマ：{theme}")
    if memo:
        st.caption(memo)

    st.subheader("株価トレンド")
    draw_price_chart(df, ma_short, ma_mid, ma_long)

    st.markdown("#### 表示設定")
    c_period, c_ma1, c_ma2, c_ma3 = st.columns([1.2, 1, 1, 1])

    with c_period:
        new_period = st.selectbox(
            "表示期間",
            ["6mo", "1y", "2y", "5y"],
            index=["6mo", "1y", "2y", "5y"].index(period) if period in ["6mo", "1y", "2y", "5y"] else 1,
            key="detail_period_select",
        )

    with c_ma1:
        new_ma_short = st.number_input(
            "短期MA",
            min_value=1,
            max_value=100,
            value=ma_short,
            step=1,
            key="detail_ma_short",
        )

    with c_ma2:
        new_ma_mid = st.number_input(
            "中期MA",
            min_value=2,
            max_value=200,
            value=ma_mid,
            step=1,
            key="detail_ma_mid",
        )

    with c_ma3:
        new_ma_long = st.number_input(
            "長期MA",
            min_value=3,
            max_value=300,
            value=ma_long,
            step=1,
            key="detail_ma_long",
        )

    if not validate_ma_values(int(new_ma_short), int(new_ma_mid), int(new_ma_long)):
        st.stop()

    if (
        new_period != period
        or int(new_ma_short) != ma_short
        or int(new_ma_mid) != ma_mid
        or int(new_ma_long) != ma_long
    ):
        st.session_state["detail_period"] = new_period
        st.session_state["ma_short"] = int(new_ma_short)
        st.session_state["ma_mid"] = int(new_ma_mid)
        st.session_state["ma_long"] = int(new_ma_long)
        st.rerun()

    with st.expander("この銘柄の判定理由を表示", expanded=False):
        st.write(f"判定：{score_result['status']} / スコア：{score_result['score']} / 12")
        for reason in score_result["reasons"]:
            st.write(f"- {reason}")

    st.subheader("補助指標")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.caption("出来高")
        draw_volume_chart(df)

    with c2:
        st.caption("RSI")
        draw_rsi_chart(df)

    with c3:
        st.caption("MACD")
        draw_macd_chart(df)

    with st.expander("取得データを表示", expanded=False):
        compact_df = df.tail(80).copy()
        compact_df = compact_df[
            [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                labels["short"],
                labels["mid"],
                labels["long"],
                "RSI",
                "MACD_DIFF",
                "Volume_Ratio",
                "Return_5d",
                "Return_20d",
            ]
        ]
        st.dataframe(compact_df, use_container_width=True)

    with st.expander("用語・見方を表示", expanded=False):
        show_beginner_guide()
