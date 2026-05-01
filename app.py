from pathlib import Path
import io
import hmac

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import ta
import yfinance as yf


st.set_page_config(
    page_title="Stock Analyzer",
    page_icon="📈",
    layout="wide",
)

WATCHLIST_PATH = Path("watchlist.csv")

HISTORY_PATH = Path("score_history.csv")


def get_app_password() -> str:
    """Streamlit Secrets からアプリ用パスワードを取得する。"""
    try:
        password = st.secrets.get("APP_PASSWORD", "")
    except Exception:
        password = ""
    return str(password) if password is not None else ""


def require_password() -> None:
    """認証が通るまで本体機能を表示しない。"""
    if st.session_state.get("authenticated", False):
        return

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { display: none; }
        [data-testid="collapsedControl"] { display: none; }
        .block-container { max-width: 720px; padding-top: 7rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Stock Analyzer")
    st.caption("株価分析ダッシュボードにアクセスするにはパスワードを入力してください。")

    with st.form("login_form", clear_on_submit=False):
        password = st.text_input("Password", type="password", placeholder="Password")
        submitted = st.form_submit_button("Unlock")

    if submitted:
        expected_password = get_app_password()

        if not expected_password:
            st.error("Streamlit Secrets に APP_PASSWORD が設定されていません。")
            return

        if hmac.compare_digest(password, expected_password):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("パスワードが違います。")

    st.stop()

require_password()

st.title("Stock Analyzer")
st.caption("株価・出来高・テクニカル指標・買い候補スコアを確認するためのダッシュボードです。")


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

    optional_text_cols = [
        "analysis_symbol",
        "analysis_name",
        "analysis_note",
        "asset_class",
        "fund_type",
        "manager",
        "sales_channel",
        "nisa_growth",
        "availability",
        "hedge",
        "settlement",
        "source_date",
        "source_url",
    ]
    for col in optional_text_cols:
        if col not in df.columns:
            df[col] = ""

    for col in ["symbol", "name", "theme", "memo"] + optional_text_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df = df[df["symbol"] != ""]
    df = df.drop_duplicates(subset=["symbol"], keep="first")
    df = df.reset_index(drop=True)

    if df.empty:
        st.error("銘柄CSVに有効な銘柄コードがありません。")
        st.stop()

    return df


@st.cache_data(ttl=3600)
def load_watchlist_from_path(path_text: str, mtime: float = 0.0) -> pd.DataFrame:
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


def is_fund_catalog(df: pd.DataFrame) -> bool:
    """投資信託カタログCSVかどうかを判定する。"""
    return "asset_class" in df.columns or df["symbol"].astype(str).str.startswith("SMBCG").all()


def show_fund_catalog(funds: pd.DataFrame) -> None:
    """yfinanceで価格取得できない投資信託CSVを、比較用カタログとして表示する。"""
    st.header("三井住友銀行 NISA成長投資枠ファンド一覧")
    st.caption(
        "このCSVは投資信託の比較・絞り込み用です。ランキングや個別トレンドでは、"
        "各ファンドに設定した分析用プロキシETF/指数ティッカーを使ってテクニカル分析します。"
    )

    df = funds.copy()

    for col in ["asset_class", "fund_type", "manager", "availability", "hedge", "settlement", "source_date"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("表示ファンド数", len(df))
    c2.metric("投資対象分類", df["asset_class"].replace("", pd.NA).dropna().nunique())
    c3.metric("運用会社数", df["manager"].replace("", pd.NA).dropna().nunique())
    c4.metric("対象予定を含む件数", int((df["availability"] != "取扱中").sum()))

    with st.expander("ファンド用フィルター", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            asset = st.selectbox("投資対象", ["すべて"] + sorted(df["asset_class"].replace("", pd.NA).dropna().unique().tolist()))
        with f2:
            manager = st.selectbox("運用会社", ["すべて"] + sorted(df["manager"].replace("", pd.NA).dropna().unique().tolist()))
        with f3:
            hedge = st.selectbox("為替ヘッジ", ["すべて"] + sorted(df["hedge"].replace("", pd.NA).dropna().unique().tolist()))
        with f4:
            availability = st.selectbox("取扱状況", ["すべて"] + sorted(df["availability"].replace("", pd.NA).dropna().unique().tolist()))
        keyword = st.text_input("ファンド名キーワード", placeholder="例：米国、AI、ゴールド、インド")

    view = df.copy()
    if asset != "すべて":
        view = view[view["asset_class"] == asset]
    if manager != "すべて":
        view = view[view["manager"] == manager]
    if hedge != "すべて":
        view = view[view["hedge"] == hedge]
    if availability != "すべて":
        view = view[view["availability"] == availability]
    if keyword.strip():
        key = keyword.strip()
        view = view[view["name"].str.contains(key, case=False, na=False)]

    st.subheader("ファンド一覧")
    display_cols = [
        col for col in [
            "symbol", "name", "asset_class", "fund_type", "manager", "hedge",
            "settlement", "availability", "analysis_symbol", "analysis_name", "analysis_note", "memo", "source_date"
        ] if col in view.columns
    ]
    st.dataframe(
        view[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "symbol": "管理ID",
            "name": "ファンド名",
            "asset_class": "投資対象",
            "fund_type": "分類",
            "manager": "運用会社",
            "hedge": "為替ヘッジ",
            "settlement": "決算頻度メモ",
            "availability": "取扱状況",
            "memo": "メモ",
            "analysis_symbol": "分析用ティッカー",
            "analysis_name": "分析用プロキシ",
            "analysis_note": "分析上の注意",
            "source_date": "資料基準日",
        },
    )

    st.subheader("分類別件数")
    left, right = st.columns(2)
    with left:
        by_asset = view["asset_class"].value_counts().reset_index()
        by_asset.columns = ["投資対象", "件数"]
        st.dataframe(by_asset, use_container_width=True, hide_index=True)
    with right:
        by_manager = view["manager"].value_counts().reset_index()
        by_manager.columns = ["運用会社", "件数"]
        st.dataframe(by_manager, use_container_width=True, hide_index=True)

    st.info(
        "この一覧は三井住友銀行のNISA成長投資枠対象ファンドをCSV化したものです。"
        "ランキング・トレンドはファンドそのものの基準価額ではなく、分析用プロキシの価格データで計算します。"
        "実際の購入前には、三井住友銀行の最新一覧、目論見書、購入時手数料、信託報酬、信託財産留保額を確認してください。"
    )


@st.cache_data(ttl=1800)
def load_price_data(ticker: str, period_value: str) -> pd.DataFrame:
    """
    yfinanceから価格データを取得する。

    Streamlit Cloud上では、ETF/指数系ティッカーで yf.download が空を返すことがあるため、
    Ticker.history もフォールバックとして試す。
    """
    ticker = str(ticker or "").replace("　", " ").strip().upper()
    if not ticker:
        return pd.DataFrame()

    def normalize_price_df(raw: pd.DataFrame) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame()

        df = raw.copy()

        # yfinanceのバージョンや取得方法により MultiIndex の向きが変わるため吸収する。
        if isinstance(df.columns, pd.MultiIndex):
            level_values = [list(map(str, df.columns.get_level_values(i))) for i in range(df.columns.nlevels)]
            price_cols = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
            if any(col in price_cols for col in level_values[0]):
                df.columns = df.columns.get_level_values(0)
            elif any(col in price_cols for col in level_values[-1]):
                df.columns = df.columns.get_level_values(-1)
            else:
                df.columns = df.columns.get_level_values(0)

        df.columns = [str(col).strip() for col in df.columns]

        if "Close" not in df.columns and "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]

        if "Close" not in df.columns:
            return pd.DataFrame()

        # 指数・一部ETFでOHLC/Volumeが欠けるケースを補完する。
        for col in ["Open", "High", "Low"]:
            if col not in df.columns:
                df[col] = df["Close"]
        if "Volume" not in df.columns:
            df["Volume"] = 0

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        df = df[required_cols].copy()
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["Close"])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df

    attempts: list[pd.DataFrame] = []

    try:
        attempts.append(
            yf.download(
                ticker,
                period=period_value,
                auto_adjust=False,
                progress=False,
                threads=False,
                timeout=20,
            )
        )
    except Exception:
        pass

    try:
        attempts.append(yf.Ticker(ticker).history(period=period_value, auto_adjust=False, timeout=20))
    except Exception:
        pass

    try:
        attempts.append(yf.Ticker(ticker).history(period=period_value, auto_adjust=True, timeout=20))
    except Exception:
        pass

    for raw in attempts:
        df = normalize_price_df(raw)
        if not df.empty:
            return df

    return pd.DataFrame()


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


def split_symbol_candidates(value: str) -> list[str]:
    text = str(value or "").replace("　", " ").strip()
    if not text:
        return []
    for sep in ["、", ",", "/", "|", ";"]:
        text = text.replace(sep, " ")
    candidates = []
    for item in text.split():
        item = item.strip().upper()
        if item and item not in candidates:
            candidates.append(item)
    return candidates


def inferred_proxy_symbols_for_row(row: pd.Series) -> list[str]:
    symbol = str(row.get("symbol", "")).strip().upper()
    text = " ".join(
        str(row.get(col, ""))
        for col in ["name", "theme", "asset_class", "fund_type", "memo", "analysis_name"]
    )

    # SMBCGxxx の管理IDはyfinanceでは取得できないため、CSV側のanalysis_symbolが欠けても動くように推定する。
    if not symbol.startswith("SMBCG"):
        return [symbol] if symbol else []

    rules = [
        (["日経", "225"], ["1321.T", "^N225", "EWJ"]),
        (["TOPIX", "トピックス", "国内株式", "日本株"], ["1306.T", "^TOPX", "EWJ"]),
        (["米国", "S&P", "Ｓ＆Ｐ", "NASDAQ", "ナスダック"], ["2558.T", "1655.T", "SPY", "VOO", "QQQ"]),
        (["全世界", "世界株", "オール", "グローバル株式"], ["2559.T", "VT", "ACWI"]),
        (["先進国"], ["2513.T", "VEA", "EFA"]),
        (["新興国", "エマージング"], ["1658.T", "VWO", "EEM"]),
        (["インド"], ["1678.T", "INDA", "EPI"]),
        (["中国"], ["FXI", "MCHI", "ASHR"]),
        (["ゴールド", "金", "黄金"], ["1540.T", "GLD", "IAU"]),
        (["REIT", "リート", "不動産"], ["1343.T", "VNQ", "IYR"]),
        (["債券", "ボンド"], ["2510.T", "AGG", "BND"]),
        (["バランス", "資産分散"], ["ACWI", "AGG", "VT"]),
    ]
    for keywords, proxies in rules:
        if any(key.lower() in text.lower() for key in keywords):
            return proxies
    return ["ACWI", "VT", "SPY"]


def analysis_candidates_for_row(row: pd.Series) -> list[str]:
    candidates = []
    candidates.extend(split_symbol_candidates(str(row.get("analysis_symbol", ""))))
    candidates.extend(inferred_proxy_symbols_for_row(row))

    # SMBCGxxx は管理IDであり価格取得対象ではない。候補から除外する。
    cleaned = []
    for item in candidates:
        if not item or item.startswith("SMBCG"):
            continue
        if item not in cleaned:
            cleaned.append(item)
    return cleaned or [str(row.get("symbol", "")).strip().upper()]


def analysis_symbol_for_row(row: pd.Series) -> str:
    candidates = analysis_candidates_for_row(row)
    return candidates[0] if candidates else str(row.get("symbol", "")).strip()


def analysis_name_for_row(row: pd.Series) -> str:
    value = str(row.get("analysis_name", "")).strip()
    return value if value else str(row.get("name", "")).strip()


def load_price_data_from_candidates(candidates, period_value: str) -> tuple[pd.DataFrame, str]:
    if isinstance(candidates, str):
        candidate_list = split_symbol_candidates(candidates) or [candidates]
    else:
        candidate_list = []
        for candidate in candidates:
            candidate_list.extend(split_symbol_candidates(str(candidate)))

    cleaned = []
    for candidate in candidate_list:
        candidate = str(candidate).strip().upper()
        if candidate and candidate not in cleaned and not candidate.startswith("SMBCG"):
            cleaned.append(candidate)

    for candidate in cleaned:
        df = load_price_data(candidate, period_value)
        if not df.empty:
            return df, candidate
    return pd.DataFrame(), cleaned[0] if cleaned else ""


def is_proxy_analysis(symbol: str, data_symbol: str) -> bool:
    return str(symbol).strip() != str(data_symbol).strip()


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
    df["Volume_Ratio"] = df["Volume_Ratio"].replace([float("inf"), float("-inf")], pd.NA).fillna(1.0)

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
    data_symbol: str | None = None,
    analysis_name: str = "",
    analysis_note: str = "",
) -> dict:
    requested_data_symbol = data_symbol if data_symbol is not None else symbol
    df, used_data_symbol = load_price_data_from_candidates(requested_data_symbol, period)
    data_symbol = used_data_symbol or str(requested_data_symbol or symbol).strip()

    if df.empty:
        return {
            "symbol": symbol,
            "name": name,
            "theme": theme,
            "analysis_symbol": data_symbol,
            "analysis_name": analysis_name,
            "analysis_note": analysis_note,
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
            "analysis_symbol": data_symbol,
            "analysis_name": analysis_name,
            "analysis_note": analysis_note,
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
        "analysis_symbol": data_symbol,
        "analysis_name": analysis_name,
        "analysis_note": analysis_note,
        "score": result["score"],
        "status": result["status"],
        "unit_price": close,
        "price_band": "プロキシ" if is_proxy_analysis(symbol, data_symbol) else price_band_label(close, symbol),
        "rsi": latest["RSI"],
        "volume_ratio": latest["Volume_Ratio"],
        "macd_diff": latest["MACD_DIFF"],
        "return_5d": latest["Return_5d"],
        "return_20d": latest["Return_20d"],
        "reasons": result["reasons"],
    }




def calculate_buy_score_at(
    df: pd.DataFrame,
    row_index: int,
    ma_short: int,
    ma_mid: int,
    ma_long: int,
) -> dict:
    """バックテスト用に、指定行時点のスコアを計算する。未来データは見ない。"""
    labels = ma_labels(ma_short, ma_mid, ma_long)
    required = [labels["short"], labels["mid"], labels["long"], "RSI", "MACD_DIFF", "Volume_Ratio", "Return_5d", "Return_20d"]

    if row_index < 1 or row_index >= len(df):
        return {"score": 0, "status": "データ不足"}

    latest = df.iloc[row_index]
    prev = df.iloc[row_index - 1]
    if latest[required].isna().any() or pd.isna(prev["MACD_DIFF"]):
        return {"score": 0, "status": "データ不足"}

    score = 0
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
    if ma_s > ma_m:
        score += 2
    if ma_m > ma_l:
        score += 2
    if 40 <= rsi <= 65:
        score += 2
    elif rsi > 75:
        score -= 2
    elif rsi < 30:
        score += 1
    if macd_diff > 0:
        score += 2
    if prev_macd_diff < 0 and macd_diff > 0:
        score += 2
    if volume_ratio >= 1.5:
        score += 2
    elif volume_ratio >= 1.2:
        score += 1
    if return_5d > 0:
        score += 1
    if return_20d > 0:
        score += 1

    score = max(0, min(int(score), 12))
    if score >= 9:
        status = "強い買い候補"
    elif score >= 6:
        status = "買い候補"
    elif score >= 3:
        status = "様子見"
    else:
        status = "弱い / 見送り"
    return {"score": score, "status": status}


@st.cache_data(ttl=1800)

@st.cache_data(ttl=1800)
def run_symbol_backtest_cached(
    symbol: str,
    name: str,
    theme: str,
    data_candidates_tuple: tuple,
    period_value: str,
    ma_short: int,
    ma_mid: int,
    ma_long: int,
    entry_score: int,
    max_hold_days: int,
    no_overlap: bool,
    exit_rule: str,
    exit_score: int,
    trailing_stop_pct: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    analysis_name: str = "",
) -> tuple[pd.DataFrame, dict]:
    """1銘柄分のバックテストを実行する。"""
    df, used_data_symbol = load_price_data_from_candidates(list(data_candidates_tuple), period_value)
    data_symbol = used_data_symbol or (data_candidates_tuple[0] if data_candidates_tuple else symbol)

    if df.empty:
        return pd.DataFrame(), {
            "symbol": symbol,
            "name": name,
            "theme": theme,
            "analysis_symbol": data_symbol,
            "analysis_name": analysis_name,
            "trades": 0,
            "status": "取得失敗",
        }

    df = add_indicators(df, ma_short, ma_mid, ma_long)
    df = df.dropna(subset=["Open", "Close"]).copy()

    labels = ma_labels(ma_short, ma_mid, ma_long)
    trend_required = [labels["short"], labels["mid"]]

    def decide_exit(j: int, entry_price: float, highest_close: float) -> str | None:
        close = float(df.iloc[j]["Close"])
        signal = calculate_buy_score_at(df, j, ma_short, ma_mid, ma_long)
        reasons = []

        if exit_rule in ["スコア悪化", "複合"]:
            if signal["score"] <= exit_score:
                reasons.append(f"スコア悪化<= {exit_score}")

        if exit_rule in ["トレンド崩れ", "複合"]:
            row = df.iloc[j]
            if not row[trend_required].isna().any():
                if close < float(row[labels["mid"]]):
                    reasons.append(f"終値<{labels['mid']}")
                if float(row[labels["short"]]) < float(row[labels["mid"]]):
                    reasons.append(f"{labels['short']}<{labels['mid']}")

        if exit_rule in ["トレーリングストップ", "複合"]:
            if highest_close > 0 and close <= highest_close * (1 - trailing_stop_pct / 100):
                reasons.append(f"高値から-{trailing_stop_pct:.1f}%")

        if exit_rule in ["利確/損切り", "複合"]:
            if entry_price > 0:
                ret_pct = (close / entry_price - 1) * 100
                if stop_loss_pct > 0 and ret_pct <= -stop_loss_pct:
                    reasons.append(f"損切り-{stop_loss_pct:.1f}%")
                if take_profit_pct > 0 and ret_pct >= take_profit_pct:
                    reasons.append(f"利確+{take_profit_pct:.1f}%")

        return " / ".join(reasons) if reasons else None

    trades = []
    i = 1
    last_signal_index = len(df) - 2

    while i <= last_signal_index:
        signal = calculate_buy_score_at(df, i, ma_short, ma_mid, ma_long)
        if signal["score"] < entry_score:
            i += 1
            continue

        entry_idx = i + 1
        if entry_idx >= len(df):
            break

        entry_price = float(df.iloc[entry_idx]["Open"])
        if entry_price <= 0 or pd.isna(entry_price):
            i += 1
            continue

        exit_limit = min(entry_idx + int(max_hold_days), len(df) - 1)
        if exit_limit <= entry_idx:
            break

        exit_idx = exit_limit
        exit_reason = "最大保有日数"
        highest_close = float(df.iloc[entry_idx]["Close"])

        if exit_rule != "最大保有日数のみ":
            for j in range(entry_idx + 1, exit_limit + 1):
                close_j = float(df.iloc[j]["Close"])
                if close_j > highest_close:
                    highest_close = close_j
                reason = decide_exit(j, entry_price, highest_close)
                if reason:
                    exit_idx = j
                    exit_reason = reason
                    break

        exit_price = float(df.iloc[exit_idx]["Close"])
        if exit_price > 0:
            return_pct = (exit_price / entry_price - 1) * 100
            realized_hold_days = int(exit_idx - entry_idx)
            trades.append({
                "symbol": symbol,
                "name": name,
                "theme": theme,
                "analysis_symbol": data_symbol,
                "analysis_name": analysis_name,
                "signal_date": df.index[i],
                "entry_date": df.index[entry_idx],
                "exit_date": df.index[exit_idx],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "max_hold_days": int(max_hold_days),
                "hold_days": realized_hold_days,
                "exit_reason": exit_reason,
                "score": signal["score"],
                "status": signal["status"],
                "return_pct": return_pct,
                "win": return_pct > 0,
            })

        i = exit_idx if no_overlap else i + 1

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        returns = trades_df["return_pct"] / 100
        compounded = (1 + returns).prod() - 1
        summary = {
            "symbol": symbol,
            "name": name,
            "theme": theme,
            "analysis_symbol": data_symbol,
            "analysis_name": analysis_name,
            "trades": int(len(trades_df)),
            "win_rate": float(trades_df["win"].mean() * 100),
            "avg_return_pct": float(trades_df["return_pct"].mean()),
            "median_return_pct": float(trades_df["return_pct"].median()),
            "best_return_pct": float(trades_df["return_pct"].max()),
            "worst_return_pct": float(trades_df["return_pct"].min()),
            "avg_hold_days": float(trades_df["hold_days"].mean()),
            "compounded_return_pct": float(compounded * 100),
            "status": "OK",
        }
    else:
        summary = {
            "symbol": symbol,
            "name": name,
            "theme": theme,
            "analysis_symbol": data_symbol,
            "analysis_name": analysis_name,
            "trades": 0,
            "win_rate": None,
            "avg_return_pct": None,
            "median_return_pct": None,
            "best_return_pct": None,
            "worst_return_pct": None,
            "avg_hold_days": None,
            "compounded_return_pct": None,
            "status": "該当シグナルなし",
        }

    if len(df) >= 2:
        first_close = float(df.iloc[0]["Close"])
        last_close = float(df.iloc[-1]["Close"])
        summary["buy_hold_return_pct"] = (last_close / first_close - 1) * 100 if first_close > 0 else None
    else:
        summary["buy_hold_return_pct"] = None
    return trades_df, summary


def show_backtest(watchlist: pd.DataFrame, is_fund_file: bool) -> None:
    st.header("バックテスト")
    st.caption(
        "過去の各営業日時点で現在と同じスコア判定を行い、条件を満たした翌営業日の始値で買います。"
        "売却は固定日数だけでなく、スコア悪化・トレンド崩れ・トレーリングストップなどで判定できます。"
    )

    with st.expander("バックテスト条件", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            period_value = st.selectbox("検証期間", ["1y", "2y", "5y", "10y"], index=2)
        with c2:
            entry_score = st.slider("買いシグナルの最低スコア", 3, 12, 9, 1)
        with c3:
            max_hold_days = st.slider(
                "最大保有営業日数",
                min_value=5,
                max_value=1250,
                value=250,
                step=5,
                help="約250営業日=約1年、約1250営業日=約5年です。売却条件に該当しない場合の最終期限として使います。",
            )
        with c4:
            no_overlap = st.checkbox("同一銘柄の重複保有をしない", value=True)

        e1, e2, e3, e4 = st.columns(4)
        with e1:
            exit_rule = st.selectbox(
                "売却ルール",
                ["複合", "スコア悪化", "トレンド崩れ", "トレーリングストップ", "利確/損切り", "最大保有日数のみ"],
                index=0,
                help="おすすめは『複合』です。トレンドが続く間は保有し、崩れたら売却します。",
            )
        with e2:
            exit_score = st.slider("売却スコア閾値", 0, 8, 4, 1, help="スコアがこの値以下になったら売却候補にします。")
        with e3:
            trailing_stop_pct = st.slider("トレーリング損切り%", 1.0, 40.0, 12.0, 0.5)
        with e4:
            stop_loss_pct = st.slider("固定損切り%", 0.0, 40.0, 10.0, 0.5, help="0%にすると無効です。")

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            take_profit_pct = st.slider("固定利確%", 0.0, 200.0, 0.0, 1.0, help="0%にすると無効です。長期トレンド検証では無効推奨です。")
        with p2:
            bt_ma_short = st.number_input("短期MA", min_value=1, max_value=100, value=int(st.session_state.get("ma_short", 5)), step=1)
        with p3:
            bt_ma_mid = st.number_input("中期MA", min_value=2, max_value=250, value=int(st.session_state.get("ma_mid", 25)), step=1)
        with p4:
            bt_ma_long = st.number_input("長期MA", min_value=3, max_value=500, value=int(st.session_state.get("ma_long", 75)), step=1)

        m1, m2 = st.columns(2)
        with m1:
            max_symbols = st.selectbox("検証対象数", ["先頭30件", "先頭100件", "すべて"], index=0 if not is_fund_file else 1)
        with m2:
            st.caption("推奨初期値：買いScore 9以上 / 最大250〜1250営業日 / 売却ルール=複合 / 利確0%。")

    if not validate_ma_values(int(bt_ma_short), int(bt_ma_mid), int(bt_ma_long)):
        st.stop()

    if max_symbols == "先頭30件":
        target_watchlist = watchlist.head(30)
    elif max_symbols == "先頭100件":
        target_watchlist = watchlist.head(100)
    else:
        target_watchlist = watchlist

    if len(target_watchlist) < len(watchlist):
        st.info(f"処理負荷を抑えるため、現在は{len(target_watchlist)}銘柄で検証しています。")

    if st.button("バックテストを実行", type="primary"):
        all_trades = []
        summaries = []
        progress = st.progress(0)
        for idx, (_, row) in enumerate(target_watchlist.iterrows(), start=1):
            candidates = tuple(analysis_candidates_for_row(row))
            trades_df, summary = run_symbol_backtest_cached(
                str(row["symbol"]),
                str(row["name"]),
                str(row["theme"]),
                candidates,
                period_value,
                int(bt_ma_short),
                int(bt_ma_mid),
                int(bt_ma_long),
                int(entry_score),
                int(max_hold_days),
                bool(no_overlap),
                str(exit_rule),
                int(exit_score),
                float(trailing_stop_pct),
                float(stop_loss_pct),
                float(take_profit_pct),
                analysis_name_for_row(row),
            )
            if not trades_df.empty:
                all_trades.append(trades_df)
            summaries.append(summary)
            progress.progress(idx / len(target_watchlist))
        progress.empty()
        st.session_state["backtest_summary_df"] = pd.DataFrame(summaries)
        st.session_state["backtest_trades_df"] = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        st.session_state["backtest_params"] = {
            "period": period_value,
            "entry_score": entry_score,
            "max_hold_days": max_hold_days,
            "exit_rule": exit_rule,
            "exit_score": exit_score,
            "trailing_stop_pct": trailing_stop_pct,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "ma": f"MA{int(bt_ma_short)}/MA{int(bt_ma_mid)}/MA{int(bt_ma_long)}",
            "no_overlap": no_overlap,
            "symbols": len(target_watchlist),
        }

    summary_df = st.session_state.get("backtest_summary_df", pd.DataFrame())
    trades_df = st.session_state.get("backtest_trades_df", pd.DataFrame())
    params = st.session_state.get("backtest_params", {})
    if summary_df.empty:
        st.info("条件を設定して『バックテストを実行』を押してください。")
        return

    valid = summary_df[summary_df["trades"] > 0].copy()
    st.subheader("検証条件")
    st.write(
        f"期間: {params.get('period')} / 買い: Score {params.get('entry_score')}以上 / "
        f"最大保有: {params.get('max_hold_days')}営業日 / 売却: {params.get('exit_rule')} / "
        f"売却Score: {params.get('exit_score')}以下 / トレーリング: {params.get('trailing_stop_pct')}% / "
        f"損切り: {params.get('stop_loss_pct')}% / 利確: {params.get('take_profit_pct')}% / "
        f"MA: {params.get('ma')} / 対象: {params.get('symbols')}件"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("検証銘柄数", len(summary_df))
    c2.metric("シグナル発生銘柄", len(valid))
    c3.metric("総トレード数", int(valid["trades"].sum()) if not valid.empty else 0)
    c4.metric("全体勝率", f"{trades_df['win'].mean() * 100:.1f}%" if not trades_df.empty else "-")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("平均リターン/回", f"{trades_df['return_pct'].mean():.2f}%" if not trades_df.empty else "-")
    c6.metric("中央値リターン/回", f"{trades_df['return_pct'].median():.2f}%" if not trades_df.empty else "-")
    c7.metric("最低リターン/回", f"{trades_df['return_pct'].min():.2f}%" if not trades_df.empty else "-")
    c8.metric("平均保有日数", f"{trades_df['hold_days'].mean():.1f}日" if not trades_df.empty else "-")

    st.subheader("銘柄別サマリー")
    display_summary = summary_df.copy()
    for col in ["win_rate", "avg_return_pct", "median_return_pct", "best_return_pct", "worst_return_pct", "avg_hold_days", "compounded_return_pct", "buy_hold_return_pct"]:
        if col in display_summary.columns:
            display_summary[col] = display_summary[col].map(lambda x: None if pd.isna(x) else round(float(x), 2))
    display_summary = display_summary.sort_values(["avg_return_pct", "win_rate", "trades"], ascending=[False, False, False], na_position="last")
    summary_cols = [
        "symbol", "name", "theme",
        *( ["analysis_symbol", "analysis_name"] if is_fund_file else [] ),
        "trades", "win_rate", "avg_return_pct", "median_return_pct", "best_return_pct", "worst_return_pct",
        "avg_hold_days", "compounded_return_pct", "buy_hold_return_pct", "status",
    ]
    summary_cols = [col for col in summary_cols if col in display_summary.columns]
    st.dataframe(
        display_summary[summary_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "symbol": "管理ID" if is_fund_file else "銘柄コード",
            "name": "ファンド名" if is_fund_file else "銘柄名",
            "theme": "テーマ",
            "analysis_symbol": "分析用ティッカー",
            "analysis_name": "分析用プロキシ",
            "trades": "取引数",
            "win_rate": "勝率%",
            "avg_return_pct": "平均%",
            "median_return_pct": "中央値%",
            "best_return_pct": "最大%",
            "worst_return_pct": "最小%",
            "avg_hold_days": "平均保有日数",
            "compounded_return_pct": "単純複利%",
            "buy_hold_return_pct": "期間保有%",
            "status": "状態",
        },
    )

    if not trades_df.empty:
        st.subheader("売却理由")
        reason_counts = trades_df["exit_reason"].value_counts().reset_index()
        reason_counts.columns = ["売却理由", "件数"]
        st.dataframe(reason_counts, use_container_width=True, hide_index=True)

        fig_reason = go.Figure()
        fig_reason.add_trace(go.Bar(x=reason_counts["売却理由"], y=reason_counts["件数"], name="売却理由"))
        fig_reason.update_layout(height=320, margin=dict(l=10, r=10, t=24, b=10), xaxis_title="売却理由", yaxis_title="件数")
        st.plotly_chart(fig_reason, use_container_width=True)

        st.subheader("トレード明細")
        detail = trades_df.sort_values("entry_date", ascending=False).copy()
        for col in ["entry_date", "exit_date", "signal_date"]:
            detail[col] = pd.to_datetime(detail[col]).dt.strftime("%Y-%m-%d")
        detail["entry_price"] = detail["entry_price"].round(2)
        detail["exit_price"] = detail["exit_price"].round(2)
        detail["return_pct"] = detail["return_pct"].round(2)
        detail_cols = [
            "signal_date", "entry_date", "exit_date", "symbol", "name",
            *( ["analysis_symbol"] if is_fund_file else [] ),
            "score", "status", "entry_price", "exit_price", "hold_days", "exit_reason", "return_pct", "win",
        ]
        detail_cols = [col for col in detail_cols if col in detail.columns]
        st.dataframe(detail[detail_cols], use_container_width=True, hide_index=True)

        st.subheader("リターン分布")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=trades_df["return_pct"], nbinsx=30, name="取引リターン"))
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=24, b=10), xaxis_title="リターン%", yaxis_title="件数")
        st.plotly_chart(fig, use_container_width=True)

        csv = detail.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("トレード明細CSVをダウンロード", data=csv, file_name="backtest_trades.csv", mime="text/csv")

    st.warning(
        "このバックテストは、現在のスコア条件が過去に出た場合の機械的な検証です。"
        "手数料・税金・スリッページ・分配金・為替ヘッジ差・投資信託本体の基準価額差は反映していません。"
        "特に投資信託CSVでは、analysis_symbolのプロキシETF/指数による近似です。"
    )

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
st.sidebar.caption("app version: backtest-20260501")

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
    selected_path_obj = Path(selected_watchlist_path)
    selected_mtime = selected_path_obj.stat().st_mtime if selected_path_obj.exists() else 0.0
    watchlist = load_watchlist_from_path(selected_watchlist_path, selected_mtime)
    active_watchlist_label = selected_path_obj.name

st.sidebar.caption(f"読込中：{active_watchlist_label} / {len(watchlist)}銘柄")

is_fund_file = is_fund_catalog(watchlist)

if "mode" not in st.session_state:
    st.session_state["mode"] = "ファンド一覧" if is_fund_file else "ランキング"

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

mode_options = ["ランキング", "バックテスト", "スコア履歴", "個別銘柄", "ファンド一覧", "用語説明"] if is_fund_file else ["ランキング", "バックテスト", "スコア履歴", "個別銘柄", "用語説明"]
if st.session_state["mode"] not in mode_options:
    st.session_state["mode"] = mode_options[0]
mode_index = mode_options.index(st.session_state["mode"])

selected_mode = st.sidebar.radio("表示モード", mode_options, index=mode_index)

if selected_mode != st.session_state["mode"]:
    st.session_state["mode"] = selected_mode
    st.rerun()

available_themes = ["すべて"] + sorted(watchlist["theme"].dropna().unique().tolist())
selected_theme = st.sidebar.selectbox("テーマ絞り込み", available_themes)

if is_fund_file:
    show_under_10000_only = False
    analysis_limit_label = st.sidebar.selectbox(
        "ランキング分析対象",
        ["先頭100件", "先頭300件", "すべて"],
        index=2,
        help="投資信託CSVでは、analysis_symbol列のプロキシを使って分析します。",
    )
    st.sidebar.divider()
    st.sidebar.caption("投資信託CSVは analysis_symbol のプロキシETF/指数でスコア分析します。")
else:
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


if is_fund_file and st.session_state["mode"] == "ファンド一覧":
    show_fund_catalog(filtered_watchlist)
    st.stop()


if st.session_state["mode"] == "用語説明":
    show_beginner_guide()
    st.stop()


if st.session_state["mode"] == "スコア履歴":
    show_score_history(watchlist)
    st.stop()


if st.session_state["mode"] == "バックテスト":
    show_backtest(filtered_watchlist, is_fund_file)
    st.stop()


if st.session_state["mode"] == "ランキング":
    st.header("買い候補ランキング")
    if is_fund_file:
        st.caption("ランキングは標準設定の 1年 / MA5・MA25・MA75 で判定します。投資信託CSVでは、各ファンドの analysis_symbol プロキシで計算します。")
    else:
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
        data_symbol = analysis_candidates_for_row(row)
        result = analyze_symbol(
            row["symbol"],
            row["name"],
            row["theme"],
            period,
            ma_short,
            ma_mid,
            ma_long,
            data_symbol=data_symbol,
            analysis_name=analysis_name_for_row(row),
            analysis_note=str(row.get("analysis_note", "")).strip(),
        )
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

    price_label = "プロキシ価格" if is_fund_file else "株価単価"
    unit_label = "概算単元金額"
    display_df[price_label] = display_df.apply(lambda row: format_price(row["unit_price"], row["analysis_symbol"]), axis=1)
    display_df[unit_label] = display_df.apply(
        lambda row: format_unit_amount(row["unit_price"], row["analysis_symbol"]),
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
                *(["analysis_symbol", "analysis_name"] if is_fund_file else []),
                "score",
                "status",
                price_label,
                "price_band",
                unit_label,
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
            "symbol": "管理ID" if is_fund_file else "銘柄コード",
            "name": "ファンド名" if is_fund_file else "銘柄名",
            "theme": "テーマ",
            "analysis_symbol": "分析用ティッカー",
            "analysis_name": "分析用プロキシ",
            "score": "スコア",
            "status": "判定",
            price_label: price_label,
            unit_label: unit_label,
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
    data_candidates = analysis_candidates_for_row(selected_row)
    data_symbol = data_candidates[0] if data_candidates else analysis_symbol_for_row(selected_row)
    analysis_name = analysis_name_for_row(selected_row)
    analysis_note = str(selected_row.get("analysis_note", "")).strip()

    period = st.session_state["detail_period"]
    ma_short = int(st.session_state["ma_short"])
    ma_mid = int(st.session_state["ma_mid"])
    ma_long = int(st.session_state["ma_long"])

    df, used_data_symbol = load_price_data_from_candidates(data_candidates, period)
    data_symbol = used_data_symbol or data_symbol

    if df.empty:
        if is_proxy_analysis(symbol, data_symbol):
            st.error(f"分析用プロキシ {data_symbol} のデータを取得できませんでした。CSVの analysis_symbol を確認してください。")
        else:
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
    if is_proxy_analysis(symbol, data_symbol):
        st.caption(f"分析用プロキシ：{data_symbol} / {analysis_name}")
        if analysis_note:
            st.caption(f"分析メモ：{analysis_note}")
    if memo:
        st.caption(memo)

    st.subheader("価格トレンド" if is_proxy_analysis(symbol, data_symbol) else "株価トレンド")
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
