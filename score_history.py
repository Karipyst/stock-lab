from pathlib import Path
from datetime import datetime

import pandas as pd
import ta
import yfinance as yf

HISTORY_PATH = Path("score_history.csv")
PERIOD = "1y"
MA_SHORT = 5
MA_MID = 25
MA_LONG = 75


def find_watchlist_files() -> list[Path]:
    return sorted(Path(".").glob("watchlist*.csv"))


def normalize_watchlist(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).replace("\ufeff", "").strip() for col in df.columns]
    if not {"symbol", "name"}.issubset(df.columns):
        raise ValueError("CSVには symbol,name 列が必要です。")
    if "theme" not in df.columns:
        df["theme"] = "未分類"
    if "memo" not in df.columns:
        df["memo"] = ""
    optional_cols = ["analysis_symbol", "analysis_name", "analysis_note", "asset_class", "fund_type"]
    for col in optional_cols:
        if col not in df.columns:
            df[col] = ""
    for col in ["symbol", "name", "theme", "memo"] + optional_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()
    df = df[df["symbol"] != ""].drop_duplicates(subset=["symbol"], keep="first")
    return df.reset_index(drop=True)


def load_price_data(ticker: str, period_value: str) -> pd.DataFrame:
    ticker = str(ticker or "").replace("　", " ").strip().upper()
    if not ticker:
        return pd.DataFrame()

    def normalize_price_df(raw: pd.DataFrame) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame()
        df = raw.copy()
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

    attempts = []
    try:
        attempts.append(yf.download(ticker, period=period_value, auto_adjust=False, progress=False, threads=False, timeout=20))
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


def ma_labels(ma_short: int, ma_mid: int, ma_long: int) -> dict[str, str]:
    return {"short": f"MA{ma_short}", "mid": f"MA{ma_mid}", "long": f"MA{ma_long}"}


def add_indicators(df: pd.DataFrame, ma_short: int, ma_mid: int, ma_long: int) -> pd.DataFrame:
    df = df.copy()
    labels = ma_labels(ma_short, ma_mid, ma_long)
    df[labels["short"]] = df["Close"].rolling(ma_short).mean()
    df[labels["mid"]] = df["Close"].rolling(ma_mid).mean()
    df[labels["long"]] = df["Close"].rolling(ma_long).mean()
    df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
    macd = ta.trend.MACD(close=df["Close"])
    df["MACD_DIFF"] = macd.macd_diff()
    df["Volume_MA20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA20"]
    df["Volume_Ratio"] = df["Volume_Ratio"].replace([float("inf"), float("-inf")], pd.NA).fillna(1.0)
    df["Return_5d"] = df["Close"].pct_change(5) * 100
    df["Return_20d"] = df["Close"].pct_change(20) * 100
    return df




def build_score_result_from_rows(latest: pd.Series, prev: pd.Series, labels: dict, include_reasons: bool = False, ma_mid: int | None = None, ma_short: int | None = None, ma_long: int | None = None) -> dict:
    """ランキングアプリ側と同じ買いスコア計算。scoreは0〜12、raw_scoreは丸め前。"""
    raw_score = 0
    reasons: list[str] = []

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

    close_gt_ma_mid = bool(close > ma_m)
    ma_short_gt_mid = bool(ma_s > ma_m)
    ma_mid_gt_long = bool(ma_m > ma_l)
    rsi_good = bool(40 <= rsi <= 65)
    rsi_overheat = bool(rsi > 75)
    rsi_oversold = bool(rsi < 30)
    macd_positive = bool(macd_diff > 0)
    macd_cross_up = bool(prev_macd_diff < 0 and macd_diff > 0)
    volume_15x = bool(volume_ratio >= 1.5)
    volume_12x = bool((volume_ratio >= 1.2) and not volume_15x)
    return_5d_positive = bool(return_5d > 0)
    return_20d_positive = bool(return_20d > 0)

    if close_gt_ma_mid:
        raw_score += 2
        if include_reasons:
            reasons.append(f"終値が{ma_mid}日移動平均線を上回っています。")
    elif include_reasons:
        reasons.append(f"終値が{ma_mid}日移動平均線を下回っています。")

    if ma_short_gt_mid:
        raw_score += 2
        if include_reasons:
            reasons.append(f"{ma_short}日移動平均線が{ma_mid}日移動平均線を上回っています。")

    if ma_mid_gt_long:
        raw_score += 2
        if include_reasons:
            reasons.append(f"{ma_mid}日移動平均線が{ma_long}日移動平均線を上回っています。")

    if rsi_good:
        raw_score += 2
        if include_reasons:
            reasons.append("RSIが40〜65の範囲です。")
    elif rsi_overheat:
        raw_score -= 2
        if include_reasons:
            reasons.append("RSIが75超で過熱気味です。")
    elif rsi_oversold:
        raw_score += 1
        if include_reasons:
            reasons.append("RSIが30未満で売られすぎ水準です。")

    if macd_positive:
        raw_score += 2
        if include_reasons:
            reasons.append("MACDがプラスです。")

    if macd_cross_up:
        raw_score += 2
        if include_reasons:
            reasons.append("MACDが直近で陽転しています。")

    if volume_15x:
        raw_score += 2
        if include_reasons:
            reasons.append("出来高が20日平均の1.5倍以上です。")
    elif volume_12x:
        raw_score += 1
        if include_reasons:
            reasons.append("出来高が20日平均の1.2倍以上です。")

    if return_5d_positive:
        raw_score += 1
        if include_reasons:
            reasons.append("直近5営業日のリターンがプラスです。")

    if return_20d_positive:
        raw_score += 1
        if include_reasons:
            reasons.append("直近20営業日のリターンがプラスです。")

    score = max(0, min(int(raw_score), 12))
    if score >= 9:
        status = "強い買い候補"
    elif score >= 6:
        status = "買い候補"
    elif score >= 3:
        status = "様子見"
    else:
        status = "弱い / 見送り"

    ma_deviation_pct = None
    try:
        ma_deviation_pct = (float(close) / float(ma_m) - 1) * 100 if float(ma_m) > 0 else None
    except Exception:
        ma_deviation_pct = None

    return {
        "score": score,
        "raw_score": int(raw_score),
        "status": status,
        "reasons": reasons,
        "latest": latest,
        "signal_close": float(close),
        "signal_rsi": float(rsi),
        "signal_volume_ratio": float(volume_ratio),
        "signal_return_5d": float(return_5d),
        "signal_return_20d": float(return_20d),
        "signal_ma_deviation_pct": ma_deviation_pct,
        "score_close_gt_ma_mid": close_gt_ma_mid,
        "score_ma_short_gt_mid": ma_short_gt_mid,
        "score_ma_mid_gt_long": ma_mid_gt_long,
        "score_rsi_good": rsi_good,
        "score_rsi_overheat": rsi_overheat,
        "score_rsi_oversold": rsi_oversold,
        "score_macd_positive": macd_positive,
        "score_macd_cross_up": macd_cross_up,
        "score_volume_15x": volume_15x,
        "score_volume_12x": volume_12x,
        "score_return_5d_positive": return_5d_positive,
        "score_return_20d_positive": return_20d_positive,
    }


def _safe_float(value, default: float | None = None) -> float | None:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def is_strictly_decreasing(values: list[float]) -> bool:
    if len(values) < 2:
        return False
    cleaned = []
    for v in values:
        if pd.isna(v):
            return False
        cleaned.append(float(v))
    return all(cleaned[i] < cleaned[i - 1] for i in range(1, len(cleaned)))


def pct_change(current: float, base: float) -> float:
    if base == 0 or pd.isna(base) or pd.isna(current):
        return 0.0
    return (float(current) / float(base) - 1) * 100


def _aa_buy_timing_score(score_result: dict) -> tuple[int, str]:
    """AA/AB系バックテスト条件を100点換算した買いタイミング評価。"""
    score = int(score_result.get("score", 0) or 0)
    raw = int(score_result.get("raw_score", score) or 0)
    ma_dev = _safe_float(score_result.get("signal_ma_deviation_pct"), None)
    rsi = _safe_float(score_result.get("signal_rsi"), None)
    ret5 = _safe_float(score_result.get("signal_return_5d"), None)
    vol = _safe_float(score_result.get("signal_volume_ratio"), None)

    points = 0.0
    notes: list[str] = []

    # 25点: Raw Score。通常scoreは12で丸められるため、丸め前Rawを重視。
    if raw == 14:
        points += 25
        notes.append("Raw14")
    elif raw in {13, 16}:
        points += 21
        notes.append(f"Raw{raw}")
    elif raw == 15:
        points += 18
        notes.append("Raw15")
    elif raw == 12:
        points += 10
    elif raw >= 11:
        points += 6

    # 25点: MA25乖離。AA以降は1.5〜2.5%が強い。
    if ma_dev is not None:
        if 1.5 <= ma_dev <= 2.5:
            points += 25
            notes.append("MA乖離◎")
        elif 1.0 <= ma_dev <= 3.0:
            points += 18
            notes.append("MA乖離○")
        elif 0.5 <= ma_dev <= 4.0:
            points += 10
        elif 0.0 <= ma_dev <= 5.0:
            points += 5

    # 20点: 出来高。1.5〜2.0倍を最重視。
    if vol is not None:
        if 1.5 <= vol <= 2.0:
            points += 20
            notes.append("出来高◎")
        elif 2.0 < vol <= 2.5:
            points += 12
            notes.append("出来高やや過熱")
        elif 1.2 <= vol < 1.5:
            points += 7
        elif 1.0 <= vol < 1.2:
            points += 2

    # 15点: RSI。AAは55〜60、ABは55〜63が良好。
    if rsi is not None:
        if 55 <= rsi <= 60:
            points += 15
            notes.append("RSI◎")
        elif 60 < rsi <= 63:
            points += 12
            notes.append("RSI○")
        elif 50 <= rsi < 55 or 63 < rsi <= 70:
            points += 6
        elif 40 <= rsi < 50:
            points += 2

    # 15点: 5日騰落。0〜3%を最重視。
    if ret5 is not None:
        if 0 <= ret5 <= 3:
            points += 15
            notes.append("5日騰落◎")
        elif -1 <= ret5 < 0 or 3 < ret5 <= 5:
            points += 7
        elif -3 <= ret5 < -1:
            points += 2

    if score < 11:
        points *= 0.65
    if raw < 13:
        points *= 0.70

    buy_score = int(round(max(0, min(100, points))))
    if buy_score >= 85:
        label = "強い買い"
    elif buy_score >= 70:
        label = "買い候補"
    elif buy_score >= 50:
        label = "監視"
    else:
        label = "見送り"
    if notes:
        label = f"{label}（{', '.join(notes[:3])}）"
    return buy_score, label


def _aa_sell_timing_score(df: pd.DataFrame, score_result: dict, ma_short: int, ma_mid: int, ma_long: int) -> tuple[int, str]:
    """現在値ベースの売りタイミング評価。ランキングアプリ側と同じロジック。"""
    if df is None or df.empty or len(df) < 6:
        return 0, "判定不可"

    labels = ma_labels(ma_short, ma_mid, ma_long)
    required = ["Close", "Open", "Low", labels["short"], labels["mid"], "MACD_DIFF", "Return_5d", "RSI", "Volume_Ratio"]
    if any(col not in df.columns for col in required):
        return 0, "判定不可"

    rows4 = df.tail(4)
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    if rows4[[labels["short"], "MACD_DIFF", "Return_5d", "Close"]].isna().any().any():
        return 0, "判定不可"

    close = _safe_float(latest.get("Close"), 0.0) or 0.0
    open_ = _safe_float(latest.get("Open"), 0.0) or 0.0
    low_prev = _safe_float(prev.get("Low"), None)
    ma_s = _safe_float(latest.get(labels["short"]), None)
    ma_m = _safe_float(latest.get(labels["mid"]), None)
    ma_s_prev = _safe_float(prev.get(labels["short"]), None)
    ret5 = _safe_float(latest.get("Return_5d"), None)
    rsi = _safe_float(latest.get("RSI"), None)
    vol = _safe_float(latest.get("Volume_Ratio"), None)
    raw = int(score_result.get("raw_score", score_result.get("score", 0)) or 0)

    macd_values = [_safe_float(v, None) for v in rows4["MACD_DIFF"].tolist()]
    macd_values_ok = all(v is not None for v in macd_values)

    short_ma_break = bool(ma_s is not None and close < ma_s)
    short_ma_slope_down = bool(ma_s is not None and ma_s_prev is not None and ma_s < ma_s_prev)
    macd_down = bool(macd_values_ok and is_strictly_decreasing([float(v) for v in macd_values]))
    return5_negative = bool(ret5 is not None and ret5 < 0)
    short_momentum_exit = bool(short_ma_break and short_ma_slope_down and macd_down and return5_negative)

    points = 0.0
    notes: list[str] = []

    # 45点: バックテストで最も機能した売却シグナル。
    if short_momentum_exit:
        points += 45
        notes.append("短期モメンタム悪化")
    else:
        if short_ma_break:
            points += 14
            notes.append("終値<MA5")
        if short_ma_slope_down:
            points += 8
        if macd_down:
            points += 10
        if return5_negative:
            points += 6

    # 15点: 中期MA割れは本命戦略ではOFFだが、警戒材料として残す。
    if ma_m is not None:
        if close < ma_m:
            points += 10
            notes.append("終値<MA25")
        elif close < ma_m * 1.01:
            points += 4

    # 15点: 出来高急増陰線・翌日安値割れ系の代替評価。
    daily_ret = pct_change(close, open_) if open_ else 0.0
    volume_down = bool(close < open_ and daily_ret <= -3.0 and vol is not None and vol >= 1.8)
    next_day_break_like = bool(low_prev is not None and close < low_prev and vol is not None and vol >= 1.5 and return5_negative)
    if volume_down or next_day_break_like:
        points += 15
        notes.append("出来高陰線警戒")

    # 15点: 過熱後の失速。
    if rsi is not None:
        if rsi >= 75 and return5_negative:
            points += 12
            notes.append("過熱失速")
        elif rsi >= 70 and return5_negative:
            points += 8
        elif rsi >= 75:
            points += 5

    # 10点: Raw Score低下は補助警戒。
    if raw <= 8 and short_ma_break:
        points += 10
        notes.append("Raw低下")
    elif raw <= 10 and short_ma_break:
        points += 5

    sell_score = int(round(max(0, min(100, points))))
    if sell_score >= 80:
        label = "強い売り"
    elif sell_score >= 60:
        label = "売り候補"
    elif sell_score >= 40:
        label = "警戒"
    elif sell_score >= 20:
        label = "軽警戒"
    else:
        label = "売り急がない"
    if notes:
        label = f"{label}（{', '.join(notes[:3])}）"
    return sell_score, label


def calculate_buy_score(df: pd.DataFrame, ma_short: int, ma_mid: int, ma_long: int) -> dict:
    labels = ma_labels(ma_short, ma_mid, ma_long)
    required = [labels["short"], labels["mid"], labels["long"], "RSI", "MACD_DIFF", "Volume_Ratio", "Return_5d", "Return_20d"]
    clean = df.dropna(subset=required)
    if clean.empty or len(clean) < 2:
        return {"score": 0, "raw_score": 0, "status": "データ不足", "latest": None}
    latest = clean.iloc[-1]
    prev = clean.iloc[-2]
    return build_score_result_from_rows(latest, prev, labels, include_reasons=False, ma_mid=ma_mid, ma_short=ma_short, ma_long=ma_long)

def yen_symbol(symbol: str) -> bool:
    return symbol.endswith(".T")


def price_band_label(value, symbol: str) -> str:
    if value is None or pd.isna(value):
        return "不明"
    if not yen_symbol(symbol):
        return "米国株/海外ETF"
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
    text = " ".join(str(row.get(col, "")) for col in ["name", "theme", "asset_class", "fund_type", "memo", "analysis_name"])
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
    lower_text = text.lower()
    for keywords, proxies in rules:
        if any(key.lower() in lower_text for key in keywords):
            return proxies
    return ["ACWI", "VT", "SPY"]


def analysis_candidates_for_row(row: pd.Series) -> list[str]:
    candidates = []
    candidates.extend(split_symbol_candidates(str(row.get("analysis_symbol", ""))))
    candidates.extend(inferred_proxy_symbols_for_row(row))
    cleaned = []
    for item in candidates:
        item = str(item).strip().upper()
        if item and not item.startswith("SMBCG") and item not in cleaned:
            cleaned.append(item)
    return cleaned or [str(row.get("symbol", "")).strip().upper()]


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
        if candidate and not candidate.startswith("SMBCG") and candidate not in cleaned:
            cleaned.append(candidate)
    for candidate in cleaned:
        df = load_price_data(candidate, period_value)
        if not df.empty:
            return df, candidate
    return pd.DataFrame(), cleaned[0] if cleaned else ""


def analyze_row(row: pd.Series, list_name: str, run_date: str) -> dict:
    symbol = str(row["symbol"]).strip()
    data_candidates = analysis_candidates_for_row(row)
    data_symbol = data_candidates[0] if data_candidates else symbol
    base = {
        "run_date": run_date,
        "list_name": list_name,
        "symbol": symbol,
        "name": row["name"],
        "theme": row.get("theme", ""),
        "analysis_symbol": data_symbol,
        "analysis_name": row.get("analysis_name", ""),
        "score": 0,
        "raw_score": 0,
        "buy_timing_score": 0,
        "buy_timing_label": "判定不可",
        "sell_timing_score": 0,
        "sell_timing_label": "判定不可",
        "ma_deviation_pct": None,
        "status": "取得失敗",
        "unit_price": None,
        "price_band": "不明",
        "rsi": None,
        "volume_ratio": None,
        "macd_diff": None,
        "return_5d": None,
        "return_20d": None,
        "score_close_gt_ma_mid": None,
        "score_ma_short_gt_mid": None,
        "score_ma_mid_gt_long": None,
        "score_rsi_good": None,
        "score_rsi_overheat": None,
        "score_rsi_oversold": None,
        "score_macd_positive": None,
        "score_macd_cross_up": None,
        "score_volume_15x": None,
        "score_volume_12x": None,
        "score_return_5d_positive": None,
        "score_return_20d_positive": None,
    }
    df, used_data_symbol = load_price_data_from_candidates(data_candidates, PERIOD)
    if used_data_symbol:
        data_symbol = used_data_symbol
        base["analysis_symbol"] = data_symbol
    if df.empty:
        return base
    df = add_indicators(df, MA_SHORT, MA_MID, MA_LONG)
    result = calculate_buy_score(df, MA_SHORT, MA_MID, MA_LONG)
    latest = result["latest"]
    if latest is None:
        base["status"] = result["status"]
        return base
    buy_timing_score, buy_timing_label = _aa_buy_timing_score(result)
    sell_timing_score, sell_timing_label = _aa_sell_timing_score(df, result, MA_SHORT, MA_MID, MA_LONG)
    base.update(
        {
            "score": result["score"],
            "raw_score": result.get("raw_score", result["score"]),
            "buy_timing_score": buy_timing_score,
            "buy_timing_label": buy_timing_label,
            "sell_timing_score": sell_timing_score,
            "sell_timing_label": sell_timing_label,
            "ma_deviation_pct": result.get("signal_ma_deviation_pct"),
            "status": result["status"],
            "unit_price": latest["Close"],
            "price_band": "プロキシ" if symbol != data_symbol else price_band_label(latest["Close"], symbol),
            "rsi": latest["RSI"],
            "volume_ratio": latest["Volume_Ratio"],
            "macd_diff": latest["MACD_DIFF"],
            "return_5d": latest["Return_5d"],
            "return_20d": latest["Return_20d"],
            "score_close_gt_ma_mid": result.get("score_close_gt_ma_mid"),
            "score_ma_short_gt_mid": result.get("score_ma_short_gt_mid"),
            "score_ma_mid_gt_long": result.get("score_ma_mid_gt_long"),
            "score_rsi_good": result.get("score_rsi_good"),
            "score_rsi_overheat": result.get("score_rsi_overheat"),
            "score_rsi_oversold": result.get("score_rsi_oversold"),
            "score_macd_positive": result.get("score_macd_positive"),
            "score_macd_cross_up": result.get("score_macd_cross_up"),
            "score_volume_15x": result.get("score_volume_15x"),
            "score_volume_12x": result.get("score_volume_12x"),
            "score_return_5d_positive": result.get("score_return_5d_positive"),
            "score_return_20d_positive": result.get("score_return_20d_positive"),
        }
    )
    return base


def main() -> None:
    files = find_watchlist_files()
    if not files:
        raise FileNotFoundError("watchlist*.csv が見つかりません。")
    run_date = datetime.now().strftime("%Y-%m-%d")
    rows = []
    for path in files:
        watchlist = normalize_watchlist(pd.read_csv(path, encoding="utf-8-sig"))
        for _, row in watchlist.iterrows():
            rows.append(analyze_row(row, path.name, run_date))
    new_df = pd.DataFrame(rows)
    if HISTORY_PATH.exists():
        old_df = pd.read_csv(HISTORY_PATH, encoding="utf-8-sig")
        old_df.columns = [str(col).replace("\ufeff", "").strip() for col in old_df.columns]
        out = pd.concat([old_df, new_df], ignore_index=True)
        key_cols = [col for col in ["run_date", "list_name", "symbol"] if col in out.columns]
        if key_cols:
            out = out.drop_duplicates(subset=key_cols, keep="last")
    else:
        out = new_df
    out.to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")
    print(f"saved {len(new_df)} rows to {HISTORY_PATH}")


if __name__ == "__main__":
    main()
