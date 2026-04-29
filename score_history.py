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
    for col in ["analysis_symbol", "analysis_name", "analysis_note"]:
        if col not in df.columns:
            df[col] = ""
    for col in ["symbol", "name", "theme", "memo", "analysis_symbol", "analysis_name", "analysis_note"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    df = df[df["symbol"] != ""].drop_duplicates(subset=["symbol"], keep="first")
    return df.reset_index(drop=True)


def load_price_data(ticker: str, period_value: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period_value, auto_adjust=False, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


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
    df["Return_5d"] = df["Close"].pct_change(5) * 100
    df["Return_20d"] = df["Close"].pct_change(20) * 100
    return df


def calculate_buy_score(df: pd.DataFrame, ma_short: int, ma_mid: int, ma_long: int) -> dict:
    labels = ma_labels(ma_short, ma_mid, ma_long)
    required = [labels["short"], labels["mid"], labels["long"], "RSI", "MACD_DIFF", "Volume_Ratio", "Return_5d", "Return_20d"]
    clean = df.dropna(subset=required)
    if clean.empty or len(clean) < 2:
        return {"score": 0, "status": "データ不足", "latest": None}
    latest = clean.iloc[-1]
    prev = clean.iloc[-2]
    score = 0
    if latest["Close"] > latest[labels["mid"]]:
        score += 2
    if latest[labels["short"]] > latest[labels["mid"]]:
        score += 2
    if latest[labels["mid"]] > latest[labels["long"]]:
        score += 2
    if 40 <= latest["RSI"] <= 65:
        score += 2
    elif latest["RSI"] > 75:
        score -= 2
    elif latest["RSI"] < 30:
        score += 1
    if latest["MACD_DIFF"] > 0:
        score += 2
    if prev["MACD_DIFF"] < 0 and latest["MACD_DIFF"] > 0:
        score += 2
    if latest["Volume_Ratio"] >= 1.5:
        score += 2
    elif latest["Volume_Ratio"] >= 1.2:
        score += 1
    if latest["Return_5d"] > 0:
        score += 1
    if latest["Return_20d"] > 0:
        score += 1
    score = max(0, min(score, 12))
    if score >= 9:
        status = "強い買い候補"
    elif score >= 6:
        status = "買い候補"
    elif score >= 3:
        status = "様子見"
    else:
        status = "弱い / 見送り"
    return {"score": score, "status": status, "latest": latest}


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


def analyze_row(row: pd.Series, list_name: str, run_date: str) -> dict:
    symbol = str(row["symbol"]).strip()
    data_symbol = str(row.get("analysis_symbol", "")).strip() or symbol
    base = {
        "run_date": run_date,
        "list_name": list_name,
        "symbol": symbol,
        "name": row["name"],
        "theme": row.get("theme", ""),
        "analysis_symbol": data_symbol,
        "analysis_name": row.get("analysis_name", ""),
        "score": 0,
        "status": "取得失敗",
        "unit_price": None,
        "price_band": "不明",
        "rsi": None,
        "volume_ratio": None,
        "macd_diff": None,
        "return_5d": None,
        "return_20d": None,
    }
    df = load_price_data(data_symbol, PERIOD)
    if df.empty:
        return base
    df = add_indicators(df, MA_SHORT, MA_MID, MA_LONG)
    result = calculate_buy_score(df, MA_SHORT, MA_MID, MA_LONG)
    latest = result["latest"]
    if latest is None:
        base["status"] = result["status"]
        return base
    base.update(
        {
            "score": result["score"],
            "status": result["status"],
            "unit_price": latest["Close"],
            "price_band": "プロキシ" if symbol != data_symbol else price_band_label(latest["Close"], symbol),
            "rsi": latest["RSI"],
            "volume_ratio": latest["Volume_Ratio"],
            "macd_diff": latest["MACD_DIFF"],
            "return_5d": latest["Return_5d"],
            "return_20d": latest["Return_20d"],
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
