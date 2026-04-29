from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse

import pandas as pd
import ta
import yfinance as yf


WATCHLIST_PATH = Path("watchlist.csv")
HISTORY_PATH = Path("score_history.csv")


def normalize_watchlist(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).replace("\ufeff", "").strip() for col in df.columns]

    if not {"symbol", "name"}.issubset(df.columns):
        raise ValueError("watchlist.csv には symbol,name 列が必要です。")

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
        raise ValueError("watchlist.csv に有効な銘柄コードがありません。")

    return df


def load_watchlist(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} が見つかりません。")

    return normalize_watchlist(pd.read_csv(path, encoding="utf-8-sig"))


def load_price_data(ticker: str, period_value: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period_value,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def yen_symbol(symbol: str) -> bool:
    return symbol.endswith(".T")


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


def add_indicators(df: pd.DataFrame, ma_short: int, ma_mid: int, ma_long: int) -> pd.DataFrame:
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


def calculate_buy_score(df: pd.DataFrame, ma_short: int, ma_mid: int, ma_long: int) -> dict:
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
            "reasons": ["指標計算に必要なデータが不足しています。"],
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

    base = {
        "symbol": symbol,
        "name": name,
        "theme": theme,
    }

    if df.empty:
        return {
            **base,
            "score": 0,
            "status": "取得失敗",
            "unit_price": None,
            "price_band": "不明",
            "rsi": None,
            "volume_ratio": None,
            "macd_diff": None,
            "return_5d": None,
            "return_20d": None,
            "reasons": "データを取得できませんでした。",
        }

    df = add_indicators(df, ma_short, ma_mid, ma_long)
    result = calculate_buy_score(df, ma_short, ma_mid, ma_long)
    latest = result["latest"]

    if latest is None:
        return {
            **base,
            "score": result["score"],
            "status": result["status"],
            "unit_price": None,
            "price_band": "不明",
            "rsi": None,
            "volume_ratio": None,
            "macd_diff": None,
            "return_5d": None,
            "return_20d": None,
            "reasons": " / ".join(result["reasons"]),
        }

    close = latest["Close"]

    return {
        **base,
        "score": result["score"],
        "status": result["status"],
        "unit_price": float(close),
        "price_band": price_band_label(close, symbol),
        "rsi": float(latest["RSI"]),
        "volume_ratio": float(latest["Volume_Ratio"]),
        "macd_diff": float(latest["MACD_DIFF"]),
        "return_5d": float(latest["Return_5d"]),
        "return_20d": float(latest["Return_20d"]),
        "reasons": " / ".join(result["reasons"]),
    }


def append_history(new_rows: pd.DataFrame, history_path: Path, run_date: str) -> pd.DataFrame:
    """同一run_dateの履歴は置き換え、別日の履歴は残す。"""
    if history_path.exists():
        old = pd.read_csv(history_path, encoding="utf-8-sig")
        old.columns = [str(col).replace("\ufeff", "").strip() for col in old.columns]
        if "run_date" in old.columns:
            old = old[old["run_date"].astype(str) != run_date]
        merged = pd.concat([old, new_rows], ignore_index=True)
    else:
        merged = new_rows

    merged = merged.sort_values(["run_date", "score", "symbol"], ascending=[True, False, True])
    history_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(history_path, index=False, encoding="utf-8-sig")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Stock score history collector")
    parser.add_argument("--watchlist", default=str(WATCHLIST_PATH), help="銘柄CSVのパス")
    parser.add_argument("--history", default=str(HISTORY_PATH), help="履歴CSVの出力先")
    parser.add_argument("--period", default="1y", help="yfinanceの取得期間")
    parser.add_argument("--ma-short", type=int, default=5)
    parser.add_argument("--ma-mid", type=int, default=25)
    parser.add_argument("--ma-long", type=int, default=75)
    parser.add_argument("--limit", type=int, default=0, help="0なら全件。テスト時は10などを指定。")
    args = parser.parse_args()

    watchlist = load_watchlist(Path(args.watchlist))
    if args.limit and args.limit > 0:
        watchlist = watchlist.head(args.limit)

    run_date = datetime.now().strftime("%Y-%m-%d")

    results = []
    for idx, row in watchlist.iterrows():
        symbol = row["symbol"]
        name = row["name"]
        print(f"[{idx + 1}/{len(watchlist)}] {symbol} {name}")
        try:
            result = analyze_symbol(
                symbol=symbol,
                name=name,
                theme=row.get("theme", "未分類"),
                period=args.period,
                ma_short=args.ma_short,
                ma_mid=args.ma_mid,
                ma_long=args.ma_long,
            )
        except Exception as exc:
            result = {
                "symbol": symbol,
                "name": name,
                "theme": row.get("theme", "未分類"),
                "score": 0,
                "status": "取得失敗",
                "unit_price": None,
                "price_band": "不明",
                "rsi": None,
                "volume_ratio": None,
                "macd_diff": None,
                "return_5d": None,
                "return_20d": None,
                "reasons": f"例外エラー: {exc}",
            }

        results.append({"run_date": run_date, **result})

    new_rows = pd.DataFrame(results)
    merged = append_history(new_rows, Path(args.history), run_date)
    print(f"saved: {args.history}")
    print(f"rows added/replaced for {run_date}: {len(new_rows)}")
    print(f"history total rows: {len(merged)}")


if __name__ == "__main__":
    main()
