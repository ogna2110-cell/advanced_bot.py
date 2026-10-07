import os
import time
import requests
import pandas as pd
import numpy as np

BOT_TOKEN = os.environ["BOT_TOKEN"]
CHAT_ID = "5446314741"

# Broker spread varsayımı: 13 pip
SPREAD_PIPS = 13.0
SPREAD = SPREAD_PIPS * 0.0001


# =========================
# YAHOO VERİSİ
# =========================

def get_data(interval, range_):
    urls = [
        f"https://query1.finance.yahoo.com/v8/finance/chart/EURUSD=X?interval={interval}&range={range_}",
        f"https://query2.finance.yahoo.com/v8/finance/chart/EURUSD=X?interval={interval}&range={range_}"
    ]

    headers = {"User-Agent": "Mozilla/5.0"}

    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=20)

            if r.status_code != 200:
                continue

            data = r.json()
            result = data["chart"]["result"]

            if not result:
                continue

            result = result[0]

            df = pd.DataFrame({
                "time": pd.to_datetime(result["timestamp"], unit="s"),
                "open": result["indicators"]["quote"][0]["open"],
                "high": result["indicators"]["quote"][0]["high"],
                "low": result["indicators"]["quote"][0]["low"],
                "close": result["indicators"]["quote"][0]["close"]
            })

            df = df.dropna()

            if len(df) >= 50:
                return df

        except Exception:
            time.sleep(1)

    raise Exception(f"{interval} verisi alınamadı.")


# =========================
# 4H VERİSİ
# =========================

def get_4h_data():
    df = get_data("1h", "3mo")

    df = df.set_index("time")

    df = df.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    })

    df = df.dropna().reset_index()

    return df


# =========================
# İNDİKATÖRLER
# =========================

def indicators(df):

    df = df.copy()

    df["EMA20"] = df["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df["EMA50"] = df["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    df["EMA200"] = df["close"].ewm(
        span=200,
        adjust=False
    ).mean()

    # RSI
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["MACD_HIST"] = (
        df["MACD"] - df["MACD_SIGNAL"]
    )

    # ATR
    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = abs(df["high"] - previous_close)
    tr3 = abs(df["low"] - previous_close)

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["ATR"] = tr.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    return df


# =========================
# ZAMAN DİLİMİ
# =========================

def analyze(df):

    df = indicators(df)

    x = df.iloc[-1]

    close = float(x["close"])
    ema20 = float(x["EMA20"])
    ema50 = float(x["EMA50"])
    ema200 = float(x["EMA200"])
    rsi = float(x["RSI"])
    macd = float(x["MACD"])
    macd_signal = float(x["MACD_SIGNAL"])
    macd_hist = float(x["MACD_HIST"])
    atr = float(x["ATR"])

    if close > ema20 and ema20 > ema50 and ema50 > ema200:
        trend = "BULLISH"

    elif close < ema20 and ema20 < ema50 and ema50 < ema200:
        trend = "BEARISH"

    elif close > ema200:
        trend = "BULLISH"

    elif close < ema200:
        trend = "BEARISH"

    else:
        trend = "NEUTRAL"

    return {
        "close": close,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "rsi": rsi,
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "atr": atr,
        "trend": trend
    }


# =========================
# DESTEK / DİRENÇ
# =========================

def support_resistance(df):

    recent = df.tail(80)

    support = float(recent["low"].min())
    resistance = float(recent["high"].max())

    return support, resistance


# =========================
# ANA ANALİZ
# =========================

def create_analysis():

    print("15M verisi alınıyor...")
    df15 = get_data("15m", "5d")

    print("1H verisi alınıyor...")
    df1h = get_data("1h", "1mo")

    print("4H verisi alınıyor...")
    df4h = get_4h_data()

    a15 = analyze(df15)
    a1h = analyze(df1h)
    a4h = analyze(df4h)

    price = a15["close"]

    # Yahoo yaklaşık piyasa fiyatı kabul edilir.
    ask = price + SPREAD / 2
    bid = price - SPREAD / 2

    support, resistance = support_resistance(df15)

    # =========================
    # SKOR
    # =========================

    buy = 0
    sell = 0

    # 4H = 2 puan
    if a4h["trend"] == "BULLISH":
        buy += 2
    elif a4h["trend"] == "BEARISH":
        sell += 2

    # 1H = 2 puan
    if a1h["trend"] == "BULLISH":
        buy += 2
    elif a1h["trend"] == "BEARISH":
        sell += 2

    # 15M = 2 puan
    if a15["trend"] == "BULLISH":
        buy += 2
    elif a15["trend"] == "BEARISH":
        sell += 2

    # RSI = 1 puan
    if 52 <= a15["rsi"] <= 68:
        buy += 1

    elif 32 <= a15["rsi"] <= 48:
        sell += 1

    # MACD = 1 puan
    if a15["macd_hist"] > 0:
        buy += 1

    elif a15["macd_hist"] < 0:
        sell += 1

    # Momentum = 1 puan
    if a1h["close"] > a1h["ema20"]:
        buy += 1

    elif a1h["close"] < a1h["ema20"]:
        sell += 1

    # Destek/direnç = 1 puan
    if price > support and price < resistance:

        support_distance = price - support
        resistance_distance = resistance - price

        if support_distance > resistance_distance:
            sell += 1
        else:
            buy += 1

    # =========================
    # SİNYAL
    # =========================

    if buy >= 8 and buy > sell:
        signal = "BUY"
        score = buy

    elif sell >= 8 and sell > buy:
        signal = "SELL"
        score = sell

    else:
        signal = "WAIT"
        score = max(buy, sell)

    # =========================
    # ENTRY / SL / TP
    # =========================

    entry = None
    sl = None
    tp1 = None
    tp2 = None
    tp3 = None

    if signal == "BUY":

        entry = ask

        stop_distance = max(
            a15["atr"] * 1.5,
            SPREAD * 1.5
        )

        sl = entry - stop_distance

        tp1 = entry + stop_distance * 1.5
        tp2 = entry + stop_distance * 2
        tp3 = entry + stop_distance * 3

    elif signal == "SELL":

        entry = bid

        stop_distance = max(
            a15["atr"] * 1.5,
            SPREAD * 1.5
        )

        sl = entry + stop_distance

        tp1 = entry - stop_distance * 1.5
        tp2 = entry - stop_distance * 2
        tp3 = entry - stop_distance * 3

    return {
        "signal": signal,
        "score": score,
        "price": price,
        "ask": ask,
        "bid": bid,
        "support": support,
        "resistance": resistance,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "a15": a15,
        "a1h": a1h,
        "a4h": a4h,
        "buy": buy,
        "sell": sell
    }


# =========================
# MESAJ
# =========================

def p(value):

    if value is None:
        return "-"

    return f"{value:.5f}"


def make_message(a):

    if a["signal"] == "BUY":
        emoji = "🟢"

    elif a["signal"] == "SELL":
        emoji = "🔴"

    else:
        emoji = "🟡"

    x15 = a["a15"]
    x1h = a["a1h"]
    x4h = a["a4h"]

    msg = f"""
EUR/USD ADVANCED SIGNAL

{emoji} {a["signal"]}

Strength Score: {a["score"]}/10

━━━━━━━━━━━━━━

CURRENT PRICE

Price: {p(a["price"])}
Estimated Ask: {p(a["ask"])}
Estimated Bid: {p(a["bid"])}

Broker spread assumption:
~{SPREAD_PIPS:.0f} pips

━━━━━━━━━━━━━━

15 MINUTE

Trend: {x15["trend"]}
RSI: {x15["rsi"]:.1f}

EMA20: {p(x15["ema20"])}
EMA50: {p(x15["ema50"])}
EMA200: {p(x15["ema200"])}

MACD: {x15["macd"]:.6f}
MACD Histogram: {x15["macd_hist"]:.6f}

ATR: {x15["atr"]:.5f}

━━━━━━━━━━━━━━

1 HOUR

Trend: {x1h["trend"]}
RSI: {x1h["rsi"]:.1f}

EMA20: {p(x1h["ema20"])}
EMA50: {p(x1h["ema50"])}
EMA200: {p(x1h["ema200"])}

MACD: {x1h["macd"]:.6f}

━━━━━━━━━━━━━━

4 HOUR

Trend: {x4h["trend"]}
RSI: {x4h["rsi"]:.1f}

EMA20: {p(x4h["ema20"])}
EMA50: {p(x4h["ema50"])}
EMA200: {p(x4h["ema200"])}

MACD: {x4h["macd"]:.6f}

━━━━━━━━━━━━━━

SUPPORT:
{p(a["support"])}

RESISTANCE:
{p(a["resistance"])}

BUY SCORE: {a["buy"]}/10
SELL SCORE: {a["sell"]}/10
"""

    if a["signal"] != "WAIT":

        msg += f"""

━━━━━━━━━━━━━━

TRADE PLAN

ENTRY:
{p(a["entry"])}

STOP LOSS:
{p(a["sl"])}

TP1:
{p(a["tp1"])}

TP2:
{p(a["tp2"])}

TP3:
{p(a["tp3"])}

━━━━━━━━━━━━━━

ACTION

{a["signal"]} yönü daha güçlü görünüyor.

Önce demo hesapta test et.

Score bir kazanma garantisi değildir.
"""

    else:

        msg += """

━━━━━━━━━━━━━━

ACTION

🟡 WAIT

Şu anda güçlü bir işlem sinyali yok.

BUY ve SELL koşulları yeterince
güçlü şekilde aynı yönde birleşmiyor.

İşlem açma.
Daha iyi hizalanma bekle.

━━━━━━━━━━━━━━

Demo hesapta test et.
"""

    return msg


# =========================
# TELEGRAM
# =========================

def send_message(text):

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    response = requests.post(
        url,
        data={
            "chat_id": CHAT_ID,
            "text": text
        },
        timeout=20
    )

    response.raise_for_status()


# =========================
# ÇALIŞTIR
# =========================

if __name__ == "__main__":

    try:

        print("EUR/USD Advanced analiz başlıyor...")

        result = create_analysis()

        message = make_message(result)

        send_message(message)

        print("Analiz Telegram'a gönderildi.")
        print("Bot tamamlandı.")

    except Exception as e:

        print("HATA:", str(e))

        try:
            send_message(
                "❌ EUR/USD Advanced Bot hata verdi:\n\n"
                + str(e)[:500]
            )
        except Exception:
            pass

        raise
