import os
import time
import requests
import pandas as pd
import numpy as np

# =========================
# TELEGRAM AYARLARI
# =========================

BOT_TOKEN = os.environ["BOT_TOKEN"]
ALLOWED_CHAT_ID = "5446314741"

# Broker spread varsayımı
SPREAD_PIPS = 13.0
SPREAD_PRICE = SPREAD_PIPS * 0.0001


# =========================
# YAHOO FINANCE VERİSİ
# =========================

def get_yahoo_data(interval="15m", range_="5d"):
    urls = [
        f"https://query1.finance.yahoo.com/v8/finance/chart/EURUSD=X?interval={interval}&range={range_}",
        f"https://query2.finance.yahoo.com/v8/finance/chart/EURUSD=X?interval={interval}&range={range_}"
    ]

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for url in urls:
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=20
            )

            if response.status_code != 200:
                continue

            data = response.json()
            result = data["chart"]["result"]

            if not result:
                continue

            timestamps = result[0]["timestamp"]
            quote = result[0]["indicators"]["quote"][0]

            df = pd.DataFrame({
                "timestamp": pd.to_datetime(timestamps, unit="s"),
                "open": quote["open"],
                "high": quote["high"],
                "low": quote["low"],
                "close": quote["close"],
                "volume": quote.get("volume", [0] * len(timestamps))
            })

            df = df.dropna(subset=["open", "high", "low", "close"])

            if len(df) >= 50:
                return df

        except Exception:
            time.sleep(1)

    raise Exception(f"EUR/USD {interval} verisi alınamadı.")


# =========================
# 4 SAATLİK VERİ OLUŞTUR
# =========================

def get_4h_data():
    df = get_yahoo_data("1h", "3mo")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    df = df.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    })

    df = df.dropna().reset_index()

    return df


# =========================
# EMA
# =========================

def add_ema(df, period):
    df[f"EMA{period}"] = df["close"].ewm(
        span=period,
        adjust=False
    ).mean()

    return df


# =========================
# RSI
# =========================

def add_rsi(df, period=14):

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))

    return df


# =========================
# MACD
# =========================

def add_macd(df):

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

    return df


# =========================
# ATR
# =========================

def add_atr(df, period=14):

    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = abs(df["high"] - previous_close)
    tr3 = abs(df["low"] - previous_close)

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["ATR"] = true_range.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    return df


# =========================
# TÜM İNDİKATÖRLER
# =========================

def prepare_dataframe(df):

    for period in [20, 50, 200]:
        df = add_ema(df, period)

    df = add_rsi(df)
    df = add_macd(df)
    df = add_atr(df)

    return df


# =========================
# DESTEK / DİRENÇ
# =========================

def find_support_resistance(df, lookback=80):

    recent = df.tail(lookback)

    support = recent["low"].min()
    resistance = recent["high"].max()

    return support, resistance


# =========================
# ZAMAN DİLİMİ ANALİZİ
# =========================

def analyze_timeframe(df):

    df = prepare_dataframe(df)

    last = df.iloc[-1]

    close = float(last["close"])
    ema20 = float(last["EMA20"])
    ema50 = float(last["EMA50"])
    ema200 = float(last["EMA200"])
    rsi = float(last["RSI"])
    macd = float(last["MACD"])
    macd_signal = float(last["MACD_SIGNAL"])
    macd_hist = float(last["MACD_HIST"])
    atr = float(last["ATR"])

    bullish = 0
    bearish = 0

    # EMA trend
    if close > ema20:
        bullish += 1
    else:
        bearish += 1

    if ema20 > ema50:
        bullish += 1
    else:
        bearish += 1

    if ema50 > ema200:
        bullish += 1
    else:
        bearish += 1

    # RSI
    if rsi > 50:
        bullish += 1
    elif rsi < 50:
        bearish += 1

    # MACD
    if macd > macd_signal and macd_hist > 0:
        bullish += 1
    elif macd < macd_signal and macd_hist < 0:
        bearish += 1

    if bullish > bearish:
        direction = "BULLISH"
    elif bearish > bullish:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

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
        "bullish": bullish,
        "bearish": bearish,
        "direction": direction
    }


# =========================
# ANA SİNYAL ANALİZİ
# =========================

def generate_signal():

    df15 = get_yahoo_data("15m", "5d")
    df1h = get_yahoo_data("1h", "1mo")
    df4h = get_4h_data()

    a15 = analyze_timeframe(df15)
    a1h = analyze_timeframe(df1h)
    a4h = analyze_timeframe(df4h)

    current_price = a15["close"]

    # Yahoo fiyatı mid/last gibi düşünülür.
    assumed_ask = current_price + SPREAD_PRICE / 2
    assumed_bid = current_price - SPREAD_PRICE / 2

    support, resistance = find_support_resistance(
        df15,
        80
    )

    # =========================
    # SKOR
    # =========================

    buy_score = 0
    sell_score = 0

    # 4H trend
    if a4h["direction"] == "BULLISH":
        buy_score += 2
    elif a4h["direction"] == "BEARISH":
        sell_score += 2

    # 1H trend
    if a1h["direction"] == "BULLISH":
        buy_score += 2
    elif a1h["direction"] == "BEARISH":
        sell_score += 2

    # 15M trend
    if a15["direction"] == "BULLISH":
        buy_score += 2
    elif a15["direction"] == "BEARISH":
        sell_score += 2

    # RSI
    if 52 <= a15["rsi"] <= 68:
        buy_score += 1

    if 32 <= a15["rsi"] <= 48:
        sell_score += 1

    # MACD
    if a15["macd_hist"] > 0:
        buy_score += 1

    elif a15["macd_hist"] < 0:
        sell_score += 1

    # Momentum
    if a1h["close"] > a1h["ema20"]:
        buy_score += 1

    elif a1h["close"] < a1h["ema20"]:
        sell_score += 1

    # =========================
    # YÖN
    # =========================

    if buy_score >= 8 and buy_score > sell_score:
        signal = "BUY"

    elif sell_score >= 8 and sell_score > buy_score:
        signal = "SELL"

    else:
        signal = "WAIT"

    # =========================
    # GİRİŞ / SL / TP
    # =========================

    if signal == "BUY":

        entry = assumed_ask

        atr = a15["atr"]

        stop_distance = max(
            atr * 1.5,
            SPREAD_PRICE * 1.5
        )

        sl = entry - stop_distance

        tp1 = entry + stop_distance * 1.5
        tp2 = entry + stop_distance * 2.0
        tp3 = entry + stop_distance * 3.0

        # Direnç çok yakınsa TP'leri aşırı zorlamama
        if resistance > entry and resistance < tp1:
            tp1 = resistance - 0.00010

        action = (
            "BUY açılabilir. "
            "Ancak işlem demo hesapta test edilmeli."
        )

        score = buy_score

    elif signal == "SELL":

        entry = assumed_bid

        atr = a15["atr"]

        stop_distance = max(
            atr * 1.5,
            SPREAD_PRICE * 1.5
        )

        sl = entry + stop_distance

        tp1 = entry - stop_distance * 1.5
        tp2 = entry - stop_distance * 2.0
        tp3 = entry - stop_distance * 3.0

        if support < entry and support > tp1:
            tp1 = support + 0.00010

        action = (
            "SELL açılabilir. "
            "Ancak işlem demo hesapta test edilmeli."
        )

        score = sell_score

    else:

        entry = current_price
        sl = None
        tp1 = None
        tp2 = None
        tp3 = None
        action = (
            "ŞU ANDA İŞLEM AÇMA. "
            "Zaman dilimleri yeterince güçlü şekilde "
            "aynı yönde değil."
        )

        score = max(
            buy_score,
            sell_score
        )

    return {
        "signal": signal,
        "score": score,
        "price": current_price,
        "ask": assumed_ask,
        "bid": assumed_bid,
        "support": support,
        "resistance": resistance,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "action": action,
        "a15": a15,
        "a1h": a1h,
        "a4h": a4h
    }


# =========================
# TELEGRAM MESAJI
# =========================

def format_price(value):

    if value is None:
        return "-"

    return f"{value:.5f}"


def create_message(result):

    signal = result["signal"]

    if signal == "BUY":
        emoji = "🟢"
    elif signal == "SELL":
        emoji = "🔴"
    else:
        emoji = "🟡"

    a15 = result["a15"]
    a1h = result["a1h"]
    a4h = result["a4h"]

    message = f"""
EUR/USD ADVANCED SIGNAL

{emoji} {signal}

Score: {result["score"]}/10

Current price:
{format_price(result["price"])}

Estimated spread:
~{SPREAD_PIPS:.0f} pips

Assumed Ask:
{format_price(result["ask"])}

Assumed Bid:
{format_price(result["bid"])}

━━━━━━━━━━━━━━

15M
Trend: {a15["direction"]}
RSI: {a15["rsi"]:.1f}
EMA20: {format_price(a15["ema20"])}
EMA50: {format_price(a15["ema50"])}
EMA200: {format_price(a15["ema200"])}
MACD: {a15["macd"]:.6f}

1H
Trend: {a1h["direction"]}
RSI: {a1h["rsi"]:.1f}
EMA20: {format_price(a1h["ema20"])}
EMA50: {format_price(a1h["ema50"])}
EMA200: {format_price(a1h["ema200"])}
MACD: {a1h["macd"]:.6f}

4H
Trend: {a4h["direction"]}
RSI: {a4h["rsi"]:.1f}
EMA20: {format_price(a4h["ema20"])}
EMA50: {format_price(a4h["ema50"])}
EMA200: {format_price(a4h["ema200"])}
MACD: {a4h["macd"]:.6f}

━━━━━━━━━━━━━━

Support:
{format_price(result["support"])}

Resistance:
{format_price(result["resistance"])}

"""

    if signal != "WAIT":

        message += f"""
ENTRY:
{format_price(result["entry"])}

STOP LOSS:
{format_price(result["sl"])}

TP1:
{format_price(result["tp1"])}

TP2:
{format_price(result["tp2"])}

TP3:
{format_price(result["tp3"])}

━━━━━━━━━━━━━━

WHAT TO DO:

{result["action"]}

Signal score is a strength score,
NOT a guaranteed win probability.

━━━━━━━━━━━━━━

Risk:
Demo account recommended.
Do not risk money you cannot afford to lose.
"""

    else:

        message += f"""
━━━━━━━━━━━━━━

WHAT TO DO:

{result["action"]}

BUY score:
{result["a15"]["bullish"]}

SELL score:
{result["a15"]["bearish"]}

Signal is not strong enough.
Wait for better alignment.

━━━━━━━━━━━━━━

Demo hesapta test et.
"""

    return message


# =========================
# TELEGRAM GÖNDER
# =========================

def send_telegram(message):

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    data = {
        "chat_id": ALLOWED_CHAT_ID,
        "text": message
    }

    response = requests.post(
        url,
        data=data,
        timeout=20
    )

    response.raise_for_status()


# =========================
# TELEGRAM KOMUTLARI
# =========================

def get_updates(offset=None):

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"

    params = {
        "timeout": 20
    }

    if offset is not None:
        params["offset"] = offset

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    return response.json()


def run_bot():

    offset = None

    while True:

        try:

            data = get_updates(offset)

            if not data.get("ok"):
                time.sleep(3)
                continue

            for update in data.get("result", []):

                offset = update["update_id"] + 1

                message = update.get("message")

                if not message:
                    continue

                chat_id = str(
                    message["chat"]["id"]
                )

                text = message.get(
                    "text",
                    ""
                ).strip().lower()

                # SADECE SEN
                if chat_id != ALLOWED_CHAT_ID:
                    continue

                if text == "/signal":

                    send_telegram(
                        "⏳ EUR/USD analiz ediliyor...\n"
                        "15M + 1H + 4H kontrol ediliyor."
                    )

                    try:

                        result = generate_signal()

                        message_text = create_message(
                            result
                        )

                        send_telegram(
                            message_text
                        )

                    except Exception as e:

                        send_telegram(
                            "❌ Analiz sırasında hata oluştu.\n\n"
                            f"Hata: {str(e)[:300]}"
                        )

                elif text == "/start":

                    send_telegram(
                        "🤖 EUR/USD Advanced Bot aktif.\n\n"
                        "Komut:\n"
                        "/signal\n\n"
                        "EUR/USD için güncel analiz al."
                    )

        except Exception:
            time.sleep(5)


# =========================
# BAŞLAT
# =========================

if __name__ == "__main__":
    run_bot()
