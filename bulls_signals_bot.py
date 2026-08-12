import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import ccxt
import pandas as pd
import numpy as np
import requests
import time
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ================= Configuration =================
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN') or os.environ.get('BOT_TOKEN')
CHANNEL_ID = os.environ.get('CHANNEL_ID')

TIMEFRAME_ENTRY = '15m'
TIMEFRAME_TREND = '1h'
TOP_N_COINS = 20
LEVERAGE = 15

# Risk & Targets
TP1_PERC = 0.70
TP2_PERC = 1.50
TP3_PERC = 2.80
TP4_PERC = 5.00
SL_ATR_MULT = 1.5

# Quality Filters
MIN_ATR_PERCENT = 0.05
VOLUME_LOOKBACK = 20
COOLDOWN_HOURS = 1
COOLDOWN_FILE = Path('cooldown.json')

STABLECOINS = ['USDC/USDT', 'TUSD/USDT', 'DAI/USDT', 'FDUSD/USDT', 'USDP/USDT', 'PYUSD/USDT']
BLACKLIST = ['WXT/USDT', 'ANTFUN/USDT', 'UPC/USDT', 'RAIN/USDT', 'USD1/USDT', 'USDE/USDT']


def _fmt(price):
    if price >= 1000:   return f"{price:,.2f}"
    elif price >= 1:    return f"{price:,.4f}"
    else:              return f"{price:,.6f}"


def load_cooldown():
    if COOLDOWN_FILE.exists():
        try:
            with open(COOLDOWN_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_cooldown(data):
    try:
        with open(COOLDOWN_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save cooldown: {e}")


def is_on_cooldown(symbol, cooldown_data):
    if symbol not in cooldown_data:
        return False
    try:
        last_time = datetime.fromisoformat(cooldown_data[symbol])
        elapsed = (datetime.now() - last_time).total_seconds() / 3600
        return elapsed < COOLDOWN_HOURS
    except Exception:
        return False


def send_telegram(message):
    if not TELEGRAM_TOKEN or not CHANNEL_ID:
        print("Missing TELEGRAM_TOKEN or CHANNEL_ID")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        'chat_id': CHANNEL_ID,
        'text': message,
        'parse_mode': 'HTML',
        'disable_web_page_preview': True
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        if not r.json().get('ok'):
            print(f"Telegram error: {r.json().get('description')}")
    except Exception as e:
        print(f"Network error sending Telegram: {e}")


def get_mexc_data(symbol, timeframe, limit=150):
    exchange = ccxt.mexc({'enableRateLimit': True})
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df


def get_top_mexc_coins(limit=20):
    print(f"Fetching top {limit} coins by volume from MEXC...")
    exchange = ccxt.mexc({'enableRateLimit': True})
    try:
        tickers = exchange.fetch_tickers()
        usdt_pairs = []
        for symbol, ticker in tickers.items():
            if symbol.endswith('/USDT') and symbol not in STABLECOINS and symbol not in BLACKLIST:
                vol = ticker.get('quoteVolume') or 0
                if vol > 500000:
                    usdt_pairs.append({'symbol': symbol, 'volume': vol})
        usdt_pairs.sort(key=lambda x: x['volume'], reverse=True)
        top = [p['symbol'] for p in usdt_pairs[:limit]]
        print(f"Top coins: {top[:5]} ... ({len(top)} total)")
        return top
    except Exception as e:
        print(f"Error fetching coins: {e}")
        return []


def calculate_atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def analyze_symbol(symbol):
    """
    Strategy: Trend Following on 1H + EMA/MACD/RSI confirmation on 15m
    """
    try:
        df_15m = get_mexc_data(symbol, TIMEFRAME_ENTRY, limit=150)
        df_1h = get_mexc_data(symbol, TIMEFRAME_TREND, limit=100)

        if len(df_15m) < 50 or len(df_1h) < 50:
            print(f"  {symbol}: Not enough data")
            return None

        # === Trend Filter on 1H ===
        df_1h['ema50'] = df_1h['close'].ewm(span=50, adjust=False).mean()
        price_1h = df_1h['close'].iloc[-1]
        ema50_1h = df_1h['ema50'].iloc[-1]
        trend_bullish = price_1h > ema50_1h
        trend_bearish = price_1h < ema50_1h

        # === Indicators on 15m ===
        df_15m['ema9'] = df_15m['close'].ewm(span=9, adjust=False).mean()
        df_15m['ema21'] = df_15m['close'].ewm(span=21, adjust=False).mean()
        df_15m['rsi'] = calculate_rsi(df_15m['close'], 14)
        _, _, df_15m['macd_hist'] = calculate_macd(df_15m['close'])
        df_15m['volume_sma'] = df_15m['volume'].rolling(window=VOLUME_LOOKBACK).mean()
        df_15m['atr'] = calculate_atr(df_15m, 14)
        df_15m['atr_pct'] = (df_15m['atr'] / df_15m['close']) * 100

        # Current values
        ema9 = df_15m['ema9'].iloc[-1]
        ema21 = df_15m['ema21'].iloc[-1]
        ema9_prev = df_15m['ema9'].iloc[-2]
        ema21_prev = df_15m['ema21'].iloc[-2]
        rsi = df_15m['rsi'].iloc[-1]
        macd_hist = df_15m['macd_hist'].iloc[-1]
        macd_hist_prev = df_15m['macd_hist'].iloc[-2]
        vol_now = df_15m['volume'].iloc[-1]
        vol_avg = df_15m['volume_sma'].iloc[-1]
        atr_pct = df_15m['atr_pct'].iloc[-1]
        current_price = df_15m['close'].iloc[-1]

        # === Quality Checks ===
        if pd.isna(atr_pct) or atr_pct < MIN_ATR_PERCENT:
            print(f"  {symbol}: ATR too low ({atr_pct:.3f}% < {MIN_ATR_PERCENT}%)")
            return None

        if pd.isna(vol_avg) or vol_now < vol_avg * 0.5:
            print(f"  {symbol}: Volume too low ({vol_now:.0f} < {vol_avg:.0f})")
            return None

        if pd.isna(rsi) or pd.isna(macd_hist):
            print(f"  {symbol}: Indicators not ready")
            return None

        # === Strategy Conditions ===
        confidence = 50
        direction = None
        reason = ""

        # LONG conditions
        if trend_bullish:
            # EMA cross or EMA alignment
            ema_aligned = ema9 > ema21
            ema_cross = (ema9_prev <= ema21_prev) and (ema9 > ema21)
            macd_rising = macd_hist > macd_hist_prev and macd_hist > -0.5 * abs(macd_hist_prev)
            rsi_ok = 35 < rsi < 70  # Not overbought, some room

            if ema_aligned and macd_rising and rsi_ok:
                direction = "LONG"
                reason = "EMA Bullish + MACD Rising"
                confidence = 65
                if ema_cross:
                    confidence += 10
                    reason = "EMA Cross + MACD Rising"
                if rsi < 55:
                    confidence += 5
                if vol_now > vol_avg * 1.2:
                    confidence += 5
                if macd_hist > 0:
                    confidence += 5

        # SHORT conditions
        elif trend_bearish:
            ema_aligned = ema9 < ema21
            ema_cross = (ema9_prev >= ema21_prev) and (ema9 < ema21)
            macd_falling = macd_hist < macd_hist_prev and macd_hist < 0.5 * abs(macd_hist_prev)
            rsi_ok = 30 < rsi < 65  # Not oversold, some room

            if ema_aligned and macd_falling and rsi_ok:
                direction = "SHORT"
                reason = "EMA Bearish + MACD Falling"
                confidence = 65
                if ema_cross:
                    confidence += 10
                    reason = "EMA Cross + MACD Falling"
                if rsi > 45:
                    confidence += 5
                if vol_now > vol_avg * 1.2:
                    confidence += 5
                if macd_hist < 0:
                    confidence += 5

        if direction is None:
            print(f"  {symbol}: No signal (Trend: {'UP' if trend_bullish else 'DOWN'}, EMA9/21: {ema9:.2f}/{ema21:.2f}, RSI: {rsi:.1f}, MACD: {macd_hist:.4f})")
            return None

        confidence = min(95, confidence)

        # === Build Signal ===
        atr = df_15m['atr'].iloc[-1]

        if direction == "LONG":
            sl = current_price - (atr * SL_ATR_MULT)
            tp1 = current_price * (1 + TP1_PERC / 100)
            tp2 = current_price * (1 + TP2_PERC / 100)
            tp3 = current_price * (1 + TP3_PERC / 100)
            tp4 = current_price * (1 + TP4_PERC / 100)
        else:
            sl = current_price + (atr * SL_ATR_MULT)
            tp1 = current_price * (1 - TP1_PERC / 100)
            tp2 = current_price * (1 - TP2_PERC / 100)
            tp3 = current_price * (1 - TP3_PERC / 100)
            tp4 = current_price * (1 - TP4_PERC / 100)

        sl_dist = abs(current_price - sl) / current_price * 100
        rr = round(TP4_PERC / sl_dist, 2) if sl_dist > 0 else 0

        return {
            'symbol': symbol,
            'direction': direction,
            'price': current_price,
            'sl': sl,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'tp4': tp4,
            'confidence': confidence,
            'rr': rr,
            'atr_pct': atr_pct,
            'trend': "BULLISH" if trend_bullish else "BEARISH",
            'reason': reason,
            'rsi': rsi,
            'macd_hist': macd_hist
        }

    except Exception as e:
        print(f"  {symbol}: Error - {e}")
        return None


def build_message(signal):
    pair = signal['symbol'].replace('/', '')
    direction = signal['direction']
    conf = signal['confidence']
    rr = signal['rr']
    trend = signal['trend']
    reason = signal['reason']

    emoji_dir = "🟢" if direction == "LONG" else "🔴"
    emoji_conf = "🔥" if conf >= 85 else "✅" if conf >= 70 else "⚡"

    msg = f"""{emoji_dir} SIGNAL #{pair}
│ {TIMEFRAME_ENTRY} │ {direction} │ {LEVERAGE}x

Confidence: {conf}% {emoji_conf}
Strategy: {reason}
Trend (1H): {trend}
Risk/Reward: 1:{rr}

📌 ENTRY: {_fmt(signal['price'])}

🎯 TP1 ➜ {_fmt(signal['tp1'])}  (+{TP1_PERC}%)
🎯 TP2 ➜ {_fmt(signal['tp2'])}  (+{TP2_PERC}%)
🎯 TP3 ➜ {_fmt(signal['tp3'])}  (+{TP3_PERC}%)
☀️ TP4 ➜ {_fmt(signal['tp4'])}  (+{TP4_PERC}%)

🛡️ SL: {_fmt(signal['sl'])}
↻ Move to BE after TP1

L E A K E D  B Y: @BULLS_SIGNALS"""

    return msg


def main():
    print("=" * 50)
    print("BULLS SIGNALS — Trend Following Bot")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    if not TELEGRAM_TOKEN or not CHANNEL_ID:
        print("ERROR: TELEGRAM_TOKEN and CHANNEL_ID must be set!")
        return

    top_coins = get_top_mexc_coins(TOP_N_COINS)
    if not top_coins:
        print("No coins fetched. Aborting.")
        return

    cooldown_data = load_cooldown()
    signals_sent = 0
    skipped_cooldown = 0
    skipped_filter = 0

    print("\nScanning coins...")
    for symbol in top_coins:
        try:
            if is_on_cooldown(symbol, cooldown_data):
                skipped_cooldown += 1
                print(f"  {symbol}: On cooldown")
                continue

            time.sleep(0.6)
            signal = analyze_symbol(symbol)

            if signal is None:
                skipped_filter += 1
                continue

            if signal['confidence'] < 60:
                print(f"  {symbol}: Confidence too low ({signal['confidence']}%)")
                skipped_filter += 1
                continue

            msg = build_message(signal)
            send_telegram(msg)
            cooldown_data[symbol] = datetime.now().isoformat()
            signals_sent += 1
            print(f"  ✅ SIGNAL SENT: {symbol} {signal['direction']} (Conf: {signal['confidence']}%)")
            time.sleep(1.5)

        except Exception as e:
            print(f"  {symbol}: Exception - {e}")

    save_cooldown(cooldown_data)
    print(f"\n🏁 Done. Signals: {signals_sent} | Cooldown skipped: {skipped_cooldown} | Filter skipped: {skipped_filter}")


if __name__ == "__main__":
    main()
