#!/usr/bin/env python3
"""
MEXC Trading Signal Bot — Supertrend Edition (Fixed)
HTF 4h Filter | Entry-based Targets | Corrected SL Cap | Extended Cooldown
"""

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

TIMEFRAME = '1h'
HTF_TIMEFRAME = '4h'           # NEW: فريم أعلى للاتجاه الكبير
TOP_N_COINS = 25
LEVERAGE = 15

# Targets (4 أهداف)
TP1_PERC = 0.70
TP2_PERC = 2.20
TP3_PERC = 3.90
TP4_PERC = 5.90

# ⚡ SL ديناميكي بناءً على ATR
ATR_SL_MULTIPLIER = 3.5
# 🔒 الحد الأقصى لوقف الخسارة (لا يتجاوز هذا الرقم)
MAX_SL_PERC = 2.2              # FIXED: كان 4.2 — الآن يتوافق مع التعليق

# Quality Filters
VOLUME_LOOKBACK = 20
COOLDOWN_HOURS = 4             # FIXED: كان 1 — الآن 4 ساعات (مناسب لفريم 1h)
COOLDOWN_FILE = Path('cooldown.json')

STABLECOINS = ['USDC/USDT', 'TUSD/USDT', 'DAI/USDT', 'FDUSD/USDT', 'USDP/USDT', 'PYUSD/USDT']
BLACKLIST = ['USD1/USDT', 'USDE/USDT', 'ISEK/USDT', 'MBG/USDT', 'AIX/USDT',
             'XPLK/USDT', '9BIT/USDT', 'CYS/USDT', 'USDGOUSDT', 'GOLD/USDT']


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
        r = requests.post(url, json=payload, timeout=30)
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


# NEW: جلب EMA20 على فريم 4h للفلتر العالي
def get_htf_ema20(symbol, tf='4h'):
    try:
        exchange = ccxt.mexc({'enableRateLimit': True})
        ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=30)
        if len(ohlcv) < 20:
            return None
        df = pd.DataFrame(ohlcv, columns=['t', 'o', 'h', 'l', 'c', 'v'])
        ema20 = df['c'].ewm(span=20, adjust=False).mean().iloc[-1]
        return ema20
    except Exception:
        return None


def get_top_mexc_coins(limit=50):
    print(f"Fetching top {limit} coins by volume from MEXC...")
    exchange = ccxt.mexc({'enableRateLimit': True})
    try:
        tickers = exchange.fetch_tickers()
        usdt_pairs = []
        for symbol, ticker in tickers.items():
            if symbol.endswith('/USDT') and symbol not in STABLECOINS and symbol not in BLACKLIST:
                vol = ticker.get('quoteVolume') or 0
                if vol > 200000:
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


def supertrend(df, atr_period=10, multiplier=3):
    hl2 = (df['high'] + df['low']) / 2
    atr = df['high'].rolling(atr_period).max() - df['low'].rolling(atr_period).min()
    atr = atr.rolling(atr_period).mean()
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    for i in range(len(df)):
        if i == 0:
            st.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = 1
        else:
            if df['close'].iloc[i] > st.iloc[i-1]:
                st.iloc[i] = max(lower_band.iloc[i], st.iloc[i-1])
                direction.iloc[i] = 1
            else:
                st.iloc[i] = min(upper_band.iloc[i], st.iloc[i-1])
                direction.iloc[i] = -1
    return st, direction


def analyze_symbol(symbol):
    try:
        df = get_mexc_data(symbol, TIMEFRAME, limit=150)
        if len(df) < 50:
            return None

        # Supertrend
        df['st_line'], df['st_dir'] = supertrend(df)

        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df['macd_hist'] = macd_line - signal_line

        # RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # ATR & Volume
        df['atr'] = calculate_atr(df, 14)
        df['vol_sma'] = df['volume'].rolling(window=VOLUME_LOOKBACK).mean()

        # Current values
        row = df.iloc[-1]
        prev = df.iloc[-2]
        st_bull = row['st_dir'] == 1
        st_bear = row['st_dir'] == -1
        macd_hist = row['macd_hist']
        macd_hist_prev = prev['macd_hist']
        rsi = row['rsi']
        vol_now = row['volume']
        vol_avg = row['vol_sma']
        current_price = row['close']
        atr = row['atr']

        if pd.isna(rsi) or pd.isna(macd_hist) or pd.isna(atr):
            print(f"  {symbol}: Indicators not ready")
            return None

        if vol_now < vol_avg * 0.3:
            print(f"  {symbol}: Volume too low")
            return None

        confidence = 50
        direction = None

        # LONG
        if st_bull:
            macd_ok = macd_hist > macd_hist_prev or macd_hist > 0
            rsi_ok = 35 < rsi < 70
            if macd_ok and rsi_ok:
                direction = "LONG"
                confidence = 65
                if macd_hist > 0: confidence += 5
                if vol_now > vol_avg * 1.2: confidence += 5
                if rsi < 55: confidence += 5

        # SHORT
        elif st_bear:
            macd_ok = macd_hist < macd_hist_prev or macd_hist < 0
            rsi_ok = 30 < rsi < 65
            if macd_ok and rsi_ok:
                direction = "SHORT"
                confidence = 65
                if macd_hist < 0: confidence += 5
                if vol_now > vol_avg * 1.2: confidence += 5
                if rsi > 45: confidence += 5

        if direction is None:
            print(f"  {symbol}: No signal (ST: {'UP' if st_bull else 'DOWN'}, RSI: {rsi:.1f}, MACD: {macd_hist:.4f})")
            return None

        confidence = min(95, confidence)
        if confidence < 55:
            print(f"  {symbol}: Confidence too low ({confidence}%)")
            return None

        # ═══════════════════════════════════════════════
        # 🆕 HTF Filter: التحقق من EMA20 على فريم 4h
        # ═══════════════════════════════════════════════
        htf_ema = get_htf_ema20(symbol, HTF_TIMEFRAME)
        if htf_ema is not None:
            diff_pct = (current_price - htf_ema) / htf_ema * 100
            if direction == "LONG" and diff_pct < -3.0:
                print(f"  {symbol}: HTF 4h too bearish ({diff_pct:.1f}%)")
                return None
            if direction == "SHORT" and diff_pct > 3.0:
                print(f"  {symbol}: HTF 4h too bullish ({diff_pct:.1f}%)")
                return None
            if direction == "LONG" and diff_pct > 0:
                confidence += 5  # السعر فوق EMA20 على 4h = إضافي
            if direction == "SHORT" and diff_pct < 0:
                confidence += 5  # السعر تحت EMA20 على 4h = إضافي

        # ═══════════════════════════════════════════════
        # ⚡ SL ديناميكي بناءً على ATR + حد أقصى 2.2%
        # ═══════════════════════════════════════════════
        sl_distance = atr * ATR_SL_MULTIPLIER
        max_sl_distance = current_price * (MAX_SL_PERC / 100)

        if sl_distance > max_sl_distance:
            sl_distance = max_sl_distance
            sl_capped = True
        else:
            sl_capped = False

        # ═══════════════════════════════════════════════
        # 🎯 نقطة دخول واحدة فقط: سعر أحسن (انتظار)
        # ═══════════════════════════════════════════════
        if direction == "LONG":
            entry = current_price - (atr * 1.0)
            sl = current_price - sl_distance
            # FIXED: الأهداف تُحسب من entry وليس من current_price
            tp1 = entry * (1 + TP1_PERC / 100)
            tp2 = entry * (1 + TP2_PERC / 100)
            tp3 = entry * (1 + TP3_PERC / 100)
            tp4 = entry * (1 + TP4_PERC / 100)
        else:
            entry = current_price + (atr * 1.0)
            sl = current_price + sl_distance
            # FIXED: الأهداف تُحسب من entry وليس من current_price
            tp1 = entry * (1 - TP1_PERC / 100)
            tp2 = entry * (1 - TP2_PERC / 100)
            tp3 = entry * (1 - TP3_PERC / 100)
            tp4 = entry * (1 - TP4_PERC / 100)

        # نسبة R:R ديناميكية (حسب الهدف الرابع)
        risk = abs(entry - sl)
        reward = abs(tp4 - entry)
        rr = round(reward / risk, 1) if risk > 0 else 0

        return {
            'symbol': symbol,
            'direction': direction,
            'accuracy': round(confidence / 10),
            'rr': rr,
            'entry': entry,
            'sl': sl,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'tp4': tp4,
            'atr': atr,
            'sl_distance_pct': round((risk / entry) * 100, 2),  # FIXED: من entry
            'sl_capped': sl_capped
        }

    except Exception as e:
        print(f"  {symbol}: Error - {e}")
        return None


def build_message(signal):
    pair = signal['symbol'].replace('/', '')

    msg = f"""✅Signal Alert: #{pair}

Direction: {signal['direction']}
Leverage: {LEVERAGE}x 

Entry: {_fmt(signal['entry'])}

Targets:
1- {_fmt(signal['tp1'])}
2- {_fmt(signal['tp2'])}
3- {_fmt(signal['tp3'])}
4- {_fmt(signal['tp4'])}

🚫 Stop Loss: {_fmt(signal['sl'])} ({signal['sl_distance_pct']}%)

━━━━━━━━━━━━━━━
🤖 AutoTrade:

No need to monitor the charts! Our Bot executes every trade in real time
Just connect and let the bot follow
Perfect for beginners or busy traders.

L E A K E D B Y: @BULLS_SIGNALS"""

    return msg


def main():
    print("=" * 50)
    print("BULLS SIGNALS v4 — Fixed Edition")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Timeframe: {TIMEFRAME} | HTF: {HTF_TIMEFRAME} | Cooldown: {COOLDOWN_HOURS}h")
    print(f"ATR Multiplier: {ATR_SL_MULTIPLIER}x | Max SL: {MAX_SL_PERC}% | Entry-based Targets")
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
                print(f"  {symbol}: On cooldown ({COOLDOWN_HOURS}h)")
                continue

            time.sleep(0.6)
            signal = analyze_symbol(symbol)

            if signal is None:
                skipped_filter += 1
                continue

            msg = build_message(signal)
            send_telegram(msg)
            cooldown_data[symbol] = datetime.now().isoformat()
            signals_sent += 1
            print(f"  ✅ SIGNAL SENT: {symbol} {signal['direction']} (Acc: {signal['accuracy']}/10, SL: {signal['sl_distance_pct']}%, RR: {signal['rr']}:1)")
            time.sleep(1.5)

        except Exception as e:
            print(f"  {symbol}: Exception - {e}")

    save_cooldown(cooldown_data)
    print(f"\n🏁 Done. Signals: {signals_sent} | Cooldown skipped: {skipped_cooldown} | Filter skipped: {skipped_filter}")


if __name__ == "__main__":
    main()
