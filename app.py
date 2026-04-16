import time
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import ccxt
import threading
import os
from http.server import HTTPServer, BaseHTTPRequestHandler

# ================= CONFIG =================
LOOKBACK = 60
SLEEP_SECONDS = 600

TELEGRAM_TOKEN = "8753161051:AAFI_4KaBPGzFQH7hLuGPy1Abos20VfcrNs"
CHANNEL_1 = "@btc10signalW"
CHANNEL_2 = "@btc10signalWVIP"

HISTORY_FILE = "trading_history.csv"
MODEL_FILE = "lstm_model.h5"
# ==========================================

# ---------------- TELEGRAM ----------------
def send_telegram(msg, chat_id):
    import requests
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": msg}
        )
    except:
        pass

# ---------------- DATA ----------------
def get_price():
    return float(ccxt.mexc().fetch_ticker("BTC/USDT")['last'])

def get_ohlcv():
    ohlcv = ccxt.mexc().fetch_ohlcv("BTC/USDT", '1h', limit=500)
    df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
    return df[['c','v']]

# ---------------- MARKET REGIME ----------------
def market_regime(df):
    close = df['c']
    returns = close.pct_change()

    trend = close.iloc[-1] - close.iloc[-20]
    volatility = returns.std()

    if trend > 100:
        regime = "BULL"
        threshold = 0.52
    elif trend < -100:
        regime = "BEAR"
        threshold = 0.52
    else:
        regime = "RANGE"
        threshold = 0.60

    return regime, threshold, volatility

# ---------------- LSTM ----------------
def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model

def prepare(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(LOOKBACK, len(df)-1):
        X.append(scaled[i-LOOKBACK:i])
        y.append(1 if scaled[i+1][0] > scaled[i][0] else 0)

    return np.array(X), np.array(y)

# ---------------- META MODEL ----------------
def train_meta_model():
    if not os.path.exists(HISTORY_FILE):
        return None, 0.5

    df = pd.read_csv(HISTORY_FILE)

    if len(df) < 5:
        return None, 0.5

    df['weight'] = np.linspace(0.1, 1.0, len(df))

    if len(df) > 15000:
        df = df.sample(15000, weights=df['weight'])

    X = df[['confidence','volatility']]
    y = df['result']
    w = df['weight']

    winrate = y.mean()

    model = XGBClassifier(n_estimators=100, max_depth=4)
    model.fit(X, y, sample_weight=w)

    return model, winrate

# ---------------- SAVE ----------------
def save_trade(data):
    pd.DataFrame([data]).to_csv(
        HISTORY_FILE,
        mode='a',
        header=not os.path.exists(HISTORY_FILE),
        index=False
    )

# ---------------- SERVER (FIXED PORT) ----------------
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"alive")

PORT = int(os.environ.get("PORT", 10000))

def run_server():
    server = HTTPServer(('0.0.0.0', PORT), Handler)
    server.serve_forever()

threading.Thread(target=run_server, daemon=True).start()

# ---------------- MEMORY ----------------
last_trade = None

# ================= MAIN LOOP =================
while True:
    try:
        df = get_ohlcv()
        X, y = prepare(df)

        if len(X) < 10:
            time.sleep(60)
            continue

        # -------- LSTM --------
        if os.path.exists(MODEL_FILE):
            lstm = load_model(MODEL_FILE)
        else:
            lstm = build_lstm((X.shape[1], X.shape[2]))

        lstm.fit(X, y, epochs=1, verbose=0)
        lstm.save(MODEL_FILE)

        lstm_prob = float(lstm.predict(X[-1].reshape(1,*X[-1].shape))[0][0])

        # -------- XGB --------
        xgb = XGBClassifier(n_estimators=30)
        xgb.fit(df.values[LOOKBACK:-1], y)
        xgb_prob = xgb.predict_proba(df.values[-1].reshape(1,-1))[0][1]

        base_prob = (lstm_prob + xgb_prob) / 2

        regime, threshold, volatility = market_regime(df)
        direction = "UP" if base_prob > threshold else "DOWN"

        price = get_price()
        current_time = datetime.now().strftime("%H:%M:%S")

        msg_normal = (
            f"📊 NORMAL SIGNAL\n"
            f"━━━━━━━━━━━━\n"
            f"Direction: {direction} {'🟢' if direction=='UP' else '🔴'}\n"
            f"Price: {price:,.2f}\n"
            f"Confidence: {base_prob:.2%}\n"
            f"Regime: {regime}\n"
            f"Time: {current_time}"
        )
        send_telegram(msg_normal, CHANNEL_1)
        )

        # VALIDATE
        if last_trade:
            correct = int(
                (last_trade["direction"]=="UP" and price>last_trade["price"]) or
                (last_trade["direction"]=="DOWN" and price<last_trade["price"])
            )

            save_trade({
                "confidence": last_trade["confidence"],
                "volatility": last_trade["volatility"],
                "result": correct
            })

        # META
        meta, winrate = train_meta_model()

        if meta:
            features = np.array([[base_prob, volatility]])
            meta_prob = meta.predict_proba(features)[0][1]

            score = (meta_prob * 0.6) + (winrate * 0.4)

            if score > 0.65:
                vip_text = (
                    f"💎 VIP SIGNAL (High Accuracy)\n"
                    f"━━━━━━━━━━━━\n"
                    f"Direction: {direction} {'🟢' if direction=='UP' else '🔴'}\n"
                    f"Entry Price: {price:,.2f}\n"
                    f"AI Score: {score:.2f}\n"
                    f"System Winrate: {winrate:.1%}\n"
                    f"Regime: {regime}\n"
                    f"Time: {current_time}"
                )
                send_telegram(vip_text, CHANNEL_2)

        last_trade = {
            "price": price,
            "direction": direction,
            "confidence": base_prob,
            "volatility": volatility
        }

        time.sleep(SLEEP_SECONDS)

    except Exception as e:
        print("ERROR:", e)
        time.sleep(60)
