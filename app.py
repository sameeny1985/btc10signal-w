import time
import numpy as np
import pandas as pd
from datetime import datetime
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

TELEGRAM_TOKEN = "YOUR_TOKEN"
CHANNEL_1 = "@btc10signalW"
CHANNEL_2 = "@btc10signalWVIP"

HISTORY_FILE = "trading_history.csv"
MODEL_FILE = "lstm_model.h5"
# ==========================================

# ---------------- TELEGRAM ----------------
def send_telegram(msg, chat_id):
    import requests
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": msg})
    except Exception as e:
        print("Telegram Error:", e)

# ---------------- DATA ----------------
def get_price():
    exchange = ccxt.mexc()
    ticker = exchange.fetch_ticker("BTC/USDT")
    return float(ticker['last'])

def get_ohlcv():
    exchange = ccxt.mexc()
    ohlcv = exchange.fetch_ohlcv("BTC/USDT", '1h', limit=500)
    df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
    return df[['c','v']]

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
        return None

    df = pd.read_csv(HISTORY_FILE)

    if len(df) < 2:
        return None

    X = df[['confidence']]
    y = df['result']

    model = XGBClassifier(n_estimators=20, max_depth=3)
    model.fit(X, y)
    return model

# ---------------- SAVE HISTORY ----------------
def save_trade(data):
    df = pd.DataFrame([data])
    df.to_csv(HISTORY_FILE, mode='a', header=not os.path.exists(HISTORY_FILE), index=False)

# ---------------- WEB SERVER ----------------
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"alive")

def run_server():
    HTTPServer(('0.0.0.0', 8000), Handler).serve_forever()

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

        # -------- LSTM LOAD / TRAIN --------
        if os.path.exists(MODEL_FILE):
            lstm = load_model(MODEL_FILE)
        else:
            lstm = build_lstm((X.shape[1], X.shape[2]))

        lstm.fit(X, y, epochs=1, verbose=0)
        lstm.save(MODEL_FILE)

        lstm_prob = float(lstm.predict(X[-1].reshape(1,*X[-1].shape))[0][0])

        # -------- XGBoost --------
        xgb = XGBClassifier(n_estimators=30, max_depth=3)
        xgb.fit(df.values[LOOKBACK:-1], y)
        xgb_prob = xgb.predict_proba(df.values[-1].reshape(1,-1))[0][1]

        # -------- FINAL --------
        final_prob = (lstm_prob + xgb_prob) / 2
        direction = "UP" if final_prob > 0.55 else "DOWN"

        price = get_price()
        now = datetime.utcnow()

        # -------- CHANNEL 1 --------
        msg = f"""
BTC SIGNAL

Price: {price}
Direction: {direction}
Confidence: {final_prob*100:.2f}
Time: {now}
"""
        send_telegram(msg, CHANNEL_1)

        # -------- VALIDATE PREVIOUS TRADE --------
        if last_trade is not None:
            new_price = price

            correct = 0
            if last_trade["direction"] == "UP" and new_price > last_trade["price"]:
                correct = 1
            elif last_trade["direction"] == "DOWN" and new_price < last_trade["price"]:
                correct = 1

            save_trade({
                "timestamp": last_trade["time"],
                "confidence": last_trade["confidence"],
                "direction": 1 if last_trade["direction"]=="UP" else 0,
                "result": correct
            })

        # -------- META MODEL (IMMEDIATE LEARNING) --------
        meta = train_meta_model()

        if meta:
            features = np.array([[final_prob]])
            meta_prob = meta.predict_proba(features)[0][1]

            if meta_prob > 0.55:
                vip_msg = f"""
VIP SIGNAL 🔥

Direction: {direction}
Confidence: {meta_prob*100:.2f}
"""
                send_telegram(vip_msg, CHANNEL_2)

        # -------- SAVE CURRENT TRADE --------
        last_trade = {
            "time": now,
            "price": price,
            "direction": direction,
            "confidence": final_prob
        }

        time.sleep(SLEEP_SECONDS)

    except Exception as e:
        print("ERROR:", e)
        send_telegram(f"ERROR: {e}", CHANNEL_1)
        time.sleep(60)
