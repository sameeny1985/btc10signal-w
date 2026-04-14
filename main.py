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

# ---------------- MARKET FEATURES ----------------
def market_features(df):
    close = df['c']
    trend = close.iloc[-1] - close.iloc[-10]
    returns = close.pct_change()

    volatility = returns.std()
    momentum = returns.mean()

    return trend, volatility, momentum

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

# ---------------- META MODEL (SMART) ----------------
def train_meta_model():
    if not os.path.exists(HISTORY_FILE):
        return None

    df = pd.read_csv(HISTORY_FILE)

    if len(df) < 5:
        return None

    # وزن‌دهی (داده جدید مهم‌تر)
    df['weight'] = np.linspace(0.1, 1.0, len(df))

    # نمونه‌گیری هوشمند
    if len(df) > 10000:
        df = df.sample(10000, weights=df['weight'])

    X = df[['confidence','trend','volatility','momentum']]
    y = df['result']
    w = df['weight']

    model = XGBClassifier(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.05
    )

    model.fit(X, y, sample_weight=w)
    return model

# ---------------- SAVE ----------------
def save_trade(data):
    pd.DataFrame([data]).to_csv(
        HISTORY_FILE,
        mode='a',
        header=not os.path.exists(HISTORY_FILE),
        index=False
    )

# ---------------- WEB SERVER ----------------
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"alive")

threading.Thread(
    target=lambda: HTTPServer(('0.0.0.0',8000), Handler).serve_forever(),
    daemon=True
).start()

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
        send_telegram(
            f"BTC SIGNAL\nPrice: {price}\nDir: {direction}\nConf: {final_prob:.2f}",
            CHANNEL_1
        )

        # -------- VALIDATION --------
        if last_trade:
            correct = int(
                (last_trade["direction"]=="UP" and price>last_trade["price"]) or
                (last_trade["direction"]=="DOWN" and price<last_trade["price"])
            )

            trend, vol, mom = market_features(df)

            save_trade({
                "confidence": last_trade["confidence"],
                "trend": trend,
                "volatility": vol,
                "momentum": mom,
                "result": correct
            })

        # -------- META MODEL --------
        meta = train_meta_model()

        if meta:
            trend, vol, mom = market_features(df)

            features = np.array([[final_prob, trend, vol, mom]])
            meta_prob = meta.predict_proba(features)[0][1]

            if meta_prob > 0.6:
                send_telegram(
                    f"VIP 🔥\nDir: {direction}\nScore: {meta_prob:.2f}",
                    CHANNEL_2
                )

        # -------- SAVE CURRENT --------
        last_trade = {
            "price": price,
            "direction": direction,
            "confidence": final_prob
        }

        time.sleep(SLEEP_SECONDS)

    except Exception as e:
        print("ERROR:", e)
        time.sleep(60)
