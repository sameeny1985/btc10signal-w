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
from supabase import create_client, Client

# ================= CONFIG (حتماً این‌ها را چک کن) =================
SUPABASE_URL = "https://tzjjbuqwwipendmimdfj.supabase.co" 
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR6ampidXF3d2lwZW5kbWltZGZqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzcxNDU2ODgsImV4cCI6MjA5MjcyMTY4OH0.Yub8Kl3pnkRIDPDsyLucAWKbORO4ndHW9oFLueQubQc" # کلید درست را اینجا بگذار
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

LOOKBACK = 60
TELEGRAM_TOKEN = "8753161051:AAFI_4KaBPGzFQH7hLuGPy1Abos20VfcrNs"
CHANNEL_1 = -1003893409389 # Normal
CHANNEL_2 = -1003698594050 # VIP

MODEL_FILE = "lstm_model.h5"
# ================================================================

def send_telegram(msg, chat_id):
    import requests
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"})
    except: pass

def get_price():
    return float(ccxt.mexc().fetch_ticker("BTC/USDT")['last'])

def get_ohlcv():
    try:
        # استفاده از تایم‌فریم 1 دقیقه‌ای برای درک نوسانات نزدیک
        # این دیتا فقط برای "آموزش مدل" استفاده می‌شود، نه قیمت ورود (Entry)
        exchange = ccxt.mexc()
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", '15m', limit=500)
        df = pd.DataFrame(ohlcv, columns=['t', 'o', 'h', 'l', 'c', 'v'])
        
        # ما فقط به قیمت بسته شدن (c) و حجم معاملات (v) برای تحلیل نیاز داریم
        return df[['c', 'v']]
    except Exception as e:
        print(f"Error fetching OHLCV: {e}")
        # بازگرداندن یک دیتای خالی یا قبلی برای جلوگیری از توقف ربات
        return pd.DataFrame()

def market_regime(df):
    returns = df['c'].pct_change()
    volatility = returns.iloc[-20:].std()
    trend = df['c'].iloc[-1] - df['c'].iloc[-20]
    regime = "BULL" if trend > 50 else "BEAR" if trend < -50 else "RANGE"
    return regime, volatility

def prepare(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    X, y = [], []
    for i in range(LOOKBACK, len(df)-1):
        X.append(scaled[i-LOOKBACK:i])
        y.append(1 if scaled[i+1][0] > scaled[i][0] else 0)
    return np.array(X), np.array(y)

# ---------------- هوش مصنوعی (META) برای پیدا کردن الگوی زمانی ----------------
def train_meta_model():
    try:
        response = supabase.table("trading_history").select("*").execute()
        data = response.data
        if len(data) < 10: return None, 0.5 # حداقل ۱۰ دیتا برای شروع
        
        df = pd.DataFrame(data)
        # یادگیری بر اساس: احتمال اولیه، نوسان، ساعت و دقیقه!
        X = df[['confidence', 'volatility', 'hour', 'minute']]
        y = df['result']
        
        winrate = y.mean()
        model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05)
        model.fit(X, y)
        return model, winrate
    except: return None, 0.5

def save_trade(data):
    try:
        supabase.table("trading_history").insert(data).execute()
    except Exception as e: print(f"DB Error: {e}")

# ---------------- زمان‌بندی دقیق ----------------
def wait_for_interval(minutes_step=10):
    while True:
        now = datetime.now()
        if now.minute % minutes_step == 0 and now.second < 2:
            return now
        time.sleep(1)

# ---------------- SERVER ----------------
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is Running...")

def run_server():
    server = HTTPServer(('0.0.0.0', int(os.environ.get("PORT", 10000))), Handler)
    server.serve_forever()

threading.Thread(target=run_server, daemon=True).start()

# ================= MAIN PROCESS =================
last_trade = None

# ================= اصلاح شده برای قیمت لحظه‌ای (TICKER) =================

while True:
    try:
        # ۱. هماهنگی با ثانیه صفر
        now_time = wait_for_interval(60)
        
        # ۲. بلافاصله گرفتن قیمت تیکر (Ticker Price) - قبل از هر کار دیگری
        # این همان عددی است که در مکسی می‌بینی
        exact_ticker_price = get_price() 
        
        # ۳. اعتبارسنجی سیگنال قبلی با قیمت تیکر جدید
        if last_trade and (now_time.minute == (last_trade["minute"] + 10) % 60):
            is_correct = 1 if (last_trade["direction"] == "UP" and exact_ticker_price > last_trade["price"]) or \
                             (last_trade["direction"] == "DOWN" and exact_ticker_price < last_trade["price"]) else 0
            
            save_trade({
                "confidence": last_trade["confidence"],
                "volatility": last_trade["volatility"],
                "hour": last_trade["hour"],
                "minute": last_trade["minute"],
                "result": is_correct
            })

        # ۴. حالا برود سراغ تحلیل سنگین (اینجا دیگر مهم نیست چقدر طول بکشد)
        df = get_ohlcv()
        regime, volatility = market_regime(df)
        X_train, y_train = prepare(df)
        
        # لود و پیش‌بینی
        if os.path.exists(MODEL_FILE): lstm = load_model(MODEL_FILE)
        else:
            lstm = Sequential([LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])), Dense(1, activation="sigmoid")])
            lstm.compile(optimizer="adam", loss="binary_crossentropy")
        
        prob = float(lstm.predict(X_train[-1].reshape(1, *X_train[-1].shape), verbose=0)[0][0])
        direction = "UP" if prob > 0.5 else "DOWN"

        # ۵. ارسال به تلگرام با همان قیمتی که در ثانیه صفر گرفتیم (نه قیمت الان!)
        msg = (f"⏱ **10-MIN BINARY SIGNAL**\n"
               f"Direction: {direction} {'🟢' if direction=='UP' else '🔴'}\n"
               f"Entry (Ticker): `{exact_ticker_price:,.2f}`\n"
               f"Time: {now_time.strftime('%H:%M:%S')}")
        send_telegram(msg, CHANNEL_1)

        # ۶. بخش VIP (الگویابی زمانی)
        meta, winrate = train_meta_model()
        if meta:
            feat = np.array([[prob, volatility, now_time.hour, now_time.minute]])
            meta_prob = meta.predict_proba(feat)[0][1]
            score = (meta_prob * 0.7) + (winrate * 0.3)
            
            if score > 0.72:
                vip_msg = (f"💎 **VIP TICKER SIGNAL**\n"
                           f"Direction: {direction}\n"
                           f"Entry: `{exact_ticker_price:,.2f}`\n"
                           f"AI Score: `{score:.2f}`")
                send_telegram(vip_msg, CHANNEL_2)

        # ۷. ذخیره قیمت ورود برای ۱۰ دقیقه بعد
        last_trade = {
            "price": exact_ticker_price, 
            "direction": direction, 
            "confidence": prob,
            "volatility": volatility, 
            "hour": now_time.hour, 
            "minute": now_time.minute
        }
        
        time.sleep(10)

    except Exception as e:
        print(f"Error: {e}")
        time.sleep(1)
