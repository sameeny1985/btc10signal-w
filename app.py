import time
import os
import numpy as np
import pandas as pd
from datetime import datetime
import requests
import ccxt
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# ================== تنظیمات نهایی (بدون دستکاری) ==================
SYMBOL = "BTC/USDT"
LOOKBACK = 60
SLEEP_SECONDS = 600  # هر 10 دقیقه بررسی مجدد
LOG_FILE = "trading_history.csv"
MODEL_PATH = "lstm_main_v25.keras"

TELEGRAM_TOKEN = "8753161051:AAFI_4KaBPGzFQH7hLuGPy1Abos20VfcrNs"
CHAT_ID_NORMAL = "@btc10signalW"  # کانال اول (سیگنال‌های خام)
CHAT_ID_VIP = "@btc10signalWVIP"       # کانال دوم (الگوهای تایید شده 80%)
# ================================================================

def send_telegram(msg, chat_id):
    """ارسال پیام به تلگرام با مدیریت خطا"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": msg}, timeout=10)
    except Exception as e:
        print(f"Telegram Error: {e}")

def get_data_from_mexc(symbol="BTC/USDT", timeframe='1h', n=500):
    """دقیقاً مطابق کد قبلی شما برای دریافت قیمت از MEXC"""
    try:
        exchange = ccxt.mexc({'enableRateLimit': True})
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=n)
        if not ohlcv: return pd.DataFrame()
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        return df[['close', 'volume']]
    except Exception as e:
        print(f"MEXC Price Error: {e}")
        return pd.DataFrame()

def update_lstm_model(X, y):
    """آموزش و بروزرسانی مغز ربات (LSTM) بدون پاک شدن اطلاعات قبلی"""
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            model.fit(X[-20:], y[-20:], epochs=2, verbose=0)
            model.save(MODEL_PATH)
            return model
        except: pass
    
    model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(X, y, epochs=5, verbose=0)
    model.save(MODEL_PATH)
    return model

# --- بخش Health Check برای جلوگیری از خوابیدن Render/Koyeb ---
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"Bot is Active and Learning...")
    def log_message(self, format, *args): return

def run_health_server():
    httpd = HTTPServer(('0.0.0.0', 8000), HealthCheckHandler)
    httpd.serve_forever()

threading.Thread(target=run_health_server, daemon=True).start()

# --- متغیرهای حافظه برای لایه دوم تحلیل ---
last_price, last_dir, last_conf, last_time = None, None, None, None

print("--- AI MASTER BOT STARTED (MEXC ENGINE) ---")

while True:
    try:
        # ۱. دریافت دیتای تازه
        df = get_data_from_mexc(symbol=SYMBOL, timeframe='1h', n=500)
        if df.empty or len(df) < LOOKBACK:
            time.sleep(30); continue
        
        current_price = df['close'].iloc[-1]
        now = datetime.utcnow()

        # ۲. لایه دوم: ثبت و بررسی نتیجه سیگنال قبلی (تکامل)
        if last_price is not None:
            # فرمول پیروزی/شکست: آیا قیمت در جهت پیش‌بینی حرکت کرد؟
            is_correct = 1 if (last_dir == "UP" and current_price > last_price) or \
                              (last_dir == "DOWN" and current_price < last_price) else 0
            
            # ذخیره تجربه در فایل CSV
            new_entry = pd.DataFrame([{
                "time": last_time, "confidence": last_conf, "direction": last_dir,
                "is_correct": is_correct, "hour": pd.to_datetime(last_time).hour,
                "day": pd.to_datetime(last_time).dayofweek
            }])
            new_entry.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)
            print(f">>> Experience Saved: {last_time} Result: {is_correct}")

        # ۳. لایه اول تحلیل: ترکیب LSTM و XGBoost
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df.values)
        
        X, y = [], []
        for i in range(LOOKBACK, len(scaled) - 1):
            X.append(scaled[i-LOOKBACK:i])
            y.append(1 if scaled[i+1][0] > scaled[i][0] else 0)
        X, y = np.array(X), np.array(y)

        # پیش‌بینی لایه اول
        lstm_model = update_lstm_model(X, y)
        lstm_prob = float(lstm_model.predict(X[-1:], verbose=0)[0][0])
        
        xgb_first = XGBClassifier(n_estimators=100, max_depth=4, verbosity=0)
        xgb_first.fit(df.values[LOOKBACK:-1], y)
        xgb_prob = float(xgb_first.predict_proba(df.values[-1:])[0][1])

        final_prob = (lstm_prob + xgb_prob) / 2
        direction = "UP" if final_prob > 0.55 else "DOWN"

        # ۴. ارسال به کانال اول (گزارش عمومی)
        msg_normal = (f"BTC iCF936_v25 (MEXC)\n"
                      f"Price: {current_price:.2f} USD\n"
                      f"Direction: {direction}\n"
                      f"Confidence: {final_prob*100:.2f}%\n"
                      f"Time: {now.strftime('%H:%M')} UTC")
        send_telegram(msg_normal, CHAT_ID_NORMAL)

        # ۵. لایه دوم تحلیل: بررسی الگوهای ۸۰٪ در تاریخچه
        if os.path.exists(LOG_FILE):
            history = pd.read_csv(LOG_FILE)
            if len(history) >= 2: # شروع از دومین سیگنال
                meta_learner = XGBClassifier(n_estimators=50, max_depth=3)
                meta_learner.fit(history[['hour', 'day', 'confidence']], history['is_correct'])
                
                # پیش‌بینی احتمال پیروزی واقعی بر اساس تجربه
                real_accuracy_pattern = meta_learner.predict_proba([[now.hour, now.weekday(), final_prob]])[0][1]
                
                if real_accuracy_pattern >= 0.80:
                    msg_vip = (f"💎 VIP GOLDEN SIGNAL\n"
                               f"Direction: {direction}\n"
                               f"Pattern Confidence: {real_accuracy_pattern*100:.2f}%\n"
                               f"Based on {len(history)} past signals.")
                    send_telegram(msg_vip, CHAT_ID_VIP)

        # ذخیره برای دور بعد (10 دقیقه دیگر)
        last_price, last_dir, last_conf, last_time = current_price, direction, final_prob, now.strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"[{now.strftime('%H:%M')}] Signal Processed. Sleeping...")
        time.sleep(SLEEP_SECONDS)

    except Exception as e:
        print(f"MAIN ERROR: {e}")
        time.sleep(30)
