import time
import os
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import requests
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# ================== تنظیمات حافظه و کانال ==================
SYMBOL = "BITCOIN"
LOOKBACK = 60
LOG_FILE = "trading_history.csv" 
MODEL_PATH = "lstm_main_v25.keras"

TOKEN = "8560780520:AAFLbdAOW8-j1mTXHmbUHOwZ6cAzudEmlj8"
CHAT_ID_NORMAL = "@rrxfs"        
CHAT_ID_VIP = "@VIP_CHANNEL"     # کانال دوم شما
# =========================================================

def send_telegram(msg, chat_id):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try: requests.post(url, json={"chat_id": chat_id, "text": msg}, timeout=10)
    except: pass

def update_model_continuous(X, y):
    """به‌روزرسانی پیوسته مغز مدل با دیتای جدید بازار"""
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            model.fit(X[-10:], y[-10:], epochs=2, verbose=0) 
            model.save(MODEL_PATH)
            return model
        except: pass
    
    model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(X, y, epochs=5, verbose=0)
    model.save(MODEL_PATH)
    return model

# متغیرهای حافظه برای مقایسه لحظه‌ای
last_price = None
last_dir = None
last_conf = None
last_time = None

print("--- ربات فعال شد: شروع یادگیری از سیگنال دوم ---")

while True:
    try:
        if not mt5.initialize(): time.sleep(5); continue
        rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 800)
        mt5.shutdown()
        
        if rates is None or len(rates) == 0: continue
        df_raw = pd.DataFrame(rates)
        current_price = df_raw['close'].iloc[-1]
        now = datetime.now()

        # --- ۱. بررسی بلافاصله سیگنال قبلی و ثبت در حافظه ---
        if last_price is not None:
            # تعیین درستی سیگنال قبلی با قیمت فعلی
            is_correct = 1 if (last_dir == "UP" and current_price > last_price) or \
                              (last_dir == "DOWN" and current_price < last_price) else 0
            
            # ذخیره تجربه جدید در فایل CSV برای تکامل الگوها
            new_log = pd.DataFrame([{
                "time": last_time, "confidence": last_conf, 
                "direction": last_dir, "is_correct": is_correct,
                "hour": pd.to_datetime(last_time).hour,
                "day": pd.to_datetime(last_time).dayofweek
            }])
            new_log.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)
            print(f"تجربه جدید ثبت شد. نتیجه سیگنال قبل: {is_correct}")

        # --- ۲. تحلیل هوش مصنوعی (LSTM + XGB) ---
        df_raw['returns'] = df_raw['close'].pct_change()
        df_raw.dropna(inplace=True)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df_raw[['returns', 'tick_volume']].values)
        
        X, y = [], []
        for i in range(LOOKBACK, len(scaled)):
            X.append(scaled[i-LOOKBACK:i]); y.append(1 if scaled[i][0] > 0 else 0)
        X, y = np.array(X), np.array(y)

        main_model = update_model_continuous(X, y)
        l_prob = float(main_model.predict(X[-1:], verbose=0)[0][0])
        
        xgb = XGBClassifier(n_estimators=50, max_depth=3, eval_metric='logloss')
        min_s = min(len(scaled[LOOKBACK-1:-1]), len(y[:-1]))
        xgb.fit(scaled[LOOKBACK-1:-1][:min_s], y[:-1][:min_s])
        x_prob = float(xgb.predict_proba(scaled[-1:])[0][1])

        final_prob = (l_prob + x_prob) / 2
        direction = "UP" if final_prob > 0.50 else "DOWN"

        # --- ۳. ارسال به کانال اول (بلافاصله) ---
        msg = f"BTC/USD\nPrice: {current_price:.2f}\nDir: {direction}\nConf: {int(final_prob*100)}%"
        send_telegram(msg, CHAT_ID_NORMAL)

        # --- ۴. بخش تکاملی: ارسال به کانال دوم (VIP) ---
        if os.path.exists(LOG_FILE):
            history = pd.read_csv(LOG_FILE)
            # اگر حداقل ۱ تجربه در حافظه باشد، شروع به الگویابی می‌کند
            if len(history) >= 1:
                # آموزش مدل الگوشناس بر اساس تمام تجربیات ثبت شده تاکنون
                meta_xgb = XGBClassifier()
                meta_xgb.fit(history[['hour', 'day', 'confidence']], history['is_correct'])
                
                # پیش‌بینی احتمال موفقیت بر اساس ساعت و روز فعلی
                pattern_acc = meta_xgb.predict_proba([[now.hour, now.weekday(), final_prob]])[0][1]
                
                # هر چه دیتا بیشتر شود، این شرط دقیق‌تر عمل می‌کند
                # در ابتدا با درصد پایین‌تر هم ارسال می‌کند، اما رفته‌رفته سخت‌گیرتر می‌شود
                vip_msg = (f"💎 سیگنال تایید شده\nجهت: {direction}\n"
                           f"تطابق با تجربه قبلی: {int(pattern_acc*100)}%\n"
                           f"تعداد تجربیات ثبت شده: {len(history)}")
                send_telegram(vip_msg, CHAT_ID_VIP)

        # ذخیره وضعیت فعلی برای مقایسه در ۱۰ دقیقه بعد
        last_price, last_dir, last_conf, last_time = current_price, direction, final_prob, now.strftime('%Y-%m-%d %H:%M:%S')

        print(f"[{now.strftime('%H:%M')}] تحلیل انجام شد. منتظر تیک بعدی...")
        time.sleep(600) # بررسی هر ۱۰ دقیقه

    except Exception as e:
        print(f"Error: {e}"); time.sleep(10)
