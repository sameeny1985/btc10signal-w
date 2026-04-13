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
CHAT_ID_VIP = "@VIP_CHANNEL"     # آیدی کانال دوم خود را اینجا بزنید
# =========================================================

def send_telegram(msg, chat_id):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": msg}, timeout=10)
    except:
        pass

def update_model_continuous(X, y):
    """مدیریت مغز ربات: لود کردن مدل قدیمی و آموزش روی دیتای جدید"""
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            # آموزش خیلی سریع روی آخرین داده‌ها برای آپدیت شدن
            model.fit(X[-10:], y[-10:], epochs=2, verbose=0) 
            model.save(MODEL_PATH)
            return model
        except:
            pass
    
    # اگر فایل مدل وجود نداشت، از صفر ساخته می‌شود
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

# متغیرهای حافظه برای مقایسه قیمت‌ها (این‌ها در رم می‌مانند)
last_price, last_dir, last_conf, last_time = None, None, None, None

print("--- AI MASTER BOT: INITIALIZED ---")

while True:
    try:
        # 1. اتصال به متاتریدر و دریافت قیمت
        if not mt5.initialize():
            time.sleep(5); continue
        
        mt5.symbol_select(SYMBOL, True)
        rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 800)
        mt5.shutdown()
        
        if rates is None or len(rates) == 0:
            time.sleep(10); continue
            
        df_raw = pd.DataFrame(rates)
        current_price = df_raw['close'].iloc[-1]
        now = datetime.now()

        # --- 2. بخش راستی‌آزمایی (Validation) ---
        # مقایسه قیمت سیگنال قبل با قیمت الان برای فهمیدن درست/غلط بودن
        if last_price is not None:
            is_correct = 1 if (last_dir == "UP" and current_price > last_price) or \
                              (last_dir == "DOWN" and current_price < last_price) else 0
            
            # ذخیره این تجربه در فایل CSV (حافظه دائمی)
            history_entry = pd.DataFrame([{
                "time": last_time, 
                "confidence": last_conf, 
                "direction": last_dir, 
                "is_correct": is_correct,
                "hour": pd.to_datetime(last_time).hour,
                "day": pd.to_datetime(last_time).dayofweek,
                "entry_price": last_price,
                "exit_price": current_price
            }])
            history_entry.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)
            print(f">>> [LOG] Signal at {last_time} was Verified as: {is_correct}")

        # --- 3. تحلیل لایه اول (LSTM + XGBoost) ---
        df_raw['returns'] = df_raw['close'].pct_change()
        df_raw.dropna(inplace=True)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_raw[['returns', 'tick_volume']].values)
        
        X, y = [], []
        for i in range(LOOKBACK, len(scaled_data)):
            X.append(scaled_data[i-LOOKBACK:i])
            y.append(1 if scaled_data[i][0] > 0 else 0)
        X, y = np.array(X), np.array(y)

        # آپدیت مغز ربات و پیش‌بینی
        main_brain = update_model_continuous(X, y)
        lstm_p = float(main_brain.predict(X[-1:], verbose=0)[0][0])
        
        xgb_init = XGBClassifier(n_estimators=50, max_depth=3, eval_metric='logloss')
        # تراز کردن ابعاد برای جلوگیری از خطای همیشگی
        train_x = scaled_data[LOOKBACK-1:-1]
        min_len = min(len(train_x), len(y[:-1]))
        xgb_init.fit(train_x[:min_len], y[:-1][:min_len])
        xgb_p = float(xgb_init.predict_proba(scaled_data[-1:])[0][1])

        # میانگین قدرت سیگنال
        final_prob = (lstm_p + xgb_p) / 2
        direction = "UP" if final_prob > 0.50 else "DOWN"

        # --- 4. ارسال گزارش به کانال اول (گزارش عمومی) ---
        normal_msg = (f"📊 تحلیل لحظه‌ای بیت‌کوین\n"
                      f"قیمت فعلی: {current_price:.2f}\n"
                      f"جهت پیش‌بینی: {direction}\n"
                      f"اطمینان اولیه: {int(final_prob*100)}%\n"
                      f"زمان: {now.strftime('%H:%M')}")
        send_telegram(normal_msg, CHAT_ID_NORMAL)

        # --- 5. لایه دوم: الگویابی حرفه‌ای برای کانال VIP ---
        if os.path.exists(LOG_FILE):
            history = pd.read_csv(LOG_FILE)
            if len(history) >= 2: # شروع الگویابی از سیگنال دوم
                # هوش مصنوعی دوم روی تاریخچه شکست‌ها و پیروزی‌ها آموزش می‌بیند
                meta_learner = XGBClassifier(n_estimators=100, max_depth=4)
                meta_learner.fit(history[['hour', 'day', 'confidence']], history['is_correct'])
                
                # تخمین شانس موفقیت واقعی بر اساس زمان و تجربه
                real_chance = meta_learner.predict_proba([[now.hour, now.weekday(), final_prob]])[0][1]
                
                # فیلتر طلایی: اگر الگو شانس موفقیت را بالای 80% تشخیص دهد
                if real_chance >= 0.80:
                    vip_msg = (f"💎 سیگنال طلایی (تایید شده با الگو)\n"
                               f"جهت پیشنهادی: {direction}\n"
                               f"شانس موفقیت واقعی: {int(real_chance*100)}%\n"
                               f"تعداد تجربیات تحلیل شده: {len(history)}\n"
                               f"قیمت ورود: {current_price:.2f}")
                    send_telegram(vip_msg, CHAT_ID_VIP)
                    print("🚀 VIP Signal Sent based on verified pattern.")

        # --- 6. ذخیره وضعیت برای مقایسه در 10 دقیقه بعد ---
        last_price = current_price
        last_dir = direction
        last_conf = final_prob
        last_time = now.strftime('%Y-%m-%d %H:%M:%S')

        print(f"[{now.strftime('%H:%M')}] Analysis Done. Sleep 10 mins...")
        time.sleep(600) 

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        time.sleep(20)
