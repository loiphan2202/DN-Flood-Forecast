from datetime import datetime

import firebase_admin
import pandas as pd
from config.config import CONFIG
from firebase_admin import credentials, db


class FirebaseHandler:
    def __init__(self):
        try:
            # Kiểm tra xem Firebase đã được khởi tạo chưa
            if not firebase_admin._apps:
                cred = credentials.Certificate(CONFIG['FIREBASE_CRED_PATH'])
                firebase_admin.initialize_app(cred, {
                    'databaseURL': CONFIG['DATABASE_URL']
                })
            self.db = db.reference()
            print("Firebase initialized successfully")
        except Exception as e:
            print(f"Firebase initialization error: {str(e)}")
            raise e

    # ... existing code ...

    def get_latest_sensor_data(self):
        """Lấy dữ liệu sensor mới nhất"""
        try:
            # Lấy dữ liệu từ cả ESP32_1 và ESP32_2
            esp32_1_data = self.db.child('ESP32_1').order_by_child('Thời gian').limit_to_last(CONFIG['REQUIRED_DATAPOINTS']).get()
            esp32_2_data = self.db.child('ESP32_2').order_by_child('Thời gian').limit_to_last(CONFIG['REQUIRED_DATAPOINTS']).get()
            
            if esp32_1_data and esp32_2_data:
                # Chuyển đổi thành DataFrame
                df1 = pd.DataFrame.from_dict(esp32_1_data, orient='index')
                df2 = pd.DataFrame.from_dict(esp32_2_data, orient='index')
                
                # Xử lý giá trị nhiệt độ (loại bỏ °C)
                if 'Nhiệt độ' in df1.columns:
                    df1['Nhiệt độ'] = df1['Nhiệt độ'].str.replace('°C', '').astype(float)
                
                # Kết hợp dữ liệu
                df = pd.merge(df1, df2, on='Thời gian', how='outer')
                df['Thời gian'] = pd.to_datetime(df['Thời gian'])
                df.set_index('Thời gian', inplace=True)
                
                return df.sort_index()
            return None
        except Exception as e:
            print(f"Lỗi lấy dữ liệu: {e}")
            return None

    def update_predictions(self, predictions, dates):
        """Cập nhật dự đoán lên Firebase"""
        prediction_data = {
            'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'predictions': [
                {
                    'date': date.strftime('%Y-%m-%d %H:%M:%S'),
                    'value': float(pred)
                } for date, pred in zip(dates, predictions[0])
            ],
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.db.child('predictions').set(prediction_data)