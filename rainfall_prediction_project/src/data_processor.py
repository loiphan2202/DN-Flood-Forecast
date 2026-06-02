import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from config.config import CONFIG

class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def prepare_data(self, df):
        """Chuẩn bị dữ liệu cho dự đoán"""
        if df is None or len(df) < CONFIG['REQUIRED_DATAPOINTS']:
            return None

        try:
            # Thêm các features phái sinh
            df = self.add_derived_features(df)
            
            # Chuẩn hóa dữ liệu
            scaled_data = self.scaler.fit_transform(df[CONFIG['FEATURES']])
            
            # Reshape cho LSTM (samples, time steps, features)
            return scaled_data.reshape(1, scaled_data.shape[0], scaled_data.shape[1])
            
        except Exception as e:
            print(f"Lỗi xử lý dữ liệu: {e}")
            return None
            
    def add_derived_features(self, df):
        """Thêm các features phái sinh"""
        df['Mưa_1h_trước'] = df['Mưa'].shift(1)
        df['Mưa_3h_trước'] = df['Mưa'].shift(3)
        df['Mưa_6h_trước'] = df['Mưa'].shift(6)
        
        # Thêm các đặc trưng thời gian
        df['Giờ'] = df.index.hour
        df['Ngày'] = df.index.day
        df['Tháng'] = df.index.month
        
        # Điền các giá trị NaN
        df = df.fillna(method='ffill')
        return df