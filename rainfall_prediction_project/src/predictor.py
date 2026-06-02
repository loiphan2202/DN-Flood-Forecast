import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from datetime import datetime, timedelta
from config.config import CONFIG
import numpy as np

class RainfallPredictor:
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        """Load model đã train"""
        try:
            return load_model(CONFIG['MODEL_PATH'])
        except Exception as e:
            print(f"Lỗi khi tải model: {e}")
            return None

    def predict(self, data):
        """Thực hiện dự đoán"""
        if self.model is None or data is None:
            return None
        
        try:
            predictions = self.model.predict(data)
            # Đảm bảo giá trị dự đoán không âm
            predictions = np.clip(predictions, 0, None)
            return predictions
        except Exception as e:
            print(f"Lỗi dự đoán: {e}")
            return None

    def generate_dates(self, num_days=7):
        """Tạo danh sách ngày cho dự đoán"""
        start_date = datetime.now()
        return [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') 
                for i in range(num_days)]