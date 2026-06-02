import time

from config.config import CONFIG
from src.data_processor import DataProcessor
from src.firebase_handler import FirebaseHandler
from src.predictor import RainfallPredictor
from utils.helpers import display_predictions


def main():
    
    firebase = FirebaseHandler()
    processor = DataProcessor()
    predictor = RainfallPredictor()

    print("Bắt đầu monitoring...")

    while True:
        try:
            # dữ liệu mới nhất
            df = firebase.get_latest_sensor_data()
            
            # Xử lý
            processed_data = processor.prepare_data(df)
            
            if processed_data is not None:
                # dự đoán
                predictions = predictor.predict(processed_data)
                dates = predictor.generate_dates()
                
                # kết quả
                display_predictions(predictions, dates)
                
                # Cập nhật lên Firebase
                firebase.update_predictions(predictions, dates)
            
            # Đợi lần cập nhật tiếp theo
            time.sleep(CONFIG['UPDATE_INTERVAL'])
            
        except Exception as e:
            print(f"Lỗi: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()