import json
import os

# Đọc cấu hình Firebase từ file JSON
config_path = os.path.join(os.path.dirname(__file__), 'firebase_config.json')
with open(config_path) as f:
    firebase_config = json.load(f)

# Cấu hình chung
CONFIG = {
    'MODEL_PATH': './models/rainfall_prediction_model.h5',
    'FIREBASE_CRED_PATH': 'config/firebase_credentials.json',  # Đổi thành file credentials service account
    'DATABASE_URL': firebase_config['databaseURL'],
    'UPDATE_INTERVAL': 300,
    'REQUIRED_DATAPOINTS': 24,
    'FEATURES': [
        'Mưa',
        'Độ ẩm đất',
        'Nhiệt độ',
        'Độ ẩm không khí',
        'Khoảng cách',
        'Lưu lượng'
    ],
    'WARNING_THRESHOLD': 30,
    'GAUGE_RANGES': {
        'Mưa': [0, 100],
        'Độ ẩm đất': [0, 100]
    }
}