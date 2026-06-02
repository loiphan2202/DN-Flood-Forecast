!pip install scikit-learn matplotlib seaborn pandas numpy
!pip install tensorflow==2.13.0

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Reshape
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import seaborn as sns
from tensorflow.keras.layers import Input
import tensorflow as tf
from google.colab import drive

# Mount Google Drive
drive.mount('/content/gdrive')

def load_data():
    """Đọc dữ liệu từ file CSV"""
    try:
        sensor_data = pd.read_csv('/content/gdrive/MyDrive/AiForLife/generated_sensor_data.csv')
        sensor_data['Datetime'] = pd.to_datetime(sensor_data['Datetime'])
        return sensor_data
    except Exception as e:
        print(f"Lỗi khi đọc dữ liệu: {e}")
        return None

def describe_data(sensor_data):
    """Hiển thị thông tin và biểu đồ về dữ liệu"""
    print("\nThông tin về dữ liệu cảm biến:")
    print(sensor_data.info())
    print("\nMô tả thống kê dữ liệu cảm biến:")
    print(sensor_data.describe())

    # Vẽ biểu đồ phân phối dữ liệu
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    sns.histplot(sensor_data['rain'])
    plt.title('Phân phối lượng mưa')

    plt.subplot(2, 2, 2)
    sns.histplot(sensor_data['Distance'])
    plt.title('Phân phối mực nước')
    
    plt.subplot(2, 2, 3)
    sns.histplot(sensor_data['humidity'])
    plt.title('Phân phối độ ẩm')
    
    plt.subplot(2, 2, 4)
    sns.histplot(sensor_data['temperature'])
    plt.title('Phân phối nhiệt độ')
    
    plt.tight_layout()
    plt.show()

def preprocess_data(sensor_data, samples_per_3hours=2160):
    """Tiền xử lý dữ liệu với một số cải tiến"""
    try:
        # Xử lý outliers trước khi gom nhóm
        Q1_rain = sensor_data['rain'].quantile(0.25)
        Q3_rain = sensor_data['rain'].quantile(0.75)
        IQR_rain = Q3_rain - Q1_rain
        
        Q1_dist = sensor_data['Distance'].quantile(0.25)
        Q3_dist = sensor_data['Distance'].quantile(0.75)
        IQR_dist = Q3_dist - Q1_dist
        
        # Lọc outliers
        sensor_data = sensor_data[
            (sensor_data['rain'] >= Q1_rain - 1.5 * IQR_rain) & 
            (sensor_data['rain'] <= Q3_rain + 1.5 * IQR_rain) &
            (sensor_data['Distance'] >= Q1_dist - 1.5 * IQR_dist) &
            (sensor_data['Distance'] <= Q3_dist + 1.5 * IQR_dist)
        ].copy()
        
        # Thêm cột timestamp để group theo 3 giờ
        sensor_data['timestamp'] = sensor_data['Datetime'].dt.floor('3H')
        
        # Thêm các features thời gian
        sensor_data['hour'] = sensor_data['Datetime'].dt.hour
        sensor_data['day_of_week'] = sensor_data['Datetime'].dt.dayofweek
        sensor_data['month'] = sensor_data['Datetime'].dt.month
        
        # Thêm moving average cho rain và Distance
        sensor_data['rain_ma'] = sensor_data['rain'].rolling(window=samples_per_3hours//24).mean()
        sensor_data['distance_ma'] = sensor_data['Distance'].rolling(window=samples_per_3hours//24).mean()
        
        # Group theo 3 giờ và tính các thống kê
        three_hourly_data = sensor_data.groupby('timestamp').agg({
            'rain': ['mean', 'max', 'std'],  # Thêm std để nắm bắt độ biến động
            'Distance': ['mean', 'min', 'max', 'std'],
            'humidity': ['mean', 'std'],
            'temperature': ['mean', 'std'],
            'soil_moisture': ['mean', 'std'],
            'rain_ma': 'mean',
            'distance_ma': 'mean',
            'hour': 'first',
            'day_of_week': 'first',
            'month': 'first'
        }).reset_index()
        
        # Làm phẳng các cột
        three_hourly_data.columns = ['timestamp', 
                                   'rain_mean', 'rain_max', 'rain_std',
                                   'distance_mean', 'distance_min', 'distance_max', 'distance_std',
                                   'humidity_mean', 'humidity_std',
                                   'temperature_mean', 'temperature_std',
                                   'soil_moisture_mean', 'soil_moisture_std',
                                   'rain_ma', 'distance_ma',
                                   'hour', 'day_of_week', 'month']
        
        print(f"Số lượng mẫu sau khi gom nhóm 3 giờ: {len(three_hourly_data)}")
        
        # Chuẩn hóa dữ liệu
        scaler = RobustScaler()
        features = [col for col in three_hourly_data.columns if col != 'timestamp']
        scaled_data = scaler.fit_transform(three_hourly_data[features])
        
        return scaled_data, scaler, three_hourly_data
    
    except Exception as e:
        print(f"Lỗi trong quá trình tiền xử lý: {e}")
        return None, None, None

def prepare_sequences(scaled_data, input_sequence_length=24, prediction_length=24):
    """Chuẩn bị sequences cho LSTM với dự đoán nhiều bước"""
    X, y = [], []
    for i in range(input_sequence_length, len(scaled_data) - prediction_length):
        X.append(scaled_data[i-input_sequence_length:i])
        # Chỉ lấy 2 features đầu tiên (rain_mean và rain_max) cho y
        y.append(scaled_data[i:i+prediction_length, :2])  # Thay đổi [0, 1] thành :2
    return np.array(X), np.array(y)

def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    """Chia dữ liệu thành tập train, validation và test"""
    n = len(X)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    # Chia dữ liệu
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    print(f"\nKích thước các tập dữ liệu:")
    print(f"Train: {len(X_train)} mẫu")
    print(f"Validation: {len(X_val)} mẫu")
    print(f"Test: {len(X_test)} mẫu")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def create_model(input_sequence_length=24, prediction_length=24, n_features=18):
    model = Sequential([
        Input(shape=(input_sequence_length, n_features)),
        
        # Tăng độ phức tạp của mô hình
        LSTM(512, return_sequences=True, activation='tanh'),
        BatchNormalization(),
        Dropout(0.2),
        
        LSTM(256, return_sequences=True, activation='tanh'),
        BatchNormalization(),
        Dropout(0.2),
        
        LSTM(128, return_sequences=True, activation='tanh'),
        BatchNormalization(),
        Dropout(0.2),
        
        LSTM(64, activation='tanh'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        
        Dense(prediction_length * 2),
        Reshape((prediction_length, 2))
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Tăng learning rate
    model.compile(
        optimizer=optimizer,
        loss='huber',  # Đổi sang Huber loss để xử lý outliers tốt hơn
        metrics=['mae', 'mape']
    )
    return model

def train_model(X_train, y_train, X_val, y_val):
    """Huấn luyện mô hình với các tham số được điều chỉnh"""
    model = create_model()
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=25,  # Tăng patience
        restore_best_weights=True,
        min_delta=1e-4
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Điều chỉnh factor
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        '/content/gdrive/MyDrive/AiForLife/best_model',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,  # Tăng số epochs
        batch_size=16,  # Giảm batch size
        callbacks=[early_stopping, reduce_lr, checkpoint],
        shuffle=True
    )
    
    return model, history

def evaluate_model(model, X_test, y_test, scaler):
    """Đánh giá mô hình trên tập test"""
    # Dự đoán
    predictions = model.predict(X_test)
    
    # Chọn một mẫu để vẽ
    sample_idx = 0
    
    # Lấy giá trị thực và dự đoán cho mẫu đã chọn
    true_values = y_test[sample_idx]
    pred_values = predictions[sample_idx]
    
    # Tạo dữ liệu giả để inverse transform
    dummy_data = np.zeros((true_values.shape[0], 11))  # Thay đổi từ 5 thành 11 features
    dummy_data[:, [0, 1]] = true_values  # rain_mean và distance_mean
    true_values_original = scaler.inverse_transform(dummy_data)[:, [0, 1]]
    
    dummy_data = np.zeros((pred_values.shape[0], 11))  # Thay đổi từ 5 thành 11 features
    dummy_data[:, [0, 1]] = pred_values
    pred_values_original = scaler.inverse_transform(dummy_data)[:, [0, 1]]
    
    # Vẽ biểu đồ
    plt.figure(figsize=(15, 10))
    
    # Vẽ biểu đồ lượng mưa
    plt.subplot(2, 1, 1)
    plt.plot(true_values_original[:, 0], label='Thực tế - Lượng mưa')
    plt.plot(pred_values_original[:, 0], label='Dự đoán - Lượng mưa')
    plt.title('So sánh lượng mưa thực tế và dự đoán')
    plt.legend()
    
    # Vẽ biểu đồ mực nước
    plt.subplot(2, 1, 2)
    plt.plot(true_values_original[:, 1], label='Thực tế - Mực nước')
    plt.plot(pred_values_original[:, 1], label='Dự đoán - Mực nước')
    plt.title('So sánh mực nước thực tế và dự đoán')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Tính các metrics
    mae_rain = mean_absolute_error(true_values_original[:, 0], pred_values_original[:, 0])
    mae_water = mean_absolute_error(true_values_original[:, 1], pred_values_original[:, 1])
    mape_rain = mean_absolute_percentage_error(true_values_original[:, 0], pred_values_original[:, 0])
    mape_water = mean_absolute_percentage_error(true_values_original[:, 1], pred_values_original[:, 1])
    
    print("\nMetrics đánh giá:")
    print(f"Lượng mưa - MAE: {mae_rain:.4f}, MAPE: {mape_rain:.4f}")
    print(f"Mực nước - MAE: {mae_water:.4f}, MAPE: {mape_water:.4f}")

def predict_next_3days(model, last_3days_data, scaler):
    """Dự đoán 3 ngày tiếp theo"""
    # Chuẩn bị input
    X = last_3days_data.reshape(1, 24, 5)
    
    # Dự đoán
    predictions = model.predict(X)
    
    # Chuyển đổi predictions về giá trị thực
    predictions = predictions.reshape(24, 2)
    
    # Tạo dữ liệu giả để inverse transform
    dummy_data = np.zeros((predictions.shape[0], 5))
    dummy_data[:, [0, 1]] = predictions
    predictions_original = scaler.inverse_transform(dummy_data)[:, [0, 1]]
    
    return predictions_original

def run_model():
    """Hàm chính để chạy toàn bộ quá trình"""
    try:
        # Đọc dữ liệu
        print("Đang đọc dữ liệu...")
        sensor_data = load_data()
        if sensor_data is None:
            return

        # Mô tả dữ liệu
        print("Đang phân tích dữ liệu...")
        describe_data(sensor_data)

        # Tiền xử lý dữ liệu
        print("Đang tiền xử lý dữ liệu...")
        scaled_data, scaler, grouped_data = preprocess_data(sensor_data)
        if scaled_data is None:
            return

        # Chuẩn bị sequences
        print("Đang chuẩn bị sequences...")
        X, y = prepare_sequences(scaled_data)
        
        # Chia dữ liệu
        print("Đang chia dữ liệu...")
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

        # Huấn luyện mô hình
        print("Đang huấn luyện mô hình...")
        model, history = train_model(X_train, y_train, X_val, y_val)

        # Vẽ biểu đồ loss
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Đánh giá mô hình
        print("Đang đánh giá mô hình...")
        evaluate_model(model, X_test, y_test, scaler)

    except Exception as e:
        print(f"Lỗi trong quá trình chạy model: {str(e)}")
        print("Vui lòng kiểm tra lại dữ liệu đầu vào và các tham số.")

if __name__ == "__main__":
    run_model()

