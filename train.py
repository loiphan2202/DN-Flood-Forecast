!pip install scikit-learn matplotlib seaborn tensorflow pandas numpy

import pandas as pd   #đọc dữ liệu
import numpy as np      #xử lý dữ liệu
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler  #chuẩn hóa dữ liệu
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score   #r2score: đo mức độ phù hợp, mean_absolute: đo sai số tuyệt đối trung bình, mean_squared: đo % sai số tuyệt đối trung bình
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential, load_model    #đầu vào, tải mô hình
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization   #LSTM: học phụ thuộc,    dropout: tránh học tủ, dense: đầu ra
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  #checkpoint là dùng để lưu mô hình huấn luyện tốt nhất
import matplotlib.pyplot as plt   #vẽ biểu đô
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Thiết lập style cho đồ thị
plt.style.use('seaborn')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

def load_data():
    """Đọc dữ liệu từ file CSV"""
    try:
        # Đọc dữ liệu cảm biến từ ESP32_1 và ESP32_2
        sensor_data = pd.read_csv('updated_sensor.csv')
        sensor_data['Datetime'] = pd.to_datetime(sensor_data['Datetime'])
        
        # Đọc dữ liệu lịch sử
        historical_df = pd.read_csv('vietnam-rainfall-1901-2015.csv',
                                  names=['pr', 'year', 'month'],
                                  skiprows=1)
        
        print("Đã đọc dữ liệu thành công!")
        return sensor_data, historical_df
    except Exception as e:
        print(f"Lỗi khi đọc dữ liệu: {e}")
        return None, None, None

def remove_outliers(df, columns):
    """Loại bỏ outliers sử dụng phương pháp IQR"""
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)
    return df

def add_time_features(df):
    """Thêm các đặc trưng thời gian"""
    df['Giờ'] = df.index.hour
    df['Ngày'] = df.index.day
    df['Tháng'] = df.index.month
    df['Ngày_trong_tuần'] = df.index.dayofweek
    return df

def create_rolling_features(df, column, windows=[1, 3, 6, 12, 24]):
    """Tạo các đặc trưng rolling"""
    for window in windows:
        df[f'{column}_mean_{window}h'] = df[column].rolling(window=window).mean()
        df[f'{column}_max_{window}h'] = df[column].rolling(window=window).max()
        df[f'{column}_min_{window}h'] = df[column].rolling(window=window).min()
    return df

def preprocess_data(sensor_data):
    """Tiền xử lý và chuẩn hóa dữ liệu"""
    # Xử lý dữ liệu thời gian
    sensor_data['Datetime'] = pd.to_datetime(sensor_data['Datetime'])
    sensor_data.set_index('Datetime', inplace=True)
    
    # Xử lý missing values
    sensor_data = sensor_data.interpolate(method='time')
    
    # Xử lý outliers cho tất cả các cột số
    numeric_columns = ['Distance', 'Flow Rate', 'humidity', 'rain', 'soil_moisture', 'temperature']
    sensor_data = remove_outliers(sensor_data, numeric_columns)
    
    # Thêm đặc trưng thời gian
    sensor_data = add_time_features(sensor_data)
    
    # Tạo các đặc trưng rolling cho lượng mưa
    sensor_data = create_rolling_features(sensor_data, 'rain')
    
    # Loại bỏ các hàng có giá trị NaN
    sensor_data = sensor_data.dropna()
    
    # Chuẩn hóa dữ liệu
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(sensor_data)
    
    features = sensor_data.columns.tolist()
    print(f"Số features: {len(features)}")
    print("Các features:", features)
    
    return scaled_data, scaler, sensor_data, features
def prepare_sequences(data, seq_length):
    """Chuẩn bị sequences cho LSTM với cải tiến"""
    X, y = [], []
    for i in range(len(data) - seq_length - 7):
        sequence = data[i:(i + seq_length)]
        # Thay đổi index target từ 0 thành 4 vì cột rain là cột thứ 5
        target = data[i + seq_length:i + seq_length + 7, 4]
        
        if not (np.isnan(sequence).any() or np.isnan(target).any()):
            X.append(sequence)
            y.append(target)
    
    return np.array(X), np.array(y)

def build_model(sequence_length, n_features):
    """Xây dựng mô hình LSTM cải tiến"""
    model = Sequential([
        # First LSTM layer
        LSTM(128, return_sequences=True, 
             input_shape=(sequence_length, n_features),
             kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Second LSTM layer
        LSTM(64, return_sequences=True,
             kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Third LSTM layer
        LSTM(32, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Dense layers
        Dense(16, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dense(7)
    ])
    
    # Sử dụng learning rate cố định thay vì schedule
    optimizer = Adam(learning_rate=0.001)
    
    model.compile(optimizer=optimizer,
                 loss='huber',
                 metrics=['mae', 'mse'])
    
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """Huấn luyện mô hình với các callbacks"""
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    # Model checkpoint
    checkpoint = ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    # Learning rate reduction
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.0001,
        verbose=1
    )
    
    # Training
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, checkpoint, reduce_lr],
        verbose=1
    )
    
    return history, model

def cross_validate_model(X, y, n_splits=5):
    """Đánh giá mô hình bằng k-fold cross validation"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    histories = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f'\nFold {fold + 1}/{n_splits}')
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = build_model(X.shape[1], X.shape[2])
        history, _ = train_model(model, X_train, y_train, X_val, y_val)
        
        # Evaluate on validation set
        val_score = model.evaluate(X_val, y_val, verbose=0)
        scores.append(val_score)
        histories.append(history.history)
        
        print(f'Fold {fold + 1} - Loss: {val_score[0]:.4f}')
    
    return scores, histories
def evaluate_model(model, X_test, y_test, scaler, combined_df):
    """Đánh giá chi tiết mô hình"""
    # Dự đoán trên tập test
    y_pred = model.predict(X_test)
    
    # Điều chỉnh vị trí cột rain trong ma trận
    y_test_reshaped = np.zeros((y_test.shape[0], combined_df.shape[1]))
    y_test_reshaped[:, 4] = y_test[:, 0]  # Đặt vào cột rain (index 4)
    
    y_pred_reshaped = np.zeros((y_pred.shape[0], combined_df.shape[1]))
    y_pred_reshaped[:, 4] = y_pred[:, 0]
    
    # Inverse transform và lấy giá trị cột rain
    y_test_real = scaler.inverse_transform(y_test_reshaped)[:, 4]
    y_pred_real = scaler.inverse_transform(y_pred_reshaped)[:, 4]
    
    # Tính các metrics
    mae = mean_absolute_error(y_test_real, y_pred_real)
    mse = mean_squared_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_real, y_pred_real)
    
    print('\nKết quả đánh giá mô hình:')
    print(f'MAE: {mae:.2f}mm')
    print(f'RMSE: {rmse:.2f}mm')
    print(f'R2 Score: {r2:.3f}')
    
    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(15, 8))
    plt.plot(y_test_real, label='Thực tế', color='blue', alpha=0.7)
    plt.plot(y_pred_real, label='Dự đoán', color='red', alpha=0.7)
    plt.title('So sánh giá trị thực tế và dự đoán', fontsize=14)
    plt.xlabel('Thời gian', fontsize=12)
    plt.ylabel('Lượng mưa (mm)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return mae, rmse, r2

def plot_training_history(history):
    """Vẽ biểu đồ quá trình training"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss qua các epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MAE plot
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title('MAE qua các epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_feature_importance(model, feature_names):
    """Phân tích tầm quan trọng của các features"""
    try:
        # Lấy trọng số từ layer LSTM đầu tiên
        weights = model.layers[0].get_weights()[0]
        
        # Tính độ lớn trung bình của trọng số
        importance = np.mean(np.abs(weights), axis=1)
        
        # Tạo DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Vẽ biểu đồ
        plt.figure(figsize=(12, 6))
        sns.barplot(data=importance_df, x='Importance', y='Feature')
        plt.title('Tầm quan trọng của các features')
        plt.xlabel('Độ quan trọng')
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    except Exception as e:
        print(f"Lỗi khi phân tích tầm quan trọng của features: {e}")
        print(f"Shape của weights: {weights.shape}")
        print(f"Số lượng features: {len(feature_names)}")
        return pd.DataFrame()
def generate_predictions_and_warnings(model, recent_data, scaler, historical_stats):
    """Tạo dự đoán và cảnh báo"""
    # Reshape dữ liệu đầu vào
    input_data = recent_data.reshape(1, recent_data.shape[0], recent_data.shape[1])
    
    # Thực hiện dự đoán
    scaled_prediction = model.predict(input_data)  # Shape: (1, 7) vì model dự đoán 7 ngày
    
    # Tạo ma trận zeros với kích thước phù hợp cho từng ngày
    predictions = []
    for i in range(7):  # 7 ngày
        prediction_reshaped = np.zeros((1, recent_data.shape[1]))
        prediction_reshaped[0, 4] = scaled_prediction[0, i]  # Lấy giá trị dự đoán cho ngày thứ i
        unscaled_pred = scaler.inverse_transform(prediction_reshaped)[0, 4]
        predictions.append(unscaled_pred)
    
    # Chuyển list thành numpy array
    predictions = np.array(predictions)
    
    # Tạo cảnh báo dựa trên giá trị dự đoán cao nhất
    warnings = []
    severity_level = 0
    max_prediction = np.max(predictions)
    
    if max_prediction > 5:  # Ngưỡng mưa lớn
        warnings.append({
            'level': 'KHẨN CẤP',
            'message': f'Dự báo lượng mưa cao nhất ({max_prediction:.1f}mm) vượt quá mức cao!'
        })
        severity_level = 3
    elif max_prediction > 3:  # Ngưỡng mưa vừa
        warnings.append({
            'level': 'CẢNH BÁO CAO',
            'message': f'Dự báo lượng mưa cao nhất ({max_prediction:.1f}mm) cao!'
        })
        severity_level = 2
    elif max_prediction > 1:  # Ngưỡng mưa nhẹ
        warnings.append({
            'level': 'CẢNH BÁO',
            'message': f'Dự báo có mưa nhẹ, cao nhất ({max_prediction:.1f}mm)!'
        })
        severity_level = 1
        
    return predictions, warnings, severity_level

if __name__ == "__main__":
    print("=== BẮT ĐẦU CHƯƠNG TRÌNH DỰ BÁO LƯỢNG MƯA ===")
    
    # 1. Đọc dữ liệu
    print("\nĐang đọc dữ liệu...")
    sensor_df, historical_df = load_data()
    
    if all(df is not None for df in [sensor_df, historical_df]):
        # 2. Tiền xử lý dữ liệu
        print("\nĐang xử lý dữ liệu...")
        scaled_data, scaler, combined_df, features = preprocess_data(sensor_df)
        
        # 3. Chuẩn bị dữ liệu cho mô hình
        sequence_length = 24  # 24 giờ dữ liệu để dự đoán
        X, y = prepare_sequences(scaled_data, sequence_length)
        
        # 4. Chia dữ liệu
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.15)
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        # 5. Xây dựng và huấn luyện mô hình
        print("\nĐang huấn luyện mô hình...")
        model = build_model(sequence_length, len(features))
        history, trained_model = train_model(model, X_train, y_train, X_val, y_val)
        
        # 6. Đánh giá mô hình
        print("\nĐang đánh giá mô hình...")
        mae, rmse, r2 = evaluate_model(trained_model, X_test, y_test, scaler, combined_df)
        
        # 7. Vẽ biểu đồ training history
        plot_training_history(history)
        
        # 8. Phân tích tầm quan trọng của features
        importance_df = analyze_feature_importance(trained_model, features)
        print("\nTop 5 features quan trọng nhất:")
        print(importance_df.head())
        
        # 9. Cross-validation
        print("\nĐang thực hiện cross-validation...")
        cv_scores, cv_histories = cross_validate_model(X, y)
        print(f"\nKết quả cross-validation - Mean Loss: {np.mean([s[0] for s in cv_scores]):.4f}")
        
        # 10. Dự đoán và cảnh báo
        print("\nĐang tạo dự đoán cho 7 ngày tới...")
        recent_data = scaled_data[-sequence_length:]
        predictions, warnings, severity = generate_predictions_and_warnings(
            trained_model, recent_data, scaler,
            {'historical_mean': historical_df['pr'].mean(),
             'historical_max': historical_df['pr'].max()}
        )
        
        # 11. Hiển thị kết quả
        print("\nDỰ BÁO 7 NGÀY TỚI:")
        for day, value in enumerate(predictions, 1):
            print(f"Ngày {day}: {value:.1f}mm")
            
        if warnings:
            print("\nCẢNH BÁO:")
            for warning in warnings:
                print(f"[{warning['level']}] {warning['message']}")
        
        # 12. Luôn lưu mô hình sau khi train xong
        print("\nĐã lưu mô hình thành công!")
            
    else:
        print("Không thể tiếp tục do lỗi đọc dữ liệu!")
    
    print("\n=== KẾT THÚC CHƯƠNG TRÌNH ===")
