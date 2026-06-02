# Hệ Thống Dự Báo Lượng Mưa & Cảnh Báo Ngập Lụt

Hệ thống sử dụng mô hình LSTM (Long Short-Term Memory) kết hợp dữ liệu cảm biến IoT từ ESP32 để dự báo lượng mưa và giám sát mực nước theo thời gian thực, hỗ trợ cảnh báo ngập lụt.

---

## Kiến trúc hệ thống

```
[ESP32 Sensors] → [Firebase Realtime DB] → [ML Model (LSTM)] → [Streamlit Dashboard]
```

- **ESP32**: Thu thập dữ liệu mưa, nhiệt độ, độ ẩm, mực nước, độ ẩm đất
- **Firebase**: Lưu trữ dữ liệu realtime và kết quả dự đoán
- **LSTM Model**: Dự báo lượng mưa 7 ngày tiếp theo
- **Streamlit**: Dashboard trực quan hóa dữ liệu

---

## Cấu trúc thư mục

```
flood2/
├── rainfall_prediction_project/     # Dự án chính (cấu trúc module)
│   ├── config/
│   │   └── config.py                 # Cấu hình Firebase & model
│   ├── src/
│   │   ├── data_processor.py         # Xử lý & chuẩn hóa dữ liệu
│   │   ├── firebase_handler.py       # Tương tác Firebase
│   │   └── predictor.py              # Dự đoán bằng LSTM
│   ├── utils/
│   │   └── helpers.py                # Hàm hiển thị kết quả
│   ├── models/
│   │   └── best_model.keras          # Model đã huấn luyện
│   ├── styles/
│   │   └── style.css                 # CSS cho dashboard
│   ├── app.py                        # Streamlit dashboard
│   ├── main.py                       # Vòng lặp monitoring
│   └── requirements.txt              # Dependencies
│
├── train.py                          # Huấn luyện model (chính)
├── wtrain.py                         # Huấn luyện trên Google Colab
├── alo.py                            # Chuyển đổi .h5 → .keras
├── alo2.py                           # Sinh dữ liệu cảm biến giả
├── www.py                            # Web app (Streamlit)
├── zf.py                             # Dashboard nâng cao (Kalman Filter)
├── zdoinam2024.py                    # Cập nhật năm trong CSV
├── test_data.py                      # Kiểm tra & trực quan dữ liệu
│
├── best_model.keras                  # Model huấn luyện sẵn
├── best_model/                       # Model dạng SavedModel
│
├── firebase_config.json              # Cấu hình Firebase
├── firebase_credentials.json         # Service account key
│
├── generated_sensor_data.csv         # Dữ liệu cảm biến sinh ra
├── generated_sensor_data_1.csv       # Dữ liệu cảm biến bổ sung
├── generated_sensor_data_2022_2023.csv
├── data_firebase2024_changed.csv     # Dữ liệu Firebase năm 2024
├── vietnam-rainfall-1901-2015.csv    # Dữ liệu mưa lịch sử Việt Nam
├── rain_level.csv                    # Dữ liệu mức mưa
└── 23-26.csv                         # Dữ liệu mẫu
```

---

## Công nghệ sử dụng

| Công nghệ | Mục đích |
|-----------|----------|
| Python 3.x | Ngôn ngữ chính |
| TensorFlow / Keras | Xây dựng & huấn luyện mô hình LSTM |
| scikit-learn | Tiền xử lý dữ liệu, chuẩn hóa |
| Firebase Admin SDK | Kết nối Firebase Realtime DB |
| Streamlit | Dashboard web |
| Plotly | Biểu đồ tương tác |
| Pandas / NumPy | Xử lý dữ liệu |
| Kalman Filter (filterpy) | Lọc nhiễu dữ liệu cảm biến |

---

## Cài đặt

```bash
# Clone repository
git clone <repo-url>
cd flood2

# Cài đặt dependencies
pip install -r rainfall_prediction_project/requirements.txt

# Bổ sung thêm (nếu chạy zf.py)
pip install streamlit-card streamlit-lottie filterpy pillow requests
```

### Firebase Setup

1. Tạo project trên [Firebase Console](https://console.firebase.google.com)
2. Tạo Realtime Database
3. Download service account key → lưu thành `firebase_credentials.json`
4. Cập nhật `firebase_config.json` với thông tin project của bạn

---

## Hướng dẫn sử dụng

### 1️. Huấn luyện model

```bash
python train.py
```

Model sử dụng LSTM 3 tầng với:
- `input_sequence_length = 51840` (~3 ngày dữ liệu 5s/mẫu)
- Dự đoán 7 bước tiếp theo
- Huber loss, Adam optimizer
- Early stopping & ReduceLROnPlateau

### 2️. Chạy dashboard

**Dashboard cơ bản:**
```bash
cd rainfall_prediction_project
streamlit run app.py
```

**Dashboard nâng cao (có Kalman Filter):**
```bash
streamlit run zf.py
```

### 3️. Giám sát tự động

```bash
cd rainfall_prediction_project
python main.py
```

### 4️. Sinh dữ liệu mẫu

```bash
python alo2.py    # Sinh dữ liệu cảm biến giả
```

---

## Mô hình LSTM

### Kiến trúc

```
Input (51840, n_features)
  → LSTM(128) → BatchNorm → Dropout(0.3)
  → LSTM(64)  → BatchNorm → Dropout(0.3)
  → LSTM(32)  → BatchNorm → Dropout(0.3)
  → Dense(16) → BatchNorm
  → Dense(7)
```

### Features đầu vào

- `Distance` - Khoảng cách (mực nước)
- `Flow Rate` - Lưu lượng dòng chảy
- `humidity` - Độ ẩm không khí
- `rain` - Lượng mưa
- `soil_moisture` - Độ ẩm đất
- `temperature` - Nhiệt độ
- Các đặc trưng thời gian (giờ, ngày, tháng)
- Rolling features (mean, max, min) theo các cửa sổ 1h, 3h, 6h, 12h, 24h

---

## API & Dữ liệu

### Cấu trúc Firebase Realtime Database

```
/
├── ESP32_1/
│   ├── {timestamp}/
│   │   ├── Mưa: float
│   │   ├── Độ ẩm đất: float
│   │   ├── Nhiệt độ: string (VD: "28°C")
│   │   ├── Độ ẩm không khí: float
│   │   ├── Khoảng cách: float
│   │   └── Lưu lượng: float
├── ESP32_2/ (tương tự)
└── predictions/
    ├── datetime: string
    ├── predictions: [{date, value}]
    └── updated_at: string
```

---

## Kết quả

- Dự báo lượng mưa 7 ngày với sai số MAE ~2-5mm
- Cảnh báo khi lượng mưa vượt ngưỡng 30mm
- Dashboard realtime cập nhật mỗi 5 phút
- Lọc nhiễu dữ liệu cảm biến bằng Kalman Filter

---

## License

MIT License
