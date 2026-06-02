import json
import os

import numpy as np
import pandas as pd

try:
    from keras.models import load_model
except ImportError:
    from keras.models import load_model
import time
from datetime import datetime, timedelta

import firebase_admin
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit_card
import tensorflow as tf
from filterpy.kalman import KalmanFilter
from firebase_admin import credentials, db
from PIL import Image
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from streamlit_lottie import st_lottie

# Khởi tạo Firebase
if not firebase_admin._apps:  # Kiểm tra xem đã có app nào được khởi tạo chưa
    # Đọc cấu hình Firebase từ file JSON
    config_path = os.path.join(os.path.dirname(__file__), './firebase_config.json')
    with open(config_path) as f:
        firebase_config = json.load(f)

    cred = credentials.Certificate("./firebase_credentials.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': firebase_config['databaseURL']
    })

# Đảm bảo folder model tồn tại trước khi load
model_path = './best_model'

if os.path.exists(model_path):
    try:
        # Load model từ folder lưu trưc đó
        model = load_model(model_path)
        st.success("Model đã được load thành công.")
        print("Cấu trúc model:")
        model.summary()
    except Exception as e:
        st.error(f"Lỗi khi load model: {str(e)}")
        model = None
else:
    st.error(f"Không tìm thấy folder model tại {model_path}")
    model = None

# Khởi tạo scaler (giả định dữ liệu cần chuẩn hóa từ 0-1)
# scaler = MinMaxScaler()

class SensorKalmanFilter:
    def __init__(self, initial_value, process_variance, measurement_variance):
        self.kf = KalmanFilter(dim_x=1, dim_z=1)
        self.kf.x = np.array([[initial_value]])  # Trạng thái ban đầu
        self.kf.P *= process_variance  # Độ không chắc chắn ban đầu
        self.kf.R = measurement_variance  # Nhiễu đo
        self.kf.Q = process_variance  # Nhiễu quá trình
        
        # Ma trận chuyển trạng thái
        self.kf.F = np.array([[1.]])
        # Ma trận đo
        self.kf.H = np.array([[1.]])

    def update(self, measurement):
        self.kf.predict()
        self.kf.update(measurement)
        return self.kf.x[0, 0]

def apply_kalman_filter(data_series):
    """Áp dụng bộ lọc Kalman cho một chuỗi dữ liệu"""
    # Khởi tạo với giá trị đầu tiên
    initial_value = data_series.iloc[0]
    
    # Tính toán phương sai cho việc tinh chỉnh bộ lọc
    measurement_variance = np.var(data_series) if len(data_series) > 1 else 1.0
    process_variance = measurement_variance * 0.1  # Thường nhỏ hơn measurement_variance
    
    # Khởi tạo bộ lọc
    kf = SensorKalmanFilter(initial_value, process_variance, measurement_variance)
    
    # Áp dụng bộ lọc
    filtered_data = []
    for value in data_series:
        filtered_value = kf.update(value)
        filtered_data.append(filtered_value)
    
    return pd.Series(filtered_data, index=data_series.index)

def clean_and_filter_data(df):
    """Làm sạch và lọc dữ liệu sử dụng Kalman Filter"""
    # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
    df_cleaned = df.copy()
    
    # Xử lý missing values
    df_cleaned = df_cleaned.dropna()
    
    # Áp dụng Kalman Filter cho từng cảm biến
    for column in df_cleaned.columns:
        df_cleaned[column] = apply_kalman_filter(df_cleaned[column])
    
    # Kiểm tra giá trị hợp lệ cho từng cảm biến
    valid_ranges = {
        'Distance': (0, 5),           # Khoảng cách từ 0-5m
        'Flow Rate': (0, 10),         # Lưu lượng từ 0-10 L/min
        'humidity': (0, 100),         # Độ ẩm không khí 0-100%
        'rain': (0, 100),             # Lượng mưa 0-100mm
        'soil_moisture': (0, 100),    # Độ ẩm đất 0-100%
        'temperature': (0, 50)        # Nhiệt độ 0-50°C
    }
    
    for column, (min_val, max_val) in valid_ranges.items():
        df_cleaned = df_cleaned[(df_cleaned[column] >= min_val) & 
                              (df_cleaned[column] <= max_val)]
    
    # Sắp xếp theo thời gian
    df_cleaned = df_cleaned.sort_index()
    
    return df_cleaned

def get_realtime_data():
    """Lấy và kết hợp dữ liệu từ cả hai ESP32"""
    try:
        # Lấy reference đến các node ESP32
        esp32_1_ref = db.reference('devices/ESP32_1/sensor_data')
        esp32_2_ref = db.reference('devices/ESP32_2/sensor_data')
        
        # Lấy dữ liệu từ cả hai ESP32
        data_1 = esp32_1_ref.get()
        data_2 = esp32_2_ref.get()
        
        if data_1 and data_2:
            # Tạo DataFrame rỗng để lưu dữ liệu kết hợp
            combined_data = []
            
            # Duyệt qua các timestamp
            for timestamp in data_1:
                if timestamp in data_2:
                    # Lấy dữ liệu từ cả hai thiết bị
                    sensor_1 = data_1[timestamp]
                    sensor_2 = data_2[timestamp]
                    
                    # Kết hợp dữ liệu
                    combined_row = {
                        'datetime': sensor_1['datetime'],
                        'Distance': float(sensor_2.get('distance', 0)),
                        'Flow Rate': float(sensor_2.get('flow_rate', 0)),
                        'humidity': float(sensor_1.get('humidity', 0)),
                        'rain': float(sensor_1.get('rain', 0)),
                        'soil_moisture': float(sensor_1.get('soil_moisture', 0)),
                        'temperature': float(sensor_1.get('temperature', 0))
                    }
                    combined_data.append(combined_row)
            
            # Tạo DataFrame từ dữ liệu kết hợp
            df = pd.DataFrame(combined_data)
            
            # Chuyển đổi cột datetime thành index
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            
            # In ra để debug
            print("DataFrame sau khi xử lý:", df)
            
            # Áp dụng Kalman Filter và làm sạch dữ liệu
            df = clean_and_filter_data(df)
            return df
            
        else:
            print("Không có dữ liệu từ một hoặc cả hai ESP32")
            return pd.DataFrame(columns=["Distance", "Flow Rate", "humidity", 
                                      "rain", "soil_moisture", "temperature"])
            
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu từ Firebase: {str(e)}")
        return pd.DataFrame(columns=["Distance", "Flow Rate", "humidity", 
                                   "rain", "soil_moisture", "temperature"])

# Thêm cache cho model và scaler
@st.cache_resource
def load_model_and_scaler():
    """Cache model và scaler để tránh load lại nhiều lần"""
    model_path = './best_model'
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            scaler = MinMaxScaler()
            # Tạo dữ liệu mẫu để fit scaler
            sample_data = np.array([
                [0, 0, 0, 0, 0, 0],  # min values
                [5, 10, 100, 100, 100, 50]  # max values từ valid_ranges
            ])
            scaler.fit(sample_data)
            return model, scaler
        except Exception as e:
            st.error(f"Lỗi khi load model: {str(e)}")
    return None, None

# Sửa decorator và tham số của hàm prepare_data_for_prediction
@st.cache_data(ttl=60)  # Cache trong 60s
def prepare_data_for_prediction(df, _scaler):  # Thêm dấu gạch dưới trước scaler
    """Chuẩn bị dữ liệu cho dự đoán với tối ưu hóa"""
    if len(df) < 24:
        return None
        
    required_columns = ["Distance", "Flow Rate", "humidity", "rain", "soil_moisture", "temperature"]
    df = df[required_columns].copy()
    
    # Sử dụng scaler đã được fit từ tham số
    scaled_data = _scaler.transform(df.values)  # Dùng transform thay vì fit_transform
    
    # Optimize feature expansion
    if len(scaled_data) > 24:
        scaled_data = scaled_data[-24:]
    elif len(scaled_data) < 24:
        padding = np.zeros((24 - len(scaled_data), scaled_data.shape[1]))
        scaled_data = np.vstack([padding, scaled_data])
    
    # Vectorized feature expansion
    expanded_data = np.zeros((24, 18))
    expanded_data[:, :6] = scaled_data
    
    # Vectorized calculations for additional features
    for i in range(6):
        expanded_data[:, 6+i] = np.convolve(scaled_data[:, i], np.ones(3)/3, mode='same')
        expanded_data[:, 12+i] = np.gradient(scaled_data[:, i])
    
    return expanded_data.reshape((1, 24, 18))

# Tương tự cho hàm make_prediction
@st.cache_data(ttl=60)
def make_prediction(_model, scaled_data):  # Thêm dấu gạch dưới
    """Thực hiện dự đoán với tối ưu hóa"""
    if _model is None:
        return None, None
        
    try:
        # Batch prediction để tăng tốc
        with tf.device('/CPU:0'):  # Force CPU usage for small batches
            prediction = _model.predict(scaled_data, batch_size=1, verbose=0)
        
        # Tối ưu việc tạo timestamps
        current_time = pd.Timestamp.now()
        future_times = pd.date_range(start=current_time, periods=3, freq='D')
        
        prediction = prediction[0, :3, :]
        
        return prediction, future_times
    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {str(e)}")
        return None, None

def display_realtime_data(df):
    """Hiển thị biểu đồ realtime với style đẹp hơn"""
    fig = make_subplots(
        rows=6, cols=1,
        subplot_titles=(
            'Khoảng cách (m)',
            'Lưu lượng (L/min)',
            'Độ ẩm không khí (%)',
            'Lượng mưa (mm)',
            'Độ ẩm đất (%)',
            'Nhiệt độ (°C)'
        ),
        vertical_spacing=0.05
    )

    colors = {
        "Distance": "#1f77b4",      # Xanh dương
        "Flow Rate": "#ff7f0e",     # Cam
        "humidity": "#2ca02c",      # Xanh lá
        "rain": "#d62728",          # Đỏ
        "soil_moisture": "#9467bd",  # Tím
        "temperature": "#8c564b"     # Nâu
    }

    metrics = ["Distance", "Flow Rate", "humidity", "rain", "soil_moisture", "temperature"]
    
    for i, metric in enumerate(metrics, start=1):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[metric],
                name=metric,
                line=dict(
                    color=colors[metric],
                    width=2
                ),
                mode='lines'
            ),
            row=i,
            col=1
        )

        # Thêm grid lines và style cho mỗi subplot
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            row=i,
            col=1,
            tickformat='%H:%M\n%d/%m/%y'
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            row=i,
            col=1
        )

    fig.update_layout(
        height=900,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=50, b=50),
        title=dict(
            text="Dữ liệu realtime từ cảm biến",
            x=0.5,
            y=0.98,
            xanchor='center',
            yanchor='top',
            font=dict(size=24)
        ),
        font=dict(
            family="Arial, sans-serif",
            size=12
        )
    )

    # Thêm hover template
    for trace in fig.data:
        trace.update(
            hovertemplate=(
                "<b>Thời gian:</b> %{x}<br>" +
                "<b>Giá trị:</b> %{y:.2f}<br>" +
                "<extra></extra>"
            )
        )

    return fig

def display_predictions(predictions, future_times, _scaler):  # Thêm dấu gạch dưới
    """Hiển thị kết quả dự đoán và cảnh báo"""
    if predictions is None or future_times is None:
        st.error("Không có dữ liệu dự đoán")
        return None
        
    try:
        temp_array = np.zeros((len(predictions), 6))
        temp_array[:, 0] = predictions[:, 0]
        temp_array[:, 3] = predictions[:, 1]
        
        # Sử dụng scaler từ tham số
        predictions_original = _scaler.inverse_transform(temp_array)
        
        distance_pred = predictions_original[:, 0]
        rain_pred = predictions_original[:, 3]
        
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Dự đoán mực nước (m)', 'Dự đoán lượng mưa (mm)'))
        
        # Vẽ biểu đồ cho mực nước
        fig.add_trace(go.Scatter(x=future_times, y=distance_pred, 
                                name='Mực nước'), row=1, col=1)
        fig.add_hline(y=3, line_dash="dash", line_color="red", 
                     annotation_text="Ngưỡng nguy hiểm: 3m", row=1, col=1)
        
        # Vẽ biểu đồ cho lượng mưa
        fig.add_trace(go.Scatter(x=future_times, y=rain_pred, 
                                name='Lượng mưa'), row=2, col=1)
        fig.add_hline(y=90, line_dash="dash", line_color="red", 
                     annotation_text="Ngưỡng nguy hiểm: 90mm", row=2, col=1)
        
        fig.update_layout(height=600, title='Dự báo cho 3 ngày tiếp theo')
        
        # Hiển thị cảnh báo
        for i in range(len(future_times)):
            date = future_times[i].strftime('%d/%m/%Y')
            if distance_pred[i] >= 3:
                st.error(f"⚠️ CẢNH BÁO: Mực nước dự báo cho ngày {date} là {distance_pred[i]:.2f}m - VƯỢT NGƯỠNG AN TOÀN!")
            if rain_pred[i] >= 90:
                st.error(f"⚠️ CẢNH BÁO: Lượng mưa dự báo cho ngày {date} là {rain_pred[i]:.2f}mm - VƯỢT NGƯỠNG AN TOÀN!")
        
        return fig
    except Exception as e:
        st.error(f"Lỗi khi hiển thị dự đoán: {str(e)}")
        return None

# Thêm hàm load animation
def load_lottie_url(url):
    """Load animation từ URL"""
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Cập nhật CSS với thêm hiệu ứng và màu sắc
def load_css():
    st.markdown("""
        <style>
        /* Main container */
        .main {
            padding: 1rem 2rem;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        /* Header styling */
        .header-container {
            background: rgba(255, 255, 255, 0.95);
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
            backdrop-filter: blur(4px);
            margin-bottom: 2rem;
            transition: all 0.3s ease;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: rgba(255, 255, 255, 0.8);
            border-radius: 15px;
            padding: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 15px 25px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #00acee 0%, #1e90ff 100%);
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,172,238,0.3);
        }
        
        /* Sensor cards */
        .sensor-card {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
            margin-bottom: 1.5rem;
            transition: transform 0.3s ease;
            border: 1px solid rgba(255, 255, 255, 0.18);
        }
        
        .sensor-card:hover {
            transform: translateY(-5px);
        }
        
        .sensor-value {
            font-size: 2rem;
            font-weight: bold;
            color: #00acee;
            margin: 0.5rem 0;
        }
        
        /* Charts container */
        .chart-container {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
            margin: 1.5rem 0;
        }
        
        /* Buttons */
        .stButton button {
            background: linear-gradient(135deg, #00acee 0%, #1e90ff 100%);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 10px;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(0,172,238,0.3);
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0,172,238,0.4);
        }
        
        /* Loading animation */
        .stSpinner {
            text-align: center;
            padding: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

def create_metric_card(icon, title, value, unit, trend=None):
    """Tạo card hiển thị metric với animation"""
    trend_html = ""
    if trend is not None:
        color = "#00c853" if trend > 0 else "#ff3d00"
        arrow = "↑" if trend > 0 else "↓"
        trend_html = f'<div style="color: {color}">{arrow} {abs(trend)}%</div>'
    
    return f"""
        <div class="sensor-card">
            <div style="font-size: 1.5rem; color: #666;">{icon} {title}</div>
            <div class="sensor-value">{value} {unit}</div>
            {trend_html}
        </div>
    """

def load_animations():
    """Load tất cả animations cần thiết"""
    return {
        'monitoring': load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_qp1q7mct.json"),
        'weather': load_lottie_url("https://assets5.lottiefiles.com/private_files/lf30_jmgekfqg.json"),
        'warning': load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_qmfs6c3i.json")
    }

def main():
    # Load CSS
    load_css()
    
    # Load model và scaler một lần duy nhất
    model, scaler = load_model_and_scaler()
    
    # Load animations/images
    animations = load_animations()
    
    # Header section với animation mượt mà
    with st.container():
        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown('<div class="header-container">', unsafe_allow_html=True)
            st.title("🌊 Hệ thống Giám sát và Dự báo Lũ lụt")
            st.markdown("##### Theo dõi và dự báo tình hình thời tiết theo thời gian thực")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st_lottie(animations['weather'], height=180, key="header_animation")

    # Tabs với hiệu ứng
    tab1, tab2 = st.tabs(["📊 Giám sát dữ liệu", "🔮 Dự báo"])
    
    with tab1:
        st.header("Dữ liệu Realtime")
        
        # Lấy dữ liệu realtime
        df_realtime = get_realtime_data()
        
        if not df_realtime.empty:
            # Hin thị metric cards
            latest_data = df_realtime.iloc[-1]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(create_metric_card(
                    "💧", 
                    "Mực nước", 
                    f"{latest_data['Distance']:.1f}", 
                    "m",
                    5
                ), unsafe_allow_html=True)
            
            with col2:
                st.markdown(create_metric_card(
                    "🌧️", 
                    "Lượng mưa", 
                    f"{latest_data['rain']:.0f}", 
                    "mm",
                    -2
                ), unsafe_allow_html=True)
            
            with col3:
                st.markdown(create_metric_card(
                    "🌡️", 
                    "Nhiệt độ", 
                    f"{latest_data['temperature']:.1f}", 
                    "°C"
                ), unsafe_allow_html=True)
            
            # Hiển thị biểu đồ
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig = display_realtime_data(df_realtime)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Không có dữ liệu realtime")
    
    with tab2:
        st.header("Dự báo từ mô hình")
        
        col1, col2 = st.columns([3,1])
        with col1:
            st.info("🤖 Model LSTM được training với dữ liệu 6 tháng gần nhất")
            predict_container = st.empty()
        with col2:
            st_lottie(animations['monitoring'], height=200, key="predict_animation")
            predict_button = st.button("🔄 Cập nhật dự báo", key="predict")
            
        if predict_button:
            with st.spinner('Đang thực hiện dự đoán...'):
                if not df_realtime.empty and model is not None and scaler is not None:
                    scaled_data = prepare_data_for_prediction(df_realtime, scaler)
                    if scaled_data is not None:
                        predictions, future_times = make_prediction(model, scaled_data)
                        if predictions is not None:
                            # Truyền scaler vào hàm display_predictions
                            fig_predictions = display_predictions(predictions, future_times, scaler)
                            if fig_predictions is not None:
                                predict_container.plotly_chart(fig_predictions, use_container_width=True)
                            else:
                                predict_container.error("Lỗi khi tạo biểu đồ dự đoán")
                        else:
                            predict_container.warning("Không thể thực hiện dự đoán")
                    else:
                        predict_container.warning("Không đủ dữ liệu để dự đoán")
                else:
                    predict_container.warning("Thiếu dữ liệu hoặc model chưa được load")

    # Footer với thông tin cập nhật
    st.markdown("---")
    st.markdown(f"🕒 Cập nhật lần cuối: **{datetime.now().strftime('%H:%M:%S %d/%m/%Y')}**")

if __name__ == "__main__":
    main()
