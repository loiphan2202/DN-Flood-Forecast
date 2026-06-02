import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import firebase_admin
from firebase_admin import credentials, db
import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Khởi tạo Firebase
cred = credentials.Certificate("path/to/your/firebase-credentials.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'your-database-url'
})

# Tải model và thông tin
model = load_model('best_model.keras')
model_info = np.load('model_info.npy', allow_pickle=True).item()
scaler = model_info['scaler']
features = model_info['features']

def get_realtime_data():
    """Lấy dữ liệu realtime từ Firebase"""
    ref = db.reference('sensor_data')  # Điều chỉnh path theo cấu trúc Firebase của bạn
    data = ref.get()
    
    # Chuyển đổi dữ liệu thành DataFrame
    df = pd.DataFrame(data).T
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    return df

def prepare_data_for_prediction(df):
    """Chuẩn bị dữ liệu cho việc dự đoán"""
    # Lấy 51840 mẫu dữ liệu gần nhất
    df_recent = df.tail(51840)
    
    # Đảm bảo các cột theo đúng thứ tự features
    df_features = df_recent[features]
    
    # Chuẩn hóa dữ liệu
    scaled_data = scaler.transform(df_features)
    
    return scaled_data

def make_prediction(scaled_data):
    """Thực hiện dự đoán"""
    # Reshape data cho model
    X = scaled_data.reshape((1, scaled_data.shape[0], scaled_data.shape[1]))
    
    # Dự đoán
    prediction = model.predict(X)
    
    # Tạo timestamps cho dự đoán (3 ngày tiếp theo)
    last_time = datetime.now()
    future_times = pd.date_range(
        start=last_time,
        periods=prediction.shape[1],
        freq='5T'  # 5 phút một lần
    )
    
    return prediction[0], future_times

def main():
    st.title("Hệ thống Giám sát và Dự báo")
    
    # Chia layout thành 2 cột
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Dữ liệu Realtime")
        # Tạo placeholder cho biểu đồ realtime
        chart_placeholder = st.empty()
        
        # Cập nhật dữ liệu realtime
        while True:
            df_realtime = get_realtime_data()
            
            # Tạo biểu đồ với plotly
            fig = make_subplots(rows=6, cols=1, 
                              subplot_titles=('Khoảng cách', 'Lưu lượng', 
                                            'Độ ẩm không khí', 'Lượng mưa',
                                            'Độ ẩm đất', 'Nhiệt độ'))
            
            # Thêm từng trace cho mỗi thông số
            fig.add_trace(go.Scatter(x=df_realtime.index, y=df_realtime['Distance'],
                                   name='Khoảng cách'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_realtime.index, y=df_realtime['Flow Rate'],
                                   name='Lưu lượng'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_realtime.index, y=df_realtime['humidity'],
                                   name='Độ ẩm KK'), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_realtime.index, y=df_realtime['rain'],
                                   name='Lượng mưa'), row=4, col=1)
            fig.add_trace(go.Scatter(x=df_realtime.index, y=df_realtime['soil_moisture'],
                                   name='Độ ẩm đất'), row=5, col=1)
            fig.add_trace(go.Scatter(x=df_realtime.index, y=df_realtime['temperature'],
                                   name='Nhiệt độ'), row=6, col=1)
            
            fig.update_layout(height=800, showlegend=False)
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Đợi 5 giây trước khi cập nhật
            st.experimental_rerun()
    
    with col2:
        st.header("Dự báo")
        if st.button("Dự đoán"):
            # Lấy dữ liệu và thực hiện dự đoán
            df_realtime = get_realtime_data()
            scaled_data = prepare_data_for_prediction(df_realtime)
            predictions, future_times = make_prediction(scaled_data)
            
            # Hiển thị kết quả dự đoán
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=future_times, y=predictions,
                                   name='Dự đoán'))
            fig.update_layout(title='Dự báo cho 3 ngày tới',
                            xaxis_title='Thời gian',
                            yaxis_title='Giá trị dự đoán')
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()