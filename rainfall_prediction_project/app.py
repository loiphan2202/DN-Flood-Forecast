from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.data_processor import DataProcessor
from src.firebase_handler import FirebaseHandler
from src.predictor import RainfallPredictor
from config import CONFIG

def init_app():
    """Initialize app components"""
    try:
        st.set_page_config(
            page_title="Dự Báo Lượng Mưa",
            page_icon="🌧️",
            layout="wide"
        )
        
        # Thử load CSS nếu có
        try:
            with open('./styles/style.css') as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        except FileNotFoundError:
            print("CSS file not found, continuing without styles")
            
    except Exception as e:
        st.error(f"Error initializing app: {str(e)}")

def create_gauge_chart(value, title):
    """Tạo biểu đồ đồng hồ"""
    try:
        range_values = CONFIG['GAUGE_RANGES'].get(title, [0, 100])
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = float(value),
            title = {'text': title},
            gauge = {
                'axis': {'range': range_values},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, range_values[1]/3], 'color': "lightgray"},
                    {'range': [range_values[1]/3, range_values[1]*2/3], 'color': "gray"},
                    {'range': [range_values[1]*2/3, range_values[1]], 'color': "darkgray"}
                ]
            }
        ))
        return fig
    except Exception as e:
        st.error(f"Error creating gauge chart: {str(e)}")
        return None

def main():
    try:
        init_app()
        
        st.title("🌧️ Hệ Thống Dự Báo Lượng Mưa")
        
        # Initialize components with error handling
        try:
            firebase = FirebaseHandler()
            processor = DataProcessor()
            predictor = RainfallPredictor()
        except Exception as e:
            st.error(f"Error initializing components: {str(e)}")
            return

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Dữ Liệu Cảm Biến Hiện Tại")
            
            df = firebase.get_latest_sensor_data()
            if df is not None and not df.empty:
                latest_data = df.iloc[-1]
                
                # Display sensor data
                for sensor in ['Mưa', 'Độ ẩm đất']:
                    if sensor in latest_data:
                        chart = create_gauge_chart(latest_data[sensor],
                                                "Cảm biến mưa" if sensor == 'Mưa' else "Độ ẩm đất")
                        if chart:
                            st.plotly_chart(chart)
                
                # Display metrics
                metrics = {
                    'Nhiệt độ': ('Nhiệt độ', '°C'),
                    'Độ ẩm không khí': ('Độ ẩm không khí', '%'),
                    'Lưu lượng': ('Lưu lượng', 'L/min'),
                    'Mưa': ('Mưa', 'mm'),
                    'Độ ẩm đất': ('Độ ẩm đất', '%'),
                    'Khoảng cách': ('Khoảng cách', 'm')
                }
                
                for key, (label, unit) in metrics.items():
                    if key in latest_data:
                        value = latest_data[key]
                        # Loại bỏ °C từ giá trị nhiệt độ nếu có
                        if key == 'Nhiệt độ':
                            value = str(value).replace('°C', '')
                        st.metric(label, f"{value}{unit}")
            else:
                st.warning("Không có dữ liệu cảm biến")

        with col2:
            st.subheader("Dự Báo 7 Ngày Tới")
            
            if df is not None and not df.empty:
                processed_data = processor.prepare_data(df)
                if processed_data is not None:
                    predictions = predictor.predict(processed_data)
                    dates = predictor.generate_dates()
                    
                    pred_df = pd.DataFrame({
                        'Ngày': dates,
                        'Lượng mưa (mm)': predictions[0]
                    })
                    
                    # Plot forecast
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=pred_df['Ngày'],
                        y=pred_df['Lượng mưa (mm)'],
                        mode='lines+markers',
                        name='Dự đoán'
                    ))
                    fig.update_layout(
                        title='Dự báo lượng mưa',
                        xaxis_title='Ngày',
                        yaxis_title='Lượng mưa (mm)'
                    )
                    st.plotly_chart(fig)
                    
                    st.dataframe(pred_df)
                    
                    if any(pred > 30 for pred in predictions[0]):
                        st.warning('⚠️ Cảnh báo: Có khả năng mưa lớn trong những ngày tới!')
                else:
                    st.warning("Không thể xử lý dữ liệu cho dự báo")
            else:
                st.warning("Không có dữ liệu để dự báo")

        st.markdown("---")
        st.markdown("Cập nhật lần cuối: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    except Exception as e:
        st.error(f"Lỗi chung: {str(e)}")

if __name__ == "__main__":
    main()