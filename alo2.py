import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Thời gian bắt đầu và kết thúc
start_date = datetime(2024, 12, 23, 0, 0, 0)  # Bắt đầu từ 00:00:00
end_date = datetime(2024, 12, 26, 23, 59, 59)
time_increment = timedelta(seconds=5)  # Mỗi dòng cách nhau 5 giây

# Tính tổng số dòng cần tạo
total_seconds = (end_date - start_date).total_seconds()
total_rows = int(total_seconds / 5) + 1

# Khởi tạo các thông số cho mưa
rain_patterns = {
    'no_rain': 0,
    'light_rain': (1, 20),
    'moderate_rain': (20, 50),
    'heavy_rain': (50, 100),
    'very_heavy_rain': (100, 150)
}

# Hàm tạo một đợt mưa
def generate_rain_event(duration, rain_type):
    if rain_type == 'no_rain':
        return [0] * duration
    min_val, max_val = rain_patterns[rain_type]
    base_value = random.uniform(min_val, max_val)
    # Tạo độ dao động nhẹ xung quanh giá trị cơ bản
    variation = np.random.normal(0, 0.5, duration)
    rain_values = np.clip(base_value + variation, min_val, max_val)
    return rain_values.tolist()

# Khởi tạo dữ liệu
data = []
current_time = start_date
current_distance = 2.35  # Giá trị mực nước ban đầu
previous_rain_type = 'no_rain'

# Tạo các đợt mưa
i = 0
while i < total_rows:
    rain_duration = random.randint(1440, 2880)  # 2-4 giờ
    rain_type = random.choice(['no_rain', 'no_rain', 'no_rain', 'light_rain', 
                             'moderate_rain', 'heavy_rain', 'very_heavy_rain'])
    
    rain_values = generate_rain_event(rain_duration, rain_type)
    
    # Xác định target_distance dựa trên loại mưa
    if rain_type == 'no_rain':
        target_distance = random.uniform(2.3, 2.45)
    elif rain_type == 'light_rain':
        target_distance = random.uniform(2.4, 2.5)
    elif rain_type == 'moderate_rain':
        target_distance = random.uniform(2.5, 2.6)
    elif rain_type == 'heavy_rain':
        target_distance = random.uniform(2.6, 2.7)
    else:  # very_heavy_rain
        target_distance = random.uniform(2.7, 3.2)
    
    # Tính toán bước tăng/giảm mực nước
    distance_step = (target_distance - current_distance) / min(rain_duration, 100)
    
    for j in range(min(rain_duration, total_rows - i)):
        rain = rain_values[j]
        
        # Điều chỉnh mực nước dần dần
        if j < 100:  # Trong 100 bước đầu, điều chỉnh dần dần
            current_distance += distance_step
        
        # Thêm dao động nhỏ để tạo tính tự nhiên
        current_distance += random.uniform(-0.01, 0.01)
        
        # Đảm bảo mực nước nằm trong khoảng cho phép
        if rain == 0:  # Không mưa
            current_distance = max(2.3, min(2.45, current_distance))
            humidity = random.uniform(50, 60)
            soil_moisture = random.uniform(5, 10)
            temp = random.uniform(29, 33)
            flow_rate = random.uniform(0.2, 0.5)
        else:
            humidity = random.uniform(85, 100)
            soil_moisture = random.uniform(85, 100)
            temp = random.uniform(26, 27)
            
            # Giới hạn mực nước theo loại mưa
            if rain < 20:  # Mưa nhỏ
                current_distance = max(2.4, min(2.5, current_distance))
                flow_rate = random.uniform(0.4, 0.7)
            elif rain < 50:  # Mưa vừa
                current_distance = max(2.5, min(2.6, current_distance))
                flow_rate = random.uniform(0.5, 0.8)
            elif rain < 100:  # Mưa to
                current_distance = max(2.6, min(2.7, current_distance))
                flow_rate = random.uniform(0.6, 0.9)
            else:  # Mưa rất to
                current_distance = max(2.7, min(3.2, current_distance))
                flow_rate = random.uniform(0.7, 1.0)
        
        row = {
            "Datetime": current_time.strftime("%Y/%m/%d %H:%M:%S"),
            "Distance": round(current_distance, 2),
            "Flow Rate": round(flow_rate, 2),
            "humidity": round(humidity, 2),
            "rain": round(rain, 2),
            "soil_moisture": round(soil_moisture, 2),
            "temperature": round(temp, 2)
        }
        data.append(row)
        current_time += time_increment
        i += 1
    
    previous_rain_type = rain_type

# Tạo DataFrame và lưu file
df = pd.DataFrame(data)
df['Datetime'] = pd.to_datetime(df['Datetime'])  # Chuyển đổi sang datetime
df['Datetime'] = df['Datetime'].dt.strftime('%Y/%m/%d %H:%M:%S')  # Định dạng lại theo yêu cầu
output_path = "23-26.csv"
df.to_csv(output_path, index=False)

print(f"File đã được lưu tại: {output_path}")