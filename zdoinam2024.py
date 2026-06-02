import pandas as pd

# Đọc file CSV hoặc Excel
df = pd.read_csv("./data_firebase.csv")  # Hoặc pd.read_csv() nếu là file CSV

# Thay đổi năm thành 2023 (giữ nguyên ngày giờ)
df['Datetime'] = pd.to_datetime(df['Datetime']).apply(lambda x: x.replace(year=2024))

# Lưu file mới
df.to_csv("data_firebase2024_changed.csv", index=False)  # Hoặc .to_csv() nếu là file CSV
