import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        return data
    except Exception as e:
        print(f"Lỗi khi đọc dữ liệu: {e}")
        return None

# Kiểm tra dữ liệu
def check_data(data):
    if data is None:
        print("Dữ liệu không tồn tại.")
        return

    # Kiểm tra kích thước dữ liệu
    print("Kích thước dữ liệu:", data.shape)

    # Kiểm tra giá trị thiếu
    print("\nGiá trị thiếu:")
    print(data.isnull().sum())

    # Kiểm tra phân phối dữ liệu
    print("\nPhân phối dữ liệu:")
    data.hist(figsize=(10, 8))
    plt.show()

    # Kiểm tra dữ liệu ngoại lai
    print("\nDữ liệu ngoại lai:")
    sns.boxplot(data=data)
    plt.show()

# Đường dẫn tới file CSV
file_path = './generated_sensor_data.csv'

# Đọc và kiểm tra dữ liệu
sensor_data = load_data(file_path)
check_data(sensor_data)