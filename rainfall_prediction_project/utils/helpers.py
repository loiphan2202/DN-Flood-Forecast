from datetime import datetime

def display_predictions(predictions, dates):
    """Hiển thị kết quả dự đoán"""
    print("\n=== DỰ BÁO LƯỢNG MƯA ===")
    print(f"Cập nhật lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 40)
    
    for date, pred in zip(dates, predictions[0]):
        print(f"{date}: {pred:.2f}mm")
    print("-" * 40)