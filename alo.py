import tensorflow as tf
import h5py
import numpy as np

input_model_path = "./rainfall_prediction_project/models/rainfall_prediction_model.h5"
output_model_path = "./rainfall_prediction_project/models/rainfall_prediction_model.keras"

# In ra cấu trúc weights để kiểm tra
with h5py.File(input_model_path, 'r') as f:
    print("Dense kernel shape:", f['model_weights/dense/sequential/dense/kernel'][:].shape)
    print("Dense bias shape:", f['model_weights/dense/sequential/dense/bias'][:].shape)

# Tạo model mới với cấu trúc tương tự
inputs = tf.keras.layers.Input(shape=(12, 7))
x = inputs

# Thêm các LSTM layers
x = tf.keras.layers.LSTM(units=64, return_sequences=True)(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.LSTM(units=32, return_sequences=True)(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.LSTM(units=16, return_sequences=False)(x)
x = tf.keras.layers.Dropout(0.2)(x)

# Dense layer cuối cùng - sửa units thành 7
outputs = tf.keras.layers.Dense(7)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Copy weights từ file h5
with h5py.File(input_model_path, 'r') as f:
    # Copy LSTM weights
    for i, lstm_name in enumerate(['lstm', 'lstm_1', 'lstm_2']):
        weights = []
        base_path = f'model_weights/{lstm_name}/sequential/{lstm_name}/lstm_cell'
        weights.append(f[f'{base_path}/kernel'][:])
        weights.append(f[f'{base_path}/recurrent_kernel'][:])
        weights.append(f[f'{base_path}/bias'][:])
        model.layers[2*i+1].set_weights(weights)
    
    # Copy Dense weights
    dense_weights = []
    dense_weights.append(f['model_weights/dense/sequential/dense/kernel'][:])
    dense_weights.append(f['model_weights/dense/sequential/dense/bias'][:])
    model.layers[-1].set_weights(dense_weights)

# Lưu model mới
model.save(output_model_path)
print("Model đã được chuyển đổi thành công!")