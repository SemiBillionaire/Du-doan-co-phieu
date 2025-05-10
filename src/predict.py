import numpy as np
import pandas as pd
from datetime import timedelta
import random
import tensorflow as tf

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

def predict_historical(model, x_train, x_test, sc):
    y_train_predict = model.predict(x_train)
    y_train_predict = sc.inverse_transform(y_train_predict)

    y_test_predict = model.predict(x_test)
    y_test_predict = sc.inverse_transform(y_test_predict)

    return y_train_predict, y_test_predict

def predict_future(model, df1, sc, timestep=50, future_days=750):
    # Kiểm tra dữ liệu đầu vào
    if df1['Lần cuối'].isnull().all():
        raise ValueError("Dữ liệu lịch sử (df1['Lần cuối']) chứa toàn NaN, không thể dự đoán.")
    
    last_data = df1.values[-timestep:]  # Lấy 50 ngày cuối cùng
    if np.isnan(last_data).all():
        raise ValueError("Dữ liệu 50 ngày cuối (last_data) chứa toàn NaN, không thể dự đoán.")

    last_data = last_data.reshape(-1, 1)
    scaled_data = sc.transform(last_data)
    if np.isnan(scaled_data).all():
        raise ValueError("Dữ liệu chuẩn hóa (scaled_data) chứa toàn NaN.")
    if np.isnan(scaled_data).any():
        print("Cảnh báo: Dữ liệu chuẩn hóa (scaled_data) chứa NaN:", scaled_data)

    future_predictions = []
    current_data = scaled_data.copy()

    last_date = df1.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]

    for i in range(future_days):
        x_future = np.array([current_data[-timestep:]])
        if np.isnan(x_future).any():
            print(f"Lỗi tại bước {i}: Dữ liệu đầu vào x_future chứa NaN:", x_future)
            break

        x_future = np.reshape(x_future, (x_future.shape[0], x_future.shape[1], 1))
        if np.isnan(x_future).any():
            print(f"Lỗi tại bước {i}: Dữ liệu đầu vào x_future chứa NaN sau reshape:", x_future)
            break
        pred = model.predict(x_future, verbose=0)
        if np.isnan(pred).any():
            print(f"Lỗi tại bước {i}: Giá trị dự đoán (pred) chứa NaN: {pred}")
            break

        future_predictions.append(pred[0, 0])
        current_data = np.append(current_data, pred)[1:]

    future_predictions = np.array(future_predictions).reshape(-1, 1)
    if np.isnan(future_predictions).all():
        raise ValueError("Dữ liệu dự đoán (future_predictions) chứa toàn NaN.")

    future_predictions = sc.inverse_transform(future_predictions)
    if np.isnan(future_predictions).all():
        raise ValueError("Dữ liệu sau khi giải chuẩn hóa chứa toàn NaN.")

    future_df = pd.DataFrame(future_predictions, columns=['Dự đoán'], index=future_dates)
    return future_df