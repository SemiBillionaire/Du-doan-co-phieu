import os
from src.data_processing import load_data, prepare_data
from src.model import build_model, train_model
from src.predict import predict_historical, predict_future
from src.visualization import plot_comparison, plot_future, visualize_data
from tensorflow.keras.models import load_model

# Danh sách các file cổ phiếu
stock_files = {
    'VNM': 'data/VNM.csv',
    'FPT': 'data/FPT.csv',
    'HPG': 'data/HPG.csv',
    'MBB': 'data/MBB.csv'
}

timestep = 50
train_ratio = 0.8
future_days = 750  # ~3 năm (ngày giao dịch)

def get_company_input():
    print("Danh sách công ty có sẵn:", list(stock_files.keys()))
    while True:
        stock_name = input("Bạn muốn truy cập dữ liệu của công ty nào? (VD: VNM, HPG, ...): ").strip().upper()
        if stock_name in stock_files:
            return stock_name
        print(f"Xin lỗi, chúng tôi chưa có dữ liệu về công ty bạn vừa nhập, bạn còn muốn kiểm tra dữ liệu của công ty nào khác không?")

# Nhận input công ty
stock_name = get_company_input()
file_path = stock_files[stock_name]
print(f"Xử lý cổ phiếu: {stock_name}")

def get_user_choice():
    while True:
        print("\nBạn muốn:")
        print(f"1. Nhận dự đoán xu hướng tài chính 3 năm tới của {stock_name}")
        print("2. Kiểm tra độ chính xác của mô hình")
        print(f"3. Trực quan hóa dữ liệu lịch sử của {stock_name}")
        choice = input("Nhập lựa chọn: ").strip()
        if choice in ['1', '2', '3']:
            return choice
        print("Lựa chọn không hợp lệ. Vui lòng nhập 1, 2 hoặc 3.")

# Tải và xử lý dữ liệu
df1 = load_data(file_path)
x_train, y_train, x_test, train_size, sc, data = prepare_data(df1, train_ratio, timestep)

# Xây dựng và huấn luyện mô hình (hoặc tải mô hình)
model_path = f'output/models/model_{stock_name}.h5'
try:
    model = load_model(model_path)
    print(f"Đã tải mô hình từ: {model_path}")
except:
    print(f"Không tìm thấy mô hình, huấn luyện mô hình mới cho {stock_name}...")
    model = build_model(timestep=timestep)
    model = train_model(model, model_path, x_train, y_train, epochs=100, batch_size=50, stock_name=stock_name)

# Nhận lựa chọn từ người dùng
choice = get_user_choice()

if choice == '1':
    # Dự đoán 3 năm tới
    future_df = predict_future(model, df1, sc, timestep, future_days)
    # Lưu dữ liệu dự đoán thành file CSV
    output_dir = 'output/predict'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    csv_path = os.path.join(output_dir, f'future_predictions_{stock_name}.csv')
    future_df.to_csv(csv_path, index_label='Ngày')
    print(f"Dữ liệu dự đoán đã được lưu tại: {csv_path}")
    # Vẽ và lưu biểu đồ
    plot_future(df1, future_df, stock_name=stock_name)

elif choice == '2':
    # Kiểm tra độ chính xác (so sánh thực tế và dự đoán)
    y_train_predict, y_test_predict = predict_historical(model, x_train, x_test, sc)
    plot_comparison(df1, y_train_predict, y_test_predict, train_size, timestep, stock_name=stock_name)
    
elif choice == '3':
    # Trực quan hóa dữ liệu
    visualize_data(df1, stock_name=stock_name)