import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import os

def plot_comparison(df1, y_train_predict, y_test_predict, train_size, timestep=50, output_dir='output/comparison', stock_name='VNM'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_data1 = df1[timestep:train_size]
    test_data1 = df1[train_size:]

    plt.figure(figsize=(24, 8))
    plt.plot(df1.index, df1['Lần cuối'], label='Giá thực tế', color='red')
    train_data1['Dự đoán'] = y_train_predict
    plt.plot(train_data1.index, train_data1['Dự đoán'], label='Giá dự đoán train', color='green')
    test_data1['Dự đoán'] = y_test_predict
    plt.plot(test_data1.index, test_data1['Dự đoán'], label='Giá dự đoán test', color='blue')
    plt.title(f'So sánh giá thực tế và dự đoán ({stock_name})')
    plt.xlabel('Thời gian')
    plt.ylabel('Giá Lần cuối (VNĐ)')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f'comparison_{stock_name}.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Biểu đồ so sánh đã được lưu tại: {output_path}")

def plot_future(df1, future_df, output_dir='output/predict', stock_name='VNM'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(24, 8))
    plt.plot(df1.index, df1['Lần cuối'], label='Giá lịch sử', color='red')
    plt.plot(future_df.index, future_df['Dự đoán'], label='Dự đoán 3 năm tới', color='blue')
    plt.title(f'Dự đoán giá cổ phiếu 3 năm tới ({stock_name})')
    plt.xlabel('Thời gian')
    plt.ylabel('Giá Lần cuối (VNĐ)')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f'predict_{stock_name}.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Biểu đồ dự đoán đã được lưu tại: {output_path}")

def visualize_data(df1, output_dir='output/data_visualization_analysis', stock_name='VNM'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Đọc lại file CSV để lấy thêm các cột
    plt.figure(figsize=(12, 6))
    plt.plot(df1.index, df1['Lần cuối'], label='Giá lịch sử', color='red')
    plt.title(f'Biểu đồ giá cổ phiếu {stock_name} theo thời gian')
    plt.xlabel('Thời gian')
    plt.ylabel('Giá Lần cuối (VNĐ)')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    
    output_path = os.path.join(output_dir, f"{stock_name}_price_visualization.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Đã lưu biểu đồ vào: {output_path}")