import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns
import pandas as pd
import os

def convert_volume(value):
    """Hàm chuyển đổi giá trị khối lượng (KL) từ dạng string sang float."""
    value = str(value).strip()  # Chuyển thành string và loại bỏ khoảng trắng
    if 'M' in value:
        return float(value.replace('M', '')) * 1000000  # Nhân với 1 triệu
    elif 'K' in value:
        return float(value.replace('K', '')) * 1000  # Nhân với 1 nghìn
    else:
        return float(value)  # Giữ nguyên nếu không có ký hiệu

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

    df_raw = pd.read_csv(f'data/{stock_name}.csv')
    df_raw['Ngày'] = pd.to_datetime(df_raw['Ngày'], format='%d/%m/%Y')
    df_raw = df_raw.sort_values(by='Ngày')
    df_raw.set_index('Ngày', inplace=True)

    numeric_cols = ['Lần cuối', 'Mở', 'Cao', 'Thấp', 'KL']
    for col in numeric_cols:
        if col == 'KL':  # Xử lý cột KL riêng
            df_raw[col] = df_raw[col].apply(convert_volume)
        else:  # Xử lý các cột giá
            df_raw[col] = df_raw[col].str.replace(',', '').astype(float)
    corr_matrix = df_raw[numeric_cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f')
    plt.title(f'Heatmap tương quan giá cổ phiếu {stock_name}')
    plt.tight_layout()

    heatmap_output_path = os.path.join(output_dir, f"{stock_name}_correlation_heatmap.png")
    plt.savefig(heatmap_output_path, dpi=300)
    plt.close()
    print(f"Đã lưu heatmap vào: {heatmap_output_path}")