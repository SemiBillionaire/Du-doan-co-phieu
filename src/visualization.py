import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns
import pandas as pd
import mplfinance as mpf
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

def visualize_data(stock_name='VNM'):
    output_dir = f'output/data_visualization_analysis/{stock_name}_analysis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(f'data/{stock_name}.csv')
    df['Ngày'] = pd.to_datetime(df['Ngày'], format='%d/%m/%Y')
    df = df.sort_values(by='Ngày')
    df.set_index('Ngày', inplace=True)

    df.rename(columns={
        'Ngày': 'Date',
        'Lần cuối': 'Close',
        'Mở': 'Open',
        'Cao': 'High',
        'Thấp': 'Low',
        'KL': 'Volume',
        '% Thay đổi': 'Percent_Change'
    }, inplace=True)

    # Chuyển đổi các cột giá thành float
    numeric_cols = ['Close', 'Open', 'High', 'Low', 'Volume', 'Percent_Change']
    for col in numeric_cols:
        if col == 'Volume':
            df[col] = df[col].apply(convert_volume)
        elif col == 'Percent_Change':
            df[col] = df[col].str.replace('%', '').astype(float)
        else:
            df[col] = df[col].str.replace(',', '').astype(float)

    # Lọc dữ liệu cho Candlestick Chart (chỉ cuối quý từ 5/2020 - 5/2025)
    start_date = '2020-05-10'
    end_date = '2025-05-10'
    quarterly_df = df[(df.index >= start_date) & (df.index <= end_date) & 
                      (df.index.month.isin([3, 6, 7, 9, 12]))]
    # Lấy ngày cuối cùng của mỗi quý
    quarterly_df = quarterly_df.resample('QE').last().dropna()

    # 1. Candlestick Chart
    candlestick_df = quarterly_df[['Open', 'High', 'Low', 'Close']].copy()
    mpf.plot(candlestick_df, type='candle', style='charles', title=f'{stock_name} Candlestick Chart',
             ylabel='Price (VNĐ)', savefig=f'{output_dir}/{stock_name}_candlestick.png')

    # 2. Heatmap
    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f')
    plt.title(f'{stock_name} Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{stock_name}_heatmap.png')
    plt.close()

    # 3. Line Chart cho Lần cuối
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Close Price', color='blue')
    plt.title(f'{stock_name} Price Trend')
    plt.xlabel('Time')
    plt.ylabel('Price (VNĐ)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{stock_name}_price_trend.png')
    plt.close()

    # 4. Scatter Plot với KL và % Thay đổi
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Volume'], df['Percent_Change'], color='green', alpha=0.5)
    plt.title(f'{stock_name} Volume vs % Change')
    plt.xlabel('Trading Volume (VNĐ)')
    plt.ylabel('Percent_Change')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{stock_name}_scatter.png')
    plt.close()

    print(f"Đã tạo các biểu đồ cho {stock_name} trong thư mục: {output_dir}")