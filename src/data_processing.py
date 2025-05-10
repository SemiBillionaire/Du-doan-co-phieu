import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Ngày'] = pd.to_datetime(df['Ngày'], format='%d/%m/%Y')
    df['Lần cuối'] = df['Lần cuối'].str.replace(',', '').astype(float)
    df = df.sort_values(by='Ngày')
    df = df.set_index('Ngày')
    df = df.resample('D').interpolate(method='linear')  # Nội suy dữ liệu thiếu
    df = df.reset_index()
    df1 = pd.DataFrame(df, columns=['Lần cuối'])
    df1.set_index(df['Ngày'], inplace=True)
    return df1

def prepare_data(df1, train_ratio=0.8, timestep=50):
    data = df1.values
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]

    sc = MinMaxScaler(feature_range=(0, 1))
    sc_train = sc.fit_transform(train_data)
    sc_test = sc.transform(test_data)

    x_train, y_train = [], []
    for i in range(timestep, len(train_data)):
        x_train.append(sc_train[i-timestep:i, 0])
        y_train.append(sc_train[i, 0])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    y_train = np.reshape(y_train, (y_train.shape[0], 1))

    test = df1[train_size-timestep:].values
    test = test.reshape(-1, 1)
    sc_test = sc.transform(test)
    x_test = []
    for i in range(timestep, len(test)):
        x_test.append(sc_test[i-timestep:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, train_size, sc, data