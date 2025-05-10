# src/model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import os

def build_model(timestep=50):
    model = Sequential()
    model.add(LSTM(units=128, input_shape=(timestep, 1), return_sequences=True))
    model.add(LSTM(units=64))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model

def train_model(model, model_path, x_train, y_train, epochs=100, batch_size=50, output_dir='output/models', stock_name='VNM'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    best_model = ModelCheckpoint(model_path, monitor='val_loss', verbose=2, save_best_only=True, mode='auto')
    model.fit(x_train, y_train, epochs=100, batch_size=50, verbose=2, callbacks=[best_model])
    model.save(os.path.join(output_dir, f'model_{stock_name}.h5'))
    return model