# 
# Script for building and training the LSTM model

from keras.models import Sequential
from keras.layers import LSTM, Dense
from config import Config
from dataset_creation import create_dataset

def build_and_train_model(trainX, trainY):
    model = Sequential()
    model.add(LSTM(Config.lstm_units, input_shape=(1, Config.look_back)))
    model.add(Dense(Config.dense_units))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=Config.epochs, batch_size=Config.batch_size, verbose=2)
    return model
