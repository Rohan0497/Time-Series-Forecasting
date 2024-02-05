# data_preprocessing.py
# Script for loading and preprocessing data

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(data_url):
    # Load dataset
    dataframe = pd.read_csv(data_url, usecols=[1], engine='python')
    dataset = dataframe.values.astype('float32')

    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    return dataset, scaler
