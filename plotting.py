# plotting.py
# Script for plotting the results

import matplotlib.pyplot as plt
import numpy as np
from config import Config
from sklearn.preprocessing import MinMaxScaler

def plot_results(dataset, trainPredict, testPredict):
    # Shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[Config.look_back:len(trainPredict) + Config.look_back, :] = trainPredict

    # Shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (Config.look_back * 2) + 1:len(dataset) - 1, :] = testPredict

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # Plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()
