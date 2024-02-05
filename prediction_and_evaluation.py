# 
# Script for making predictions and evaluating the model

import math
from sklearn.metrics import mean_squared_error
from config import Config

def make_predictions_and_evaluate(model, trainX, testX, scaler, trainY, testY):
    # Make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # Invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # Calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))

    return trainPredict, testPredict, trainScore, testScore
