# main.py
# Main script for executing the entire process

from config import Config
from data_preprocessing import load_and_preprocess_data
from dataset_creation import create_dataset
from model_building import build_and_train_model
from prediction_and_evaluation import make_predictions_and_evaluate
from plotting import plot_results
import numpy as np

# Load and preprocess data
dataset, scaler = load_and_preprocess_data(Config.data_url)

# Split into train and test sets
train_size = int(len(dataset) * 0.67)
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# Create datasets
trainX, trainY = create_dataset(train, Config.look_back)
testX, testY = create_dataset(test, Config.look_back)

# Reshape input
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Build and train the LSTM model
model = build_and_train_model(trainX, trainY)

# Make predictions and evaluate the model
trainPredict, testPredict, trainScore, testScore = make_predictions_and_evaluate(model, trainX, testX, scaler, trainY, testY)

# Plot the results
plot_results(dataset, trainPredict, testPredict)

# Display evaluation scores
print(f'Train Score: {trainScore} RMSE')
print(f'Test Score: {testScore} RMSE')
