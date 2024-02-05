# Time Series Prediction with LSTM

This repository contains Python scripts for predicting time series data using a Long Short-Term Memory (LSTM) neural network. The code is organized into modular scripts for data preprocessing, LSTM model creation and training, and visualization of results.

## Files

1. **`config.py`**: Configuration file containing hyperparameters and settings.

2. **`data_preprocessing.py`**: Module for loading and normalizing time series data.

3. **`lstm_model.py`**: Module for creating and training the LSTM model.

4. **`visualization.py`**: Module for visualizing the results of the time series prediction.

5. **`main_script.py`**: Main script that imports the modules and executes the complete workflow.

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/time-series-prediction-lstm.git
   cd time-series-prediction-lstm
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the `main_script.py`:

   ```bash
   python main_script.py
   ```

4. Adjust hyperparameters and settings in `config.py` for experimentation.

## Configuration (`config.py`)

- **Data Configuration**:
  - `data_url`: URL for the time series data.
  - `feature_range`: Tuple specifying the feature range for Min-Max scaling.

- **LSTM Model Configuration**:
  - `lstm_units`: Number of LSTM units in the model.
  - `dense_units`: Number of units in the Dense layer.
  - `loss_function`: Loss function for model training.
  - `optimizer`: Optimizer for model training.

- **Training Configuration**:
  - `epochs`: Number of epochs for model training.
  - `batch_size`: Batch size for model training.
  - `verbose`: Verbosity level during training.

## Results

The script generates visualizations that include the original time series data, training predictions, and test predictions. Additionally, it prints the root mean squared error (RMSE) for both training and testing phases.



Feel free to use, modify, and distribute the code as per the license terms. If you find this code helpful, consider giving it a star!