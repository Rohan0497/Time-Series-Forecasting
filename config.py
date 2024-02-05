# config.py
# Configuration file for hyperparameters

class Config:
    # Model configuration
    look_back = 1
    lstm_units = 4
    dense_units = 1
    epochs = 100
    batch_size = 1

    # Data configuration
    data_url = 'https://raw.githubusercontent.com/amankharwal/Website-data/master/airline-passengers.csv'
