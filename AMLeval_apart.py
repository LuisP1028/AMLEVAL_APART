import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Input
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import sys
from tqdm import tqdm
from tqdm.keras import TqdmCallback

# UNIVARIATE MODEL EVALUATION SUITE FOR 
# ARIMA, MLP, LSTM, CNN

# Function to create input-output sequences from univariate time series data
def create_sequences(data, input_length):
    X, y = [], []
    for i in range(len(data) - input_length):
        X.append(data[i:i + input_length])  # Slice of input_length for X
        y.append(data[i + input_length])    # The next value after the slice for y
    return np.array(X), np.array(y)

# Prepare input for the chosen model type
def prepare_input(data, input_length, model_type):
    X, y = create_sequences(data, input_length)  # Create input-output pairs

    if model_type == 'MLP':
        return X, y  # MLP expects 2D input
    elif model_type == 'LSTM' or model_type == 'CNN':
        return X.reshape((X.shape[0], X.shape[1], 1)), y  # 3D input for LSTM/CNN
    elif model_type == 'ARIMA':
        return data  # ARIMA uses 1D input
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Generalized model creation function
def create_model(model_type, input_length, config=None):
    if model_type == 'MLP':
        return create_mlp_model(input_length)
    elif model_type == 'LSTM':
        return create_lstm_model(input_length)
    elif model_type == 'CNN':
        return create_cnn_model(input_length, config)
    elif model_type == 'ARIMA':
        return create_arima_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# MLP Model Creation
def create_mlp_model(input_length):
    model = Sequential()
    model.add(Input(shape=(input_length,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# LSTM Model Creation
def create_lstm_model(input_length):
    model = Sequential()
    model.add(Input(shape=(input_length, 1)))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# CNN Model Creation
def create_cnn_model(input_length, config):
    n_filters, n_kernel, n_pool = config
    model = Sequential()
    model.add(Input(shape=(input_length, 1)))
    model.add(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu'))
    model.add(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu'))
    model.add(MaxPooling1D(pool_size=n_pool))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

# ARIMA Model Creation
def create_arima_model():
    return ARIMA  # ARIMA model will be instantiated later during training

# RMSE Calculation
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))

# Walk-forward validation for univariate data (works for ARIMA, MLP, LSTM, CNN)
def walk_forward_validation(data, n_test, input_length, epochs, batch_size, model_type='LSTM', config=None):
    predictions = []
    # Split dataset into training and test sets
    train, test = train_test_split(data, test_size=n_test/len(data), shuffle=False)

    if model_type == 'ARIMA':
        history = [x for x in train]
        for i in tqdm(range(len(test)), desc='Forecasting', unit='step'):
            model = ARIMA(history, order=(5,1,0))
            model_fit = model.fit(method_kwargs={"disp": 0})
            yhat = model_fit.forecast()[0]
            predictions.append(yhat)
            history.append(test[i])
    else:
        train_x, train_y = prepare_input(train, input_length, model_type)
        model = create_model(model_type, input_length, config)
        model.fit(
            train_x,
            train_y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[TqdmCallback(verbose=1)]
        )
        history = [x for x in train]

        for i in tqdm(range(len(test)), desc='Forecasting', unit='step'):
            x_input = np.array(history[-input_length:]).reshape(1, input_length)
            if model_type == 'LSTM' or model_type == 'CNN':
                x_input = x_input.reshape((1, input_length, 1))

            yhat = model.predict(x_input, verbose=0)[0][0]
            predictions.append(yhat)
            history.append(test[i])

    error = measure_rmse(test, predictions)
    return error

# Train and evaluate each model independently
def evaluate_final_models_separately(data, n_test, input_length, epochs, batch_size, n_models=5, model_type='LSTM', config=None):
    errors = []

    for i in tqdm(range(n_models), desc='Models', unit='model'):
        error = walk_forward_validation(data, n_test, input_length, epochs, batch_size, model_type, config)
        errors.append(error)
        print(f'Model {i + 1}/{n_models} RMSE: {error:.3f}')
        sys.stdout.flush()

    avg_error = np.mean(errors)
    print(f'Average RMSE across all models: {avg_error:.3f}')

    # Plotting box and whisker chart for RMSE scores
    plt.boxplot(errors)
    plt.title(f'Box and Whisker Plot of RMSE Scores for {model_type}')
    plt.ylabel('RMSE')
    plt.show()

    return avg_error

# Load dataset here
series = pd.read_csv(r'', header=0, index_col=0)
data = series.values.flatten()

# Parameters
n_test = int(len(data) * 0.2)
input_length = 24
epochs = 100
batch_size = 32
n_models = 5
model_type = 'CNN'  # You can switch this to 'MLP', 'LSTM', 'CNN', or 'ARIMA'
cnn_config = [256, 3, 2]  # CNN configuration (filters, kernel size, pooling size)

# Run the evaluation process and plot box and whisker chart
average_error = evaluate_final_models_separately(
    data,
    n_test,
    input_length,
    epochs,
    batch_size,
    n_models,
    model_type,
    cnn_config
)
