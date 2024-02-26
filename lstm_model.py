import math  # Mathematical functions
import numpy as np  # Fundamental package for scientific computing with Python
import pandas as pd  # Additional functions for analyzing and manipulating data

from datetime import datetime  # Date functions

import matplotlib.pyplot as plt  # Important packages for visualization

from sqlalchemy import create_engine, text  # Database connection and query execution

from sklearn.metrics import mean_absolute_error  # Packages for measuring model performance

from tensorflow.keras import Sequential  # Deep learning library, used for neural networks
from tensorflow.keras import layers  # Deep learning classes for recurrent and regular densely-connected layers
from tensorflow.keras.callbacks import EarlyStopping  # EarlyStopping during model training
from sklearn.preprocessing import MinMaxScaler  # This scaler removes the median and scales the data according to the quantile range to normalize the price data

import tensorflow as tf  # Deep learning library
from tensorflow.keras.optimizers import Adam  # Optimization algorithm for training the model

from tensorflow import keras  # Deep learning library
from tensorflow.keras import load_model  # Load a pre-trained model

import seaborn as sns  # Visualization

sns.set_style('white', {'axes.spines.right': False, 'axes.spines.top': False})

LSTM = layers.LSTM  # Long Short-Term Memory layer for recurrent neural networks
Dense = layers.Dense  # Regular densely-connected neural network layer

# For access across functions
global sequence_length
sequence_length = 5

global end_date
end_date = datetime.today().strftime('%Y-%m-%d')

  
def get_model(ticker, market_cap):
    """
    Loads and returns the LSTM model for a given ticker and market cap.

    Args:
        ticker (str): The ticker symbol of the stock.
        market_cap (str): The market capitalization category of the stock.

    Returns:
        keras.Model: The loaded LSTM model.
    """    
    
    print("Loading model...")
    
    # Load the saved model
    model = load_model(f'lstm_models/{market_cap}/{ticker}.h5')
    return model

def return_tickers():
    """
    Returns the tickers of companies with a market cap of 'Mega'.

    Returns:
        pandas.DataFrame: DataFrame containing the tickers of companies with a market cap of 'Mega'
    """    
    
    print("Retrieving tickers...")
    
    # Define the SQL query and connect to the MySQL database
    engine = create_engine('mysql+mysqlconnector://root:mnmsstar15@localhost:3306/StockMarket')
    sql_query = text(f"SELECT DISTINCT Ticker FROM market_data WHERE Market_Cap = 'Mega'")
    df = pd.read_sql_query(sql_query, engine)
    
    return df
          
def retrieve_data(ticker):
    """ Takes a ticker and retrieves the data for that ticker from the MySQL database.

    Args:
        ticker (str): Ticker symbol of the stock.

    Returns:
        pandas.DataFrame: DataFrame containing the stock data for the given ticker.
        list: List of features to be used in training.
        str: Market capitalization category of the stock.
    """    
    
    print("Retrieving data...")
    
    # Connect to MySQL
    engine = create_engine('mysql+mysqlconnector://root:mnmsstar15@localhost:3306/StockMarket')
    
    # Determine query based on ticker provided
    if ticker:
        # Define SQL query to be run with parameter placeholder
        sql_query = text(f"SELECT * FROM daily_data WHERE Ticker = '{ticker}';")
    else:
        # If no ticker provided, retrieve all rows
        sql_query = text("SELECT Ticker, Date, Market_Cap, Close, others_cr, High, Low, trend_ichimoku_conv, volatility_kcl, Open, trend_ema_fast, trend_sma_fast, volatility_dcl FROM daily_data WHERE Market_Cap = 'Mega';")

    # Define batch size
    batch_size = 100000 

    # Initialize an empty list to store batches
    data_batches = []

    iter = 0 
    
    # Execute the SQL query in batches and store the result in batches
    for chunk in pd.read_sql_query(sql_query, engine, chunksize=batch_size):
        iter = iter + batch_size
        print(f"{iter} processed...", end='\r')
        data_batches.append(chunk)

    # Concatenate the batches to form the final DataFrame
    df = pd.concat(data_batches)
    print(df.head())

    # Sort by date and get rid of NULL values
    train_df = df.sort_values(by=['Date']).copy()
    train_df = train_df.dropna()
    
    # Features to be used in training
    #FEATURES = ['Close', 'others_cr', 'High', 'Low', 'trend_ichimoku_conv', 'volatility_kcl', 'Open', 'trend_ema_fast', 'trend_sma_fast', 'volatility_dcl']
    FEATURES = ['Open', 'High', 'Low', 'Volume','Increase_Decrease','Returns', 'Dividends', 'Stock Splits', 'volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi', 'volume_em', 'volume_sma_em', 'volume_vpt', 'volume_vwap', 'volume_mfi', 'volume_nvi', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl', 'volatility_bbw', 'volatility_bbp', 'volatility_bbhi', 'volatility_bbli', 'volatility_kcc', 'volatility_kch', 'volatility_kcl', 'volatility_kcw', 'volatility_kcp', 'volatility_kchi', 'volatility_kcli', 'volatility_dcl', 'volatility_dch', 'volatility_dcm', 'volatility_dcw', 'volatility_dcp', 'volatility_atr', 'volatility_ui', 'trend_macd', 'trend_macd_signal', 'trend_macd_diff', 'trend_sma_fast', 'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow', 'trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_ind_diff', 'trend_trix', 'trend_mass_index', 'trend_dpo', 'trend_kst', 'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_conv', 'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b', 'trend_stc', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg', 'trend_cci', 'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b', 'trend_aroon_up', 'trend_aroon_down', 'trend_aroon_ind', 'trend_psar_up', 'trend_psar_down', 'trend_psar_up_indicator', 'trend_psar_down_indicator', 'momentum_rsi', 'momentum_stoch_rsi', 'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d', 'momentum_tsi', 'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr', 'momentum_ao', 'momentum_roc', 'momentum_ppo', 'momentum_ppo_signal', 'momentum_ppo_hist', 'momentum_pvo', 'momentum_pvo_signal', 'momentum_pvo_hist', 'momentum_kama', 'others_dr', 'others_dlr', 'others_cr', 'Close']

    market_cap = df['Market_Cap'].iloc[0]
    
    return train_df, FEATURES, market_cap

def preprocess_data(train_df, FEATURES):
    """ Preprocesses the data for training the LSTM model.

    Args:
        train_df (pandas.DataFrame): DataFrame containing the stock data for the given ticker.
        FEATURES (list): List of features to be used in training.

    Returns:
        numpy.Array: Numpy arrays containing the training and testing data.
    """    
    
    print("Preprocessing data...")
    
    # Define train/test split
    train_data_len = math.ceil(train_df.shape[0] * 0.7)
    
    # Define the sequence length in this scope
    global sequence_length
    
    # Partition into train and test sets
    train_data = train_df.iloc[0:train_data_len,:]
    test_data = train_df.iloc[train_data_len - sequence_length:, :]
    
    test_data_len = math.ceil(test_data.shape[0] * 0.5)
    val_data = test_data.iloc[0:test_data_len,:]
    test_data = test_data.iloc[test_data_len - sequence_length, :]
    
    # Sort by ticker once split into train and test sets
    train_data.sort_values(by='Ticker', inplace=True)
    test_data.sort_values(by='Ticker',inplace=True)
    val_data.sort_values(by='Ticker', inplace=True)
    
    # Filter data to only contain features
    train_data_filtered = train_data[FEATURES]
    test_data_filtered = test_data[FEATURES]
    val_data_filtered = val_data[FEATURES]

    # Create a new DataFrame to store the predictions
    train_filtered_ext = train_data_filtered.copy()
    test_filtered_ext = test_data_filtered.copy()
    val_filtered_ext = val_data_filtered.copy()
    
    # Add a Prediction column to the DataFrames
    train_filtered_ext['Prediction'] = train_filtered_ext['Close']
    test_filtered_ext['Prediction'] = test_filtered_ext['Close']
    val_filtered_ext['Prediction'] = val_filtered_ext['Close']
    
    print(train_data_filtered.tail())
    
    # Get the number of rows to train the model on
    train_rows = train_data_filtered.shape[0]
    test_rows = test_data_filtered.shape[0]
    val_rows = val_data_filtered.shape[0]

    # Convert the data to a numpy array
    train_data_unscaled = np.array(train_data_filtered)
    np_train = np.reshape(train_data_unscaled,(train_rows, -1))
    test_data_unscaled = np.array(test_data_filtered)
    np_test = np.reshape(test_data_unscaled,(test_rows, -1))
    val_data_unscaled = np.array(val_data_filtered)
    np_val = np.reshape(val_data_unscaled,(val_rows, -1))

    print(np_train.shape)
    print(np_test.shape)
    print(np_val.shape)

    # Scale the data
    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data_unscaled)
    test_data_scaled = scaler.fit_transform(test_data_unscaled)
    val_data_scaled = scaler.fit_transform(val_data_unscaled)

    # Scale the prediction data
    scaler_pred = MinMaxScaler()    
    train_pred = pd.DataFrame(train_filtered_ext['Prediction'])
    test_pred = pd.DataFrame(test_filtered_ext['Prediction'])
    val_pred = pd.DataFrame(val_filtered_ext['Prediction'])

    train_pred_scaled = scaler_pred.fit_transform(train_pred)
    test_pred_scaled = scaler_pred.fit_transform(test_pred)
    val_pred_scaled = scaler_pred.fit_transform(val_pred)
    ##############################################################
    
    # Find the index of the 'Close' column
    index_close = train_data_filtered.columns.get_loc('Close')
    
    # Create the sequences
    def partition_dataset(sequence_length, data, prediction_days=5):
        x, y = [],[]
        data_len = data.shape[0]
        # Partition the dataset into sequences
        for i in range(sequence_length, data_len - prediction_days):
            x.append(data[i-sequence_length:i,:])
            y.append(data[i:i+prediction_days, index_close])
        x = np.array(x)
        y = np.array(y)
        return x,y
    
    # Partition the dataset into x_train, y_train, x_test, and y_test
    x_train, y_train = partition_dataset(sequence_length, train_data_scaled)
    x_test, y_test = partition_dataset(sequence_length, test_data_scaled)
    x_val, y_val = partition_dataset(sequence_length, val_data_scaled)
    
    return x_train, x_test, y_train, y_test, train_data_filtered, test_data_filtered, scaler, scaler_pred, x_val, y_val

def train_model(x_train, x_test, y_train, y_test):
    """ Create and Train the LSTM model.

    Args:
        x_train (np.Array): Features to train the model on.
        x_test (np.Array): Target to train the model on.
        y_train (np.Array): Validation features.
        y_test (np.Array): Validation target.

    Returns:
        keras: 
    """    
    
    print("Training model...")
    
    # Train Model
    model = Sequential()

    n_neurons = 64  # Reduced number of neurons
    model.add(LSTM(n_neurons, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(n_neurons, return_sequences=True)) 
    model.add(LSTM(n_neurons, return_sequences=False))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(5))

    model.compile(optimizer='adam', loss='mse')

    epochs = 100  # Reduced number of epochs
    batch_size = 128  # Increased batch size
    early_stop = EarlyStopping(monitor='val_loss', patience=450, verbose=1)
    
    trained_model = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, 
                              validation_data=(x_test, y_test), callbacks=[early_stop])

    return model

def make_predictions(ticker, model):
    """_summary_

    Args:
        ticker (_type_): _description_
        model (_type_): _description_
    """    
    
    print("Making predictions...")
    
    df, FEATURES, market_cap = retrieve_data(ticker)
    
    x_train, x_test, y_train, y_test, train_data_filtered, test_data_filtered, scaler, scaler_pred = preprocess_data(df, FEATURES)
    
    y_pred_scaled = model.predict(x_test)

    y_pred = scaler_pred.inverse_transform(y_pred_scaled)
    y_test_unscaled = scaler_pred.inverse_transform(y_test)

    # Median Absolute Error
    MAE = mean_absolute_error(y_test_unscaled, y_pred)
    print(f'Median Absolute Error (MAE): {np.round(MAE,2)}')
    
    # Mean Absolute Percentage Error
    MAPE = np.mean((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled))) * 100
    print(f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')

    # Median Absolute Percentage Error (MDAPE)
    MDAPE = np.median((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled)) ) * 100
    print(f'Median Absolute Percentage Error (MDAPE): {np.round(MDAPE, 2)} %')

    display_start_date = pd.to_datetime("2022-01-01")

    # Add the difference between the valid and predicted prices
    train = pd.DataFrame(train_data_filtered['Close']).rename(columns={'Close': 'y_train'})
    valid = pd.DataFrame(test_data_filtered['Close'][:len(test_data_filtered)-1]).rename(columns={'Close': 'y_test'})

    # Trim 'valid' to match the length of 'y_pred' and insert 'y_pred'
    valid = valid.iloc[:len(y_pred)]
    for i in range(5):
        valid.insert(i+1, f"y_pred_day_{i+1}", y_pred[:, i], True)
    df_union = pd.concat([train, valid])

    # Create the lineplot
    fig, ax1 = plt.subplots(figsize=(16, 8))
    plt.title("y_pred vs y_test")
    sns.set_palette(["#090364", "#1960EF", "#EF5919"])
    sns.lineplot(data=df_union[['y_pred_day_1', 'y_pred_day_2', 'y_pred_day_3', 'y_pred_day_4', 'y_pred_day_5', 'y_train', 'y_test']], linewidth=1.0, dashes=False, ax=ax1)

    plt.legend()
    plt.show()

    df_temp = df[-sequence_length:]
    new_df = df_temp.filter(FEATURES)

    N = sequence_length

    # Get the last N day closing price values and scale the data to be values between 0 and 1
    last_N_days = new_df[-sequence_length:].values
    last_N_days_scaled = scaler.transform(last_N_days)

    # Create an empty list and Append past N days
    X_test_new = []
    X_test_new.append(last_N_days_scaled)

    # Convert the X_test data set to a numpy array and reshape the data
    pred_price_scaled = model.predict(np.array(X_test_new))
    pred_price_unscaled = scaler_pred.inverse_transform(pred_price_scaled.reshape(-1, 1))

    # Print last price and predicted price for the next day
    price_today = np.round(new_df['Close'].iloc[-1], 2)
    predicted_prices = np.round(pred_price_unscaled.ravel(), 2)
    print(f'The close price for {ticker} at {end_date} was {price_today}')
    
    previous_price = price_today
    for i, predicted_price in enumerate(predicted_prices):
        change_percent = np.round(100 - (previous_price * 100)/predicted_price, 2)
        sign = '+' if change_percent > 0 else '-'
        print(f'The predicted close price for day {i+1} is {predicted_price} ({sign}{abs(change_percent)}%)')
        previous_price = predicted_price

def main():
    """
    Entry point of the program. 

    Retrieves the data for a specific stock, preprocesses the data, trains an LSTM model on the training data, 
    and makes predictions on the test data. The user is prompted to choose a stock to predict.
    """
    
    print("Program starting...")
    #ticker_list = return_tickers()
    #for index, ticker in ticker_list.iterrows():
        #ticker_value = ticker.iloc[0]
    
    # 'Open' 'volatility_kch' 'volatility_kcl' 'trend_sma_fast' 'trend_ema_fast' 'High' ‘Low’ 'others_cr' 'Close' 'trend_psar_down'
    # Neurons: 256; Batch Size: 256; Epochs: 50
    # Neurons: 128; Batch Size: 128; Epochs: 50
    df, features, market_cap = retrieve_data('KO')
    x_train, x_test, y_train, y_test, train_data_filtered, test_data_filtered, scaler, scaler_pred = preprocess_data(df, features)
    
    model = train_model(x_train, x_test, y_train, y_test)
        #model.save(f'lstm_models/{market_cap}/{ticker_value}.h5')
    ticker_value = input(f"Choose a stock to predict: ")
    #model = get_model(ticker_value, market_cap)
    make_predictions(ticker_value, model)

if __name__ == "__main__":
    main()