# Author: Alex Johnston
# Purpose: Retrieve stock market data from MySQL and preprocess the data so that it can be used to train an LSTM model.
# Last Updated: Mar 19, 2024

# Commented out code may be relevant for future use, but is not currently in use.

# TODO: Get rid of class and make functions standalone
# TODO: Implement training on multiple stocks
# TODO: Alter data retrieval to new database schema

from openai import chatgpt
import math  # Mathematical functions
import numpy as np  # Fundamental package for scientific computing with Python
import pandas as pd  # Additional functions for analyzing and manipulating data

from datetime import datetime  # Date functions

import matplotlib.pyplot as plt  # Important packages for visualization

from sqlalchemy import create_engine, text  # Database connection and query execution

from sklearn.metrics import mean_absolute_error  # Packages for measuring model performance

from sklearn.preprocessing import MinMaxScaler  # This scaler removes the median and scales the data according to the quantile range to normalize the price data

import tensorflow as tf  # Deep learning library

from tensorflow import keras  # Deep learning library

from keras import load_model  # Load a pre-trained model

from keras.optimizers import Adam  # Optimization algorithm for training the model

from keras import Sequential, layers, EarlyStopping 
# EarlyStopping: EarlyStopping during model training  
# layers: Deep learning classes for recurrent and regular densely-connected layers 
# Sequential: Deep learning library, used for neural networks

import seaborn as sns  # Visualization

sns.set_style('white', {'axes.spines.right': False, 'axes.spines.top': False})

LSTM = layers.LSTM  # Long Short-Term Memory layer for recurrent neural networks
Dense = layers.Dense  # Regular densely-connected neural network layer

# For access across functions
global sequence_length
sequence_length = 5

global end_date
end_date = datetime.today().strftime('%Y-%m-%d')

  
class Model:
    def get_model():
        model = load_model('model.h5')
        return model
    
    def return_tickers():
        engine = create_engine('sqlite:///db.sqlite')
        sql_query = text(f"SELECT DISTINCT Ticker FROM stock_data")
        df = pd.read_sql_query(sql_query, engine)
        return df

    def retrieve_data(self):
        engine = create_engine('sqlite:///../database/db.sqlite')
        
        # Determine query based on ticker provided
        if self.ticker:
            # Define SQL query to be run with parameter placeholder
            sql_query = text(f"SELECT * FROM daily_data WHERE Ticker = '{self.ticker}';")
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
    
    def preprocess_data(self, train_df, FEATURES):
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
        # Create Model
        model = Sequential()
        
        n_neurons = 128 

        # ADD INDUSTRY

        
        model.add(LSTM(n_neurons, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(LSTM(n_neurons, return_sequences=True)) 
        model.add(LSTM(n_neurons, return_sequences=False))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(5))

        model.compile(optimizer='adam', loss='mse')

        epochs = 50  
        batch_size = 128  
        early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        
        trained_model = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, 
                                validation_data=(x_test, y_test), callbacks=[early_stop])

        return model
    
    def make_predictions(self, model):
        """ Makes predictions on the test data using the trained LSTM model.

        Args:
            ticker (str): Ticker symbol of the stock.
            model (keras.LSTM): The trained LSTM model.
        """    
        
        print("Making predictions...")
        
        df, FEATURES, market_cap = self.retrieve_data(self.ticker)
        
        x_train, x_test, y_train, y_test, train_data_filtered, test_data_filtered, scaler, scaler_pred = self.preprocess_data(df, FEATURES)
        
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
        print(f'The close price for {self.ticker} at {end_date} was {price_today}')
        
        previous_price = price_today
        for i, predicted_price in enumerate(predicted_prices):
            change_percent = np.round(100 - (previous_price * 100)/predicted_price, 2)
            sign = '+' if change_percent > 0 else '-'
            print(f'The predicted close price for day {i+1} is {predicted_price} ({sign}{abs(change_percent)}%)')
            previous_price = predicted_price

    def __init__(self, ticker):
        self.ticker = ticker
    
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
    obj = Model('AAPL')
    df, features, market_cap = obj.retrieve_data('KO')
    x_train, x_test, y_train, y_test, train_data_filtered, test_data_filtered, scaler, scaler_pred = obj.preprocess_data(df, features)
    
    model = obj.train_model(x_train, x_test, y_train, y_test)
        #model.save(f'lstm_models/{market_cap}/{ticker_value}.h5')
    ticker_value = input(f"Choose a stock to predict: ")
    #model = get_model(ticker_value, market_cap)
    obj.make_predictions(ticker_value, model)

if __name__ == "__main__":
    main()
    

"""# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('seaborn-whitegrid')

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error

# Some functions to help out with
def return_rmse(test,predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {}.".format(rmse))
    
import os
fileList = os.listdir("../input")

companyList = []
for file in fileList:
    companyName = file.split("_")[0]
    if companyName != "all":
        companyList.append(companyName)
print(companyList)

# First, we get the data
stockList = ["GE", "MSFT", "GOOGL", "AAPL", "AMZN", "IBM", "CSCO"]
df_ = {}
for i in stockList:
    df_[i] = pd.read_csv("../input/" + i + "_2006-01-01_to_2018-01-01.csv", index_col="Date", parse_dates=["Date"])
    
def split(dataframe, border, col):
    return dataframe.loc[:border,col], dataframe.loc[border:,col]

df_new = {}
for i in stockList:
    df_new[i] = {}
    df_new[i]["Train"], df_new[i]["Test"] = split(df_[i], "2015", "Close")
    
for i in stockList:
    plt.figure(figsize=(14,4))
    plt.plot(df_new[i]["Train"])
    plt.plot(df_new[i]["Test"])
    plt.ylabel("Price")
    plt.xlabel("Date")
    plt.legend(["Training Set", "Test Set"])
    plt.title(i + " Closing Stock Price")

# Scaling the training set
transform_train = {}
transform_test = {}
scaler = {}

for num, i in enumerate(stockList):
    sc = MinMaxScaler(feature_range=(0,1))
    a0 = np.array(df_new[i]["Train"])
    a1 = np.array(df_new[i]["Test"])
    a0 = a0.reshape(a0.shape[0],1)
    a1 = a1.reshape(a1.shape[0],1)
    transform_train[i] = sc.fit_transform(a0)
    transform_test[i] = sc.fit_transform(a1)
    scaler[i] = sc
    
del a0
del a1

for i in transform_train.keys():
    print(i, transform_train[i].shape)
print("\n")    
for i in transform_test.keys():
    print(i, transform_test[i].shape)

trainset = {}
testset = {}
for j in stockList:
    trainset[j] = {}
    X_train = []
    y_train = []
    for i in range(60,2516):
        X_train.append(transform_train[j][i-60:i,0])
        y_train.append(transform_train[j][i,0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    trainset[j]["X"] = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
    trainset[j]["y"] = y_train
    
    testset[j] = {}
    X_test = []
    y_test = []    
    for i in range(60, 755):
        X_test.append(transform_test[j][i-60:i,0])
        y_test.append(transform_test[j][i,0])
    X_test, y_test = np.array(X_test), np.array(y_test)
    testset[j]["X"] = np.reshape(X_test, (X_test.shape[0], X_train.shape[1], 1))
    testset[j]["y"] = y_test
    
arr_buff = []
for i in stockList:
    buff = {}
    buff["X_train"] = trainset[i]["X"].shape
    buff["y_train"] = trainset[i]["y"].shape
    buff["X_test"] = testset[i]["X"].shape
    buff["y_test"] = testset[i]["y"].shape
    arr_buff.append(buff)

pd.DataFrame(arr_buff, index=stockList)

# The LSTM architecture
regressor = Sequential()
# First LSTM layer with Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))
# Second LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Third LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.5))
# Fourth LSTM layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.5))
# The output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='rmsprop', loss='mean_squared_error')
# Fitting to the training set
for i in stockList:
    print("Fitting to", i)
    regressor.fit(trainset[i]["X"], trainset[i]["y"], epochs=10, batch_size=200)

pred_result = {}
for i in stockList:
    y_true = scaler[i].inverse_transform(testset[i]["y"].reshape(-1,1))
    y_pred = scaler[i].inverse_transform(regressor.predict(testset[i]["X"]))
    MSE = mean_squared_error(y_true, y_pred)
    pred_result[i] = {}
    pred_result[i]["True"] = y_true
    pred_result[i]["Pred"] = y_pred
    
    plt.figure(figsize=(14,6))
    plt.title("{} with MSE {:10.4f}".format(i,MSE))
    plt.plot(y_true)
    plt.plot(y_pred)

time_index = df_new["CSCO"]["Test"][60:].index
def lagging(df, lag, time_index):
    df_pred = pd.Series(df["Pred"].reshape(-1), index=time_index)
    df_true = pd.Series(df["True"].reshape(-1), index=time_index)
    
    df_pred_lag = df_pred.shift(lag)
    
    print("MSE without Lag", mean_squared_error(np.array(df_true), np.array(df_pred)))
    print("MSE with Lag 5", mean_squared_error(np.array(df_true[:-5]), np.array(df_pred_lag[:-5])))

    plt.figure(figsize=(14,4))
    plt.title("Prediction without Lag")
    plt.plot(df_true)
    plt.plot(df_pred)

    MSE_lag = mean_squared_error(np.array(df_true[:-5]), np.array(df_pred_lag[:-5]))
    plt.figure(figsize=(14,4))
    plt.title("Prediction with Lag")
    plt.plot(df_true)
    plt.plot(df_pred_lag)

lagging(pred_result["IBM"], -5, time_index)

# The GRU architecture
regressorGRU = Sequential()
# First GRU layer with Dropout regularisation
regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
regressorGRU.add(Dropout(0.2))
# Second GRU layer
regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
regressorGRU.add(Dropout(0.2))
# Third GRU layer
regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
regressorGRU.add(Dropout(0.2))
# Fourth GRU layer
regressorGRU.add(GRU(units=50, activation='tanh'))
regressorGRU.add(Dropout(0.2))
# The output layer
regressorGRU.add(Dense(units=1))
# Compiling the RNN
regressorGRU.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False),loss='mean_squared_error')
# Fitting to the training set
for i in stockList:
    print("Fitting to", i)
    regressorGRU.fit(trainset[i]["X"], trainset[i]["y"],epochs=50,batch_size=150)

for i in stockList:
    y_true = scaler[i].inverse_transform(testset[i]["y"].reshape(-1,1))
    y_pred = scaler[i].inverse_transform(regressorGRU.predict(testset[i]["X"]))
    MSE = mean_squared_error(y_true, y_pred)
    
    plt.figure(figsize=(14,6))
    plt.title("{} with MSE {:10.4f}".format(i,MSE))
    plt.plot(y_true)
    plt.plot(y_pred)

"""