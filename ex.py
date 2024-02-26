import math # Mathematical functions
import numpy as np # Fundamental package for scientific computing with Python
import pandas as pd # Additional functions for analyzing and manipulating data

from datetime import date, timedelta, datetime # Date functions
from pandas.plotting import register_matplotlib_converters # This function adds plotting functions for calendar dates

import matplotlib.pyplot as plt # Important packages for visualization
import matplotlib.dates as mdates # Formatting dates

import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error # Packages for measuring model performance
from tensorflow.keras import Sequential # Deep learning library, used for neural networks
from tensorflow.keras.layers import LSTM, Dense, Dropout # Deep learning classes for recurrent and regular densely-connected layers
from tensorflow.keras.callbacks import EarlyStopping # EarlyStopping during model training
from sklearn.preprocessing import RobustScaler, MinMaxScaler # This scaler removes the median and scales
# the data according to the quantile range to normalize the price data

import seaborn as sns # Visualization
sns.set_style('white', {'axes.spines.right': False, 'axes.spines.top': False})

end_date = datetime.today().strftime('%Y-%m-%d')

def load_model():
    from tensorflow.keras.models import load_model
    print("Loading model...")
    # Load the saved model
    model = load_model('stock_model.h5')

def retrieve_data():
    print("Retrieving data...")
    from sqlalchemy import create_engine, text
    engine = create_engine('mysql+mysqlconnector://root:mnmsstar15@localhost:3306/StockMarket')
    #sql_query = text("SELECT * FROM stock_data WHERE market_cap = 'Mega';")
    sql_query = text("SELECT * FROM market_data WHERE Ticker = 'KO';")
    # Define batch size
    batch_size = 100000  # You can adjust this based on your memory constraints

    # Initialize an empty list to store batches
    data_batches = []

    print("Retrieving SQL data...")
    # Execute the SQL query in batches and store the result in batches
    for chunk in pd.read_sql_query(sql_query, engine, chunksize=batch_size):
        data_batches.append(chunk)

    # Concatenate the batches to form the final DataFrame
    df = pd.concat(data_batches)
    print(df.head())

    train_df = df.sort_values(by=['Date']).copy()
    train_df = train_df.dropna()
    #FEATURES = ['Open', 'High', 'Low', 'Volume','Increase_Decrease','Returns', 'Dividends', 'Stock Splits', 'volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi', 'volume_em', 'volume_sma_em', 'volume_vpt', 'volume_vwap', 'volume_mfi', 'volume_nvi', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl', 'volatility_bbw', 'volatility_bbp', 'volatility_bbhi', 'volatility_bbli', 'volatility_kcc', 'volatility_kch', 'volatility_kcl', 'volatility_kcw', 'volatility_kcp', 'volatility_kchi', 'volatility_kcli', 'volatility_dcl', 'volatility_dch', 'volatility_dcm', 'volatility_dcw', 'volatility_dcp', 'volatility_atr', 'volatility_ui', 'trend_macd', 'trend_macd_signal', 'trend_macd_diff', 'trend_sma_fast', 'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow', 'trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_ind_diff', 'trend_trix', 'trend_mass_index', 'trend_dpo', 'trend_kst', 'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_conv', 'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b', 'trend_stc', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg', 'trend_cci', 'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b', 'trend_aroon_up', 'trend_aroon_down', 'trend_aroon_ind', 'trend_psar_up', 'trend_psar_down', 'trend_psar_up_indicator', 'trend_psar_down_indicator', 'momentum_rsi', 'momentum_stoch_rsi', 'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d', 'momentum_tsi', 'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr', 'momentum_ao', 'momentum_roc', 'momentum_ppo', 'momentum_ppo_signal', 'momentum_ppo_hist', 'momentum_pvo', 'momentum_pvo_signal', 'momentum_pvo_hist', 'momentum_kama', 'others_dr', 'others_dlr', 'others_cr', 'Close']

    FEATURES = ['Close', 'others_cr', 'High', 'Low', 'trend_ichimoku_conv', 'volatility_kcl', 'Open', 'trend_ema_fast', 'trend_sma_fast', 'volatility_dcl']
    print([f for f in FEATURES])

    data = pd.DataFrame(train_df)
    data_filtered = data[FEATURES]

    data_filtered_ext = data_filtered.copy()
    data_filtered_ext['Prediction'] = data_filtered_ext['Close']
    print(data_filtered_ext.tail())
    return data_filtered_ext, data_filtered, FEATURES

def train_model(data_filtered_ext, data_filtered):
    print("Training model...")
    nrows = data_filtered.shape[0]

    np_data_unscaled = np.array(data_filtered)
    np_data = np.reshape(np_data_unscaled,(nrows, -1))
    print(np_data.shape)

    # Standardize the data
    scaler = MinMaxScaler()
    np_data_scaled = scaler.fit_transform(np_data_unscaled)

    scaler_pred = MinMaxScaler()
    df_close = pd.DataFrame(data_filtered_ext['Close'])
    np_close_scaled = scaler_pred.fit_transform(df_close)

    sequence_length = 1 #200

    index_close = data_filtered.columns.get_loc('Close')
    
    train_data_len = math.ceil(np_data_scaled.shape[0] * 0.7)

    train_data = np_data_scaled[0:train_data_len,:]
    test_data = np_data_scaled[train_data_len - sequence_length:, :]
    
    # Create the sequences
    def partition_dataset(sequence_length, data):
        x, y = [],[]
        data_len = data.shape[0]
        for i in range(sequence_length, data_len):
            x.append(data[i-sequence_length:i,:])
            y.append(data[i, index_close])
        x = np.array(x)
        y = np.array(y)
        return x,y
    
    x_train, y_train = partition_dataset(sequence_length, train_data)
    x_test, y_test = partition_dataset(sequence_length, test_data)

    # Train Model
    model = Sequential()

    n_neurons = 64
    print(n_neurons, x_train.shape[1],x_train.shape[2])
    model.add(LSTM(n_neurons,return_sequences=True, input_shape=(x_train.shape[1],x_train.shape[2])))
    model.add(LSTM(n_neurons,return_sequences=True))
    model.add(LSTM(n_neurons,return_sequences=False))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam',loss='mse')

    epochs = 500
    batch_size = 128
    early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
    trained_model = model.fit(x_train, y_train, batch_size=batch_size,epochs=epochs, validation_data = (x_test,y_test))
    
    # Plot Model Loss Over Epochs
    fig, ax = plt.subplots(figsize=(16,5),sharex=True)
    sns.lineplot(data=trained_model.history["loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    ax.xaxis.set_major_locator(plt.MaxNLocator(epochs))
    plt.legend(["Train","Test"], loc="upper left")
    plt.grid()
    plt.show()
    return model, x_train, x_test, y_train, y_test, scaler, scaler_pred

def make_predictions(model, x_train, x_test, y_train, y_test, scaler, scaler_pred, FEATURES, data_filtered_ext):
    print("Making predictions...")
    
    y_pred_scaled = model.predict(x_test)

    y_pred = scaler_pred.inverse_transform(y_pred_scaled)
    y_test_unscaled = scaler_pred.inverse_transform(y_test.reshape(-1,1))

    MAE = mean_absolute_error(y_test_unscaled, y_pred)
    print(f'Median Absolute Error (MAE): {np.round(MAE,2)}')
    #########################
    MAPE = np.mean((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled))) * 100
    print(f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')

    # Median Absolute Percentage Error (MDAPE)
    MDAPE = np.median((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled)) ) * 100
    print(f'Median Absolute Percentage Error (MDAPE): {np.round(MDAPE, 2)} %')

    display_start_date = pd.to_datetime("2022-01-01")

    # Add the difference between the valid and predicted prices
    train = pd.DataFrame(data_filtered_ext['Close'][:train_data_len + 1]).rename(columns={'Close': 'y_train'})
    valid = pd.DataFrame(data_filtered_ext['Close'][train_data_len:]).rename(columns={'Close': 'y_test'})
    valid.insert(1, "y_pred", y_pred, True)
    valid.insert(1, "residuals", valid["y_pred"] - valid["y_test"], True)
    df_union = pd.concat([train, valid])

    # Zoom in to a closer timeframe
    #df_union_zoom = df_union[df_union.index > display_start_date]

    # Create the lineplot
    fig, ax1 = plt.subplots(figsize=(16, 8))
    plt.title("y_pred vs y_test")
    #plt.ylabel(stockname, fontsize=18)
    sns.set_palette(["#090364", "#1960EF", "#EF5919"])
    sns.lineplot(data=df_union[['y_pred', 'y_train', 'y_test']], linewidth=1.0, dashes=False, ax=ax1)

    # Create the bar plot with the differences
    df_sub = ["#2BC97A" if x > 0 else "#C92B2B" for x in df_union["residuals"].dropna()]
    ax1.bar(height = 100, x=df_union['residuals'].dropna().index, width=3, label='residuals', color=df_sub)
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
    predicted_price = np.round(pred_price_unscaled.ravel()[0], 2)
    change_percent = np.round(100 - (price_today * 100)/predicted_price, 2)

    ticker = df['Ticker'].iloc[1]
    
    global end_date
    
    plus = '+'; minus = ''
    print(f'The close price for {ticker} at {end_date} was {price_today}')
    print(f'The predicted close price is {predicted_price} ({plus if change_percent > 0 else minus}{change_percent}%)')
    return True

def main():
    # Main code logic goes here
    print("Program starting...")
    data_filtered_ext, data_filtered, features = retrieve_data()
    model, x_train, x_test, y_train, y_test = train_model(data_filtered_ext, data_filtered)
    prediction = make_predictions(model, x_train, x_test, y_train, y_test, scaler, scaler_pred, FEATURES, data_filtered_ext)
    

if __name__ == "__main__":
    main()