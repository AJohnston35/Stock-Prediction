import random
import numpy as np # For data preprocessing

import pandas as pd # For dataframes

import requests # For API requests
from ratelimit import limits, sleep_and_retry # For rate limiting

import mysql.connector # For database connection
from sqlalchemy import create_engine

import concurrent.futures # For parallel processing
from threading import Semaphore # For concurrency control

endpoint = 'stockaide-db.cluster-czcooo8241qk.us-east-2.rds.amazonaws.com'
username = 'admin'
password = 'mnmsstar15'
database = 'stockaide'
engine = create_engine(f'mysql+mysqlconnector://{username}:{password}@{endpoint}/{database}', echo=False)

conn = mysql.connector.connect(user=username, password=password,
                              host=endpoint, database=database)
cursor = conn.cursor()

from news_sentiment import Sentiment

import ta # For generating technical indicators on market data

# TODO: Make the start and end date dynamic

"""create_table_query = '''CREATE TABLE subset_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    Ticker TEXT,
    Date DATETIME,
    Open FLOAT(53),
    High FLOAT(53),
    Low FLOAT(53),
    Close FLOAT(53),
    Adjusted_close FLOAT(53),
    Volume BIGINT,
    Weekday BIGINT,
    Weighted_avg FLOAT(53),
    DFF FLOAT(53),
    volume_adi FLOAT(53),
    volume_obv BIGINT,
    volume_cmf FLOAT(53),
    volume_fi FLOAT(53),
    volume_em FLOAT(53),
    volume_sma_em FLOAT(53),
    volume_vpt FLOAT(53),
    volume_vwap FLOAT(53),
    volume_mfi FLOAT(53),
    volume_nvi FLOAT(53),
    volatility_bbm FLOAT(53),
    volatility_bbh FLOAT(53),
    volatility_bbl FLOAT(53),
    volatility_bbw FLOAT(53),
    volatility_bbp FLOAT(53),
    volatility_bbhi FLOAT(53),
    volatility_bbli FLOAT(53),
    volatility_kcc FLOAT(53),
    volatility_kch FLOAT(53),
    volatility_kcl FLOAT(53),
    volatility_kcw FLOAT(53),
    volatility_kcp FLOAT(53),
    volatility_kchi FLOAT(53),
    volatility_kcli FLOAT(53),
    volatility_dcl FLOAT(53),
    volatility_dch FLOAT(53),
    volatility_dcm FLOAT(53),
    volatility_dcw FLOAT(53),
    volatility_dcp FLOAT(53),
    volatility_atr FLOAT(53),
    volatility_ui FLOAT(53),
    trend_macd FLOAT(53),
    trend_macd_signal FLOAT(53),
    trend_macd_diff FLOAT(53),
    trend_sma_fast FLOAT(53),
    trend_sma_slow FLOAT(53),
    trend_ema_fast FLOAT(53),
    trend_ema_slow FLOAT(53),
    trend_vortex_ind_pos FLOAT(53),
    trend_vortex_ind_neg FLOAT(53),
    trend_vortex_ind_diff FLOAT(53),
    trend_trix FLOAT(53),
    trend_mass_index FLOAT(53),
    trend_dpo FLOAT(53),
    trend_kst FLOAT(53),
    trend_kst_sig FLOAT(53),
    trend_kst_diff FLOAT(53),
    trend_ichimoku_conv FLOAT(53),
    trend_ichimoku_base FLOAT(53),
    trend_ichimoku_a FLOAT(53),
    trend_ichimoku_b FLOAT(53),
    trend_stc FLOAT(53),
    trend_adx FLOAT(53),
    trend_adx_pos FLOAT(53),
    trend_adx_neg FLOAT(53),
    trend_cci FLOAT(53),
    trend_visual_ichimoku_a FLOAT(53),
    trend_visual_ichimoku_b FLOAT(53),
    trend_aroon_up FLOAT(53),
    trend_aroon_down FLOAT(53),
    trend_aroon_ind FLOAT(53),
    trend_psar_up FLOAT(53),
    trend_psar_down FLOAT(53),
    trend_psar_up_indicator FLOAT(53),
    trend_psar_down_indicator FLOAT(53),
    momentum_rsi FLOAT(53),
    momentum_stoch_rsi FLOAT(53),
    momentum_stoch_rsi_k FLOAT(53),
    momentum_stoch_rsi_d FLOAT(53),
    momentum_tsi FLOAT(53),
    momentum_uo FLOAT(53),
    momentum_stoch FLOAT(53),
    momentum_stoch_signal FLOAT(53),
    momentum_wr FLOAT(53),
    momentum_ao FLOAT(53),
    momentum_roc FLOAT(53),
    momentum_ppo FLOAT(53),
    momentum_ppo_signal FLOAT(53),
    momentum_ppo_hist FLOAT(53),
    momentum_pvo FLOAT(53),
    momentum_pvo_signal FLOAT(53),
    momentum_pvo_hist FLOAT(53),
    momentum_kama FLOAT(53),
    others_dr FLOAT(53),
    others_dlr FLOAT(53),
    others_cr FLOAT(53),
    Increase_Decrease BIGINT,
    Returns FLOAT(53)
);'''

cursor.execute(create_table_query)
conn.commit()
"""

class Stock:
    ONE_MINUTE = 60
    MAX_CALLS_PER_MINUTE = 1000
    
    # Class to represent a stock
    def process_ticker(self):
        historical_data = self.get_stock_data()
        
        if historical_data.empty:
            print(f"No data available for {self.ticker} in the past trading week")
        else:
            historical_data = self.add_sentiment(historical_data)
            historical_data = self.add_interest_rates(historical_data)
            historical_data = ta.add_all_ta_features(historical_data, open='Open', high='High', low='Low', close='Close', volume='Volume', fillna=True)
            historical_data['Increase_Decrease'] = np.where(historical_data['Volume'].shift(-1) > historical_data['Volume'], 1, 0)
            historical_data['Returns'] = historical_data['Close'].pct_change()*100
            return historical_data

    @sleep_and_retry
    @limits(calls=MAX_CALLS_PER_MINUTE, period=ONE_MINUTE)
    def get_stock_data(self):
        url = f'https://eodhd.com/api/eod/{self.ticker}.US?api_token=65f5adcce659e6.99552900&fmt=json&from={self.start_date}&to={self.end_date}'
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame()
        for item in data:
            date = item['date']
            open = item['open']
            high = item['high']
            low = item['low']
            close = item['close']
            adjusted_close = item['adjusted_close']
            volume = item['volume']
            new_row = {'Ticker': self.ticker, 'Date': date, 'Open': open, 'High': high, 'Low': low, 'Close': close, 'Adjusted_close': adjusted_close, 'Volume': volume}
            new_row = pd.DataFrame(new_row, index=[0])
            df = pd.concat([df, new_row], ignore_index=True)
        
        return df 
    
    def add_interest_rates(self, df):
    # Load the interest rates CSV file into a DataFrame
        interest_df = pd.read_csv('backend/csv_files/extrapolated_interest_rates.csv')
        interest_df['DATE'] = pd.to_datetime(interest_df['DATE'])
        # Merge the main DataFrame with the interest rates DataFrame based on the DATE column
        merged_df = pd.merge(df, interest_df, how='left', left_on='Date', right_on='DATE')

        # Drop the extra DATE column and rename FEDFUNDS to interest_rates
        merged_df.drop(columns=['DATE'], inplace=True)
        merged_df.rename(columns={'FEDFUNDS': 'interest_rates'}, inplace=True)

        return merged_df
        
    def add_sentiment(self, df):
        obj = Sentiment(self.ticker, self.start_date, self.end_date)
        sentiment_df = obj.process_ticker()
        df['Date'] = pd.to_datetime(df['Date'])
        df = pd.merge(df, sentiment_df, how='left', on='Date')
        df = df.drop(columns=['Ticker_y'])
        df = df.rename(columns={'Ticker_x': 'Ticker'})
        return df
    
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    
def main():
    start_date = '2014-03-24'
    end_date = '2024-03-24'
    
    max_threads = 20  # Adjust this based on your system's capabilities 
    
    tickers = pd.read_csv('backend/csv_files/stock_list.csv').Code.unique()

    ticker_subset = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'META', 'NVDA', 'PYPL', 'ADBE', 'NFLX']
    
    list_dfs = []
    
    # Use ThreadPoolExecutor to create a pool of threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Submit tasks for each ticker to the executor
        futures = {executor.submit(Stock(ticker, start_date, end_date).process_ticker): ticker for ticker in ticker_subset}
        processed = 0
        total = len(futures)
        # Iterate over completed futures and print results
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                list_dfs = list_dfs.append(result)
            processed += 1
            print(f'{processed}/{total} completed', end='\r')
            
    all_dfs = pd.concat(list_dfs, ignore_index=True)
    
    all_dfs.to_sql('subset_data', con=engine, if_exists='append', index=False, chunksize=50000)

if __name__ == "__main__":
    main()