import yfinance as yf # For stock data

import numpy as np # For data preprocessing

import pandas as pd # For dataframes

import requests # For API requests
from ratelimit import limits, sleep_and_retry # For rate limiting

import mysql.connector # For database connection

from threading import Semaphore # For concurrency control

endpoint = 'stockaide-db.cluster-czcooo8241qk.us-east-2.rds.amazonaws.com'
username = 'admin'
password = 'mnmsstar15'
database = 'stockaide'

conn = mysql.connector.connect(user=username, password=password,
                              host=endpoint, database=database)
cursor = conn.cursor()


import ta # For generating technical indicators on market data

class Stock:
    ONE_MINUTE = 60
    MAX_CALLS_PER_MINUTE = 1000
    MAX_CONCURRENT_INSERTS = 1
    
    semaphore = Semaphore(MAX_CONCURRENT_INSERTS)
    
    # Class to represent a stock
    def stock_data(self):
        historical_data = self.ticker.history(period='10y')
        
        if historical_data.empty:
            print(f"No data available for {self.symbol} in the past trading week")
        else:
            historical_data.insert(0, 'Ticker', self.symbol)
            historical_data.insert(1, 'Date', historical_data.index)
            # Calculate technical indicators
            
            historical_data = ta.add_all_ta_features(historical_data, open='Open', high='High', low='Low', close='Close', volume='Volume', fillna=True)
            historical_data['Increase_Decrease'] = np.where(historical_data['Volume'].shift(-1) > historical_data['Volume'], 1, 0)
            historical_data['Returns'] = historical_data['Close'].pct_change()
            historical_data.to_sql('stock_data', con=conn, if_exists='append', index=False)

    @sleep_and_retry
    @limits(calls=MAX_CALLS_PER_MINUTE, period=ONE_MINUTE)
    def access_rate_limited_api(self):
        print("Retrieving sentiment data...")
        url = f'https://eodhd.com/api/eod/{self.ticker}.US?api_token=65f5adcce659e6.99552900&fmt=json&from={self.start_date}&to={self.end_date}'
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame()
        for item in data.get(f"{self.ticker}.US", []):
            date = item['date']
            count = item['count']
            normalized = item['normalized']
            new_row = {'ticker': self.ticker, 'date': date, 'count': count, 'normalized': normalized}
            new_row = pd.DataFrame(new_row, index=[0])
            df = pd.concat([df, new_row], ignore_index=True)
        
        return df 
    
    def stock_info(self):
        info = self.ticker.info
        print(info)
        
    def add_sentiment(self):
        print("Hello")
        # FILL IN GAPS IN SENTIMENT DATA
        # CREATE COLUMN "WEIGHTED AVERAGE" FOR ENTIRE DATASET
        # ASSIGN WEEKDAY VALUE TO EACH DAY
        # THEN TAKE AVERAGE OF FRIDAY - SUNDAY REPORTS
        # MERGE WITH STOCK DATA
    
    def write_to_database(self, df):
        print("Writing to database...")
        
        with self.semaphore:
            # Convert DataFrame rows to tuples
            data_tuples = [tuple(row) for row in df.to_numpy()]
            # Execute the INSERT INTO statement for each row in the DataFrame
            cursor.executemany('''INSERT INTO sentiment_data (ticker, date, weekday, weighted_avg) VALUES (%s, %s, %s, %s)''', data_tuples)
            conn.commit()
    
    def __init__(self, symbol, name, ticker, start_date, end_date):
        self.symbol = symbol
        self.name = name
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    
def main():
    print("Retrieving stock data")
    df = pd.read_sql_query("SELECT * FROM sentiment_data ORDER BY ticker", conn)

    """df = pd.read_csv('csv_files/stock_list.csv')

    i = 0
    for index, row in df.iterrows():
        ticker = row['Code']
        name = row['Name']
        obj = Stock(ticker, name)
        obj.stock_data()
        i += 1 
        print(str(i) + " out of " + str(len(df)))
        """

if __name__ == "__main__":
    main()