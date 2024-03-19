# For stock data
import yfinance as yf
# For data preprocessing
import numpy as np
# For dataframes
import pandas as pd
import sqlite3
conn = sqlite3.connect('database/sentiment_data.db')
# For generating technical indicators on market data
import ta

# READ SENTIMENT DATA FROM sentiment_data.db
# WRITE SENTIMENT DATA TO stock_data.db
# READ INTEREST RATES FROM FEDFUNDS.csv
# WRITE INTEREST RATES TO stock_data.db

class Stock:
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
    
    def __init__(self, symbol, name):
        self.symbol = symbol
        self.name = name
        self.ticker = yf.Ticker(symbol)

def main():
    print("Retrieving stock data")
    df = pd.read_csv('csv_files/stock_list.csv')

    i = 0
    for index, row in df.iterrows():
        ticker = row['Code']
        name = row['Name']
        obj = Stock(ticker, name)
        obj.stock_data()
        i += 1 
        print(str(i) + " out of " + str(len(df)))
        
if __name__ == "__main__":
    main()
