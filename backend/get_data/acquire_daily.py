# Author: Alex Johnston
# Purpose: Retrieve stock market data from Yahoo! Finance and move that data to MySQL table
# Up to date as of: Feb 16, 2024


# Acquire data from yfinance
# Calculate metrics from data
# Write data to SQL

# For stock data
import yfinance as yf
# For data preprocessing
import numpy as np
# For dataframes
import pandas as pd
# For reading/writing to csv files
import csv
# For current date
import datetime
# Connection to MySQL
from sqlalchemy import create_engine, text
# Define your MySQL connection details
engine = create_engine('mysql+mysqlconnector://root:mnmsstar15@localhost:3306/StockMarket')
from datetime import date, timedelta, datetime # Date functions
# For generating technical indicators on market data
import ta


def retrieve_stock_data ():
    print("Retrieving stock data")
    df = pd.read_csv('csv_data/correct_stock_data_filtered.csv')

    i = 0
    for index, row in df.iterrows():
        ticker = row['Ticker']
        market_cap = row['MarketCapCategory']
        ticker_object = yf.Ticker(ticker)
        if ticker_object is not None and ticker_object.info is not None:
            company_info = ticker_object.info
        else:
            print(f"Failed to retrieve info for {ticker}")
        try:
            ticker_symbol = company_info.get('underlyingSymbol', None)
            industry = company_info.get('industryKey', company_info.get('quoteType', None))
        except ValueError as e:
            print(f"Error with ticker {ticker}: {e}")
            continue
        
        historical_data = yf.Ticker(ticker).history(period = '5y')
        
        if historical_data.empty:
            print(f"No data available for {ticker} in the past trading week")
            continue
        
        historical_data.insert(0,'Ticker',ticker_symbol)
        historical_data.insert(1,'Market_Cap', market_cap)
        historical_data.insert(2,'Date', historical_data.index)
        
        # Calculate technical indicators
        historical_data = ta.add_all_ta_features(historical_data, open='Open', high='High', low='Low', close='Close', volume='Volume', fillna=True)
        historical_data['Increase_Decrease'] = np.where(historical_data['Volume'].shift(-1) > historical_data['Volume'],1,0)
        historical_data['Returns'] = historical_data['Close'].pct_change()

        historical_data.to_sql('daily_data', con=engine, if_exists='append', index=False)

        i += 1
        print(str(i) + " out of " + str(len(df)))


def main():
    df = retrieve_stock_data()


if __name__ == "__main__":
    main()