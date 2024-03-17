# For stock data
import yfinance as yf
# For data preprocessing
import numpy as np
# For dataframes
import pandas as pd
# Connection to SQLite
from sqlalchemy import create_engine
engine = create_engine('sqlite:///database/db.sqlite')
# For generating technical indicators on market data
import ta

from news_sentiment import Sentiment

class Stock:
    # Class to represent a stock
    def stock_data(self):
        historical_data = self.ticker.history(period='10y')
        
        if historical_data.empty:
            print(f"No data available for {self.symbol} in the past trading week")
        else:
            historical_data.insert(0, 'Ticker', self.symbol)
            historical_data.insert(1, 'Market_Cap', self.market_cap)
            historical_data.insert(2, 'Date', historical_data.index)
            sentiment = self.stock_sentiment()
            # Calculate technical indicators
            historical_data = ta.add_all_ta_features(historical_data, open='Open', high='High', low='Low', close='Close', volume='Volume', fillna=True)
            historical_data['Increase_Decrease'] = np.where(historical_data['Volume'].shift(-1) > historical_data['Volume'], 1, 0)
            historical_data['Returns'] = historical_data['Close'].pct_change()
            merged_df = pd.merge(historical_data, sentiment, left_on='Date', right_on='date', how='left')
            merged_df.to_sql('stock_data', con=engine, if_exists='append', index=False)

    def stock_sentiment(self):
        obj = Sentiment(self.symbol, self.name, 3650)
        news_df = obj.load_news()
        news_df.fillna(0)
        #  Perform sentiment analysis
        results_df = obj.perform_sentiment_analysis(news_df)
        #  Group by date
        grouped_df = results_df.set_index('date').groupby(pd.Grouper(freq='D')).sum()
        return grouped_df
    
    def stock_info(self):
        info = self.ticker.info
        print(info)
        
    def __init__(self, symbol, market_cap, name):
        self.symbol = symbol
        self.market_cap = market_cap
        self.name = name
        self.ticker = yf.Ticker(symbol)

def main():
    print("Retrieving stock data")
    df = pd.read_csv('csv_files/correct_stock_data_filtered.csv')

    i = 0
    for index, row in df.iterrows():
        ticker = row['Ticker']
        market_cap = row['MarketCapCategory']
        name = row['Name']
        obj = Stock(ticker, market_cap, name)
        obj.stock_data()
        i += 1 
        print(str(i) + " out of " + str(len(df)))
        
if __name__ == "__main__":
    main()

'''
msft = yf.Ticker("MSFT")

# get all stock info
msft.info

# get historical market data
hist = msft.history(period="1mo")

# show meta information about the history (requires history() to be called first)
msft.history_metadata

# show actions (dividends, splits, capital gains)
msft.actions
msft.dividends
msft.splits
msft.capital_gains  # only for mutual funds & etfs

# show share count
msft.get_shares_full(start="2022-01-01", end=None)

# show financials:
# - income statement
msft.income_stmt
msft.quarterly_income_stmt
# - balance sheet
msft.balance_sheet
msft.quarterly_balance_sheet
# - cash flow statement
msft.cashflow
msft.quarterly_cashflow
# see `Ticker.get_income_stmt()` for more options

# show holders
msft.major_holders
msft.institutional_holders
msft.mutualfund_holders
msft.insider_transactions
msft.insider_purchases
msft.insider_roster_holders

# show recommendations
msft.recommendations
msft.recommendations_summary
msft.upgrades_downgrades

# Show future and historic earnings dates, returns at most next 4 quarters and last 8 quarters by default. 
# Note: If more are needed use msft.get_earnings_dates(limit=XX) with increased limit argument.
msft.earnings_dates

# show ISIN code - *experimental*
# ISIN = International Securities Identification Number
msft.isin

# show options expirations
msft.options

# show news
msft.news

# get option chain for specific expiration
opt = msft.option_chain('YYYY-MM-DD')
# data available via: opt.calls, opt.puts
'''