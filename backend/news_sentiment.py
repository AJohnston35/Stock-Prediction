import pandas as pd
import requests
import concurrent.futures
from ratelimit import limits, sleep_and_retry
import sqlite3
from datetime import datetime, timedelta

# Sample list of tickers
tickers = pd.read_csv('csv_files/stock_list.csv')['Code']

ONE_MINUTE = 60
MAX_CALLS_PER_MINUTE = 1000

@sleep_and_retry
@limits(calls=MAX_CALLS_PER_MINUTE, period=ONE_MINUTE)
def access_rate_limited_api(ticker):
    url = f'https://eodhd.com/api/sentiments?s={ticker}&from=2014-03-19&to=2024-03-20&api_token=65f5adcce659e6.99552900&fmt=json'
    response = requests.get(url)
    data = response.json()
    
    for item in data.get(f"{ticker}.US", []):
        date = item['date']
        count = item['count']
        normalized = item['normalized']
        # Write data to SQLite database
        write_to_database(ticker, date, count, normalized)
    
    return "Processed " + ticker

def generate_dates(start_date, end_date):
    dates = []
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    while current_date <= end_date:
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    return dates

def process_ticker(ticker):
    try:
        return access_rate_limited_api(ticker)
    except Exception as e:
        return f"Error processing {ticker}: {e}"

def write_to_database(ticker, date, count, normalized):
    conn = sqlite3.connect('database/sentiment_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sentiment_data
                 (ticker TEXT, date TEXT, count INTEGER, normalized REAL)''')
    c.execute("INSERT INTO sentiment_data (ticker, date, count, normalized) VALUES (?, ?, ?, ?)", (ticker, date, count, normalized))
    conn.commit()
    conn.close()

def fill_missing(ticker, ticker_df, start_date, end_date):
    all_dates = generate_dates(start_date, end_date)
    for i in range(len(all_dates)):
        if all_dates[i] not in ticker_df.date.values:
            new_row = {'ticker': ticker, 'date': all_dates[i], 'count': 0, 'normalized': 0}
            new_row = pd.DataFrame(new_row, index=[0])
            ticker_df = pd.concat([ticker_df, new_row], ignore_index=True)
    return ticker_df

def add_weekday(df):
    df['date'] = pd.to_datetime(df['date'])
    df['weekday'] = df['date'].dt.dayofweek
    return df

def weekend_average(df):
    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Add 'weekday' column
    df['weekday'] = df['date'].dt.dayofweek

    # Separate the weekdays and weekends
    df_weekday = df[df['weekday'] < 4]
    df_weekend = df[df['weekday'] >= 4]
    
    rows_to_drop = []
    
    for i in range(0, len(df_weekend)):
        if i + 2 >= len(df_weekend) or i + 1 >= len(df_weekend):
            break
        if df_weekend.iloc[i]['weekday'] == 4:
            sum_count = df_weekend.iloc[i]['count'] + df_weekend.iloc[i+1]['count'] + df_weekend.iloc[i+2]['count']
            if sum_count != 0:
                avg = ((df_weekend.iloc[i]['normalized'] * df_weekend.iloc[i]['count']) + 
                    (df_weekend.iloc[i+1]['normalized'] * df_weekend.iloc[i+1]['count']) + 
                    (df_weekend.iloc[i+2]['normalized'] * df_weekend.iloc[i+2]['count'])) / sum_count
            else:
                avg = 0
            new_row = {'ticker': df_weekend.iloc[i]['ticker'], 'date': df_weekend.iloc[i]['date'], 'count': sum_count, 'normalized': avg, 'weekday': 4}
            new_row = pd.DataFrame(new_row, index=[0])
            df_weekend = pd.concat([df_weekend, new_row], ignore_index=True)
            rows_to_drop.extend([i, i+1, i+2])
            i += 3
        else: continue
    
    df_weekend = df_weekend.drop(rows_to_drop).reset_index(drop=True)
        
    df_merged = pd.concat([df_weekday, df_weekend], ignore_index=True)
    
    # Sort the dataframe by 'ticker' and 'date'
    df_merged.sort_values(['ticker', 'date'], inplace=True)

    return df_merged


def main():
    # Define the maximum number of threads
    """df = pd.read_csv('csv_files/stock_list.csv')
    tickers = df.Code.unique()"""
    tickers = ['AAPL','NVDA','KO']
    max_threads = 3  # Adjust this based on your system's capabilities
    
    # Use ThreadPoolExecutor to create a pool of threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Submit tasks for each ticker to the executor
        futures = {executor.submit(process_ticker, ticker): ticker for ticker in tickers}
        
        # Iterate over completed futures and print results
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(result)       

    # After processing all tickers, fill missing dates in the table
    # Connect to SQLite database
    conn = sqlite3.connect('database/sentiment_data.db')

    # Read all data from sentiment_data table
    df = pd.read_sql_query("SELECT * FROM sentiment_data ORDER BY ticker ASC, date ASC", conn)

    # tickers = df.ticker.unique()
    # ticker_df = df[df.ticker == ticker]
    filled_dfs = pd.DataFrame()
    
    for ticker in tickers:
        ticker_df = df[df.ticker == ticker]
        filled_df = fill_missing(ticker, ticker_df, '2014-03-19', '2024-03-20')
        filled_dfs = pd.concat([filled_dfs, filled_df])
        
    sorted_df = filled_dfs.sort_values(by=['ticker', 'date'])
    #sorted_df = add_weekday(sorted_df)
    trading_days = weekend_average(sorted_df)
    trading_days.to_csv('csv_files/filled_sentiment_data.csv')
    #filled_dfs.to_sql('sentiment_data', conn, if_exists='replace', index=False)
    
    # Close the database connection
    conn.close()
    

if __name__ == "__main__":
    main()


# TODO: TAKE AVERAGE OF FRIDAY - SUNDAY REPORTS
# TODO: CREATE COLUMN "WEIGHTED AVERAGE" FOR ENTIRE DATASET
# TODO: MERGE WITH STOCK DATA
# TODO: MERGE INTEREST RATES WITH STOCK DATA
# TODO: CONVERT SCRIPT TO CLASS FORMAT
# TODO: CREATE SINGLE DATABASE WITH STOCK DATA, SENTIMENT DATA, AND INTEREST RATES

# FOR TRAINING, TAKE RANDOM SUBSET OF TICKERS ACCORDING TO PERCENTAGE COMPOSITION IN STOCK MARKET