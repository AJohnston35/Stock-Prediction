import pandas as pd
import requests
import concurrent.futures
from ratelimit import limits, sleep_and_retry
import sqlite3

# Sample list of tickers
tickers = pd.read_csv('csv_files/stock_list.csv')['Code']

ONE_MINUTE = 60
MAX_CALLS_PER_MINUTE = 1000

@sleep_and_retry
@limits(calls=MAX_CALLS_PER_MINUTE, period=ONE_MINUTE)
def access_rate_limited_api(ticker):
    url = f'https://eodhd.com/api/sentiments?s={ticker}&from=2014-03-19&to=2024-03-19&api_token=65f5adcce659e6.99552900&fmt=json'
    response = requests.get(url)
    data = response.json()
    
    for item in data.get(f"{ticker}.US", []):
        date = item['date']
        count = item['count']
        normalized = item['normalized']
        # Write data to SQLite database
        write_to_database(ticker, date, count, normalized)
    
    return "Processed " + ticker


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

def main():
    # Define the maximum number of threads
    max_threads = 50  # Adjust this based on your system's capabilities
    
    # Use ThreadPoolExecutor to create a pool of threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Submit tasks for each ticker to the executor
        futures = {executor.submit(process_ticker, ticker): ticker for ticker in tickers}
        
        # Iterate over completed futures and print results
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(result)       

if __name__ == "__main__":
    main()
