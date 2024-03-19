import pandas as pd
import math
import numpy as np
import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
US_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())
from yahoo_fin import stock_info as si
import os
import time
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import itertools
import concurrent.futures
from ratelimit import limits, RateLimitException, sleep_and_retry

MAX_TICKER_THREADS = 8
MAX_URL_THREADS = 32

API_KEY = "65f5adcce659e6.99552900"

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Class functions
class Sentiment:    
    def get_biz_days_delta_date(self, start_date_str, delta_days):
        start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = start_date + (delta_days * US_BUSINESS_DAY)
        end_date_str = datetime.datetime.strftime(end_date, "%Y-%m-%d")
        return end_date_str

    def fetch_news(self, start_date_str, end_date_str, limit, offset=0):
        url = f'https://eodhistoricaldata.com/api/news?api_token={API_KEY}&s={self.symbol}&limit={limit}&offset={offset}&from={start_date_str}&to={end_date_str}'
        print(url)
        news_json = requests.get(url).json()
        results_df = pd.DataFrame()
        for item in news_json:
            title = item['title']
            desc = item['content']
            date = pd.to_datetime(item["date"], utc=True)
            result_df = pd.DataFrame({"title": [title], "desc": [desc], "date": [date]})
            results_df = pd.concat([results_df, result_df], axis=0, ignore_index=True)
        time.sleep(1)
        return results_df

    def load_news(self):
        limit = 8
        today = datetime.datetime.today()
        today_date_str = today.strftime("%Y-%m-%d")
        news_df = pd.DataFrame()

        # Define a function to fetch news for a specific day
        def fetch_news_for_day(start_date, end_date):
            return self.fetch_news(start_date, end_date, limit, 0)

        # Create a list of date ranges to fetch news for each day
        date_ranges = []
        for i in range(0, self.past_days):
            start_day = self.past_days - i
            end_day = self.past_days - i - 1
            start_date_str = self.get_biz_days_delta_date(today_date_str, -start_day)
            end_date_str = self.get_biz_days_delta_date(today_date_str, -end_day)
            date_ranges.append((start_date_str, end_date_str))

        # Use multithreading to fetch news for each day
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_URL_THREADS) as executor:
            results = list(executor.map(lambda x: fetch_news_for_day(x[0], x[1]), date_ranges))

        # Concatenate the results into a single DataFrame
        news_df = pd.concat(results, ignore_index=True)
        news_df.to_csv(f'csv_files/news/{self.symbol}_news.csv', index=False)
        return news_df

    def is_empty_string(self, str):
        if str == '' or (isinstance(str, float) and math.isnan(str)):
            return True
        else:
            return False

    def batch(self, iterable, size):
        """
        Generator function to split an iterable into batches of the given size.
        """
        it = iter(iterable)
        while True:
            batch_items = list(itertools.islice(it, size))
            if not batch_items:
                break
            yield batch_items

    def perform_sentiment_analysis(self, df, batch_size=50):
        """
        Perform sentiment analysis on headlines in batches.
        """
        results_df = pd.DataFrame([], columns=['date', 'positive', 'negative'])
        count = 0
        for batch_headlines in self.batch(df.iterrows(), batch_size):
            headlines = []
            dates = []
            for index, row in batch_headlines:
                count += 1
                if count % 10 == 0:
                    print(f"Performing sentiment analysis {count} of {len(df)}")
                date = row["date"]
                title = row["title"]
                desc = row["desc"]
                if self.is_empty_string(title) or self.is_empty_string(desc):
                    continue
                headlines.append(title)
                dates.append(date)
            if len(headlines) == 0:
                continue
            #  Run sentiment analysis on batch
            inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors='pt')
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            positive = predictions[:,0].tolist()
            negative = predictions[:,1].tolist()
            neutral = predictions[:,2].tolist()
            table = {"headline": headlines,
                    "positive": positive,
                    "negative": negative,
                    "neutral": neutral,
                    "date": dates}
            batch_results_df = pd.DataFrame(table, columns=["positive", "negative", "neutral", "date"])
            results_df = pd.concat([results_df, batch_results_df], ignore_index=True)
        return results_df

    def __init__(self, symbol, past_days):
        self.symbol = symbol
        self.past_days = past_days


        


def merge_csvs():
    nasdaq_df = pd.read_csv('csv_files/NASDAQ_stock_list.csv')
    nyse_df = pd.read_csv('csv_files/NYSE_stock_list.csv')

    # Filter NYSE tickers that are not in NASDAQ
    nyse_unique = nyse_df[~nyse_df['Code'].isin(nasdaq_df['Code'])]

    # Concatenate NASDAQ and unique NYSE tickers
    combined_df = pd.concat([nasdaq_df, nyse_unique], ignore_index=True)

    # Save the merged DataFrame to a CSV file
    combined_df.to_csv('csv_files/stock_list.csv', index=False)

    return combined_df

def retrieve_nasdaq():
    url = f'https://eodhd.com/api/exchange-symbol-list/NASDAQ?api_token=65f5adcce659e6.99552900&fmt=json'
    data = requests.get(url).json()
    print(data)
    data = pd.DataFrame(data)
    data.to_csv('csv_files/NASDAQ_stock_list.csv')
    return data

def retrieve_nyse():
    url = f'https://eodhd.com/api/exchange-symbol-list/NYSE?api_token=65f5adcce659e6.99552900&fmt=json'
    data = requests.get(url).json()
    print(data)
    data = pd.DataFrame(data)
    data.to_csv('csv_files/NYSE_stock_list.csv')
    return data
    
   
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
    df = pd.DataFrame(data)
    df.to_csv(f'csv_files/news/{ticker}_sentiment.csv')
    return "Processed " + ticker

def process_ticker(ticker):
    try:
        return access_rate_limited_api(ticker)
    except Exception as e:
        return f"Error processing {ticker}: {e}"

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