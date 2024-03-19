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
MAX_THREADS = 30

API_KEY = "65f5adcce659e6.99552900"

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

class Sentiment:
    def retrieve_nasdaq(self):
        url = f'https://eodhd.com/api/exchange-symbol-list/NASDAQ?api_token=65f5adcce659e6.99552900&fmt=json'
        data = requests.get(url).json()
        print(data)
        data = pd.DataFrame(data)
        data.to_csv('csv_files/NASDAQ_stock_list.csv')
        return data
    
    def retrieve_nyse(self):
        url = f'https://eodhd.com/api/exchange-symbol-list/NYSE?api_token=65f5adcce659e6.99552900&fmt=json'
        data = requests.get(url).json()
        print(data)
        data = pd.DataFrame(data)
        data.to_csv('csv_files/NYSE_stock_list.csv')
        return data
    
    def get_biz_days_delta_date(self, start_date_str, delta_days):
        start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = start_date + (delta_days * US_BUSINESS_DAY)
        end_date_str = datetime.datetime.strftime(end_date, "%Y-%m-%d")
        return end_date_str

    def fetch_news(self, start_date_str, end_date_str, limit, offset=0):
        url = f'https://eodhistoricaldata.com/api/news?api_token={API_KEY}&s={self.symbol}&limit={limit}&offset={offset}&from={start_date_str}&to={end_date_str}'
        news_json = requests.get(url).json()
        results_df = pd.DataFrame()
        for item in news_json:
            title = item['title']
            desc = item['content']
            date = pd.to_datetime(item["date"], utc=True)
            result_df = pd.DataFrame({"title": [title], "desc": [desc], "date": [date]})
            results_df = pd.concat([results_df, result_df], axis=0, ignore_index=True)
        results_df.to_csv(f'csv_files/{self.symbol}_news.csv')
        time.sleep(0.25)
        return results_df

    def load_news(self):
        limit = 40
        today = datetime.datetime.today()
        today_date_str = today.strftime("%Y-%m-%d")
        news_df = pd.DataFrame()
        #  Get news for each day
        for i in range(0, self.past_days):
            start_day = self.past_days - i
            end_day = self.past_days - i - 1
            start_date_str = self.get_biz_days_delta_date(today_date_str, - start_day)
            end_date_str = self.get_biz_days_delta_date(today_date_str, - end_day)
            #  Fetch news
            if i % 5 == 0:
                print(f"Fetching news for day {start_day} of {self.past_days}")
            day_news_df = self.fetch_news(start_date_str, end_date_str, limit, 0)
            news_df = pd.concat([news_df, day_news_df])
            #  Throttle requests
            if i % 20 == 0:
                time.sleep(2)
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

    def __init__(self, symbol, name, past_days):
        self.symbol = symbol
        self.name = name
        self.past_days = past_days
        
def download_news(stock_df):
    threads = min(MAX_THREADS, len(stock_df))
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        executor.map(Sentiment(stock_df['Code'], stock_df['Name'], 3650).load_news, stock_df)

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

    

def main():
    t0 = time.time()
    df = merge_csvs()
    #download_news(df)
    t1 = time.time()
    print(f"Time taken: {t1 - t0} seconds")
    
    """obj = Sentiment("AAPL", "Apple Inc.", 3650)
    news_df = obj.load_news()
    news_df.fillna(0)
    results_df = obj.perform_sentiment_analysis(news_df)
    grouped_df = results_df.set_index('date').groupby(pd.Grouper(freq='D')).sum()
    print(grouped_df)"""
    
    

if __name__ == "__main__":
    main()