# FOR TRAINING, TAKE RANDOM SUBSET OF TICKERS ACCORDING TO PERCENTAGE COMPOSITION IN STOCK MARKET

import pandas as pd # For data manipulation

import requests # For making API requests

from datetime import datetime, timedelta # For date manipulation

import concurrent.futures # For parallel processing
from ratelimit import limits, sleep_and_retry # For rate limiting
from threading import Semaphore # For concurrency control

import mysql.connector # For database connection

endpoint = 'stockaide-db.cluster-czcooo8241qk.us-east-2.rds.amazonaws.com'
username = 'admin'
password = 'mnmsstar15'
database = 'stockaide'

conn = mysql.connector.connect(user=username, password=password,
                              host=endpoint, database=database)
cursor = conn.cursor()

class Sentiment:
    ONE_MINUTE = 60
    MAX_CALLS_PER_MINUTE = 1000
    MAX_CONCURRENT_INSERTS = 1
    
    semaphore = Semaphore(MAX_CONCURRENT_INSERTS)

    def process_ticker(self):
        try:
            # Retrieve sentiment data from the API
            df = self.get_sentiment_data()
            # Fill missing dates in the dataframe
            filled_df = self.fill_missing(df)
            # Sort the dataframe by date
            filled_df = filled_df.sort_values(by=['Date'])
            # Average the sentiment data for weekends
            trading_days = self.weekend_average(filled_df)
            # Calculate the weighted average of the sentiment data
            weighted_df = self.weighted_average(trading_days, 0.7)
            # Write the data to the database
            #self.write_to_database(weighted_df)
            return weighted_df
        except Exception as e:
            print(f"Error processing {self.ticker}: {e}")
            raise e

    @sleep_and_retry
    @limits(calls=MAX_CALLS_PER_MINUTE, period=ONE_MINUTE)
    def get_sentiment_data(self):
        url = f'https://eodhd.com/api/sentiments?s={self.ticker}&from={self.start_date}&to={self.end_date}&api_token=65f5adcce659e6.99552900&fmt=json'
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame()
        for item in data.get(f"{self.ticker}.US", []):
            date = item['date']
            count = item['count']
            normalized = item['normalized']
            new_row = {'Ticker': self.ticker, 'Date': date, 'Count': count, 'Normalized': normalized}
            new_row = pd.DataFrame(new_row, index=[0])
            df = pd.concat([df, new_row], ignore_index=True)
        
        return df
        
    def generate_dates(self):
        dates = []
        current_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        finish_date = datetime.strptime(self.end_date, "%Y-%m-%d")
        while current_date <= finish_date:
            dates.append(current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=1)
        return dates
    
    def fill_missing(self, ticker_df):
        all_dates = self.generate_dates()
        for i in range(len(all_dates)):
            if all_dates[i] not in ticker_df.Date.values:
                new_row = {'Ticker': self.ticker, 'Date': all_dates[i], 'Count': 0, 'Normalized': 0}
                new_row = pd.DataFrame(new_row, index=[0])
                ticker_df = pd.concat([ticker_df, new_row], ignore_index=True)
        return ticker_df

    def weekend_average(self, df):
        # Convert 'date' column to datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Add 'weekday' column
        df['Weekday'] = df['Date'].dt.dayofweek

        # Separate the weekdays and weekends
        df_weekday = df[df['Weekday'] < 4]
        df_weekend = df[df['Weekday'] >= 4]
        
        rows_to_drop = []
        
        for i in range(0, len(df_weekend)):
            if i + 2 >= len(df_weekend) or i + 1 >= len(df_weekend):
                break
            if df_weekend.iloc[i]['Weekday'] == 4:
                sum_count = df_weekend.iloc[i]['Count'] + df_weekend.iloc[i+1]['Count'] + df_weekend.iloc[i+2]['Count']
                if sum_count != 0:
                    avg = ((df_weekend.iloc[i]['Normalized'] * df_weekend.iloc[i]['Count']) + 
                        (df_weekend.iloc[i+1]['Normalized'] * df_weekend.iloc[i+1]['Count']) + 
                        (df_weekend.iloc[i+2]['Normalized'] * df_weekend.iloc[i+2]['Count'])) / sum_count
                else:
                    avg = 0
                new_row = {'Ticker': df_weekend.iloc[i]['Ticker'], 'Date': df_weekend.iloc[i]['Date'], 'Count': sum_count, 'Normalized': avg, 'Weekday': 4}
                new_row = pd.DataFrame(new_row, index=[0])
                df_weekend = pd.concat([df_weekend, new_row], ignore_index=True)
                rows_to_drop.extend([i, i+1, i+2])
                i += 3
            else: continue
        
        df_weekend = df_weekend.drop(rows_to_drop).reset_index(drop=True)
            
        df_merged = pd.concat([df_weekday, df_weekend], ignore_index=True)
        
        # Sort the dataframe by 'ticker' and 'date'
        df_merged.sort_values(['Ticker', 'Date'], inplace=True)

        return df_merged

    def weighted_average(self, df, weight):
        df['Weighted_avg'] = (df['Normalized'] * weight) + (df['Count']  * (1 - weight))
        df.drop(['Count', 'Normalized'], axis=1, inplace=True)
        return df
    
    def write_to_database(self, df):
        with self.semaphore:
            # Convert DataFrame rows to tuples
            data_tuples = [tuple(row) for row in df.to_numpy()]
            # Execute the INSERT INTO statement for each row in the DataFrame
            cursor.executemany('''INSERT INTO sentiment_data (Ticker, Date, Weekday, Weighted_avg) VALUES (%s, %s, %s, %s)''', data_tuples)
            conn.commit()
        
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
    