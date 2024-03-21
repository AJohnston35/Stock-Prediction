# TODO: CHANGE STOCK DATA RETRIEVAL TO USE EODHD API
# TODO: MERGE WITH STOCK DATA
# TODO: MERGE INTEREST RATES WITH STOCK DATA
# TODO: CREATE SINGLE DATABASE WITH STOCK DATA, SENTIMENT DATA, AND INTEREST RATES
# FOR TRAINING, TAKE RANDOM SUBSET OF TICKERS ACCORDING TO PERCENTAGE COMPOSITION IN STOCK MARKET

import pandas as pd
import requests
import concurrent.futures
from ratelimit import limits, sleep_and_retry
from datetime import datetime, timedelta
from threading import Semaphore
import mysql.connector

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
            df = self.access_rate_limited_api()
            # Fill missing dates in the dataframe
            filled_df = self.fill_missing(df)
            # Sort the dataframe by date
            filled_df = filled_df.sort_values(by=['date'])
            # Average the sentiment data for weekends
            trading_days = self.weekend_average(filled_df)
            # Calculate the weighted average of the sentiment data
            weighted_df = self.weighted_average(trading_days, 0.7)
            # Write the data to the database
            self.write_to_database(weighted_df)
            print(f"Successfully processed {self.ticker}")
        except Exception as e:
            return f"Error processing {self.ticker}: {e}"

    @sleep_and_retry
    @limits(calls=MAX_CALLS_PER_MINUTE, period=ONE_MINUTE)
    def access_rate_limited_api(self):
        print("Retrieving sentiment data...")
        url = f'https://eodhd.com/api/sentiments?s={self.ticker}&from={self.start_date}&to={self.end_date}&api_token=65f5adcce659e6.99552900&fmt=json'
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
        
    def generate_dates(self):
        dates = []
        current_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        finish_date = datetime.strptime(self.end_date, "%Y-%m-%d")
        while current_date <= finish_date:
            dates.append(current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=1)
        return dates
    
    def fill_missing(self, ticker_df):
        print("Filling missing dates...")
        all_dates = self.generate_dates()
        for i in range(len(all_dates)):
            if all_dates[i] not in ticker_df.date.values:
                new_row = {'ticker': self.ticker, 'date': all_dates[i], 'count': 0, 'normalized': 0}
                new_row = pd.DataFrame(new_row, index=[0])
                ticker_df = pd.concat([ticker_df, new_row], ignore_index=True)
        return ticker_df

    def weekend_average(self, df):
        print("Averaging weekend sentiment data...")
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

    def weighted_average(self, df, weight):
        print("Calculating weighted average...")
        df['weighted_avg'] = (df['normalized'] * weight) + (df['count']  * (1 - weight))
        df.drop(['count', 'normalized'], axis=1, inplace=True)
        return df
    
    def write_to_database(self, df):
        print("Writing to database...")
        
        with self.semaphore:
            # Convert DataFrame rows to tuples
            data_tuples = [tuple(row) for row in df.to_numpy()]
            # Execute the INSERT INTO statement for each row in the DataFrame
            cursor.executemany('''INSERT INTO sentiment_data (ticker, date, weekday, weighted_avg) VALUES (%s, %s, %s, %s)''', data_tuples)
            conn.commit()
        
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
    

def main():
    # Define the maximum number of threads
    """df = pd.read_csv('csv_files/stock_list.csv')
    tickers = df.Code.unique()"""
    
    tickers = ['AAPL','NVDA','KO']
    start_date = '2014-03-19'
    end_date = '2024-03-20'
    
    max_threads = 3  # Adjust this based on your system's capabilities
    
    create_table_query = '''
    CREATE TABLE IF NOT EXISTS sentiment_data (
        id INT AUTO_INCREMENT PRIMARY KEY,
        ticker VARCHAR(10),
        date DATE,
        weekday INT,
        weighted_avg FLOAT
    )
    '''
    cursor.execute(create_table_query)
    conn.commit()
    
    # Use ThreadPoolExecutor to create a pool of threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Submit tasks for each ticker to the executor
        futures = {executor.submit(Sentiment(ticker, start_date, end_date).process_ticker): ticker for ticker in tickers}
        
        # Iterate over completed futures and print results
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(result)  
                
    query = '''SELECT * FROM sentiment_data'''
    cursor.execute(query)
    cursor.fetchall()
    rows = cursor.fetchall()
    print(rows)
    # Close the database connection
    conn.close()
    

if __name__ == "__main__":
    main()
