import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import yfinance as yf 
from scipy.optimize import minimize
from alpha_vantage.timeseries import TimeSeries
import requests
import os 

# extract api key
alpha_key = os.getenv('API_KEY')

def stock_to_df(stock_metadata) :
    """Function to took data from alpha vantage api then change it more readable shape and dataframe"""
    r = requests.get(stock_metadata)
    stock_json = r.json()
    # Extract the time series data
    time_series_data = stock_json['Time Series (Daily)']

    # Create a DataFrame from the time series data
    df = pd.DataFrame.from_dict(time_series_data, orient='index')

    # Rename the columns for better readability
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Reset the index to have the date as a column
    df = df.reset_index()
    stock_df = df.rename(columns={'index': 'Date'})
    
    return stock_df

# url for 5 stocks
meta_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=META&outputsize=full&apikey={alpha_key}'
amazon_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AMZN&outputsize=full&apikey={alpha_key}'
apple_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&outputsize=full&apikey={alpha_key}'
netflix_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=NFLX&outputsize=full&apikey={alpha_key}'
google_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=GOOGL&outputsize=full&apikey={alpha_key}'

# fetch 5 stocks data
df_META = stock_to_df(meta_url)
df_AMZN = stock_to_df(amazon_url)
df_AAPL = stock_to_df(apple_url)
df_NFLX = stock_to_df(netflix_url)
df_GOOGL = stock_to_df(google_url)

# took the last 2 years data (2023-2024)
meta_2y = df_META.query('Date <= "2024-12-31" and Date >= "2023-01-01"')['Close']
amazon_2y = df_AMZN.query('Date <= "2024-12-31" and Date >= "2023-01-01"')['Close']
apple_2y = df_AAPL.query('Date <= "2024-12-31" and Date >= "2023-01-01"')['Close']
netflix_2y = df_NFLX.query('Date <= "2024-12-31" and Date >= "2023-01-01"')['Close']
google_2y = df_GOOGL.query('Date <= "2024-12-31" and Date >= "2023-01-01"')['Close']
# combine all stocks into one dataframe
all_stocks = pd.concat([meta_2y, amazon_2y, apple_2y, netflix_2y, google_2y], axis=1)

# calculate daily returns
stock_daily_return = all_stocks.astype('float64').pct_change().dropna()
print(stock_daily_return)

