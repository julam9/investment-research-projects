import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import yfinance as yf 
from scipy.optimize import minimize
from alpha_vantage.timeseries import TimeSeries
import requests
import os 
## Using MAANG (Meta, Amazon, Apple, Netflix, and Google) Stocks
#tickers = ['META', 'AMZN', 'AAPL', 'NFLX', 'GOOGL']

## Take the data from 2010-2019, then calculate return, then calculate mean return, then rank the mean return descendingly
#stock_data = yf.download('AMZN', period='1y', interval='1d') #["Adj Close"]      
#print(stock_data)
# calculate daily return
#sp500_daily_returns = stock_data.pct_change().dropna()
#print(sp500_daily_returns)

#stock = yf.Ticker("NFLX")
#df = stock.history(period="1mo")
#print(df)

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

# extract api key
alpha_key = os.getenv('API_KEY')
meta_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=META&outputsize=full&apikey={alpha_key}'
amazon_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AMZN&outputsize=full&apikey={alpha_key}'
apple_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&outputsize=full&apikey={alpha_key}'
netflix_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=NFLX&outputsize=full&apikey={alpha_key}'
google_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=GOOGL&outputsize=full&apikey={alpha_key}'

# function to tidy up the json to dataframe 
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

# fetch 5 stocks data
df_META = stock_to_df(meta_url)
df_AMAZON = stock_to_df(amazon_url)
df_APPLE = stock_to_df(apple_url)
df_NETFLIX = stock_to_df(netflix_url)
df_GOOGLE = stock_to_df(google_url)

print(df_META.tail())
print(df_AMAZON.tail())
print(df_APPLE.tail())
print(df_NETFLIX.tail())
print(df_GOOGLE.tail())