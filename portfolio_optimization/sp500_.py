import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import yfinance as yf 
from scipy.optimize import minimize
from alpha_vantage.timeseries import TimeSeries
import requests

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

url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=META&outputsize=full&apikey=Y2FE60C8AV6LY2QH'
r = requests.get(url)
data = r.json()

# function to tidy up the json to dataframe 
def tidy_json(stock_json) :
    # Extract the time series data
    time_series_data = stock_json['Time Series (Daily)']

    # Create a DataFrame from the time series data
    df = pd.DataFrame.from_dict(time_series_data, orient='index')

    # Rename the columns for better readability
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Reset the index to have the date as a column
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Date'}, inplace=True)
    
    return df