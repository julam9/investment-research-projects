import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import yfinance as yf 
from scipy.optimize import minimize
from alpha_vantage.timeseries import TimeSeries
import requests
import os 
import matplotlib.dates as mdates

""" FETCH DATA USING ALPHA VANTAGE API WITHIN DESIRED DATE"""
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

# # took the last 2 years data (2023-2024)
meta_2y = df_META.query('Date <= "2024-12-31" and Date >= "2023-01-01"')[['Date', 'Close']]
amazon_2y = df_AMZN.query('Date <= "2024-12-31" and Date >= "2023-01-01"')['Close']
apple_2y = df_AAPL.query('Date <= "2024-12-31" and Date >= "2023-01-01"')['Close']
netflix_2y = df_NFLX.query('Date <= "2024-12-31" and Date >= "2023-01-01"')['Close']
google_2y = df_GOOGL.query('Date <= "2024-12-31" and Date >= "2023-01-01"')['Close']

# # combine all stocks into one dataframe
all_stocks = pd.concat([meta_2y, amazon_2y, apple_2y, netflix_2y, google_2y], axis=1)

# # since api request limited per day, save the data into csv
all_stocks.to_csv('./data/maang_2y.csv', index=False)



""" TIDYING UP THE DATA """
# import data from csv
stock_data = pd.read_csv('./data/maang_2y.csv').set_index('Date')
# rename the column 
stock_data = stock_data.rename(columns={'Close':'META', 'Close.1':'AMZN', 'Close.2':'AAPL', 'Close.3':'NFLX', 'Close.4':'GOOGL'})

# calculate daily returns
stock_daily_return = stock_data.pct_change().dropna()



""" EDA """
# plot the return of five stocks
dates = pd.to_datetime(stock_daily_return.index)
plt.figure(figsize=(12, 8))
plt.plot(dates, stock_daily_return['META'], label='META')
plt.plot(dates, stock_daily_return['AMZN'], label='AMZN')
plt.plot(dates, stock_daily_return['AAPL'], label='AAPL')
plt.plot(dates, stock_daily_return['NFLX'], label='NFLX')
plt.plot(dates, stock_daily_return['GOOGL'], label='GOOGL')

# Customize ticks, titles and labels
plt.title('Daily Return of last 2 years MAANG')
plt.xlabel('Date')
plt.ylabel('Values')

# Set major ticks format
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # Major ticks every month
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Format as 'Jan 2020'

# Show legend
plt.legend(title='Columns')

# Save the plot
plt.savefig('./image/MAANG Return Chart.jpg')

# Show the plot 
plt.show()



""" SIMULATION FOR PORTFOLIO OPTIMIZATION """

# name of the stocks 
maang_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

# calculate daily returns
stock_daily_return = stock_data.pct_change().dropna()

# calculate expected return per stocks and covariance matrix (both annualized) with assumption of 252 days of trading days in a year
mean_return_annual = stock_daily_return.mean()*252
cov_matrix_annual = stock_daily_return.cov()*252

# number of stocks
num_stocks = stock_daily_return.shape[1]

# number of portfolios generated through simulations
simuls = 100000
# array for storing the results
results = np.zeros((3, simuls))

# simulation
for i in range(simuls):
    weights = np.random.random(num_stocks)
    weights /= np.sum(weights)

    portfolio_return = np.sum(weights*mean_return_annual)
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_annual, weights)))
    sharpe_ratio = portfolio_return/portfolio_stddev

    results[0, i] = portfolio_return
    results[1, i] = portfolio_stddev
    results[2, i] = sharpe_ratio

# find optimal portfolio
def optimal_portfolio(weights):
    return -np.sum(weights*mean_return_annual)/np.sqrt(np.dot(weights.T, np.dot(cov_matrix_annual, weights)))

# set the constraints
constraints = ({'type':'eq', 'fun':lambda x:np.sum(x)-1}) # sum of weights equal to 1
bounds = tuple((0, 1) for _ in range(num_stocks)) # weights can't be negative
initial_guess = np.ones(num_stocks)/num_stocks

optimal_port = minimize(optimal_portfolio, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = optimal_port.x

# optimal portfolio return and risk
optimal_port_return = np.sum(optimal_weights*mean_return_annual)
optimal_port_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix_annual, optimal_weights)))
optimal_port_sharpe = optimal_port_return/optimal_port_risk

# plot efficient frontier
plt.scatter(results[1], results[0], c=results[2], cmap="viridis", alpha=0.5)
plt.colorbar(label="Sharpe Ratio")
plt.scatter(optimal_port_risk, optimal_port_return, c="red", marker="*", s=200, label="Optimal Portfolio")
plt.xlabel("Risk (Standard Deviation)")
plt.ylabel("Expected Return")
plt.title("Efficient Frontier") 
plt.legend()
plt.savefig('./image/Efficient Frontier.jpg')

# Print Results
print("Optimal Portfolio Weights:")
for stock, weight in zip(maang_stocks, optimal_weights):
    print(f"{stock}: {weight:.2%}")

print(f"\nExpected Return: {optimal_port_return:.2%}")
print(f"Risk (Standard Deviation): {optimal_port_risk:.2%}")
print(f"Sharpe Ratio: {optimal_port_sharpe:.2f}")
