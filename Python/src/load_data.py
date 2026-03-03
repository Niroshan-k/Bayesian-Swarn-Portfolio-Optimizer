import yfinance as yf
import pandas as pd
import numpy as np

def get_market_statistics(tickers, period="1y"):
    data = yf.download(tickers, period=period)['Close']
    
    # 2. Calculate daily percentage change
    # This turns $150, $155 into 0.033 (3.3%)
    returns = data.pct_change().dropna()
    
    # expected_returns list
    expected_returns = returns.mean() * 252 # (252 trading days in a year)
    
    # covariance_matrix' (list of lists)
    cov_matrix = returns.cov() * 252
    
    return expected_returns, cov_matrix, returns