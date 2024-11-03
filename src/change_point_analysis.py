import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

def detect_change_points(prices):
    """
    Detects change points in time series data using Bayesian inference.
    
    Parameters:
        prices (pd.Series): Time series data (Brent oil prices).
    
    Returns:
        trace: Trace object containing the MCMC results.
    """
    # Normalize the price data for stable modeling
    normalized_prices = (prices - prices.mean()) / prices.std()

    with pm.Model() as model:
        # Change point location (uniform prior)
        tau = pm.DiscreteUniform('tau', lower=0, upper=len(normalized_prices) - 1)
        
        # Different means before and after the change point
        mu1 = pm.Normal('mu1', mu=0, sigma=1)
        mu2 = pm.Normal('mu2', mu=0, sigma=1)
        
        # Standard deviation of observations
        sigma = pm.HalfNormal('sigma', sigma=1)
        
        # Observations
        idx = np.arange(len(normalized_prices))
        mu = pm.math.switch(tau > idx, mu1, mu2)
        obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=normalized_prices)
        
        # Sampling
        trace = pm.sample(1000, cores=2)
    
    return trace

def plot_change_points(prices, trace):
    """
    Plots the time series data with detected change points.
    
    Parameters:
        prices (pd.Series): Original time series data.
        trace: Trace object from PyMC3 MCMC sampling.
    """
    tau_posterior = trace['tau']
    tau_mean = int(np.mean(tau_posterior))
    
    plt.figure(figsize=(10, 5))
    plt.plot(prices.index, prices, label='Brent Oil Price')
    plt.axvline(prices.index[tau_mean], color='red', linestyle='--', label=f'Change Point (Day {tau_mean})')
    plt.legend()
    plt.title("Brent Oil Price with Detected Change Point")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.show()

if __name__ == "__main__":
    # Load cleaned data from local storage
    data_path = "C:\\Users\\hp\\Desktop\\KAIM\\Week 5\\brent_oil_prices_cleaned.csv"
    df = pd.read_csv(data_path, parse_dates=['Date'])
    prices = df['Price']
    
    # Detect change points
    trace = detect_change_points(prices)
    
    # Plot results
    plot_change_points(df.set_index('Date')['Price'], trace)
