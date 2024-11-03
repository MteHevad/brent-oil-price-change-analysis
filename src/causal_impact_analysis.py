import pymc3 as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def estimate_causal_impact(prices, event_index):
    """
    Estimates the causal impact of an event on time series data using Bayesian inference.
    
    Parameters:
        prices (pd.Series): Time series data (Brent oil prices).
        event_index (int): Index of the event in the time series data.

    Returns:
        trace: Trace object containing the MCMC results.
    """
    pre_event = prices[:event_index]
    post_event = prices[event_index:]

    with pm.Model() as model:
        # Priors for pre-event mean and standard deviation
        mu_pre = pm.Normal('mu_pre', mu=pre_event.mean(), sigma=pre_event.std())
        sigma_pre = pm.HalfNormal('sigma_pre', sigma=pre_event.std())

        # Priors for post-event mean
        mu_post = pm.Normal('mu_post', mu=pre_event.mean(), sigma=pre_event.std())
        
        # Observed data
        pre_obs = pm.Normal('pre_obs', mu=mu_pre, sigma=sigma_pre, observed=pre_event)
        post_obs = pm.Normal('post_obs', mu=mu_post, sigma=sigma_pre, observed=post_event)
        
        # Sampling
        trace = pm.sample(1000, cores=2)
    
    return trace

def plot_causal_impact(prices, trace, event_date):
    """
    Plots the time series with estimated causal impact due to the event.
    
    Parameters:
        prices (pd.Series): Time series data.
        trace: Trace object from PyMC3.
        event_date (str): Date of the event in the time series data.
    """
    post_mean = trace['mu_post'].mean()
    pre_mean = trace['mu_pre'].mean()
    
    plt.figure(figsize=(10, 5))
    plt.plot(prices.index, prices, label="Brent Oil Price")
    plt.axvline(pd.to_datetime(event_date), color='red', linestyle='--', label="Event Date")
    plt.hlines([pre_mean, post_mean], prices.index[0], prices.index[-1], colors=['blue', 'orange'], linestyles='--')
    plt.legend()
    plt.title("Causal Impact Analysis")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.show()

if __name__ == "__main__":
    # Load data and set event date
    data_path = "C:\\Users\\hp\\Desktop\\KAIM\\Week 5\\brent_oil_prices_cleaned.csv"
    df = pd.read_csv(data_path, parse_dates=['Date'])
    prices = df.set_index('Date')['Price']
    
    # Specify the event index (for example, event occurs at index 100)
    event_index = 100  # Change as appropriate
