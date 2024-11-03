import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import matplotlib.pyplot as plt

def arima_model(prices, order=(1, 1, 1)):
    """
    Fits an ARIMA model to the data.
    
    Parameters:
        prices (pd.Series): Time series data (Brent oil prices).
        order (tuple): The (p, d, q) order of the ARIMA model.

    Returns:
        model_fit: Fitted ARIMA model.
    """
    model = ARIMA(prices, order=order)
    model_fit = model.fit()
    return model_fit

def garch_model(prices, p=1, q=1):
    """
    Fits a GARCH model to the data.
    
    Parameters:
        prices (pd.Series): Time series data (Brent oil prices).
        p (int): Lag for ARCH terms.
        q (int): Lag for GARCH terms.

    Returns:
        model_fit: Fitted GARCH model.
    """
    model = arch_model(prices, vol='Garch', p=p, q=q)
    model_fit = model.fit()
    return model_fit

def plot_forecast(model_fit, steps=30):
    """
    Plots forecasted values from a fitted ARIMA or GARCH model.
    
    Parameters:
        model_fit: Fitted time series model (ARIMA or GARCH).
        steps (int): Number of steps to forecast.
    """
    forecast = model_fit.forecast(steps=steps)
    forecast_mean = forecast.predicted_mean

    plt.figure(figsize=(10, 5))
    plt.plot(forecast_mean, label="Forecast")
    plt.title("Brent Oil Price Forecast")
    plt.xlabel("Days")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Load cleaned data from local storage
    data_path = "C:\\Users\\hp\\Desktop\\KAIM\\Week 5\\brent_oil_prices_cleaned.csv"
    df = pd.read_csv(data_path, parse_dates=['Date'])
    prices = df['Price']

    # Fit ARIMA model and plot forecast
    arima_fit = arima_model(prices)
    plot_forecast(arima_fit)

    # Fit GARCH model and print summary
    garch_fit = garch_model(prices)
    print(garch_fit.summary())
