import pandas as pd

def load_and_clean_data(filepath):
    """
    Loads and cleans the Brent oil prices data.
    
    Parameters:
        filepath (str): Path to the CSV file containing the data.

    Returns:
        pd.DataFrame: Cleaned data with sorted dates.
    """
    df = pd.read_csv(filepath, parse_dates=['Date'], dayfirst=True)
    df.dropna(inplace=True)  # Drop rows with missing values
    df.sort_values(by='Date', inplace=True)  # Sort by date
    return df

def save_cleaned_data(df, output_path):
    """
    Saves the cleaned data to a specified path.
    
    Parameters:
        df (pd.DataFrame): Cleaned DataFrame.
        output_path (str): Output path for saving the CSV file.
    """
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Set paths for local storage
    data_path = "C:\\Users\\hp\\Desktop\\KAIM\\Week 5\\BrentOilPrices.csv"
    output_path = "C:\\Users\\hp\\Desktop\\KAIM\\Week 5\\brent_oil_prices_cleaned.csv"

    # Load, clean, and save the data
    df = load_and_clean_data(data_path)
    save_cleaned_data(df, output_path)
