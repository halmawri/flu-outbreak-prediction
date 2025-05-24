import pandas as pd
import numpy as np
import requests
import json
import time
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pytrends.request import TrendReq
import warnings
warnings.filterwarnings('ignore')

# Set up plotting styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)

# Function to create synthetic CDC ILI data for testing
def create_synthetic_ili_data(start_year=2017, end_year=2024):
    """
    Creates a synthetic ILI dataset for testing purposes.
    
    Parameters:
    -----------
    start_year : int
        The starting year for data.
    end_year : int
        The ending year for data.
        
    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing synthetic ILI data.
    """
    print(f"Creating synthetic ILI data from {start_year} to {end_year}...")
    
    # Create date range
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='W-MON')
    
    # Create dataframe with dates
    df = pd.DataFrame({
        'date': date_range,
        'year': date_range.year,
        'week': date_range.isocalendar().week
    })
    
    # Generate synthetic ILI percentage with seasonal pattern
    # Higher in winter, lower in summer
    df['day_of_year'] = df['date'].dt.dayofyear
    df['ili_percent'] = 2 + 3 * np.sin(2 * np.pi * (df['day_of_year'] - 15) / 365)
    
    # Add some random variation
    np.random.seed(42)  # For reproducibility
    df['ili_percent'] += np.random.normal(0, 0.5, size=len(df))
    df['ili_percent'] = np.abs(df['ili_percent'])  # Ensure non-negative
    
    # Add total patients column (synthetic)
    df['total_patients'] = np.random.randint(5000, 15000, size=len(df))
    
    # Drop temporary columns
    df = df.drop(columns=['day_of_year'])
    
    print(f"Created synthetic ILI data with {len(df)} records.")
    return df

# Function to fetch Google Trends data for flu-related search terms
def fetch_google_trends_data(keywords, start_date, end_date, geo='US'):
    """
    Fetches search volume data from Google Trends for specified keywords.
    
    Parameters:
    -----------
    keywords : list
        A list of keywords to fetch data for.
    start_date : str
        The start date for the data (format: 'YYYY-MM-DD').
    end_date : str
        The end date for the data (format: 'YYYY-MM-DD').
    geo : str
        The geographical area to fetch data for (default: 'US').
        
    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the Google Trends data.
    """
    print(f"Fetching Google Trends data for {keywords} from {start_date} to {end_date}...")
    
    # Initialize pytrends
    pytrends = TrendReq(hl='en-US', tz=360)
    
    try:
        # Build the payload
        pytrends.build_payload(
            kw_list=keywords,
            cat=0,
            timeframe=f'{start_date} {end_date}',
            geo=geo
        )
        
        # Get interest over time
        trends_data = pytrends.interest_over_time()
        
        # Reset index to make date a column
        if not trends_data.empty:
            trends_data = trends_data.reset_index()
            trends_data.rename(columns={'date': 'trends_date'}, inplace=True)
            
            # Create week number and year columns
            trends_data['year'] = trends_data['trends_date'].dt.year
            trends_data['week'] = trends_data['trends_date'].dt.isocalendar().week
            
            print(f"Successfully fetched Google Trends data: {len(trends_data)} records.")
            return trends_data
        else:
            print(f"Error: No data returned from Google Trends.")
            return None
            
    except Exception as e:
        print(f"Error fetching Google Trends data: {str(e)}")
        return None

# Function to preprocess and merge CDC ILI data and Google Trends data
def preprocess_and_merge_data(cdc_df, trends_df):
    """
    Preprocesses and merges CDC ILI data with Google Trends data.
    
    Parameters:
    -----------
    cdc_df : pandas.DataFrame
        The CDC ILI data.
    trends_df : pandas.DataFrame
        The Google Trends data.
        
    Returns:
    --------
    pandas.DataFrame
        A merged DataFrame containing both CDC ILI and Google Trends data.
    """
    print("Preprocessing and merging CDC ILI and Google Trends data...")
    
    if cdc_df is None or trends_df is None:
        print("Error: Cannot merge data because one or both datasets are missing.")
        return None
    
    try:
        # Ensure all numeric columns are float
        for col in cdc_df.columns:
            if col not in ['date', 'year', 'week']:
                cdc_df[col] = pd.to_numeric(cdc_df[col], errors='coerce')
        
        # Handle missing values in CDC data
        cdc_df['ili_percent'].fillna(cdc_df['ili_percent'].mean(), inplace=True)
        cdc_df['total_patients'].fillna(cdc_df['total_patients'].mean(), inplace=True)
        
        # Create a copy of trends_df to avoid modification warnings
        trends_df_copy = trends_df.copy()
        
        # Handle missing values in Google Trends data
        for col in trends_df_copy.columns:
            if col not in ['trends_date', 'year', 'week', 'isPartial']:
                trends_df_copy[col].fillna(trends_df_copy[col].mean(), inplace=True)
        
        # Drop the isPartial column if it exists
        if 'isPartial' in trends_df_copy.columns:
            trends_df_copy.drop('isPartial', axis=1, inplace=True)
        
        # Merge the datasets on year and week
        merged_df = pd.merge(cdc_df, trends_df_copy, on=['year', 'week'], how='inner')
        
        # Sort by date
        merged_df.sort_values('date', inplace=True)
        
        print(f"Successfully merged data: {len(merged_df)} records.")
        return merged_df
    
    except Exception as e:
        print(f"Error preprocessing and merging data: {str(e)}")
        return None

# Function to save data to CSV
def save_data(df, file_path):
    """
    Saves the DataFrame to a CSV file.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to save.
    file_path : str
        The path where to save the CSV file.
        
    Returns:
    --------
    bool
        True if successful, False otherwise.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        df.to_csv(file_path, index=False)
        print(f"Data successfully saved to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving data to {file_path}: {str(e)}")
        return False

# Main function to create and save synthetic data
def main():
    # Define parameters
    start_year = 2017
    end_year = 2024
    
    # Calculate start_date and end_date for Google Trends
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    
    # Define flu-related search terms
    flu_keywords = ['flu symptoms', 'fever', 'cough', 'sore throat', 'influenza']
    
    # Create directories if they don't exist
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./reports', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    
    # Step 1: Create synthetic CDC ILI data
    cdc_df = create_synthetic_ili_data(start_year, end_year)
    
    # Save synthetic CDC data
    if cdc_df is not None:
        save_data(cdc_df, "./data/cdc_ili_data.csv")
    
    # Step 2: Fetch Google Trends data
    trends_df = fetch_google_trends_data(flu_keywords, start_date, end_date)
    
    # Save Google Trends data
    if trends_df is not None:
        save_data(trends_df, "./data/google_trends_data.csv")
    
    # Step 3: Preprocess and merge the data
    merged_df = preprocess_and_merge_data(cdc_df, trends_df)
    
    # Save merged data
    if merged_df is not None:
        save_data(merged_df, "./data/flu_trends_merged_data.csv")
        
        # Optional: Create a simple visualization to verify the data
        plt.figure(figsize=(12, 8))
        plt.plot(merged_df['date'], merged_df['ili_percent'], 'b-', label='ILI Percentage')
        plt.title('CDC Influenza-Like Illness (ILI) Percentage Over Time (Synthetic Data)')
        plt.xlabel('Date')
        plt.ylabel('ILI Percentage')
        plt.grid(True)
        plt.legend()
        plt.savefig('./reports/synthetic_ili_data.png')
        plt.close()
        
        # Create a correlation heatmap
        plt.figure(figsize=(12, 10))
        numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
        corr_matrix = merged_df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap (Synthetic ILI Data and Google Trends)')
        plt.tight_layout()
        plt.savefig('./reports/synthetic_correlation_heatmap.png')
        plt.close()
    
    print("\nData generation and preparation complete!")
    print("You can now run the main.py script with --skip-data-collection to use this data.")

if __name__ == "__main__":
    main()