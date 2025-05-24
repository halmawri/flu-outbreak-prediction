import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pytrends.request import TrendReq
import warnings
warnings.filterwarnings('ignore')

# Set up plotting styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)

# Function to fetch CDC ILI (Influenza-Like Illness) data directly using the Delphi Epidata API
def fetch_cdc_ili_data(start_year=2017, end_year=2024):
    """
    Fetches weekly influenza-like illness (ILI) data directly from the Delphi Epidata API.
    
    Parameters:
    -----------
    start_year : int
        The starting year for data collection.
    end_year : int
        The ending year for data collection.
        
    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the CDC ILI data.
    """
    print(f"Fetching CDC ILI data from {start_year} to {end_year} using direct API call...")
    
    # Create a list to store the data
    ili_data = []
    
    # Since fetching all data at once might fail, let's fetch year by year
    for year in range(start_year, end_year + 1):
        # Calculate start and end epiweeks for this year
        start_epiweek = int(f"{year}01")  # First week of the year
        end_epiweek = int(f"{year}53")    # Last possible week of the year (some years have 53 weeks)
        
        # Build the API URL
        api_url = "https://api.delphi.cmu.edu/epidata/fluview"
        params = {
            "regions": "national",
            "epiweeks": f"{start_epiweek}-{end_epiweek}"
        }
        
        try:
            # Make the API request
            response = requests.get(api_url, params=params)
            
            # Check if the request was successful
            if response.status_code == 200:
                data = response.json()
                
                # Check if the API returned a successful result
                if data.get('result') != 1:
                    print(f"API returned an error for year {year}: {data.get('message')}")
                    continue  # Try the next year
                
                # Process the data
                year_records = len(data.get('epidata', []))
                if year_records > 0:
                    print(f"Received {year_records} records for year {year}")
                else:
                    print(f"No data available for year {year}")
                    continue  # Try the next year
                
                for item in data.get('epidata', []):
                    epiweek = item.get('epiweek')
                    if epiweek:
                        week = int(str(epiweek)[4:])
                        
                        # Create a date from the year and week
                        try:
                            date_str = f"{year}-W{week:02d}-1"  # ISO week date format
                            date = datetime.strptime(date_str, "%Y-W%W-%w").date()
                        except ValueError:
                            # Handle weeks beyond the standard calendar
                            date = datetime(year, 12, 28).date()  # Last week of the year
                        
                        # Extract ILI percentage and total patients
                        ili_percent = item.get('wili', None)  # Weighted ILI
                        total_patients = item.get('ilitotal', None)
                        
                        ili_data.append({
                            'date': date,
                            'year': year,
                            'week': week,
                            'ili_percent': ili_percent,
                            'total_patients': total_patients
                        })
            else:
                print(f"Failed to fetch data for year {year}: HTTP {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"Error fetching CDC ILI data for year {year}: {str(e)}")
    
    # Convert the list to a DataFrame
    df = pd.DataFrame(ili_data)
    
    # Sort the DataFrame by date
    if not df.empty:
        df.sort_values('date', inplace=True)
        print(f"Successfully fetched CDC ILI data: {len(df)} records across {start_year}-{end_year}.")
        return df
    else:
        print("No CDC ILI data retrieved for any year.")
        return None

# Alternative function to fetch CDC ILI data using a different endpoint
def fetch_cdc_ili_data_alt(start_year=2017, end_year=2024):
    """
    Alternative method to fetch CDC ILI data using another Delphi API endpoint.
    
    Parameters:
    -----------
    start_year : int
        The starting year for data collection.
    end_year : int
        The ending year for data collection.
        
    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the CDC ILI data.
    """
    print(f"Fetching CDC ILI data from {start_year} to {end_year} using alternative API endpoint...")
    
    # Try using the Delphi ILINet endpoint
    api_url = "https://api.delphi.cmu.edu/epidata/ilinet"
    
    # Create a list to store the data
    ili_data = []
    
    # Fetch year by year to improve reliability
    for year in range(start_year, end_year + 1):
        # Calculate start and end epiweeks for this year
        start_epiweek = int(f"{year}01")  # First week of the year
        end_epiweek = int(f"{year}53")    # Last possible week of the year
        
        # Build the API URL and parameters
        params = {
            "regions": "nat",  # National data
            "epiweeks": f"{start_epiweek}-{end_epiweek}"
        }
        
        try:
            # Make the API request
            response = requests.get(api_url, params=params)
            
            # Check if the request was successful
            if response.status_code == 200:
                data = response.json()
                
                # Check if the API returned a successful result
                if data.get('result') != 1:
                    print(f"API returned an error for year {year}: {data.get('message')}")
                    continue  # Try the next year
                
                # Process the data
                year_records = len(data.get('epidata', []))
                if year_records > 0:
                    print(f"Received {year_records} records for year {year} (alternative method)")
                else:
                    print(f"No data available for year {year} (alternative method)")
                    continue  # Try the next year
                
                for item in data.get('epidata', []):
                    epiweek = item.get('epiweek')
                    if epiweek:
                        week = int(str(epiweek)[4:])
                        
                        # Create a date from the year and week
                        try:
                            date_str = f"{year}-W{week:02d}-1"  # ISO week date format
                            date = datetime.strptime(date_str, "%Y-W%W-%w").date()
                        except ValueError:
                            # Handle weeks beyond the standard calendar
                            date = datetime(year, 12, 28).date()  # Last week of the year
                        
                        # Extract ILI percentage and total patients
                        ili_percent = item.get('ili', None)  # ILI percentage
                        total_patients = item.get('patients', None)
                        
                        ili_data.append({
                            'date': date,
                            'year': year,
                            'week': week,
                            'ili_percent': ili_percent,
                            'total_patients': total_patients
                        })
            else:
                print(f"Failed to fetch data for year {year} (alternative method): HTTP {response.status_code}")
                
        except Exception as e:
            print(f"Error fetching CDC ILI data for year {year} (alternative method): {str(e)}")
    
    # Convert the list to a DataFrame
    df = pd.DataFrame(ili_data)
    
    # Sort the DataFrame by date
    if not df.empty:
        df.sort_values('date', inplace=True)
        print(f"Successfully fetched CDC ILI data (alternative method): {len(df)} records.")
        return df
    else:
        print("No CDC ILI data retrieved (alternative method).")
        return None

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

# Function to create a synthetic CDC ILI dataset (if all other methods fail)
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

# Function to explore and visualize data
def explore_data(df, output_file=None):
    """
    Performs basic exploratory data analysis on the merged dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The merged dataset to explore.
    output_file : str, optional
        If provided, saves the exploration results to this file.
        
    Returns:
    --------
    None
    """
    if df is None:
        print("Error: Cannot explore data because the dataset is missing.")
        return
    
    print("Performing exploratory data analysis...")
    
    # Create a summary of the data
    summary = {
        "Dataset Shape": df.shape,
        "Column Names": df.columns.tolist(),
        "Data Types": df.dtypes.to_dict(),
        "Missing Values": df.isnull().sum().to_dict(),
        "Descriptive Statistics": df.describe().to_dict()
    }
    
    # Print the summary
    print("\nDataset Summary:")
    print(f"Dataset Shape: {summary['Dataset Shape']}")
    print("\nColumn Names:")
    for col in summary["Column Names"]:
        print(f"  - {col}")
    
    print("\nMissing Values:")
    for col, count in summary["Missing Values"].items():
        print(f"  - {col}: {count}")
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Save the summary to a file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write("Dataset Summary:\n")
            f.write(f"Dataset Shape: {summary['Dataset Shape']}\n\n")
            
            f.write("Column Names:\n")
            for col in summary["Column Names"]:
                f.write(f"  - {col}\n")
            
            f.write("\nMissing Values:\n")
            for col, count in summary["Missing Values"].items():
                f.write(f"  - {col}: {count}\n")
            
            f.write("\nDescriptive Statistics:\n")
            f.write(df.describe().to_string())
            
            f.write("\n\nFirst 5 rows:\n")
            f.write(df.head().to_string())
        
        print(f"\nExploration results saved to {output_file}")

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
        df.to_csv(file_path, index=False)
        print(f"Data successfully saved to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving data to {file_path}: {str(e)}")
        return False

# Main function to orchestrate the entire process
def main():
    """
    Main function to orchestrate the entire data collection, preprocessing, and exploration process.
    """
    # Define parameters
    start_year = 2017
    end_year = 2024  # Adjust to current year
    
    # Calculate start_date and end_date for Google Trends
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    
    # Define flu-related search terms
    flu_keywords = ['flu symptoms', 'fever', 'cough', 'sore throat', 'influenza']
    
    # Create necessary directories if they don't exist
    import os
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./reports', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    
    # Step 1: Try to fetch CDC ILI data with multiple methods
    cdc_df = None
    
    # Try the direct Delphi API method first
    cdc_df = fetch_cdc_ili_data(start_year, end_year)
    
    # If the first method fails, try the alternative CDC API
    if cdc_df is None:
        print("First method failed, trying alternative CDC API method...")
        cdc_df = fetch_cdc_ili_data_alt(start_year, end_year)
    
    # If both methods fail, use synthetic data for testing
    if cdc_df is None:
        print("Both CDC API methods failed, using synthetic data for demonstration...")
        cdc_df = create_synthetic_ili_data(start_year, end_year)
    
    # Save CDC data to a separate file
    if cdc_df is not None:
        save_data(cdc_df, "./data/cdc_ili_data.csv")
    
    # Step 2: Fetch Google Trends data
    trends_df = fetch_google_trends_data(flu_keywords, start_date, end_date)
    
    # Save Google Trends data to a separate file
    if trends_df is not None:
        save_data(trends_df, "./data/google_trends_data.csv")
    
    # Step 3: Preprocess and merge the data
    merged_df = preprocess_and_merge_data(cdc_df, trends_df)
    
    # Step 4: Explore the merged data
    if merged_df is not None:
        explore_data(merged_df, "./results/data_exploration_results.txt")
        
        # Step 5: Save the merged data
        save_data(merged_df, "./data/flu_trends_merged_data.csv")
    
    return merged_df

# Execute the main function if this script is run directly
if __name__ == "__main__":
    merged_data = main()
    
    # Optional: Display correlation heatmap
    if merged_data is not None:
        plt.figure(figsize=(12, 10))
        
        # Select only numeric columns for correlation analysis
        numeric_cols = merged_data.select_dtypes(include=[np.number]).columns
        
        # Calculate the correlation matrix
        corr_matrix = merged_data[numeric_cols].corr()
        
        # Create a heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap between CDC ILI Data and Google Trends')
        plt.tight_layout()
        plt.savefig('./reports/correlation_heatmap.png')
        plt.show()
        
        # Create a time series plot of ILI percentages and search trends
        plt.figure(figsize=(14, 8))
        
        # Plot ILI percentages
        plt.subplot(2, 1, 1)
        plt.plot(merged_data['date'], merged_data['ili_percent'], 'b-', linewidth=2)
        plt.title('CDC Influenza-Like Illness (ILI) Percentage Over Time')
        plt.ylabel('ILI Percentage')
        plt.grid(True)
        
        # Plot search trends
        plt.subplot(2, 1, 2)
        search_terms = [col for col in merged_data.columns if col not in 
                       ['date', 'trends_date', 'year', 'week', 'ili_percent', 'total_patients']]
        
        for term in search_terms:
            plt.plot(merged_data['date'], merged_data[term], linewidth=2, label=term)
        
        plt.title('Google Search Trends Over Time')
        plt.xlabel('Date')
        plt.ylabel('Search Volume (Relative)')
        plt.grid(True)
        plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig('./reports/time_series_comparison.png')
        plt.show()
        
        # Create a seasonal pattern plot
        plt.figure(figsize=(12, 8))
        
        # Add month column
        merged_data['month'] = pd.DatetimeIndex(merged_data['date']).month
        
        # Calculate monthly averages
        monthly_avg = merged_data.groupby('month')[['ili_percent'] + search_terms].mean().reset_index()
        
        # Plot monthly averages
        plt.subplot(2, 1, 1)
        plt.plot(monthly_avg['month'], monthly_avg['ili_percent'], 'bo-', linewidth=2, markersize=8)
        plt.title('Average Monthly ILI Percentage')
        plt.ylabel('ILI Percentage')
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        for term in search_terms:
            plt.plot(monthly_avg['month'], monthly_avg[term], 'o-', linewidth=2, markersize=8, label=term)
        
        plt.title('Average Monthly Search Volume')
        plt.xlabel('Month')
        plt.ylabel('Search Volume (Relative)')
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.grid(True)
        plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig('./reports/seasonal_patterns.png')
        plt.show()
        
        print("\nAnalysis completed successfully!")