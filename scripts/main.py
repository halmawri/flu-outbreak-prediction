import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import the data collection functions from the data_collection.py script
try:
    # First try to import from the current directory
    from data_collection import (
        fetch_cdc_ili_data, 
        fetch_cdc_ili_data_alt,
        create_synthetic_ili_data,
        fetch_google_trends_data, 
        preprocess_and_merge_data, 
        explore_data, 
        save_data
    )
except ImportError:
    # If that fails, try to import from the scripts directory
    try:
        sys.path.append('scripts')
        from data_collection import (
            fetch_cdc_ili_data, 
            fetch_cdc_ili_data_alt,
            create_synthetic_ili_data,
            fetch_google_trends_data, 
            preprocess_and_merge_data, 
            explore_data, 
            save_data
        )
    except ImportError:
        print("Error: Could not import functions from data_collection.py")
        print("Please make sure the file exists in the current directory or in 'scripts/'")
        sys.exit(1)

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
    --------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Flu Outbreak Prediction using Google Trends and CDC Data")
    
    # Add arguments
    parser.add_argument('--start-year', type=int, default=2017, 
                        help='Starting year for data collection (default: 2017)')
    parser.add_argument('--end-year', type=int, default=2024, 
                        help='Ending year for data collection (default: 2024)')
    parser.add_argument('--skip-data-collection', action='store_true',
                        help='Skip data collection step (use existing data)')
    parser.add_argument('--skip-eda', action='store_true',
                        help='Skip exploratory data analysis step')
    parser.add_argument('--skip-modeling', action='store_true',
                        help='Skip model building step')
    
    return parser.parse_args()

def create_directory_structure():
    """
    Create the necessary directory structure for the project.
    
    Returns:
    --------
    dict
        Dictionary containing paths to different directories.
    """
    # Define directory names
    dirs = {
        'data': 'data',
        'reports': 'reports',
        'results': 'results',
        'models': 'models'
    }
    
    # Create directories if they don't exist
    for dir_name in dirs.values():
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"Created directory: {dir_name}")
    
    return dirs

def collect_and_process_data(start_year, end_year, dirs):
    """
    Collect and process data from CDC and Google Trends.
    
    Parameters:
    -----------
    start_year : int
        Starting year for data collection.
    end_year : int
        Ending year for data collection.
    dirs : dict
        Dictionary containing directory paths.
        
    Returns:
    --------
    pandas.DataFrame
        The merged dataset.
    """
    print("\n" + "="*80)
    print(f"Collecting and Processing Data ({start_year}-{end_year})")
    print("="*80)
    
    # Calculate start_date and end_date for Google Trends
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    
    # Define flu-related search terms
    flu_keywords = ['flu symptoms', 'fever', 'cough', 'sore throat', 'influenza']
    
    # Try to fetch CDC ILI data with multiple methods
    cdc_df = None
    
    # Try the direct Delphi API method first
    print("Trying primary CDC data collection method...")
    cdc_df = fetch_cdc_ili_data(start_year, end_year)
    
    # If the first method fails, try the alternative CDC API
    if cdc_df is None:
        print("First method failed, trying alternative CDC API method...")
        cdc_df = fetch_cdc_ili_data_alt(start_year, end_year)
    
    # If both methods fail, use synthetic data for testing
    if cdc_df is None:
        print("Both CDC API methods failed, using synthetic data for demonstration...")
        cdc_df = create_synthetic_ili_data(start_year, end_year)
    
    # If CDC data was fetched successfully, save it
    if cdc_df is not None:
        save_data(cdc_df, os.path.join(dirs['data'], 'cdc_ili_data.csv'))
    
    # Fetch Google Trends data
    trends_df = fetch_google_trends_data(flu_keywords, start_date, end_date)
    
    # If Google Trends data was fetched successfully, save it
    if trends_df is not None:
        save_data(trends_df, os.path.join(dirs['data'], 'google_trends_data.csv'))
    
    # Preprocess and merge the data
    merged_df = preprocess_and_merge_data(cdc_df, trends_df)
    
    # Save the merged data
    if merged_df is not None:
        save_data(merged_df, os.path.join(dirs['data'], 'flu_trends_merged_data.csv'))
    
    return merged_df

def run_exploratory_data_analysis(merged_df, dirs):
    """
    Run exploratory data analysis on the merged dataset.
    
    Parameters:
    -----------
    merged_df : pandas.DataFrame
        The merged dataset.
    dirs : dict
        Dictionary containing directory paths.
    """
    print("\n" + "="*80)
    print("Running Exploratory Data Analysis")
    print("="*80)
    
    if merged_df is None:
        print("Error: No data available for analysis.")
        return
    
    # Perform EDA and save results
    explore_data(merged_df, os.path.join(dirs['results'], 'data_exploration_results.txt'))
    
    # Create visualizations
    
    # 1. Correlation heatmap
    plt.figure(figsize=(12, 10))
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
    corr_matrix = merged_df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap between CDC ILI Data and Google Trends')
    plt.tight_layout()
    plt.savefig(os.path.join(dirs['reports'], 'correlation_heatmap.png'))
    plt.close()
    
    # 2. Time series comparison
    plt.figure(figsize=(14, 8))
    
    # Plot ILI percentages
    plt.subplot(2, 1, 1)
    plt.plot(merged_df['date'], merged_df['ili_percent'], 'b-', linewidth=2)
    plt.title('CDC Influenza-Like Illness (ILI) Percentage Over Time')
    plt.ylabel('ILI Percentage')
    plt.grid(True)
    
    # Plot search trends
    plt.subplot(2, 1, 2)
    search_terms = [col for col in merged_df.columns if col not in 
                   ['date', 'trends_date', 'year', 'week', 'ili_percent', 'total_patients']]
    
    for term in search_terms:
        plt.plot(merged_df['date'], merged_df[term], linewidth=2, label=term)
    
    plt.title('Google Search Trends Over Time')
    plt.xlabel('Date')
    plt.ylabel('Search Volume (Relative)')
    plt.grid(True)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(dirs['reports'], 'time_series_comparison.png'))
    plt.close()
    
    # 3. Seasonal patterns
    plt.figure(figsize=(12, 8))
    
    # Add month column
    merged_df['month'] = pd.DatetimeIndex(merged_df['date']).month
    
    # Calculate monthly averages
    monthly_avg = merged_df.groupby('month')[['ili_percent'] + search_terms].mean().reset_index()
    
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
    plt.savefig(os.path.join(dirs['reports'], 'seasonal_patterns.png'))
    plt.close()
    
    # 4. Lag analysis (correlation between search terms and future ILI percentages)
    max_lag = 8  # Maximum number of weeks to lag
    lag_corrs = pd.DataFrame(index=range(max_lag+1))
    
    for term in search_terms:
        for lag in range(max_lag+1):
            # Shift the search term data by 'lag' periods
            if lag == 0:
                lag_corr = merged_df['ili_percent'].corr(merged_df[term])
            else:
                lag_corr = merged_df['ili_percent'].corr(merged_df[term].shift(lag))
            lag_corrs.loc[lag, term] = lag_corr
    
    # Plot lag correlations
    plt.figure(figsize=(14, 8))
    for term in search_terms:
        plt.plot(lag_corrs.index, lag_corrs[term], marker='o', linewidth=2, label=term)
    
    plt.title('Lag Analysis: Correlation between Search Terms and ILI Percentage')
    plt.xlabel('Lag (Weeks)')
    plt.ylabel('Correlation Coefficient')
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(dirs['reports'], 'lag_analysis.png'))
    plt.close()
    
    # Save the lag correlations to a CSV file
    lag_corrs.to_csv(os.path.join(dirs['results'], 'lag_correlations.csv'))
    
    print("Exploratory data analysis completed successfully.")

def run_model_building(merged_df, dirs):
    """
    Run model building and evaluation.
    This is a placeholder for now - will be implemented in a separate module.
    
    Parameters:
    -----------
    merged_df : pandas.DataFrame
        The merged dataset.
    dirs : dict
        Dictionary containing directory paths.
    """
    print("\n" + "="*80)
    print("Model Building and Evaluation")
    print("="*80)
    
    print("This functionality will be implemented later.")
    print("For now, please refer to the model_building.py script.")

def load_existing_data(dirs):
    """
    Load existing data files if data collection is skipped.
    
    Parameters:
    -----------
    dirs : dict
        Dictionary containing directory paths.
        
    Returns:
    --------
    pandas.DataFrame
        The merged dataset.
    """
    merged_file = os.path.join(dirs['data'], 'flu_trends_merged_data.csv')
    
    if os.path.exists(merged_file):
        try:
            merged_df = pd.read_csv(merged_file)
            
            # Convert date columns to datetime
            if 'date' in merged_df.columns:
                merged_df['date'] = pd.to_datetime(merged_df['date'])
            if 'trends_date' in merged_df.columns:
                merged_df['trends_date'] = pd.to_datetime(merged_df['trends_date'])
                
            print(f"Successfully loaded existing data from {merged_file}: {len(merged_df)} records.")
            return merged_df
        except Exception as e:
            print(f"Error loading existing data from {merged_file}: {str(e)}")
            return None
    else:
        print(f"Error: Merged data file not found at {merged_file}")
        
        # Check if individual files exist
        cdc_file = os.path.join(dirs['data'], 'cdc_ili_data.csv')
        trends_file = os.path.join(dirs['data'], 'google_trends_data.csv')
        
        if os.path.exists(cdc_file) and os.path.exists(trends_file):
            try:
                # Load CDC data
                cdc_df = pd.read_csv(cdc_file)
                if 'date' in cdc_df.columns:
                    cdc_df['date'] = pd.to_datetime(cdc_df['date'])
                
                # Load Google Trends data
                trends_df = pd.read_csv(trends_file)
                if 'trends_date' in trends_df.columns:
                    trends_df['trends_date'] = pd.to_datetime(trends_df['trends_date'])
                
                # Merge the data
                merged_df = preprocess_and_merge_data(cdc_df, trends_df)
                
                if merged_df is not None:
                    # Save the merged data
                    save_data(merged_df, merged_file)
                    
                return merged_df
            except Exception as e:
                print(f"Error loading individual data files: {str(e)}")
                return None
        else:
            print("Error: No existing data files found. Please run data collection first.")
            return None

def run_workflow():
    """
    Run the complete workflow for flu outbreak prediction.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Create directory structure
    dirs = create_directory_structure()
    
    # Current timestamp for logging
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n" + "="*80)
    print(f"Flu Outbreak Prediction using Time Series Analysis and Machine Learning")
    print(f"Started at: {timestamp}")
    print("="*80)
    
    # Step 1: Collect and process data (or load existing data)
    if not args.skip_data_collection:
        merged_df = collect_and_process_data(args.start_year, args.end_year, dirs)
        if merged_df is None:
            print("Error: Data collection and processing failed. Exiting.")
            return
    else:
        print("\n" + "="*80)
        print("Skipping data collection (using existing data)")
        print("="*80)
        
        merged_df = load_existing_data(dirs)
        if merged_df is None:
            print("Error: Failed to load existing data. Exiting.")
            return
    
    # Step 2: Run exploratory data analysis
    if not args.skip_eda:
        # Import required libraries for EDA
        import seaborn as sns
        run_exploratory_data_analysis(merged_df, dirs)
    else:
        print("\n" + "="*80)
        print("Skipping exploratory data analysis")
        print("="*80)
    
    # Step 3: Build and evaluate models
    if not args.skip_modeling:
        run_model_building(merged_df, dirs)
    else:
        print("\n" + "="*80)
        print("Skipping model building and evaluation")
        print("="*80)
    
    # Final timestamp
    end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n" + "="*80)
    print(f"Workflow completed successfully!")
    print(f"Started at: {timestamp}")
    print(f"Completed at: {end_timestamp}")
    print("="*80)

if __name__ == "__main__":
    # Add support for optional import of seaborn in case it's not installed
    try:
        import seaborn as sns
    except ImportError:
        print("Warning: seaborn package not found. Some visualizations may not work.")
        print("Install seaborn with: pip install seaborn")
    
    run_workflow()