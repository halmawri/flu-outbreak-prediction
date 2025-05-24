import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import statsmodels, but continue if not available
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    print("Warning: statsmodels package not installed. Some time series analyses will be skipped.")
    print("To install statsmodels, run: pip install statsmodels")
    STATSMODELS_AVAILABLE = False

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
colors = plt.cm.tab10.colors

# Create output directories if they don't exist
os.makedirs('./reports', exist_ok=True)
os.makedirs('./results', exist_ok=True)

print("========== Exploratory Data Analysis Report ==========")

# Load the dataset
print("\nLoading and preprocessing the data...")
df = pd.read_csv('./data/flu_trends_merged_data.csv')

# Convert date columns to datetime
df['date'] = pd.to_datetime(df['date'])
df['trends_date'] = pd.to_datetime(df['trends_date'])

# Sort by date
df = df.sort_values('date')

# Add some useful columns for analysis
df['month'] = df['date'].dt.month
df['month_name'] = df['date'].dt.strftime('%b')  # Month abbreviation
df['season'] = df['month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else 
                                          'Spring' if x in [3, 4, 5] else 
                                          'Summer' if x in [6, 7, 8] else 'Fall')

# Create a report for a comprehensive EDA
print("1. Dataset Overview")
print(f"   Shape: {df.shape}")
print(f"   Time period: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
print(f"   Years covered: {df['year'].min()} to {df['year'].max()}")

print("\n2. Summary Statistics")
numerical_columns = df.select_dtypes(include=[np.number]).columns
print(df[numerical_columns].describe().round(2).to_string())

# Check for missing values
missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    print("\n3. Missing Values")
    print(missing_values[missing_values > 0].to_string())
else:
    print("\n3. Missing Values: None")

# Analyze seasonal patterns of ILI percentage
print("\n4. Seasonal Patterns of ILI Percentage")
seasonal_avg = df.groupby('month')[['ili_percent']].mean().reset_index()
seasonal_avg['month_name'] = pd.to_datetime(seasonal_avg['month'], format='%m').dt.strftime('%b')
seasonal_avg = seasonal_avg.sort_values('month')
print(seasonal_avg.to_string(index=False))

# Analyze search term patterns by season
print("\n5. Seasonal Patterns of Search Terms")
search_terms = ['flu symptoms', 'fever', 'cough', 'sore throat', 'influenza']
seasonal_search = df.groupby('season')[search_terms].mean().reset_index()
print(seasonal_search.to_string(index=False))

# Correlation analysis
print("\n6. Correlation Analysis")
correlation_matrix = df[numerical_columns].corr()
print(correlation_matrix.round(2).to_string())

# Lag correlation analysis
print("\n7. Lag Correlation Analysis")
max_lag = 8
search_terms = ['flu symptoms', 'fever', 'cough', 'sore throat', 'influenza']
lag_results = pd.DataFrame(index=range(max_lag+1))

for term in search_terms:
    for lag in range(max_lag+1):
        if lag == 0:
            corr = df['ili_percent'].corr(df[term])
        else:
            # Apply lag within the correct year to avoid mixing years
            # This is a simplification - a more complex approach would handle year boundaries better
            lagged_term = df.groupby('year')[term].shift(lag)
            corr = df['ili_percent'].corr(lagged_term)
        lag_results.loc[lag, term] = corr

print(lag_results.round(3).to_string())

print("\n8. ILI Percentage Statistics by Year")
yearly_stats = df.groupby('year')['ili_percent'].agg(['mean', 'std', 'min', 'max']).reset_index()
print(yearly_stats.round(2).to_string(index=False))

# Perform time series decomposition if statsmodels is available
if STATSMODELS_AVAILABLE:
    print("\n9. Time Series Decomposition of ILI Percentage")
    # Set date as index for decomposition
    df_ts = df.set_index('date')
    # Check if we have enough data points for decomposition
    if len(df_ts) >= 24:  # Need at least 2 years of data for meaningful seasonal decomposition
        result = seasonal_decompose(df_ts['ili_percent'], model='multiplicative', period=12)
        print("   Decomposition completed successfully.")
    else:
        print("   Not enough data points for seasonal decomposition.")

    # Stationarity test
    print("\n10. Stationarity Test for ILI Percentage")
    result = adfuller(df['ili_percent'].dropna())
    print(f"   ADF Statistic: {result[0]:.3f}")
    print(f"   p-value: {result[1]:.3f}")
    print(f"   Critical Values:")
    for key, value in result[4].items():
        print(f"      {key}: {value:.3f}")
    if result[1] <= 0.05:
        print("   The series is stationary (suitable for time series modeling without differencing)")
    else:
        print("   The series is non-stationary (may require differencing for time series modeling)")
else:
    print("\n9-10. Advanced Time Series Analysis: Skipped (statsmodels not available)")

print("\n========== End of Report ==========")

print("\nGenerating visualizations...")

# 1. Time Series Plot of ILI Percentage
plt.figure(figsize=(14, 7))
plt.plot(df['date'], df['ili_percent'], 'b-', linewidth=2)
plt.title('CDC Influenza-Like Illness (ILI) Percentage Over Time', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('ILI Percentage', fontsize=14)
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./reports/1_ili_time_series.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Seasonal Pattern of ILI Percentage
plt.figure(figsize=(12, 6))
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_avg = df.groupby('month_name')['ili_percent'].mean().reindex(month_order)
plt.bar(monthly_avg.index, monthly_avg.values, color='darkblue', alpha=0.7)
plt.title('Average ILI Percentage by Month', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Average ILI Percentage', fontsize=14)
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('./reports/2_ili_seasonal_pattern.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Search Terms Seasonal Pattern
plt.figure(figsize=(14, 7))
monthly_search = df.groupby('month_name')[search_terms].mean().reindex(month_order)
for i, term in enumerate(search_terms):
    plt.plot(monthly_search.index, monthly_search[term], marker='o', linewidth=2, 
             color=colors[i], label=term)
plt.title('Average Search Volume by Month', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Average Search Volume', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('./reports/3_search_terms_seasonal.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title('Correlation Heatmap', fontsize=16)
plt.tight_layout()
plt.savefig('./reports/4_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Lag Correlation Analysis
plt.figure(figsize=(14, 7))
for i, term in enumerate(search_terms):
    plt.plot(lag_results.index, lag_results[term], marker='o', linewidth=2, 
             color=colors[i], label=term)
plt.title('Lag Analysis: Correlation between Search Terms and ILI Percentage', fontsize=16)
plt.xlabel('Lag (Months)', fontsize=14)
plt.ylabel('Correlation Coefficient', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('./reports/5_lag_correlation.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Time Series Decomposition Plot (if statsmodels is available)
if STATSMODELS_AVAILABLE and len(df_ts) >= 24:
    result = seasonal_decompose(df_ts['ili_percent'], model='multiplicative', period=12)
    fig = plt.figure(figsize=(14, 10))
    fig = result.plot()
    fig.suptitle('Time Series Decomposition of ILI Percentage', fontsize=16)
    plt.tight_layout()
    plt.savefig('./reports/6_time_series_decomposition.png', dpi=300, bbox_inches='tight')
    plt.close()

# 7. Time Series with ACF and PACF (if statsmodels is available)
if STATSMODELS_AVAILABLE:
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
    
    # Plot time series
    axes[0].plot(df['date'], df['ili_percent'], 'b-', linewidth=2)
    axes[0].set_title('ILI Percentage Time Series', fontsize=14)
    axes[0].set_xlabel('Date', fontsize=12)
    axes[0].set_ylabel('ILI Percentage', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Plot ACF
    plot_acf(df['ili_percent'].dropna(), lags=36, ax=axes[1])
    axes[1].set_title('Autocorrelation Function (ACF)', fontsize=14)
    
    # Plot PACF
    plot_pacf(df['ili_percent'].dropna(), lags=36, ax=axes[2])
    axes[2].set_title('Partial Autocorrelation Function (PACF)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('./reports/7_acf_pacf.png', dpi=300, bbox_inches='tight')
    plt.close()

# 8. Scatter Plot Matrix of Search Terms vs ILI Percentage
plt.figure(figsize=(16, 14))
sns.pairplot(df[['ili_percent'] + search_terms], height=2.5)
plt.suptitle('Scatter Plot Matrix: ILI Percentage vs Search Terms', fontsize=20, y=1.02)
plt.tight_layout()
plt.savefig('./reports/8_scatter_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 9. Box Plot by Season
plt.figure(figsize=(12, 7))
sns.boxplot(x='season', y='ili_percent', data=df, 
            order=['Winter', 'Spring', 'Summer', 'Fall'])
plt.title('ILI Percentage Distribution by Season', fontsize=16)
plt.xlabel('Season', fontsize=14)
plt.ylabel('ILI Percentage', fontsize=14)
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('./reports/9_seasonal_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

# 10. Combined Time Series Plot
fig, axs = plt.subplots(len(search_terms) + 1, 1, figsize=(14, 12), sharex=True)

# Plot ILI percentage
axs[0].plot(df['date'], df['ili_percent'], 'b-', linewidth=2)
axs[0].set_title('CDC ILI Percentage', fontsize=14)
axs[0].set_ylabel('Percentage', fontsize=12)
axs[0].grid(True, alpha=0.3)

# Plot each search term
for i, term in enumerate(search_terms):
    axs[i+1].plot(df['date'], df[term], color=colors[i], linewidth=2)
    axs[i+1].set_title(f'Google Search Volume: {term}', fontsize=14)
    axs[i+1].set_ylabel('Search Volume', fontsize=12)
    axs[i+1].grid(True, alpha=0.3)

# Set x-label for the bottom subplot
axs[-1].set_xlabel('Date', fontsize=14)
axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
axs[-1].xaxis.set_major_locator(mdates.YearLocator())
plt.xticks(rotation=45)

plt.suptitle('Flu Metrics Over Time: ILI Percentage and Search Terms', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.savefig('./reports/10_combined_time_series.png', dpi=300, bbox_inches='tight')
plt.close()

# 11. Year-over-Year Comparison
plt.figure(figsize=(14, 7))
years = sorted(df['year'].unique())
for year in years:
    year_data = df[df['year'] == year]
    plt.plot(year_data['month'], year_data['ili_percent'], 
             marker='o', linewidth=2, label=str(year))
plt.title('Year-over-Year Comparison of ILI Percentage', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('ILI Percentage', fontsize=14)
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12, title='Year')
plt.tight_layout()
plt.savefig('./reports/11_year_over_year.png', dpi=300, bbox_inches='tight')
plt.close()

# Save lag correlations to a CSV file
lag_results.to_csv('./results/lag_correlations.csv')

print("\nExploratory Data Analysis completed successfully!")
print("11 visualizations have been saved to the './reports' directory.")
print("Lag correlation results saved to './results/lag_correlations.csv'.")