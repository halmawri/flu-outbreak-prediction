import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("========== Feature Engineering for Predictive Modeling ==========")

# Load data
print("\nLoading data...")
df = pd.read_csv('./data/flu_trends_merged_data.csv')

# Convert date columns to datetime
df['date'] = pd.to_datetime(df['date'])
df['trends_date'] = pd.to_datetime(df['trends_date'])

# Sort by date
df = df.sort_values('date')

print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Step 1: Create time-based features
print("\nCreating time-based features...")

# Extract time components
df['year_numeric'] = df['year'] - df['year'].min()  # Normalized year (0, 1, 2...)
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter

# Create season indicator
df['season'] = df['month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else 
                                         'Spring' if x in [3, 4, 5] else 
                                         'Summer' if x in [6, 7, 8] else 'Fall')

# Create dummy variables for season
df_with_dummies = pd.get_dummies(df, columns=['season'], drop_first=True)

# Create cyclical features for month (to preserve circular nature)
df_with_dummies['month_sin'] = np.sin(2 * np.pi * df['month']/12)
df_with_dummies['month_cos'] = np.cos(2 * np.pi * df['month']/12)

# Create continuous time index
df_with_dummies['time_index'] = range(len(df_with_dummies))

# Step 2: Create lagged features
print("Creating lagged features for search terms...")

search_terms = ['flu symptoms', 'fever', 'cough', 'sore throat', 'influenza']

# Create lags 1-8 for each search term
for term in search_terms:
    for lag in range(1, 9):
        # Shift within each year to avoid mixing years
        df_with_dummies[f'{term}_lag{lag}'] = df_with_dummies.groupby(['year'])[term].shift(lag)

# Also add lags for the target variable (ili_percent)
for lag in range(1, 4):  # Lags 1-3 for ili_percent
    df_with_dummies[f'ili_percent_lag{lag}'] = df_with_dummies.groupby(['year'])['ili_percent'].shift(lag)

# Step 3: Create interaction features
print("Creating interaction features...")

# Interaction between search terms and seasonal indicators
for term in search_terms:
    # Interaction with month (seasonal effects)
    df_with_dummies[f'{term}_X_month_sin'] = df_with_dummies[term] * df_with_dummies['month_sin']
    df_with_dummies[f'{term}_X_month_cos'] = df_with_dummies[term] * df_with_dummies['month_cos']

# Step 4: Create polynomial features for key search terms
print("Creating polynomial features...")

# Based on your lag analysis, 'flu symptoms' and 'influenza' appear most important
for term in ['flu symptoms', 'influenza']:
    # Create squared term for optimal lag (based on your findings)
    optimal_lag = 3  # From your lag analysis
    df_with_dummies[f'{term}_lag{optimal_lag}_squared'] = df_with_dummies[f'{term}_lag{optimal_lag}']**2

# Step 5: Create rolling window features
print("Creating rolling window features...")

# Computing rolling averages for the search terms (3-month window)
window_size = 3
for term in search_terms:
    df_with_dummies[f'{term}_rolling_avg'] = df_with_dummies.groupby(['year'])[term].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )

# Step 6: Handle missing values from feature creation
print("Handling missing values...")

# Count missing values before handling
missing_before = df_with_dummies.isnull().sum().sum()

# Fill missing values in the lagged features with appropriate values
# For the first few rows of each year that don't have lags within the year
for col in df_with_dummies.columns:
    if '_lag' in col or '_rolling' in col:
        # Fill with column mean within the same month across years
        month_means = df_with_dummies.groupby('month')[col].transform('mean')
        df_with_dummies[col].fillna(month_means, inplace=True)
        
        # If any remaining NaNs, fill with overall mean
        df_with_dummies[col].fillna(df_with_dummies[col].mean(), inplace=True)

# Count missing values after handling
missing_after = df_with_dummies.isnull().sum().sum()

print(f"Missing values before: {missing_before}")
print(f"Missing values after: {missing_after}")

# Step 7: Feature scaling
print("Scaling numerical features...")

# Identify numeric columns for scaling (excluding the target variable and date columns)
numeric_cols = df_with_dummies.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ['year', 'week', 'ili_percent']]

# Create scaled versions of the numerical features
scaler = StandardScaler()
df_with_dummies[['scaled_' + col for col in numeric_cols]] = scaler.fit_transform(df_with_dummies[numeric_cols])

# Step 8: Create train/test split markers
print("Creating train/test split markers...")

# Mark the last 20% of data as test set (by date)
split_idx = int(len(df_with_dummies) * 0.8)
df_with_dummies['data_split'] = 'train'
df_with_dummies.iloc[split_idx:, df_with_dummies.columns.get_loc('data_split')] = 'test'

# Step 9: Export the engineered dataset
print("Exporting engineered dataset...")

# Create data directory if it doesn't exist
os.makedirs('./data', exist_ok=True)

# Save the full featured dataset
df_with_dummies.to_csv('./data/flu_trends_features.csv', index=False)

# Also create a version with only the features we'll use for modeling
# (exclude original date and intermediate columns)
model_cols = ['year', 'week', 'month', 'quarter', 'season_Spring', 'season_Summer', 
              'season_Winter', 'month_sin', 'month_cos', 'time_index', 'ili_percent', 
              'data_split']

# Add all lag columns
lag_cols = [col for col in df_with_dummies.columns if '_lag' in col]
model_cols.extend(lag_cols)

# Add interaction features
interaction_cols = [col for col in df_with_dummies.columns if '_X_' in col]
model_cols.extend(interaction_cols)

# Add polynomial features
poly_cols = [col for col in df_with_dummies.columns if '_squared' in col]
model_cols.extend(poly_cols)

# Add rolling features
rolling_cols = [col for col in df_with_dummies.columns if '_rolling' in col]
model_cols.extend(rolling_cols)

# Create and save the modeling dataset
modeling_df = df_with_dummies[model_cols]
modeling_df.to_csv('./data/flu_trends_modeling.csv', index=False)

# Step 10: Generate feature information report
print("Generating feature information report...")

# Create empty lists for feature info
features = []
types = []
missing_values = []
unique_values = []
description_values = []

# Define feature descriptions
descriptions = {
    'year': 'Year of observation',
    'week': 'Week number within year',
    'month': 'Month (1-12)',
    'quarter': 'Quarter (1-4)',
    'season_Spring': 'Indicator for Spring season',
    'season_Summer': 'Indicator for Summer season',
    'season_Winter': 'Indicator for Winter season',
    'month_sin': 'Sine transform of month (cyclical feature)',
    'month_cos': 'Cosine transform of month (cyclical feature)',
    'time_index': 'Continuous time index (0, 1, 2...)',
    'ili_percent': 'Target variable - CDC ILI percentage',
    'data_split': 'Train/test split indicator'
}

# Add descriptions for lag features
for term in search_terms:
    for lag in range(1, 9):
        descriptions[f'{term}_lag{lag}'] = f'{term} search volume lagged by {lag} months'

# Add descriptions for interaction features
for term in search_terms:
    descriptions[f'{term}_X_month_sin'] = f'Interaction between {term} and month_sin'
    descriptions[f'{term}_X_month_cos'] = f'Interaction between {term} and month_cos'

# Add descriptions for polynomial features
for term in ['flu symptoms', 'influenza']:
    optimal_lag = 3
    descriptions[f'{term}_lag{optimal_lag}_squared'] = f'Squared value of {term} lagged by {optimal_lag} months'

# Add descriptions for rolling features
for term in search_terms:
    descriptions[f'{term}_rolling_avg'] = f'3-month rolling average of {term} search volume'

# Add descriptions for ili lags
for lag in range(1, 4):
    descriptions[f'ili_percent_lag{lag}'] = f'ILI percentage lagged by {lag} months'

# Gather feature info iteratively
for col in modeling_df.columns:
    features.append(col)
    types.append(str(modeling_df[col].dtypes))  # Using correct attribute
    missing_values.append(modeling_df[col].isnull().sum())
    unique_values.append(modeling_df[col].nunique())
    description_values.append(descriptions.get(col, 'N/A'))

# Create DataFrame from lists
feature_info = pd.DataFrame({
    'Feature': features,
    'Type': types,
    'Missing_Values': missing_values,
    'Unique_Values': unique_values,
    'Description': description_values
})

# Save feature info to CSV
feature_info.to_csv('./data/feature_info.csv', index=False)

# Create directories for results if they don't exist
os.makedirs('./results', exist_ok=True)

# Write a more detailed markdown report
with open('./results/feature_engineering_report.md', 'w') as f:
    f.write("# Feature Engineering Report\n\n")
    
    f.write("## Overview\n\n")
    f.write(f"This report summarizes the feature engineering process for the flu outbreak prediction model.\n")
    f.write(f"Original dataset: {df.shape[0]} rows, {df.shape[1]} columns\n")
    f.write(f"Engineered dataset: {modeling_df.shape[0]} rows, {modeling_df.shape[1]} columns\n\n")
    
    f.write("## Feature Categories\n\n")
    
    f.write("### Time-Based Features\n\n")
    time_features = ['year', 'month', 'quarter', 'season_Spring', 'season_Summer', 
                     'season_Winter', 'month_sin', 'month_cos', 'time_index']
    f.write("These features capture temporal patterns and seasonality:\n\n")
    for feature in time_features:
        if feature in descriptions:
            f.write(f"- **{feature}**: {descriptions[feature]}\n")
    
    f.write("\n### Lagged Search Terms\n\n")
    f.write("These features capture the delayed effect of search behavior on flu outbreaks:\n\n")
    
    # Group by search term for better organization
    for term in search_terms:
        f.write(f"**{term}** lags:\n")
        for lag in range(1, 9):
            feature = f'{term}_lag{lag}'
            if feature in descriptions:
                f.write(f"- {feature}: {descriptions[feature]}\n")
        f.write("\n")
    
    f.write("\n### Interaction Features\n\n")
    f.write("These features capture how the effect of search terms may vary by season:\n\n")
    for feature in interaction_cols:
        if feature in descriptions:
            f.write(f"- **{feature}**: {descriptions[feature]}\n")
    
    f.write("\n### Polynomial Features\n\n")
    f.write("These features capture non-linear relationships:\n\n")
    for feature in poly_cols:
        if feature in descriptions:
            f.write(f"- **{feature}**: {descriptions[feature]}\n")
    
    f.write("\n### Rolling Window Features\n\n")
    f.write("These features capture recent trends in search behavior:\n\n")
    for feature in rolling_cols:
        if feature in descriptions:
            f.write(f"- **{feature}**: {descriptions[feature]}\n")
    
    f.write("\n### Target Variable Lags\n\n")
    f.write("Past values of ILI percentage can help predict future values:\n\n")
    for lag in range(1, 4):
        feature = f'ili_percent_lag{lag}'
        if feature in descriptions:
            f.write(f"- **{feature}**: {descriptions[feature]}\n")
    
    f.write("\n## Feature Engineering Steps\n\n")
    f.write("1. **Time-Based Features**: Created month, quarter, season indicators, and cyclical encodings\n")
    f.write("2. **Lagged Features**: Created 1-8 month lags for each search term\n")
    f.write("3. **Interaction Features**: Created interactions between search terms and seasonal indicators\n")
    f.write("4. **Polynomial Features**: Created squared terms for the most important search terms\n")
    f.write("5. **Rolling Window Features**: Created 3-month rolling averages\n")
    f.write("6. **Missing Value Handling**: Filled missing values with seasonal means\n")

    f.write("\n## Next Steps\n\n")
    f.write("The engineered dataset is now ready for predictive modeling. The recommended approach is to:\n\n")
    f.write("1. Use the `./data/flu_trends_modeling.csv` file for model training\n")
    f.write("2. Train multiple model types (linear regression, random forest, gradient boosting, etc.)\n")
    f.write("3. Evaluate using time-based validation to respect the temporal nature of the data\n")
    f.write("4. Select features based on importance analysis to reduce dimensionality\n")
    f.write("5. Fine-tune the best performing models\n")

print("Feature engineering complete!")
print("Engineered datasets saved to:")
print("  - ./data/flu_trends_features.csv (full dataset)")
print("  - ./data/flu_trends_modeling.csv (modeling features only)")
print("Feature information saved to:")
print("  - ./data/feature_info.csv")
print("  - ./results/feature_engineering_report.md")