# Feature Engineering Report

## Overview

This report summarizes the feature engineering process for the flu outbreak prediction model.
Original dataset: 96 rows, 15 columns
Engineered dataset: 96 rows, 136 columns

## Feature Categories

### Time-Based Features

These features capture temporal patterns and seasonality:

- **year**: Year of observation
- **month**: Month (1-12)
- **quarter**: Quarter (1-4)
- **season_Spring**: Indicator for Spring season
- **season_Summer**: Indicator for Summer season
- **season_Winter**: Indicator for Winter season
- **month_sin**: Sine transform of month (cyclical feature)
- **month_cos**: Cosine transform of month (cyclical feature)
- **time_index**: Continuous time index (0, 1, 2...)

### Lagged Search Terms

These features capture the delayed effect of search behavior on flu outbreaks:

**flu symptoms** lags:
- flu symptoms_lag1: flu symptoms search volume lagged by 1 months
- flu symptoms_lag2: flu symptoms search volume lagged by 2 months
- flu symptoms_lag3: flu symptoms search volume lagged by 3 months
- flu symptoms_lag4: flu symptoms search volume lagged by 4 months
- flu symptoms_lag5: flu symptoms search volume lagged by 5 months
- flu symptoms_lag6: flu symptoms search volume lagged by 6 months
- flu symptoms_lag7: flu symptoms search volume lagged by 7 months
- flu symptoms_lag8: flu symptoms search volume lagged by 8 months

**fever** lags:
- fever_lag1: fever search volume lagged by 1 months
- fever_lag2: fever search volume lagged by 2 months
- fever_lag3: fever search volume lagged by 3 months
- fever_lag4: fever search volume lagged by 4 months
- fever_lag5: fever search volume lagged by 5 months
- fever_lag6: fever search volume lagged by 6 months
- fever_lag7: fever search volume lagged by 7 months
- fever_lag8: fever search volume lagged by 8 months

**cough** lags:
- cough_lag1: cough search volume lagged by 1 months
- cough_lag2: cough search volume lagged by 2 months
- cough_lag3: cough search volume lagged by 3 months
- cough_lag4: cough search volume lagged by 4 months
- cough_lag5: cough search volume lagged by 5 months
- cough_lag6: cough search volume lagged by 6 months
- cough_lag7: cough search volume lagged by 7 months
- cough_lag8: cough search volume lagged by 8 months

**sore throat** lags:
- sore throat_lag1: sore throat search volume lagged by 1 months
- sore throat_lag2: sore throat search volume lagged by 2 months
- sore throat_lag3: sore throat search volume lagged by 3 months
- sore throat_lag4: sore throat search volume lagged by 4 months
- sore throat_lag5: sore throat search volume lagged by 5 months
- sore throat_lag6: sore throat search volume lagged by 6 months
- sore throat_lag7: sore throat search volume lagged by 7 months
- sore throat_lag8: sore throat search volume lagged by 8 months

**influenza** lags:
- influenza_lag1: influenza search volume lagged by 1 months
- influenza_lag2: influenza search volume lagged by 2 months
- influenza_lag3: influenza search volume lagged by 3 months
- influenza_lag4: influenza search volume lagged by 4 months
- influenza_lag5: influenza search volume lagged by 5 months
- influenza_lag6: influenza search volume lagged by 6 months
- influenza_lag7: influenza search volume lagged by 7 months
- influenza_lag8: influenza search volume lagged by 8 months


### Interaction Features

These features capture how the effect of search terms may vary by season:

- **flu symptoms_X_month_sin**: Interaction between flu symptoms and month_sin
- **flu symptoms_X_month_cos**: Interaction between flu symptoms and month_cos
- **fever_X_month_sin**: Interaction between fever and month_sin
- **fever_X_month_cos**: Interaction between fever and month_cos
- **cough_X_month_sin**: Interaction between cough and month_sin
- **cough_X_month_cos**: Interaction between cough and month_cos
- **sore throat_X_month_sin**: Interaction between sore throat and month_sin
- **sore throat_X_month_cos**: Interaction between sore throat and month_cos
- **influenza_X_month_sin**: Interaction between influenza and month_sin
- **influenza_X_month_cos**: Interaction between influenza and month_cos

### Polynomial Features

These features capture non-linear relationships:

- **flu symptoms_lag3_squared**: Squared value of flu symptoms lagged by 3 months
- **influenza_lag3_squared**: Squared value of influenza lagged by 3 months

### Rolling Window Features

These features capture recent trends in search behavior:

- **flu symptoms_rolling_avg**: 3-month rolling average of flu symptoms search volume
- **fever_rolling_avg**: 3-month rolling average of fever search volume
- **cough_rolling_avg**: 3-month rolling average of cough search volume
- **sore throat_rolling_avg**: 3-month rolling average of sore throat search volume
- **influenza_rolling_avg**: 3-month rolling average of influenza search volume

### Target Variable Lags

Past values of ILI percentage can help predict future values:

- **ili_percent_lag1**: ILI percentage lagged by 1 months
- **ili_percent_lag2**: ILI percentage lagged by 2 months
- **ili_percent_lag3**: ILI percentage lagged by 3 months

## Feature Engineering Steps

1. **Time-Based Features**: Created month, quarter, season indicators, and cyclical encodings
2. **Lagged Features**: Created 1-8 month lags for each search term
3. **Interaction Features**: Created interactions between search terms and seasonal indicators
4. **Polynomial Features**: Created squared terms for the most important search terms
5. **Rolling Window Features**: Created 3-month rolling averages
6. **Missing Value Handling**: Filled missing values with seasonal means

## Next Steps

The engineered dataset is now ready for predictive modeling. The recommended approach is to:

1. Use the `./data/flu_trends_modeling.csv` file for model training
2. Train multiple model types (linear regression, random forest, gradient boosting, etc.)
3. Evaluate using time-based validation to respect the temporal nature of the data
4. Select features based on importance analysis to reduce dimensionality
5. Fine-tune the best performing models
