# Flu Outbreak Prediction using Time Series Analysis and Machine Learning

> **GMU MDS Capstone Project**  
> **Author:** Hakim Almaweri  
> **Student ID:** 21144453  
> **Date:** May 2025

## Project Overview

This capstone project aims to predict influenza outbreaks using CDC surveillance data and Google Trends search patterns through advanced time series analysis and machine learning techniques. The goal is to provide early detection of flu activity 2-3 months before traditional surveillance methods, helping public health officials respond more effectively to outbreaks.

### Problem Statement

Seasonal influenza continues to be a major health concern worldwide, with 3-5 million severe cases reported annually according to the World Health Organization. Traditional flu surveillance methods such as the CDC's ILINet rely on data from healthcare providers and laboratories, which can be delayed by 1-2 weeks. This delay can have serious consequences, as demonstrated during the 2017-2018 flu season when late detection contributed to 61,000 deaths in the U.S. alone.

This project leverages the fact that people often search for symptoms online before visiting a doctor, creating a potential early warning signal that can be detected through analysis of search trends.

## Project Structure

```
flu-prediction/
├── data/                           # Data storage directory
│   ├── cdc_ili_data.csv            # CDC influenza-like illness data
│   ├── google_trends_data.csv      # Google Trends search data
│   ├── flu_trends_merged_data.csv  # Initial merged dataset
│   ├── flu_trends_features.csv     # Feature engineered dataset
│   ├── flu_trends_modeling.csv     # Final modeling dataset
│   └── feature_info.csv            # Feature documentation
├── reports/                        # Visualization outputs
│   ├── models/                     # Model comparison visualizations
│   ├── validation/                 # Model validation charts
│   ├── correlation_heatmap.png     # Data correlation analysis
│   ├── time_series_comparison.png  # Time series visualizations
│   ├── seasonal_patterns.png       # Seasonal trend analysis
│   └── lag_analysis.png            # Lag correlation analysis
├── results/                        # Analysis results and reports
│   ├── data_exploration_results.txt # EDA summary
│   ├── feature_engineering_report.md # Feature engineering documentation
│   ├── model_comparison_results.csv # Model performance comparison
│   ├── cross_validation_results.csv # CV validation results
│   ├── model_validation_report.md  # Comprehensive validation analysis
│   └── lag_correlations.csv        # Lag analysis results
├── models/                         # Trained model files
│   ├── xgboost_model.pkl           # Best performing model
│   ├── random_forest_model.pkl     # Random Forest model
│   ├── linear_regression_model.pkl # Linear models
│   └── feature_scaler.pkl          # Feature scaling transformation
├── scripts/                        # Analysis scripts
│   ├── data_collection.py          # Data collection and preprocessing
│   ├── exploratory_data_analysis.py # EDA and visualization
│   ├── feature_engineering.py      # Feature creation and processing
│   ├── model_development.py        # Model training and comparison
│   ├── model_validation.py         # Model validation and optimization
│   └── main.py                     # Main orchestration script
├── requirements.txt                # Required Python packages
├── setup.sh                        # Linux/Mac setup script
├── setup.bat                       # Windows setup script
└── README.md                       # This file
```

## Key Features

### Data Pipeline
- **Automated Data Collection**: Fetches and combines data from CDC surveillance and Google Trends
- **Robust Feature Engineering**: Creates 150+ features including temporal, lagged, and interaction features
- **Data Quality Assurance**: Comprehensive cleaning and validation procedures

### Analytical Capabilities
- **Exploratory Data Analysis**: Deep insights into seasonal patterns and correlations
- **Time Series Analysis**: ARIMA/SARIMA modeling with seasonal decomposition
- **Machine Learning**: Multiple algorithms including XGBoost, Random Forest, LSTM
- **Model Validation**: Cross-validation, SHAP analysis, and seasonal performance testing

### Early Warning System
- **2-3 Month Lead Time**: Predictions available well before traditional surveillance
- **Seasonal Intelligence**: Models adapted for different flu seasons
- **Interpretable Results**: SHAP analysis explains prediction drivers

## Methodology

This project follows a systematic approach based on the capstone proposal:

### Step 1: Data Collection
- Synthetic CDC ILI (Influenza-Like Illness) data with realistic seasonal patterns
- Real Google Trends data for flu-related search terms
- Data covering 2017-2024 period for comprehensive analysis

### Step 2: Exploratory Data Analysis
- Seasonal pattern identification
- Correlation analysis between search terms and flu activity
- Lag analysis determining optimal prediction timeframes

### Step 3: Feature Engineering
- Time-based features (seasonality, trends)
- Lagged search terms (1-8 months)
- Interaction and polynomial features
- Rolling window statistics

### Step 4: Model Development
- Linear models (Regression, Ridge, Lasso)
- Tree-based models (Random Forest, XGBoost)
- Neural networks (LSTM)
- Ensemble methods

### Step 5: Model Validation
- Time series cross-validation
- Seasonal performance analysis
- SHAP interpretability analysis
- Comprehensive metric evaluation

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Git for cloning the repository

### Quick Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/flu-outbreak-prediction.git
   cd flu-outbreak-prediction
   ```

2. **Run setup script:**
   
   **Linux/Mac:**
   ```bash
   bash setup.sh
   ```
   
   **Windows:**
   ```cmd
   setup.bat
   ```

3. **Or manual installation:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Required Packages
```
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
pytrends>=4.8.0
scikit-learn>=1.0.0
statsmodels>=0.13.0
xgboost>=1.7.0
tensorflow>=2.8.0
shap>=0.41.0
requests>=2.27.0
```

## Usage

### Complete Workflow

Run the entire analysis pipeline:

```bash
python scripts/main.py
```

### Individual Components

**1. Data Collection:**
```bash
python scripts/data_collection.py
```

**2. Exploratory Data Analysis:**
```bash
python scripts/exploratory_data_analysis.py
```

**3. Feature Engineering:**
```bash
python scripts/feature_engineering.py
```

**4. Model Development:**
```bash
python scripts/model_development.py
```

**5. Model Validation:**
```bash
python scripts/model_validation.py
```

### Main Script Options

```bash
python scripts/main.py --help

Options:
  --start-year YEAR     Starting year for data collection (default: 2017)
  --end-year YEAR       Ending year for data collection (default: 2024)
  --skip-data-collection Skip data collection step
  --skip-eda            Skip exploratory data analysis
  --skip-modeling       Skip model building step
```

## Key Results

### Model Performance
- **Best Model**: XGBoost with engineered features
- **Prediction Accuracy**: MAE < 0.5 percentage points
- **Lead Time**: 2-3 months before traditional surveillance
- **Cross-Validation**: Robust performance across time periods

### Key Findings
1. **Strong Predictive Signal**: Google search terms like "flu symptoms" and "influenza" show strong correlation with future flu activity
2. **Optimal Lag Structure**: 3-month lag provides best prediction accuracy
3. **Seasonal Patterns**: Model performance varies by season, reflecting flu epidemiology
4. **Feature Importance**: Lagged search terms consistently rank as top predictors

### Visualization Highlights
- Time series comparison showing search trends leading ILI percentages
- Seasonal pattern analysis revealing flu outbreak timing
- Cross-validation results demonstrating model robustness
- SHAP analysis explaining prediction factors

## Data Sources

1. **CDC Surveillance Data**: 
   - Influenza-like illness percentages (synthetic data modeled after CDC FluView)
   - Weekly granularity with seasonal patterns

2. **Google Trends API**: 
   - Search volume for flu-related terms
   - Real data covering multiple flu seasons
   - Terms: "flu symptoms", "fever", "cough", "sore throat", "influenza"

## Model Architecture

### Ensemble Approach
- Multiple algorithms trained and compared
- Best performers combined for robust predictions
- Weighted averaging based on validation performance

### Feature Engineering Pipeline
- 150+ engineered features
- Temporal features capturing seasonality
- Lagged features for predictive relationships
- Interaction terms for complex relationships

### Validation Framework
- Time series cross-validation
- Multiple evaluation metrics (MAE, RMSE, R², MAPE)
- Seasonal performance analysis
- SHAP-based interpretability

## Project Impact

### Public Health Applications
- **Early Warning System**: 2-3 month lead time for intervention planning
- **Resource Allocation**: Better hospital staffing and supply preparation
- **Vaccination Campaigns**: Optimized timing and targeting
- **Public Awareness**: Proactive communication strategies

### Technical Contributions
- Comprehensive feature engineering for flu prediction
- Ensemble modeling approach combining multiple algorithms
- Seasonal-aware validation methodology
- Interpretable AI through SHAP analysis

## Future Enhancements

### Data Expansion
- Real-time CDC data integration
- Additional search terms and sources
- Regional and state-level predictions
- Social media sentiment analysis

### Model Improvements
- Deep learning architectures (Transformer models)
- Multi-output predictions (severity, duration)
- Uncertainty quantification
- Automated model retraining

### Operational Features
- Real-time dashboard
- Automated alerting system
- API for external integrations
- Mobile application for health officials

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{almaweri2025flu,
  title={Predicting Infectious Disease Outbreaks Using Time Series Analysis and Machine Learning},
  author={Almaweri, Hakim},
  year={2025},
  school={GMU- Guglielmo Marconi University },
  type={Master's Capstone Project}
}
```

## Acknowledgments

- **GMU- Guglielmo Marconi University** for academic support and resources
- **CDC** for providing the conceptual framework for flu surveillance
- **Google Trends** for making search data available for research
- **Open Source Community** for the tools and libraries that made this project possible

## Contact

**Hakim Almaweri**  
Email: hakimma3001@gmail.com  
LinkedIn: [Connect with me](linkedin.com/in/hakim-almaweri-7239963)  
GitHub: [Project Repository](https://github.com/halmawri/flu-outbreak-prediction)

---

## Quick Start Guide

For those who want to quickly explore the project:

1. **Setup**: Run `setup.sh` (Linux/Mac) or `setup.bat` (Windows)
2. **Generate Data**: `python scripts/data_collection.py`
3. **Run Analysis**: `python scripts/main.py --skip-data-collection`
4. **View Results**: Check `./results/` and `./reports/` directories

The entire pipeline takes approximately 10-15 minutes to complete on a modern computer.

---

*This project demonstrates the power of combining traditional epidemiological data with digital surveillance methods to create more responsive public health systems.*
