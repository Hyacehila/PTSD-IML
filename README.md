# PTSD-IML: Interpretable Machine Learning Study on PTSD Influencing Factors

## Project Overview

This project is a comprehensive interpretable machine learning study on the influencing factors of PTSD (Post-Traumatic Stress Disorder). By analyzing clinical and demographic characteristics of rescue workers, we build random forest models to predict PCL (PTSD Checklist) total scores and apply various interpretable machine learning methods to deeply analyze PTSD influencing factors and their interactions.

## Core Features

- **Complete Interpretable Analysis**: Integrates multiple IML methods including Permutation Feature Importance, SHAP values, Friedman's H-statistic, Accumulated Local Effects (ALE) plots, and Partial Dependence Plots (PDP)
- **Dual Implementation Paths**: Provides complete implementations in both R and Python to support different research needs
- **Modular Script Design**: Separates data preprocessing, model training, and interpretability analysis for easy maintenance and extension
- **Comprehensive Factor Analysis**: Covers key factors including ASD scores, psychological resilience, disaster scene exposure, age, etc.

## Project Structure

```
PTSD-IML/
├── data/                           # Data files directory
│   ├── PTSD.xlsx                  # Original Excel dataset
│   ├── data.csv                   # Preprocessed CSV data (Python)
│   └── data.pkl                   # Serialized data file (Python)
├── models/                        # Model files directory
│   └── final_random_forest_model.pkl  # Random forest model (Python)
├── scripts/                       # R analysis scripts directory
│   ├── data_preprocessing.R        # Data preprocessing script
│   ├── model_training.R            # Random forest model training script
│   ├── interpretability_analysis.R # Complete interpretability analysis script
│   ├── iml_analysis.R              # Analysis script based on IML textbook
│   └── main_pipeline.R             # Complete pipeline execution script
├── notebooks/                     # Jupyter analysis notebooks
│   ├── data_pre.ipynb            # Python data preprocessing and feature engineering
│   └── prediction.ipynb          # Python model training, prediction and SHAP analysis
├── results/                       # Analysis results directory (generated)
│   ├── *.png                      # Various analysis charts
│   ├── *.csv                      # Analysis data results
│   └── *.RData                    # R result objects
├── README.md                      # Project documentation
├── LICENSE                        # Open source license
└── .gitignore                     # Git version control ignore file
```

## Quick Start

### R Language Analysis Pipeline

#### 1. Complete Pipeline Execution (Recommended)
```r
# Run the complete analysis pipeline
source("scripts/main_pipeline.R")
```

#### 2. Step-by-Step Execution
```r
# Data preprocessing
source("scripts/data_preprocessing.R")

# Model training
source("scripts/model_training.R")

# Interpretability analysis
source("scripts/interpretability_analysis.R")

# Or run analysis based on IML textbook
source("scripts/iml_analysis.R")
```

### Python Analysis Pipeline

```python
# Using Jupyter notebooks
# 1. Run notebooks/data_pre.ipynb for data preprocessing
# 2. Run notebooks/prediction.ipynb for model training and analysis
```

## Requirements

### R Language Environment

```r
# Install required R packages
install.packages(c(
  "readxl", "ggplot2", "corrplot", "reshape2", "DescTools", 
  "forecast", "pander", "knitr", "mlr3", "mlr3learners", 
  "mlr3pipelines", "iml", "mlr3extralearners", "ggcorrplot"
))
```

### Python Environment

```bash
# Install required Python packages
pip install pandas numpy scikit-learn matplotlib seaborn shap jupyter openpyxl joblib scipy
```

## Data Description

### Original Dataset
- **File**: `data/PTSD.xlsx`
- **Content**: Clinical assessment and demographic data of rescue workers
- **Main Features**:
  - Demographic information: age, gender, education level, marital status, etc.
  - Clinical assessments: ASD total score, PCL total score, psychological resilience, etc.
  - Disaster exposure: disaster scene exposure, witness to casualties, etc.
  - Health history: psychiatric history, medication use, smoking and alcohol consumption, etc.

### Data Preprocessing
1. **Feature Engineering**: Remove low-correlation features (height, weight, BMI)
2. **Variable Selection**: Remove redundant features (ASD subscale scores, ASD properties)
3. **Data Type Conversion**: Convert categorical variables to factors, correctly order ordinal variables
4. **Missing Value Handling**: Check and handle missing data
5. **Column Name Standardization**: R version uses English column names, Python version uses Chinese to English translation

## Main Analysis Methods

### 1. Feature Importance Analysis
- **Permutation Feature Importance**: Evaluate the contribution of each feature to model prediction
- **SHAP Value Analysis**: Explain attribution of individual predictions (Python implementation)

### 2. Interaction Effect Analysis
- **Friedman's H-statistic**: Quantify the strength of interactions between features
- **Second-order Partial Dependence Plots**: Visualize interaction effects of feature pairs

### 3. Feature Effect Analysis
- **Accumulated Local Effects (ALE) Plots**: Analyze the average impact of individual features on prediction
- **Partial Dependence Plots (PDP)**: Show marginal effects of features

## Key Findings

Main findings based on interpretability analysis:

1. **ASD Score** is the most important factor for predicting PTSD, higher scores indicate greater PTSD risk
2. **Psychological Resilience** has an important protective effect, rescue workers with strong psychological endurance have lower PTSD risk
3. **Disaster Scene Exposure** significantly increases PTSD risk, but rear personnel have relatively lower risk
4. **Age Factor** shows a non-linear relationship, with different risk levels across age groups
5. **Interactions**: Significant interactions exist between age and ASD scores, psychological resilience and disaster scene exposure, etc.

## Practical Application Recommendations

Recommendations based on analysis results:

- **Focus Attention** on rescue workers showing acute stress reactions
- **Strengthen Training** for psychological endurance of rescue workers
- **Reasonable Arrangement** of rescue tasks to reduce unnecessary disaster scene exposure
- **Optimal Configuration** considering age factors for reasonable personnel deployment
- **Personalized Attention** developing targeted intervention measures based on different feature combinations

## File Descriptions

### Core Scripts

- `scripts/data_preprocessing.R`: Complete R data preprocessing pipeline, including data cleaning, feature engineering, variable selection
- `scripts/model_training.R`: R random forest model construction and performance evaluation
- `scripts/interpretability_analysis.R`: Comprehensive R interpretable machine learning analysis
- `scripts/iml_analysis.R`: Analysis implementation based on original research IML textbook sections
- `scripts/main_pipeline.R`: R one-click execution of complete analysis pipeline

### Jupyter Notebooks

- `notebooks/data_pre.ipynb`: Python data preprocessing and feature engineering, including correlation analysis and feature selection
- `notebooks/prediction.ipynb`: Python model training, prediction and SHAP analysis, generating visualized feature importance plots

### Data Files

- `data/PTSD.xlsx`: Original Excel dataset containing various features and PCL scores of rescue workers
- `data/data.csv`: Preprocessed CSV format data (Python)
- `data/data.pkl`: Serialized preprocessed data (Python)

### Model Files

- `models/final_random_forest_model.pkl`: Random forest model trained with Python

### Result Outputs

- `results/feature_importance*.png`: Feature importance visualizations
- `results/interaction*.png`: Interaction effect analysis plots
- `results/ale*.png`: Accumulated local effects plots
- `results/pdp*.png`: Partial dependence plots
- `results/shap*.png`: SHAP value explanation plots (Python generated)
- `results/*.csv`: Numerical results of various analyses

## Technical Features

1. **Dual Implementation**: Complete implementations in both R and Python to meet different user needs
2. **Modular Design**: Each analysis step is an independent script for easy debugging and maintenance
3. **Error Handling**: Complete exception handling mechanisms to ensure pipeline stability
4. **Result Saving**: Automatically save all analysis results for subsequent analysis
5. **Scalability**: Clear code structure for easy addition of new analysis methods

## Version Differences

### R Version Features
- Uses mlr3 ecosystem for machine learning modeling
- Complete IML interpretable analysis suite
- Automated pipeline control and result saving
- Support for multiple visualization output formats

### Python Version Features
- Uses scikit-learn for random forest modeling
- Focuses on SHAP interpretability analysis
- Interactive Jupyter notebook environment
- Modern data visualization style

## Notes

1. Ensure R version ≥ 4.0.0, Python version ≥ 3.8
2. Large-scale data analysis requires sufficient memory support
3. Interpretability analysis (especially SHAP) may require considerable time
4. It is recommended to execute scripts in order to ensure data consistency
5. Please ensure all dependency packages are correctly installed before running

## License

MIT

