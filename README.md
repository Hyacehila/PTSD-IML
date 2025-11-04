# PTSD-IML: PTSD Prediction Research Project Based on Machine Learning

## Project Overview

This project is a complete machine learning research project for PTSD (Post-Traumatic Stress Disorder) prediction. It analyzes various clinical and demographic features to build a random forest model for predicting PCL (PTSD Checklist) total scores. The project includes a complete workflow of data preprocessing, feature engineering, model training, performance evaluation, and interpretability analysis.

## Project Structure

```
PTSD-IML/
├── data/                           # Data files directory
│   ├── PTSD.xlsx                  # Original Excel dataset
│   ├── data.csv                   # Preprocessed CSV format data
│   └── data.pkl                   # Serialized Python data file
├── models/                        # Model files directory
│   └── final_random_forest_model.pkl  # Trained random forest model
├── notebooks/                     # Jupyter analysis notebooks
│   ├── data_pre.ipynb            # Data preprocessing and feature engineering
│   └── prediction.ipynb          # Model training, prediction and SHAP analysis
├── README.md                      # Project documentation
├── LICENSE                        # Open source license
└── .gitignore                     # Git version control ignore file
