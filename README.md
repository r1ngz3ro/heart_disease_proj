# Heart Disease Prediction - Data Science Lab Project

## Overview
This project analyzes the UCI Heart Disease Dataset to predict the presence or absence of heart disease using machine learning models. The analysis includes comprehensive Exploratory Data Analysis (EDA), data preprocessing, and modeling with Logistic Regression and Random Forest.

## Dataset
The dataset contains 303 patient records with 14 attributes, but only 13 features are used for prediction after converting the target variable to binary classification.

### Original vs Processed Data
Two versions of the Cleveland data are available:
- `cleveland.data`: The original unprocessed data with 76 attributes in space-separated format
- `processed.cleveland.data`: The cleaned version with only 14 key attributes in CSV format

Key differences:
- Original: Space-separated values with missing values marked as -9
- Processed: Comma-separated values with proper missing value handling
- Processed: Only the 14 most commonly used attributes for heart disease prediction
- Processed: This version is standard for machine learning applications but still requires preprocessing, feature scaling, and target engineering before modeling

The choice to use the processed version was made to focus on analysis and modeling rather than data extraction, following standard practice in academic settings. Despite being processed, the dataset still contains missing values and requires preprocessing steps such as imputation, scaling, and target variable engineering.

## Project Structure
- `heart_disease_analysis.ipynb`: Main Jupyter Notebook containing the complete analysis
- `process_data.py`: Script explaining the data processing steps from original to processed format
- `processed.cleveland.data`: The processed Cleveland heart disease dataset
- `heart-disease.names`: Documentation file describing the dataset attributes

## Features
- Age: patient's age in years
- Sex: gender (1 = male; 0 = female)
- Chest pain type: type of chest pain experienced
- Resting blood pressure: resting blood pressure (in mm Hg)
- Serum cholesterol: serum cholestoral in mg/dl
- Fasting blood sugar: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
- Resting ECG results: resting electrocardiographic results
- Maximum heart rate achieved: maximum heart rate achieved during exercise
- Exercise induced angina: (1 = yes; 0 = no)
- ST depression: ST depression induced by exercise relative to rest
- Slope: the slope of the peak exercise ST segment
- Number of vessels: number of major vessels colored by fluoroscopy
- Thal: results of thallium stress test (3 = normal; 6 = fixed defect; 7 = reversable defect)

## Target Variable
- `num`: diagnosis of heart disease (angiographic disease status)
  - Value 0: < 50% diameter narrowing (no heart disease)
  - Value 1: > 50% diameter narrowing (heart disease present)

## Analysis Components

### 1. Problem Definition & Objectives
- Clear statement of the binary classification problem
- Motivation for early detection of heart disease

### 2. Exploratory Data Analysis (EDA)
- Summary statistics
- Distribution plots of key variables
- Correlation analysis
- Missing value analysis
- Class balance discussion

### 3. Data Cleaning & Preprocessing
- Handling missing values with median imputation
- Creating binary target variable
- Feature scaling for Logistic Regression
- Train/test split with stratification

### 4. Modeling
- Logistic Regression model
- Random Forest model (optional)

### 5. Model Evaluation
- Accuracy scores
- Confusion matrices
- ROC-AUC scores
- Feature importance analysis

### 6. Conclusion
- Key findings summary
- Limitations of the study
- Possible improvements

## Requirements
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## How to Run
1. Make sure you have all required packages installed:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

2. Run the analysis script:
   ```bash
   python heart_disease_analysis.py
   ```

3. To understand the data processing steps:
   ```bash
   python process_data.py
   ```

## Results
The analysis shows that both Logistic Regression and Random Forest models perform well on this dataset, with the Random Forest achieving slightly higher accuracy (88.52%) and both models having high ROC-AUC scores (>0.95). The most important features for predicting heart disease include maximum heart rate achieved, chest pain type, thallium stress test results, and number of major vessels colored by fluoroscopy.

## Key Findings
- The dataset has a relatively balanced class distribution (45.87% with heart disease, 54.13% without)
- Maximum heart rate achieved (thalach) is consistently identified as an important predictor
- Chest pain type (cp) and thallium stress test results (thal) are also significant factors
- The models achieve good performance despite the relatively small dataset size

## Limitations
- Small dataset size (303 records) may limit model generalization
- Missing values were imputed using median values
- Some categorical variables were treated as continuous

## Possible Improvements
- Collect more data to improve model generalization
- Perform more sophisticated imputation techniques
- Engineer new features based on domain knowledge
- Try more advanced models with hyperparameter tuning
