#!/usr/bin/env python
# coding: utf-8

"""
Data Processing Script for UCI Heart Disease Dataset

This script explains the processing steps applied to convert the original
Cleveland data into the format used for machine learning analysis.
"""

import pandas as pd
import numpy as np

def explain_processing_steps():
    """
    Explain the processing steps that were applied to create the processed Cleveland data.
    """
    print("=== HEART DISEASE DATA PROCESSING EXPLANATION ===\n")

    print("ORIGINAL DATA CHARACTERISTICS:")
    print("- Format: Space-separated values")
    print("- Contains 76 attributes (though only 14 are commonly used)")
    print("- Missing values marked as -9 (which can be confused with actual data)")
    print("- Mixed format with some text values like 'name' at the end of records\n")

    print("PROCESSING STEPS APPLIED:")
    print("1. SELECTED 14 KEY ATTRIBUTES:")
    selected_attributes = [
        'age',           # Age in years
        'sex',           # Sex (1 = male; 0 = female)
        'cp',            # Chest pain type
        'trestbps',      # Resting blood pressure
        'chol',          # Serum cholestoral
        'fbs',           # Fasting blood sugar
        'restecg',       # Resting electrocardiographic results
        'thalach',       # Maximum heart rate achieved
        'exang',         # Exercise induced angina
        'oldpeak',       # ST depression induced by exercise
        'slope',         # Slope of the peak exercise ST segment
        'ca',            # Number of major vessels colored by fluoroscopy
        'thal',          # Thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
        'num'            # Diagnosis of heart disease (target variable)
    ]

    for i, attr in enumerate(selected_attributes, 1):
        print(f"   {i:2d}. {attr}")

    print("\n2. FORMATTING CHANGES:")
    print("   - Converted from space-separated to comma-separated values (CSV)")
    print("   - Standardized numeric formatting")
    print("   - Removed text values like 'name'\n")

    print("3. MISSING VALUE HANDLING:")
    print("   - Original: Missing values marked as -9")
    print("   - Processed: Missing values properly identified and handled")
    print("   - For our analysis: Missing values filled with median/mode as appropriate\n")

    print("4. FINAL FORMAT CHARACTERISTICS:")
    print("   - CSV format ready for pandas/DataFrame operations")
    print("   - 303 patient records")
    print("   - 14 attributes (13 features + 1 target)")
    print("   - Clean numeric values suitable for machine learning algorithms\n")

    # Load and show the processed data
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                    'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']

    try:
        df = pd.read_csv('processed.cleveland.data', names=column_names, na_values='?')

        print("PROCESSED DATA SAMPLE:")
        print(df.head())
        print(f"\nProcessed data shape: {df.shape}")
        print(f"Missing values in processed data: {df.isnull().sum().sum()}")

        # Show missing values per column
        missing_vals = df.isnull().sum()
        print("\nMissing values per column:")
        for col, count in missing_vals[missing_vals > 0].items():
            print(f"  {col}: {count}")

    except FileNotFoundError:
        print("Note: processed.cleveland.data file not found in current directory")

    print("\n" + "="*60)
    print("This processing makes the data suitable for machine learning")
    print("while preserving the important clinical relationships.")


def demonstrate_processing_techniques():
    """
    Demonstrate different techniques for handling missing values and data cleaning.
    """
    print("\n=== DATA PROCESSING TECHNIQUES DEMONSTRATION ===\n")

    print("1. MISSING VALUE IDENTIFICATION:")
    print("   - Original: -9 used as placeholder (ambiguous)")
    print("   - Solution: Replace -9 with NaN for clear identification\n")

    print("2. MISSING VALUE HANDLING STRATEGIES:")
    strategies = [
        ("Deletion", "Remove rows/columns with missing values - Not recommended for small datasets"),
        ("Mean Imputation", "Replace with mean - Good for normally distributed data"),
        ("Median Imputation", "Replace with median - Robust to outliers (used in our analysis)"),
        ("Mode Imputation", "Replace with mode - Good for categorical data"),
        ("KNN Imputation", "Replace based on similar records - More sophisticated")
    ]

    for method, description in strategies:
        print(f"   {method}: {description}")

    print("\n3. CATEGORICAL VARIABLE HANDLING:")
    print("   - Some variables are numeric but represent categories")
    print("   - Chest pain type: 1=typical angina, 2=atypical angina, 3=non-anginal pain, 4=asymptomatic")
    print("   - Resting ECG: 0=normal, 1=ST-T wave abnormality, 2=left ventricular hypertrophy")
    print("   - Thal: 3=normal, 6=fixed defect, 7=reversable defect")

    print("\n4. FEATURE SCALING CONSIDERATIONS:")
    print("   - Different features have different scales")
    print("   - Age: ~30-80, Cholesterol: ~125-564, Blood pressure: ~90-200")
    print("   - Scaling important for algorithms sensitive to feature magnitude")


if __name__ == "__main__":
    explain_processing_steps()
    demonstrate_processing_techniques()

    print("\n" + "="*60)
    print("Data processing script completed successfully!")
    print("This script explains how the original Cleveland data was transformed")
    print("into the format used for our heart disease prediction analysis.")