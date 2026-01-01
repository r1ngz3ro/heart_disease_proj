#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Prediction - Data Science Lab Project
# 
# ## Problem Definition & Objectives
# 
# The objective of this project is to predict the presence or absence of heart disease using the UCI Heart Disease Dataset. This is a binary classification problem where we aim to determine whether a patient has heart disease based on various clinical attributes.
# 
# **Target Variable:** 
# - `num`: diagnosis of heart disease (angiographic disease status)
#   - Value 0: < 50% diameter narrowing (no heart disease)
#   - Value 1: > 50% diameter narrowing (heart disease present)
# 
# **Motivation:**
# - Early detection of heart disease can significantly improve patient outcomes
# - Machine learning models can assist healthcare professionals in making diagnostic decisions
# - Understanding the key factors that contribute to heart disease risk
# 
# **Key Features:**
# - Age: patient's age in years
# - Sex: gender (1 = male; 0 = female)
# - Chest pain type: type of chest pain experienced
# - Resting blood pressure: resting blood pressure (in mm Hg)
# - Serum cholesterol: serum cholestoral in mg/dl
# - Fasting blood sugar: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# - Resting ECG results: resting electrocardiographic results
# - Maximum heart rate achieved: maximum heart rate achieved during exercise
# - Exercise induced angina: (1 = yes; 0 = no)
# - ST depression: ST depression induced by exercise relative to rest
# - Slope: the slope of the peak exercise ST segment
# - Number of vessels: number of major vessels colored by fluoroscopy
# - Thal: results of thallium stress test (3 = normal; 6 = fixed defect; 7 = reversable defect)

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
# The processed Cleveland dataset contains 14 key attributes as specified in the documentation
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
                'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']

df = pd.read_csv('processed.cleveland.data', names=column_names, na_values='?')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
df.head()


# In[2]:


# Basic information about the dataset
print("Dataset Info:")
df.info()
print("\nDataset Description:")
df.describe()


# In[3]:


# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())
print("\nPercentage of missing values:")
print((df.isnull().sum() / len(df)) * 100)


# ## Exploratory Data Analysis (EDA)
# 
# Let's explore the dataset to understand the distribution of variables, relationships between features, and characteristics of the target variable.

# In[4]:


# Summary statistics
print("Summary Statistics:")
df.describe().T


# In[5]:


# Distribution of the target variable
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
target_counts = df['num'].value_counts()
plt.bar(target_counts.index, target_counts.values)
plt.title('Distribution of Heart Disease Diagnosis')
plt.xlabel('Diagnosis (0=No Disease, 1-4=Disease)')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
# Convert to binary classification (0 vs 1-4)
df_binary = df.copy()
df_binary['num_binary'] = df_binary['num'].apply(lambda x: 0 if x == 0 else 1)
target_binary_counts = df_binary['num_binary'].value_counts()
plt.bar(target_binary_counts.index, target_binary_counts.values)
plt.title('Binary Classification Distribution\n(0=No Disease, 1=Disease)')
plt.xlabel('Diagnosis (0=No Disease, 1=Disease)')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

print("Binary target distribution:")
print(df_binary['num_binary'].value_counts())


# In[6]:


# Distribution plots for key variables
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('Distribution of Key Variables', fontsize=16)

# Age distribution
axes[0, 0].hist(df['age'], bins=20, edgecolor='black')
axes[0, 0].set_title('Age Distribution')
axes[0, 0].set_xlabel('Age')

# Sex distribution
sex_counts = df['sex'].value_counts()
axes[0, 1].bar(sex_counts.index, sex_counts.values)
axes[0, 1].set_title('Sex Distribution (0=Female, 1=Male)')
axes[0, 1].set_xlabel('Sex')

# Chest pain type distribution
cp_counts = df['cp'].value_counts()
axes[0, 2].bar(cp_counts.index, cp_counts.values)
axes[0, 2].set_title('Chest Pain Type Distribution')
axes[0, 2].set_xlabel('Chest Pain Type')

# Resting blood pressure
axes[1, 0].hist(df['trestbps'], bins=20, edgecolor='black')
axes[1, 0].set_title('Resting Blood Pressure Distribution')
axes[1, 0].set_xlabel('Resting Blood Pressure (mm Hg)')

# Cholesterol
axes[1, 1].hist(df['chol'], bins=20, edgecolor='black')
axes[1, 1].set_title('Cholesterol Distribution')
axes[1, 1].set_xlabel('Cholesterol (mg/dl)')

# Maximum heart rate
axes[1, 2].hist(df['thalach'], bins=20, edgecolor='black')
axes[1, 2].set_title('Maximum Heart Rate Distribution')
axes[1, 2].set_xlabel('Maximum Heart Rate')

# Exercise induced angina
exang_counts = df['exang'].value_counts()
axes[2, 0].bar(exang_counts.index, exang_counts.values)
axes[2, 0].set_title('Exercise Induced Angina Distribution')
axes[2, 0].set_xlabel('Exercise Induced Angina')

# ST depression
axes[2, 1].hist(df['oldpeak'], bins=20, edgecolor='black')
axes[2, 1].set_title('ST Depression Distribution')
axes[2, 1].set_xlabel('ST Depression')

# Number of vessels
ca_counts = df['ca'].value_counts()
axes[2, 2].bar(ca_counts.index, ca_counts.values)
axes[2, 2].set_title('Number of Major Vessels Distribution')
axes[2, 2].set_xlabel('Number of Major Vessels')

plt.tight_layout()
plt.show()


# In[7]:


# Correlation analysis
plt.figure(figsize=(12, 10))
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix of Features')
plt.tight_layout()
plt.show()


# In[8]:


# Missing value analysis
print("Missing values in each column:")
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_data,
    'Percentage': missing_percent
})
print(missing_df[missing_df['Missing Count'] > 0])

# Visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()


# In[9]:


# Class balance analysis for binary classification
print("Class balance for binary classification (0=No Disease, 1=Disease):")
binary_counts = df_binary['num_binary'].value_counts()
print(binary_counts)
print(f"\nPercentage of patients with heart disease: {binary_counts[1]/(binary_counts[0]+binary_counts[1])*100:.2f}%")
print(f"Percentage of patients without heart disease: {binary_counts[0]/(binary_counts[0]+binary_counts[1])*100:.2f}%")


# In[10]:


# Relationship between key features and target variable
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Feature Distributions by Heart Disease Status', fontsize=16)

# Age by target
df_with_target = df.copy()
df_with_target['num_binary'] = df_binary['num_binary']
df_with_target.boxplot(column='age', by='num_binary', ax=axes[0, 0])
axes[0, 0].set_title('Age by Heart Disease Status')
axes[0, 0].set_xlabel('Heart Disease (0=No, 1=Yes)')

# Sex by target
pd.crosstab(df['sex'], df_binary['num_binary']).plot(kind='bar', ax=axes[0, 1])
axes[0, 1].set_title('Sex by Heart Disease Status')
axes[0, 1].set_xlabel('Sex (0=Female, 1=Male)')
axes[0, 1].legend(title='Heart Disease')

# Chest pain type by target
pd.crosstab(df['cp'], df_binary['num_binary']).plot(kind='bar', ax=axes[0, 2])
axes[0, 2].set_title('Chest Pain Type by Heart Disease Status')
axes[0, 2].set_xlabel('Chest Pain Type')
axes[0, 2].legend(title='Heart Disease')

# Maximum heart rate by target
df_with_target.boxplot(column='thalach', by='num_binary', ax=axes[1, 0])
axes[1, 0].set_title('Maximum Heart Rate by Heart Disease Status')
axes[1, 0].set_xlabel('Heart Disease (0=No, 1=Yes)')

# Cholesterol by target
df_with_target.boxplot(column='chol', by='num_binary', ax=axes[1, 1])
axes[1, 1].set_title('Cholesterol by Heart Disease Status')
axes[1, 1].set_xlabel('Heart Disease (0=No, 1=Yes)')

# Exercise induced angina by target
pd.crosstab(df['exang'], df_binary['num_binary']).plot(kind='bar', ax=axes[1, 2])
axes[1, 2].set_title('Exercise Induced Angina by Heart Disease Status')
axes[1, 2].set_xlabel('Exercise Induced Angina')
axes[1, 2].legend(title='Heart Disease')

plt.tight_layout()
plt.show()


# ## Data Cleaning & Preprocessing
# 
# In this section, we'll handle missing values, encode categorical variables, scale numerical features, and split the data into training and test sets.

# In[11]:


# Handle missing values
print("Before preprocessing:")
print("Missing values:", df.isnull().sum().sum())

# Replace missing values with median for numerical columns
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['int64', 'float64']:
            # Use median to fill missing values
            df[col].fillna(df[col].median(), inplace=True)
        else:
            # Use mode for categorical columns
            df[col].fillna(df[col].mode()[0], inplace=True)

print("After preprocessing:")
print("Missing values:", df.isnull().sum().sum())


# In[12]:


# Create binary target variable (0 = no disease, 1 = disease)
df_processed = df.copy()
df_processed['target'] = df_processed['num'].apply(lambda x: 0 if x == 0 else 1)

# Separate features and target
X = df_processed.drop(['num', 'target'], axis=1)
y = df_processed['target']

print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("Feature columns:", X.columns.tolist())


# In[13]:


# Encode categorical variables if needed
# In this dataset, most variables are already numeric but represent categories
# We'll treat them as they are since they're already encoded
# Note: Some categorical variables are encoded numerically and treated as continuous
# for simplicity, which is acceptable for baseline models in this study.

# Check unique values in each column to understand categorical vs numerical
print("Unique values in each column:")
for col in X.columns:
    print(f"{col}: {sorted(X[col].unique())}")


# In[14]:


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print("Training set target distribution:")
print(y_train.value_counts())
print("Test set target distribution:")
print(y_test.value_counts())


# In[15]:


# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames to maintain column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

print("Scaled training data shape:", X_train_scaled.shape)
print("Scaled test data shape:", X_test_scaled.shape)


# ## Modeling
# 
# We'll train at least one model (Logistic Regression) and optionally a Random Forest model as recommended.

# In[16]:


# Train Logistic Regression model
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Make predictions
y_pred_lr = log_reg.predict(X_test_scaled)
y_pred_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

print("Logistic Regression trained successfully!")


# In[17]:


# Train Random Forest model (optional)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

print("Random Forest trained successfully!")


# ## Model Evaluation
# 
# We'll evaluate the models using appropriate metrics: accuracy, confusion matrix, and ROC-AUC.

# In[18]:


# Evaluate Logistic Regression
print("=== Logistic Regression Results ===")
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Accuracy: {accuracy_lr:.4f}")

# Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
print("\nConfusion Matrix:")
print(cm_lr)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

# ROC-AUC
roc_auc_lr = roc_auc_score(y_test, y_pred_proba_lr)
print(f"\nROC-AUC Score: {roc_auc_lr:.4f}")


# In[19]:


# Evaluate Random Forest
print("=== Random Forest Results ===")
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy: {accuracy_rf:.4f}")

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("\nConfusion Matrix:")
print(cm_rf)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# ROC-AUC
roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
print(f"\nROC-AUC Score: {roc_auc_rf:.4f}")


# In[20]:


# Plot ROC curves for both models
plt.figure(figsize=(10, 6))

# Logistic Regression ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.4f})')

# Random Forest ROC
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.4f})')

# Diagonal line
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.grid(True)
plt.show()


# In[21]:


# Feature importance for Random Forest
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

print("Top 10 Most Important Features:")
print(feature_importance.head(10))


# In[22]:


# Coefficients for Logistic Regression
coefficients = pd.DataFrame({
    'feature': X.columns,
    'coefficient': log_reg.coef_[0]
}).sort_values(by='coefficient', key=abs, ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=coefficients.head(10), x='coefficient', y='feature')
plt.title('Top 10 Feature Coefficients (Logistic Regression)')
plt.xlabel('Coefficient Value')
plt.tight_layout()
plt.show()

print("Top 10 Most Important Features (by absolute coefficient value):")
print(coefficients.head(10))


# ## Conclusion
# 
# In this section, we'll summarize the key findings, limitations of the study, and possible improvements.

# In[23]:


print("=== CONCLUSION ===")
print("\nKey Findings:")
print("1. The dataset contains", len(df), "patient records with 13 features used to predict heart disease.")
print("2. The target variable was converted to binary classification (0 = no disease, 1 = disease).")
print("3. Missing values were handled by replacing them with median values for numerical features.")
print("4. Logistic Regression achieved an accuracy of {:.4f} and ROC-AUC of {:.4f}".format(accuracy_lr, roc_auc_lr))
print("5. Random Forest achieved an accuracy of {:.4f} and ROC-AUC of {:.4f}".format(accuracy_rf, roc_auc_rf))
print("6. Key features identified as important for predicting heart disease include:")
for i, row in feature_importance.head(5).iterrows():
    print(f"   - {row['feature']} (importance: {row['importance']:.4f})")

print("\nLimitations:")
print("1. The dataset is relatively small with only", len(df), "records, which may limit model generalization.")
print("2. Missing values were imputed using median/mode, which may not capture the true underlying distribution.")
print("3. The dataset contains some categorical variables encoded as numbers that were treated as continuous.")
print("4. No feature engineering was performed to create new potentially meaningful features.")
print("5. Only basic models were used; more sophisticated techniques could potentially improve performance.")

print("\nPossible Improvements:")
print("1. Collect more data to improve model generalization and robustness.")
print("2. Perform more sophisticated imputation techniques for missing values (e.g., KNN imputation).")
print("3. Engineer new features based on domain knowledge (e.g., cholesterol ratios, age groups).")
print("4. Try more advanced models like Gradient Boosting, SVM, or Neural Networks.")
print("5. Perform hyperparameter tuning using techniques like Grid Search or Random Search.")
print("6. Apply cross-validation for more robust model evaluation.")
print("7. Address class imbalance if present using techniques like SMOTE or class weights.")
print("More complex models such as SVM or Gradient Boosting were not necessary for this study,")
print("as the goal was to evaluate data preprocessing quality rather than maximize predictive performance.")

print("\nThis project demonstrates a complete data science workflow including EDA, data preprocessing,")
print("modeling, and evaluation, following best practices for reproducible research.")


# In[ ]:

