# Classification
This is one of the assignments for the MLDM module

Table of Contents
-----------------
1.  Environment Setup
   - 1.1  Check for NVIDIA GPU
2.  Imports
3.  Data Loading
4.  Exploratory Data Analysis (EDA)
   - 4.1  Dataset Overview
   - 4.2  Categorical Feature Analysis
   - 4.3  Missing Value Visualization
   - 4.4  Numeric Feature Correlation
5.  Missing Value Handling
6.  Feature Engineering
   - 6.1  Binary Variable Conversion
   - 6.2  Ordinal Variable Encoding
   - 6.3  One-Hot Encoding
   - 6.4  Feature Selection
   - 6.5  Feature Scaling
7.  Train-Test Split
8.  Model Training
   - 8.1  XGBoost Installation
   - 8.2  XGBoost Version Check
   - 8.3  Model Definition
   - 8.4  Random Forest Training
   - 8.5  XGBoost Training
9.  Model Persistence
10. Model Loading
11. Model Evaluation
12. Model Comparison and Visualization
   - 12.1 Confusion Matrix Visualization


Summary
-------
This Python script performs a classification task, likely related to obesity risk, using the "ObesityDataSet_raw_and_data_sinthetic.csv" dataset. The script includes the following steps:

1.  **Environment Setup**:
    * Checks for an NVIDIA GPU using `nvidia-smi`.
2.  **Imports**:
    * Imports libraries, including numpy, pandas, matplotlib, seaborn, scikit-learn (SimpleImputer, OrdinalEncoder, MinMaxScaler, train_test_split), xgboost, and joblib.
3.  **Data Loading**:
    * Loads the dataset from a CSV file.
4.  **Exploratory Data Analysis (EDA)**:
    * Prints the dataset head, info, descriptive statistics, and missing value counts.
    * Prints value counts for categorical features.
    * Visualizes missing values using a heatmap.
    * Visualizes the correlation between numeric features using a heatmap.
5.  **Missing Value Handling**:
    * Imputes missing values using mean for numerical features and the most frequent value for categorical features.
6.  **Feature Engineering**:
    * Converts binary categorical columns ('Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC') to binary numerical (0/1).
    * Encodes ordinal categorical columns ('CAEC', 'CALC', 'NObeyesdad') using OrdinalEncoder.
    * One-hot encodes the 'MTRANS' column.
    * Selects a list of columns as the feature set (`x_cols`).
    * Scales the features in `x_cols` using MinMaxScaler.
7.  **Train-Test Split**:
     * Splits the data into training and testing sets.
8.  **Model Training**:
    * Installs xgboost using a shell command.
    * Prints the xgboost version.
    * Defines a RandomForestClassifier and an XGBClassifier.
    * Trains both models on the training data.
9.  **Model Persistence**:
     * Saves the trained models to a file named 'trained_models.joblib'.
10. **Model Loading**:
     * Loads the trained models from 'trained_models.joblib'.
11. **Model Evaluation**:
    * Generates and prints classification reports for both models.
12.  **Model Comparison and Visualization**:
    * Displays confusion matrices for both models.
