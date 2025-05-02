#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('nvidia-smi')


# In[2]:


import numpy as np
import pandas as pd
# Load the dataset
df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")


# In[3]:


# EDA: Basic overview
print("Dataset Head:\n", df.head())
print("\nDataset Info:\n")
df.info()
print("\nSummary Statistics:\n", df.describe())
print("\nMissing Values Count:\n", df.isnull().sum())


# In[4]:


# Check cardinality of categorical features
num_features = tuple(df.select_dtypes(include=['float64']).columns)
cat_features = tuple(df.select_dtypes(include=['object']).columns)
for col in cat_features:
    print(f'{col} value counts')
    print(df[col].value_counts())
    print()


# In[5]:


# Visualizing missing values
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()


# In[6]:


# Correlation Heatmap for numeric columns
numeric_features = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_features].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[7]:


from sklearn.impute import SimpleImputer
# Handling Missing Values
# Impute missing values based on data type
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')


# In[8]:


# Convert Binary Vars to 0/1
# Gender
df['Gender'] = (df.Gender == 'Male').astype(int)

# family_history_with_overweight
df.family_history_with_overweight = (df.family_history_with_overweight == 'yes').astype(int)

# FAVC
df.FAVC = (df.FAVC == 'yes').astype(int)

# SMOKE 
df.SMOKE = (df.SMOKE == 'yes').astype(int)

# SCC
df.SCC = (df.SCC == 'yes').astype(int)


# In[9]:


# Encode Ordinal Vars
from sklearn.preprocessing import OrdinalEncoder
ord_encoders = {}
ord_vars = ('CAEC', 'CALC', 'NObeyesdad')
ord_vals = [(('no', 'Sometimes', 'Frequently', 'Always'),),
            (('no', 'Sometimes', 'Frequently', 'Always'),),
            (('Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I',
              'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'),)]
for i, key in enumerate(ord_vars):
    print(f"Encoding {key} with categories: {ord_vals[i]}")
    
    # Extract the inner tuple and wrap it in a list
    categories = [ord_vals[i][0]]
    
    # Initialize the OrdinalEncoder with the correctly structured categories
    ord_encoders[key] = OrdinalEncoder(categories=categories)
    
    # Fit the encoder on the data
    ord_encoders[key].fit(df[[key]])
    
    # Transform the data and create a new encoded column
    col = 'ord_' + key
    df[col] = ord_encoders[key].transform(df[[key]])


# In[10]:


# One-Hot Encode MTRANS
df = pd.concat([df, pd.get_dummies(df.MTRANS).add_prefix('MTRANS_')], axis=1)
df.head()


# In[11]:


# Collect Final X vars
df.columns


# In[12]:


# Define final X columns
x_cols = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
          'FAVC', 'FCVC', 'NCP','SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'ord_CAEC',
          'ord_CALC','MTRANS_Automobile', 'MTRANS_Bike', 'MTRANS_Motorbike',
          'MTRANS_Public_Transportation', 'MTRANS_Walking']
df[['ord_NObeyesdad'] + x_cols]


# In[13]:


# Min Max Scale X Vars
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[x_cols] = scaler.fit_transform(df[x_cols])
df[x_cols].head()


# In[14]:


# Test Train Split
from sklearn.model_selection import train_test_split
X = df[x_cols]
y = df['ord_NObeyesdad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'X Train {X_train.shape}')
print(f'X Test {X_test.shape}')
print(f'y Train {y_train.shape}')
print(f'y test {y_test.shape}')


# In[15]:


# Install xgboost
get_ipython().system('pip install xgboost')
# Restart the kernel (manually)
# Import the library and check the version
import xgboost
print(xgboost.__version__)


# In[17]:


# Import RandomForest and XGBoost Classifiers
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Define the models with their respective hyperparameters
rf_model = RandomForestClassifier(n_jobs=-1, random_state=42)
xgb_model = XGBClassifier(n_jobs=-1, use_label_encoder=False, eval_metric='mlogloss', random_state=42)


# In[18]:


# Train the Random Forest model
rf_model.fit(X_train, y_train)
print("Random Forest training completed.\n")


# In[19]:


# Train the XGBoost model
xgb_model.fit(X_train, y_train)
print("XGBoost training completed.\n")


# In[21]:


import joblib
# Save the trained models
joblib.dump({'rf': rf_model, 'xgb': xgb_model}, 'trained_models.joblib')


# In[22]:


# Load trained models
best_estimators = joblib.load('trained_models.joblib')
rf_model = best_estimators['rf']
xgb_model = best_estimators['xgb']


# In[23]:


# Evaluate the models
evaluate = {'Random Forest': rf_model, 'XGBoost': xgb_model}
y_test_str = ord_encoders['NObeyesdad'].inverse_transform(np.asarray(y_test).reshape(-1,1))

for model_name, model in evaluate.items():
    # Predict using the model
    y_pred = ord_encoders['NObeyesdad'].inverse_transform(model.predict(X_test).reshape(-1,1))
    
    # Print classification report
    print(f'Classification Report for {model_name}:\n')
    print(classification_report(y_test_str, y_pred))
    print('-' * 80)


# In[25]:


# Compare the models based on their performance
# Plot Confusion Matrices
labels = list(ord_encoders['NObeyesdad'].categories_[0])  # Get category labels

# Confusion Matrix for XGBoost
plt.figure(figsize=(10, 8))
ConfusionMatrixDisplay.from_estimator(
    xgb_model,
    X_test,
    y_test,
    display_labels=labels,
    cmap=plt.cm.Greens,
    xticks_rotation='vertical'
)
plt.title('Confusion Matrix for XGBoost Classifier', fontweight='bold')
plt.grid(False)
plt.show()

# Confusion Matrix for Random Forest
plt.figure(figsize=(10, 8))
ConfusionMatrixDisplay.from_estimator(
    rf_model,
    X_test,
    y_test,
    display_labels=labels,
    cmap=plt.cm.Blues,
    xticks_rotation='vertical'
)
plt.title('Confusion Matrix for Random Forest Classifier', fontweight='bold')
plt.grid(False)
plt.show()

