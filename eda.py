# eda.py

import pandas as pd

# Load the dataset
df = pd.read_csv("C:/Users/sahit/Desktop/Synthetic_Financial_datasets_log.csv")

# Dataset basic info
print("--- Dataset Info ---")
print(df.info())

# Check for missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Check for fraud class distribution using 'isFraud'
print("\n--- Fraud Class Distribution ---")
if 'isFraud' in df.columns:
    print(df["isFraud"].value_counts())
    print("\nFraud Percentage: {:.4f}%".format(df["isFraud"].mean() * 100))
else:
    print("❌ 'isFraud' column not found in the dataset.")

# Check for flagged fraud (optional analysis)
print("\n--- Flagged Fraud Distribution ---")
if 'isFlaggedFraud' in df.columns:
    print(df["isFlaggedFraud"].value_counts())
else:
    print("❌ 'isFlaggedFraud' column not found in the dataset.")

# Summary statistics
print("\n--- Summary Statistics ---")
print(df.describe())
