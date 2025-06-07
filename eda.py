# eda.py

import pandas as pd

# Load the dataset
df = pd.read_csv(r"C:\Users\sahit\Desktop\bs140513_032310.csv")

# Dataset basic info
print("--- Dataset Info ---")
print(df.info())

# Check for missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Check for fraud class distribution using 'fraud' column
print("\n--- Fraud Class Distribution ---")
if 'fraud' in df.columns:
    print(df["fraud"].value_counts())
    print("\nFraud Percentage: {:.4f}%".format(df["fraud"].mean() * 100))
else:
    print("❌ 'fraud' column not found in the dataset.")

# Summary statistics for numerical columns
print("\n--- Summary Statistics ---")
print(df.describe())

# Additional analysis for categorical columns
print("\n--- Categorical Columns Analysis ---")
categorical_cols = ['customer', 'gender', 'zipcodeOri', 'merchant', 'zipMerchant', 'category']
for col in categorical_cols:
    if col in df.columns:
        print(f"\n{col} value counts:")
        print(df[col].value_counts().head())  # Show top 5 values to avoid long output
    else:
        print(f"❌ {col} column not found in the dataset.")

# Transaction amount analysis by fraud status
if 'fraud' in df.columns and 'amount' in df.columns:
    print("\n--- Transaction Amount Analysis by Fraud Status ---")
    print(df.groupby('fraud')['amount'].describe())