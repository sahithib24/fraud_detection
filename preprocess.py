import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

def preprocess_data():
    # Load the dataset
    df = pd.read_csv(r"C:\Users\sahit\Desktop\bs140513_032310.csv")

    # Create directory for processed data if it doesn't exist
    os.makedirs('processed_data', exist_ok=True)

    # Drop irrelevant identifier columns
    df = df.drop(["customer", "zipcodeOri", "merchant", "zipMerchant"], axis=1)

    # Encode and clean 'gender' column
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str).str.replace("'", "").str.strip()
        df["gender"] = df["gender"].map({"M": 1, "F": 0, "U": -1, "1": 1, "0": 0, "-1": -1})
        df["gender"] = df["gender"].fillna(-1)  # If mapping fails

    # Clean and convert 'age'
    if "age" in df.columns:
        df["age"] = df["age"].astype(str).str.replace("'", "").str.strip()
        df["age"] = pd.to_numeric(df["age"], errors='coerce')
        df["age"] = df["age"].fillna(df["age"].median())

    # One-hot encode 'category'
    if "category" in df.columns:
        df["category"] = df["category"].astype(str).str.replace("'", "").str.strip()
        df = pd.get_dummies(df, columns=["category"], drop_first=True)

    # Convert 'step' and 'amount' to numeric
    for col in ["step", "amount"]:
        df[col] = df[col].astype(str).str.replace("'", "").str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    # Separate features and label
    X = df.drop("fraud", axis=1)
    y = df["fraud"]

    # Ensure all remaining object columns are cleaned and converted
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype(str).str.replace("'", "").str.strip()
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # Final NaN cleanup
    X = X.fillna(X.median())

    # Print class distribution
    print("\n--- Before SMOTE ---")
    print(y.value_counts())

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    print("\n--- After SMOTE ---")
    print(y_res.value_counts())

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler and preprocessed data
    joblib.dump(scaler, "processed_data/scaler.pkl")
    joblib.dump((X_train_scaled, X_test_scaled, y_train, y_test),
                "processed_data/processed_data.pkl")
    
    # After creating dummies for category, save dummy column names:
    category_dummies = [col for col in X.columns if col.startswith("category_")]
    joblib.dump(category_dummies, "processed_data/category_dummies.pkl")
    dummy_columns = df.columns[df.columns.str.startswith("category_")]
    print(dummy_columns)


    print("\n--- Preprocessing Complete ---")
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    print("Data saved to 'processed_data/'")

if __name__ == "__main__":
    preprocess_data()
