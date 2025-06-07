from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Load your model and scaler once on startup
model = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("processed_data/scaler.pkl")

# Fixed list of one-hot encoded category columns your model expects
CATEGORY_COLUMNS = [
    'category_es_contents', 'category_es_fashion', 'category_es_food',
    'category_es_health', 'category_es_home', 'category_es_hotelservices',
    'category_es_hyper', 'category_es_leisure', 'category_es_otherservices',
    'category_es_sportsandtoys', 'category_es_tech', 'category_es_transportation',
    'category_es_travel', 'category_es_wellnessandbeauty'
]

# Pydantic model for input validation
class Transaction(BaseModel):
    step: int
    customer: str
    age: int
    gender: Literal["M", "F", "U"]  # Assuming these are possible values
    zipcodeOri: str
    merchant: str
    zipMerchant: str
    category: str
    amount: float

def preprocess_input(data: Transaction):
    # Convert gender to numeric
    gender_map = {"M": 1, "F": 0, "U": -1}
    gender_num = gender_map.get(data.gender, -1)

    # Create dict for fixed features
    input_dict = {
        "step": data.step,
        "age": data.age,
        "gender": gender_num,
        "amount": data.amount,
    }

    # Create one-hot encoded category columns all zero initially
    for col in CATEGORY_COLUMNS:
        input_dict[col] = 0

    # Set the category column to 1 if it matches one of the known categories
    cat_col_name = f"category_{data.category}"
    if cat_col_name in CATEGORY_COLUMNS:
        input_dict[cat_col_name] = 1
    else:
        # Unknown category - leave all zero or handle differently
        pass

    # Convert dict to DataFrame (1 row)
    df = pd.DataFrame([input_dict])

    # Scale features using your scaler
    X_scaled = scaler.transform(df)

    return X_scaled

"""@app.post("/predict")
def predict(transaction: Transaction):
    try:
        X = preprocess_input(transaction)
        proba = float(model.predict_proba(X)[0, 1])  # Convert numpy.float32 to float
        pred = int(proba >= 0.5)  # Threshold 0.5 for classification

        return {
            "fraud_probability": proba,
            "is_fraud": pred,
            "status": "success"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
"""

import sqlite3

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        X = preprocess_input(transaction)
        proba = model.predict_proba(X)[0, 1]
        pred = int(proba >= 0.5)

        # Insert the transaction into the SQLite DB
        conn = sqlite3.connect("fraud.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO transactions (step, customer, age, gender, zipcodeOri, merchant, zipMerchant, category, amount, is_fraud)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            transaction.step, transaction.customer, transaction.age,
            transaction.gender, transaction.zipcodeOri, transaction.merchant,
            transaction.zipMerchant, transaction.category,
            transaction.amount, pred
        ))
        conn.commit()
        conn.close()

        return {
            "fraud_probability": float(proba),
            "is_fraud": pred,
            "status": "success"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
