# 💳 Fraud Detection System 🔍

This project is an end-to-end **Fraud Detection API and Dashboard** built using **FastAPI**, **XGBoost**, **Streamlit**, and **SQLite**. It allows real-time transaction analysis, predicts potential fraud, logs results, and visualizes trends on an interactive dashboard.

---

## 📌 Project Features

- 🧠 **Machine Learning Model** (XGBoost) trained on transaction data  
- ⚡ **FastAPI** server to predict fraud in real-time  
- 💾 **SQLite** integration to log transactions and predictions  
- 📊 **Streamlit Dashboard** to visualize fraud stats and recent activity  
- 🐳 **Dockerized** for easy deployment  
- 📁 Modular code with separate scripts for training, API, database, and dashboard

---

## 📁 Project Structure

```bash
fraud_detection/
│
├── models/                  # Trained model .pkl files
│   └── xgb_model.pkl
│
├── processed_data/          # Scaler and encoder objects
│   └── scaler.pkl
│
├── app.py
├── dashboard.py             # Streamlit visualization app
├── train.py                 # ML training script
├── preprocess.py            # Data cleaning and transformation
├── fraud.db                 # SQLite database (auto-created)
├── Dockerfile               # Docker setup
├── requirements.txt         # Python dependencies
└── README.md
````

---

## 🚀 Quick Start

### 1️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

### 2️⃣ Run the API

```bash
cd api
uvicorn app:app --reload
```

Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive Swagger UI.

### 3️⃣ Test the API

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"step\":0,\"customer\":\"C123\",\"age\":30,\"gender\":\"M\",\"zipcodeOri\":\"10001\",\"merchant\":\"M456\",\"zipMerchant\":\"10001\",\"category\":\"es_transportation\",\"amount\":250.0}"
```

### 4️⃣ Launch Streamlit Dashboard

```bash
streamlit run dashboard.py
```

---

## 🗄️ Database (SQLite)

* File: `fraud.db`
* Table: `transactions`
* Columns: `id`, `amount`, `is_fraud`

Each prediction is automatically logged in the DB via the FastAPI backend.

---

## 🐳 Docker Deployment

### ✅ Prerequisites

* Docker installed
* Virtual Machine Platform enabled (for Windows)

### 🔧 Build and Run

```bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
```

---

## 📈 Model Details

* **Algorithm**: XGBoost Classifier
* **Features**: Step, Age, Gender, Amount, and One-Hot Encoded Categories
* **Preprocessing**:

  * Gender Mapping (`M=1, F=0, U=-1`)
  * Category one-hot encoding
  * Scaled with `StandardScaler`

---

## 📌 API Endpoint

### `/predict` (POST)

| Field       | Type  | Description                        |
| ----------- | ----- | ---------------------------------- |
| step        | int   | Transaction step                   |
| customer    | str   | Customer ID                        |
| age         | int   | Age bucket (e.g. 1-10)             |
| gender      | str   | Gender: "M", "F", or "U"           |
| zipcodeOri  | str   | Zipcode of origin                  |
| merchant    | str   | Merchant ID                        |
| zipMerchant | str   | Zipcode of merchant                |
| category    | str   | Category (e.g. es\_transportation) |
| amount      | float | Transaction amount                 |

### ✅ Response

```json
{
  "fraud_probability": 0.8742,
  "is_fraud": true,
  "status": "success"
}
```

---

## 📊 Dashboard Overview

The Streamlit dashboard shows:

* Total transactions
* Fraud vs Legitimate cases (pie chart)
* Live table of recent transactions

---

