import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

# Page title
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("💳 Fraud Detection Dashboard")

# Connect to the SQLite database
conn = sqlite3.connect("fraud.db")
query = "SELECT * FROM transactions"
try:
    df = pd.read_sql(query, conn)
except Exception as e:
    st.error(f"Error reading from database: {e}")
    df = pd.DataFrame()  # fallback to empty

conn.close()

# Check if DataFrame is not empty and contains 'is_fraud'
if not df.empty and 'is_fraud' in df.columns:
    # Basic Metrics
    total = len(df)
    fraud_cases = int(df["is_fraud"].sum())
    non_fraud_cases = total - fraud_cases

    st.metric("Total Transactions", total)
    st.metric("Fraud Cases", fraud_cases)
    st.metric("Legit Transactions", non_fraud_cases)

    # Show Data Table
    st.subheader("Recent Transactions")
    st.dataframe(df.sort_values(by="id", ascending=False).reset_index(drop=True))

    # Add visualizations
    if fraud_cases + non_fraud_cases > 0:
        labels = ["Fraud", "Legit"]
        sizes = [fraud_cases, non_fraud_cases]

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.subheader("Fraud Distribution")
        st.pyplot(fig)
    else:
        st.warning("No transaction data available to plot.")
else:
    st.warning("No transactions found or 'is_fraud' column is missing.")
