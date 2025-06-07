import sqlite3

# Connect (or create) the database
conn = sqlite3.connect("fraud.db")
cursor = conn.cursor()

# Create the transactions table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    step INTEGER,
    customer TEXT,
    age INTEGER,
    gender TEXT,
    zipcodeOri TEXT,
    merchant TEXT,
    zipMerchant TEXT,
    category TEXT,
    amount REAL,
    is_fraud INTEGER
)
""")

conn.commit()
conn.close()
print("Database and table initialized successfully.")
