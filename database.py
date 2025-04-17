import sqlite3

DB_FILE = "data.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Table to store soil moisture readings
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS soil_moisture_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            zone TEXT,
            moisture_mm REAL
        )
    ''')

    # Table to store irrigation events
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS irrigation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            zone TEXT,
            irrigation_mm REAL
        )
    ''')

    conn.commit()
    conn.close()
