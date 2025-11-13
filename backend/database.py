import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'predictions.db')


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            input_hash TEXT,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            actual_label TEXT
        )
    ''')
    conn.commit()
    conn.close()


def log_prediction(input_hash, prediction, confidence):
    conn = get_connection()
    conn.execute(
        'INSERT INTO predictions (timestamp, input_hash, prediction, confidence) VALUES (?, ?, ?, ?)',
        (datetime.now().isoformat(), input_hash, prediction, confidence)
    )
    conn.commit()
    conn.close()


def get_recent_predictions(n=50):
    conn = get_connection()
    rows = conn.execute(
        'SELECT * FROM predictions ORDER BY id DESC LIMIT ?', (n,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


# create table on import
init_db()
