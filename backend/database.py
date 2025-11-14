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


def get_metrics():
    conn = get_connection()
    row = conn.execute('''
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN prediction = 'pass' THEN 1 ELSE 0 END) as pass_count,
            SUM(CASE WHEN prediction = 'fail' THEN 1 ELSE 0 END) as fail_count,
            AVG(confidence) as avg_confidence
        FROM predictions
    ''').fetchone()

    # predictions in the last hour
    per_hour = conn.execute('''
        SELECT COUNT(*) FROM predictions
        WHERE timestamp > datetime('now', '-1 hour')
    ''').fetchone()[0]

    conn.close()

    total = row['total']
    return {
        'total_predictions': total,
        'pass_count': row['pass_count'] or 0,
        'fail_count': row['fail_count'] or 0,
        'pass_rate': round((row['pass_count'] or 0) / total, 4) if total > 0 else 0,
        'fail_rate': round((row['fail_count'] or 0) / total, 4) if total > 0 else 0,
        'avg_confidence': round(row['avg_confidence'] or 0, 4),
        'predictions_last_hour': per_hour
    }


def update_actual_label(prediction_id, actual_label):
    """Set the ground truth label for a prediction. Returns True if the row existed."""
    conn = get_connection()
    cursor = conn.execute(
        'UPDATE predictions SET actual_label = ? WHERE id = ?',
        (actual_label, prediction_id)
    )
    conn.commit()
    updated = cursor.rowcount > 0
    conn.close()
    return updated


# create table on import
init_db()
