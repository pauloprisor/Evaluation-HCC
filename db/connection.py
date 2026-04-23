import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results.db')
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), 'schema.sql')

def initialize_schema():
    with open(SCHEMA_PATH, 'r') as f:
        schema = f.read()
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(schema)
        conn.commit()

def get_connection() -> sqlite3.Connection:
    """Returnează o conexiune SQLite la results.db. Inițializează schema dacă fișierul nu există."""
    needs_init = not os.path.exists(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    if needs_init:
        # Avoid recursive calls but do it cleanly
        with open(SCHEMA_PATH, 'r') as f:
            schema = f.read()
        conn.executescript(schema)
        conn.commit()
        
    return conn
