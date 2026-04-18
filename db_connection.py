import pandas as pd
from sqlalchemy import create_engine, text

# --- CONFIGURATION ---
DB_USER = 'root'
DB_PASSWORD = ''
DB_HOST = 'localhost'
DB_PORT = '3306'
DB_NAME = 'grad_project'

connection_string = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

try:
    db_engine = create_engine(connection_string, echo=False)
except Exception as e:
    print(f"❌ DB Engine Creation Error: {e}")
    db_engine = None

# --- CRUD OPERATIONS ---
def fetch_all_items():
    """Fetches all items for training and display."""
    if db_engine is None: return pd.DataFrame()
    try:
        return pd.read_sql("SELECT * FROM items_ml", db_engine)
    except:
        return pd.DataFrame()

def insert_new_item(data):
    """Saves a new item to the database."""
    if db_engine is None: return False
    try:
        pd.DataFrame([data]).to_sql('items_ml', con=db_engine, if_exists='append', index=False)
        return True
    except: 
        return False

def delete_item(part_id):
    """Deletes an item by its ID."""
    if db_engine is None: return False
    try:
        with db_engine.connect() as conn:
            conn.execute(text("DELETE FROM items_ml WHERE part_id = :id"), {"id": part_id})
            conn.commit()
        return True
    except: 
        return False