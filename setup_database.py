import pandas as pd
import os
from sqlalchemy import create_engine, text

# --- 1. CONFIGURATION ---
DB_USER = 'root'
DB_PASSWORD = ''
DB_HOST = 'localhost'
DB_PORT = '3306'
DB_NAME = 'grad_project'
FILE_NAME = 'data_en.xlsx'  # Make sure this matches your file name exactly

# --- 2. CONNECT TO DATABASE ---
print(f"--> Connecting to database '{DB_NAME}'...")
connection_string = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

try:
    engine = create_engine(connection_string)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("✅ Database Connection Successful.")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not connect to MySQL. Is XAMPP running?\nError: {e}")
    exit()

# --- 3. READ EXCEL FILE ---
print(f"--> Looking for file: {FILE_NAME}...")
if not os.path.exists(FILE_NAME):
    # Try CSV backup if xlsx is missing
    csv_name = FILE_NAME.replace('.xlsx', '.csv')
    if os.path.exists(csv_name):
        print(f"⚠️ '{FILE_NAME}' not found, but found '{csv_name}'. Using that.")
        FILE_NAME = csv_name
    else:
        print(f"❌ CRITICAL ERROR: File '{FILE_NAME}' not found in this folder.")
        print("   Please copy your Excel file here and try again.")
        exit()

try:
    if FILE_NAME.endswith('.csv'):
        df = pd.read_csv(FILE_NAME)
    else:
        # Requires openpyxl
        df = pd.read_excel(FILE_NAME, engine='openpyxl')
        
    print(f"✅ File Read Successfully. Found {len(df)} rows.")
    print(f"   Columns found: {list(df.columns)}")
    
    # Clean column names (remove spaces)
    df.columns = df.columns.str.strip()
    
except ImportError:
    print("❌ ERROR: Missing Python library.")
    print("   Please run: pip install openpyxl")
    exit()
except Exception as e:
    print(f"❌ File Error: {e}")
    exit()

# --- 4. PREPARE DATA ---
# Filter for the exact columns we need for the ML model
required_cols = ['part_id', 'comp_level', 'size', 'paint_used', 'w_i', 'h_i', 'l_i', 'p_i']

# Check if columns exist
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    print(f"❌ ERROR: The file is missing these columns: {missing_cols}")
    print("   Please rename the columns in your Excel file to match exactly.")
    exit()

df_final = df[required_cols].copy()

# --- 5. UPLOAD TO SQL ---
print("--> Uploading to MySQL table 'items_ml'...")

try:
    # 'replace' drops the old table and creates a fresh one
    df_final.to_sql('items_ml', con=engine, if_exists='replace', index=False)
    
    # Add Primary Key for performance
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE items_ml ADD PRIMARY KEY (part_id)"))
        count = conn.execute(text("SELECT COUNT(*) FROM items_ml")).scalar()
        
    print("="*40)
    print(f"✅ SUCCESS! Database updated.")
    print(f"✅ Total Items in Database: {count}")
    print("="*40)
    print("You can now run 'streamlit run app.py' and the data will be there.")

except Exception as e:
    print(f"❌ SQL Upload Error: {e}")