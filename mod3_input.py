import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import os

# --- 1. CONFIGURATION ---
DB_USER = 'root'
DB_PASSWORD = ''          # <--- Update with your actual password
DB_HOST = 'localhost'
DB_NAME = 'grad_project'

connection_string = f'mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}'
engine = create_engine(connection_string)

print(f"--> Connecting to MySQL database: {DB_NAME}")

# --- 2. DEFINE SCHEMA ---
create_schema_sql = """
SET FOREIGN_KEY_CHECKS = 0;

DROP TABLE IF EXISTS items;
DROP TABLE IF EXISTS oven_supported_recipes;
DROP TABLE IF EXISTS shelves;
DROP TABLE IF EXISTS system_config;
DROP TABLE IF EXISTS ovens;
DROP TABLE IF EXISTS recipes;

SET FOREIGN_KEY_CHECKS = 1;

-- 1. RECIPES (No Temperature)
CREATE TABLE recipes (
    recipe_id INT PRIMARY KEY,
    recipe_name VARCHAR(50),
    description VARCHAR(255)
);

-- 2. OVENS (Oven_1, Oven_2)
CREATE TABLE ovens (
    oven_id INT PRIMARY KEY,
    oven_name VARCHAR(50)
);

-- 3. SHELVES
CREATE TABLE shelves (
    oven_id INT,
    shelf_id INT,
    width_cm DECIMAL(10,2),
    length_cm DECIMAL(10,2),
    height_clearance_cm DECIMAL(10,2),
    FOREIGN KEY (oven_id) REFERENCES ovens(oven_id)
);

-- 4. SYSTEM CONFIG (Lambda = 20)
CREATE TABLE system_config (
    param_name VARCHAR(50),
    param_value DECIMAL(10,2)
);

-- 5. SUPPORTED RECIPES
CREATE TABLE oven_supported_recipes (
    oven_id INT,
    recipe_id INT,
    FOREIGN KEY (oven_id) REFERENCES ovens(oven_id),
    FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id)
);

-- 6. ITEMS
CREATE TABLE items (
    item_id INT PRIMARY KEY,
    w_i DECIMAL(10,2),
    l_i DECIMAL(10,2),
    h_i DECIMAL(10,2),
    
    -- Extra Data Columns
    m_i DECIMAL(10,2),
    c_i DECIMAL(10,2),      -- UPDATED: Changed from b_i to c_i
    comp_level INT,
    paint_used INT,
    area DECIMAL(10,2),
    temp DECIMAL(10,2),
    
    -- Scheduling params
    p_i DECIMAL(10,2),      -- Processing Time (minutes)
    r_i DECIMAL(10,2),      -- Arrival Time (minutes)
    
    recipe_id INT,
    delta_t INT,            -- Setup/Lead time
    
    FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id)
);
"""

# --- 3. APPLY SCHEMA & STATIC DATA ---
def init_database():
    with engine.connect() as conn:
        print("--> Dropping and Recreating Tables...")
        statements = create_schema_sql.split(';')
        for stmt in statements:
            if stmt.strip():
                conn.execute(text(stmt))
        conn.commit()

        # B. Insert Recipes (No Temperature column)
        print("--> Inserting Recipes...")
        recipes_data = [
            (1, 'Low-Temp Composite Cure', 'Standard curing for basic composite materials'),
            (2, 'Mid-Temp Structural Bond', 'Structural bonding for aerospace parts'),
            (3, 'High-Temp Rapid Cycle', 'High heat cycle for advanced resins')
        ]
        for r in recipes_data:
            conn.execute(text(
                "INSERT INTO recipes (recipe_id, recipe_name, description) VALUES (:id, :name, :desc)"
            ), {"id": r[0], "name": r[1], "desc": r[2]})

        # C. Insert Ovens
        print("--> Inserting Ovens...")
        ovens_data = [
            (1, 'Oven_1'),
            (2, 'Oven_2')
        ]
        for o in ovens_data:
            conn.execute(text("INSERT INTO ovens (oven_id, oven_name) VALUES (:id, :name)"), {"id": o[0], "name": o[1]})

        # D. Insert Shelves (4 Shelves per Oven, 288x504x110)
        print("--> Inserting Shelves (4 per oven)...")
        shelves_data = []
        
        # Oven 1 (Shelves 1-4)
        for s_id in range(1, 5):
            shelves_data.append((1, s_id, 288, 504, 110))
            
        # Oven 2 (Shelves 1-4)
        for s_id in range(1, 5):
            shelves_data.append((2, s_id, 288, 504, 110))
        
        for s in shelves_data:
            conn.execute(text(
                "INSERT INTO shelves (oven_id, shelf_id, width_cm, length_cm, height_clearance_cm) VALUES (:oid, :sid, :w, :l, :h)"
            ), {"oid": s[0], "sid": s[1], "w": s[2], "l": s[3], "h": s[4]})

        # E. Insert Supported Recipes
        for o_id in [1, 2]:
            for r_id in [1, 2, 3]:
                conn.execute(text("INSERT INTO oven_supported_recipes VALUES (:oid, :rid)"), {"oid": o_id, "rid": r_id})

        # F. Insert System Config
        print("--> Inserting System Config (Lambda=20)...")
        config_data = [
            ('start_time', 0),
            ('end_time', 480),
            ('lambda_penalty', 20.0) 
        ]
        for c in config_data:
            conn.execute(text("INSERT INTO system_config VALUES (:p, :v)"), {"p": c[0], "v": c[1]})
        
        conn.commit()

# --- 4. LOAD DATA FROM EXCEL/CSV ---
def load_items_from_file():
    file_path = 'data.xlsx'
    
    if not os.path.exists(file_path):
        if os.path.exists('data.csv'):
            file_path = 'data.csv'
        else:
            print(f"--> WARNING: '{file_path}' not found. Cannot load items.")
            return

    print(f"--> Loading items from {file_path}...")
    
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        # 1. Generate Item IDs if missing
        if 'item_id' not in df.columns:
            df['item_id'] = range(1, len(df) + 1)

        # 2. ASSIGN RECIPE ID BASED ON TEMP RANGES
        print("--> Assigning Recipe IDs based on Temp (120-135->1, 135-150->2, 150-160->3)...")
        
        def get_recipe_from_temp(t):
            if 120 <= t < 135: return 1
            elif 135 <= t < 150: return 2
            elif 150 <= t <= 160: return 3
            else: return 1 
            
        df['recipe_id'] = df['temp'].apply(get_recipe_from_temp)

        # 3. ASSIGN RANDOM DELTA T (Independent)
        print("--> Assigning Random Delta T (30, 45, 60)...")
        df['delta_t'] = np.random.choice([30, 45, 60], size=len(df))

        # 4. TIME CONVERSION (Seconds -> Minutes)
        print("--> Converting p_i, r_i, m_i, c_i to Minutes...")
        
        # Convert Processing Time
        if 'p_i' in df.columns:
            df['p_i'] = df['p_i'] / 60.0
        
        # Convert Arrival Time
        if 'r_i' in df.columns:
            df['r_i'] = df['r_i'] / 60.0

        # Convert Manual Time (m_i)
        if 'm_i' in df.columns:
            df['m_i'] = df['m_i'] / 60.0
            
        # Convert Curing/Cooling/Other Time (c_i) - Renamed from b_i
        if 'c_i' in df.columns:
            df['c_i'] = df['c_i'] / 60.0

        # 5. Select Columns (Updated b_i -> c_i)
        cols_to_keep = [
            'item_id', 'w_i', 'l_i', 'h_i', 
            'm_i', 'c_i', 'comp_level', 'paint_used', 'area', 'temp',
            'p_i', 'r_i', 'recipe_id', 'delta_t'
        ]
        
        # Ensure all columns exist (fill 0 if missing in Excel)
        for col in cols_to_keep:
            if col not in df.columns:
                df[col] = 0
                
        df_final = df[cols_to_keep].copy()

        # 6. Bulk Upload
        df_final.to_sql('items', con=engine, if_exists='append', index=False)
        print(f"--> SUCCESS: Loaded {len(df_final)} items.")
        
        # Preview
        print(df_final[['item_id', 'temp', 'recipe_id', 'delta_t', 'p_i', 'm_i', 'c_i']].head(10))

    except Exception as e:
        print(f"--> ERROR loading file: {e}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    init_database()
    load_items_from_file()
    print("--> Database setup complete.")