import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import os

# --- 1. CONFIGURATION ---
DB_USER = 'root'
DB_PASSWORD = ''          # <--- Update with your actual password
DB_HOST = 'localhost'
DB_NAME = 'grad_project'

# Thermal Constants (Minutes)
DELTA_HEAT = 15   # Time to raise temperature
DELTA_COOL = 30   # Time to lower temperature
AMBIENT_TEMP = 25 # Initial state (Job 0)

# Connect to DB
connection_string = f'mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}'
engine = create_engine(connection_string)

def patch_database_schema():
    """Ensures the recipes table in DB has temperature data."""
    print("--> Checking Database Schema (Recipes)...")
    with engine.connect() as conn:
        try:
            result = conn.execute(text("SHOW COLUMNS FROM recipes LIKE 'temperature_c'"))
            if result.fetchone():
                print("   [OK] 'recipes' table has temperature data.")
            else:
                print("   [FIX] Adding missing 'temperature_c' column...")
                conn.execute(text("ALTER TABLE recipes ADD COLUMN temperature_c INT DEFAULT 160"))
                # Aligning temperatures with your new Recipe categories
                conn.execute(text("UPDATE recipes SET temperature_c = 160 WHERE recipe_id = 1")) # Low-Temp
                conn.execute(text("UPDATE recipes SET temperature_c = 180 WHERE recipe_id = 2")) # Mid-Temp
                conn.execute(text("UPDATE recipes SET temperature_c = 210 WHERE recipe_id = 3")) # High-Temp
                conn.commit()
        except Exception as e:
            print(f"   [ERROR] Database patch failed: {e}")

def generate_module4_csv():
    print("\n--> Generating Module 4 Data (Output -> CSV)...")

    # --- STEP 1: READ INPUT (Updated to look for Global Solution) ---
    # prioritized list of source files
    # UPDATED: Check for Optimal, then GRASP, then Tuning outputs
    sources = ['mod3_optimal_solution.csv', 'mod3_grasp_solution.csv', 'mod3_tuning.csv']
    
    file_to_load = None
    for src in sources:
        if os.path.exists(src):
            file_to_load = src
            break

    if not file_to_load:
        print("--> [ERROR] No input CSV found. Please ensure a solution file (mod3_*.csv) is in the folder.")
        return

    print(f"--> [SOURCE] Reading: {file_to_load}")
    df_batches = pd.read_csv(file_to_load)
    
    if df_batches.empty:
        print("--> [ERROR] Input CSV is empty.")
        return

    # --- STEP 2: GET RECIPES (From DB) ---
    with engine.connect() as conn:
        df_recipes = pd.read_sql("SELECT recipe_id, temperature_c FROM recipes", conn)
        recipe_temps = df_recipes.set_index('recipe_id')['temperature_c'].to_dict()

    # --- STEP 3: GENERATE JOBS ---
    jobs_list = []
    
    # Dummy Job 0: Ambient state
    # UPDATED: Added m_j=0, c_j=0 for dummy job
    jobs_list.append({
        'job_id': 0, 'p_j': 0, 'r_j': 0, 'recipe_id': 0, 
        'temp_c': AMBIENT_TEMP, 'item_count': 0,
        'm_j': 0, 'c_j': 0
    })

    unique_batches = sorted(df_batches['Batch'].unique())
    print(f"--> Processing {len(unique_batches)} batches...")

    for b_id in unique_batches:
        batch_items = df_batches[df_batches['Batch'] == b_id]
        
        # Aggregate batch properties
        # p_j: Max processing time in batch
        # r_j: Max arrival time (batch ready when last item arrives)
        # m_j: Max manual time (assuming concurrent prep or bottleneck item)
        # c_j: Max cooling time
        p_j = batch_items['p_i'].max()
        r_j = batch_items['r_i'].max()
        
        # Check if columns exist, else default to 0
        m_j = batch_items['m_i'].max() if 'm_i' in batch_items.columns else 0
        c_j = batch_items['c_i'].max() if 'c_i' in batch_items.columns else 0
        
        rec_id = batch_items['Recipe'].iloc[0]
        temp = recipe_temps.get(rec_id, 180) 
        
        jobs_list.append({
            'job_id': int(b_id),
            'p_j': float(p_j),
            'r_j': float(r_j),
            'recipe_id': int(rec_id),
            'temp_c': float(temp),
            'item_count': len(batch_items),
            'm_j': float(m_j),
            'c_j': float(c_j)
        })

    df_jobs = pd.DataFrame(jobs_list)

    # --- STEP 4: GENERATE SETUP MATRIX ---
    setup_list = []
    all_jobs = df_jobs['job_id'].tolist()

    for i in all_jobs:
        for j in all_jobs:
            if i == j:
                s_ij = 0
            else:
                temp_i = df_jobs.loc[df_jobs['job_id'] == i, 'temp_c'].values[0]
                temp_j = df_jobs.loc[df_jobs['job_id'] == j, 'temp_c'].values[0]
                
                # Logic for heating/cooling transition times
                if temp_j > temp_i: s_ij = DELTA_HEAT 
                elif temp_j < temp_i: s_ij = DELTA_COOL
                else: s_ij = 0 
            
            setup_list.append({'from_job': int(i), 'to_job': int(j), 'setup_time': float(s_ij)})

    df_setup = pd.DataFrame(setup_list)

    # --- STEP 5: SAVE TO CSV ---
    df_jobs.to_csv('mod4_jobs.csv', index=False)
    df_setup.to_csv('mod4_setup.csv', index=False)
    
    print(f"\n--> [SUCCESS] Files Saved: mod4_jobs.csv and mod4_setup.csv")
    # Preview
    print(df_jobs[['job_id', 'p_j', 'r_j', 'm_j', 'c_j', 'temp_c']].head())

if __name__ == "__main__":
    patch_database_schema() 
    generate_module4_csv()