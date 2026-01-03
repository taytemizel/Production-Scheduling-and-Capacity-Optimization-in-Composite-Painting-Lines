import pandas as pd
import numpy as np
import random
import time
import math
from sqlalchemy import create_engine

# --- 1. CONFIGURATION ---
DB_USER = 'root'
DB_PASSWORD = ''          # <--- Update with your password
DB_HOST = 'localhost'
DB_NAME = 'grad_project'

# TUNING SETTINGS
ALPHA_VALUES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]  # Parameters to test
ITERATIONS_PER_CONFIG = 50               # How many GRASP runs per Alpha
MAX_MOVE_ITERATIONS = 100                # Max attempts for Move Operator

# COMPATIBILITY CONSTRAINTS
W_MAX = 225  # Max Waiting Time (minutes)

# Output File
OUTPUT_FILE = 'mod3_tuning.csv'

# Connect to DB
connection_string = f'mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}'
engine = create_engine(connection_string)

def load_data():
    print("--> Loading data from MySQL...")
    with engine.connect() as conn:
        items = pd.read_sql("SELECT * FROM items", conn)
        shelves = pd.read_sql("SELECT * FROM shelves", conn)
    
    # --- SAFETY: Force numeric columns to float ---
    # This prevents errors if MySQL returns 'Decimal' objects
    # UPDATED: Added m_i, c_i, temp to match mod3_input.py schema
    cols_to_float = ['w_i', 'l_i', 'h_i', 'p_i', 'r_i', 'delta_t', 'm_i', 'c_i', 'temp']
    for col in cols_to_float:
        if col in items.columns:
            items[col] = items[col].astype(float)
            
    shelf_float_cols = ['width_cm', 'length_cm', 'height_clearance_cm']
    for col in shelf_float_cols:
        if col in shelves.columns:
            shelves[col] = shelves[col].astype(float)
            
    return items, shelves

# --- 2. DATA PRE-PROCESSING ---
df_items_raw, df_shelves_raw = load_data()

# CHECK: Ensure critical temporal columns exist
required_cols = ['p_i', 'r_i', 'delta_t']
missing_cols = [c for c in required_cols if c not in df_items_raw.columns]

if missing_cols:
    raise ValueError(f"CRITICAL ERROR: Columns {missing_cols} missing in DB. Please run input.py and connection.py first.")

# Calculate Area
df_items_raw['area'] = df_items_raw['w_i'] * df_items_raw['l_i']

# Pre-calculate Capacities
SYS_CAP = (df_shelves_raw['width_cm'] * df_shelves_raw['length_cm']).sum()
OVEN_CAPACITIES = df_shelves_raw.groupby('oven_id').apply(
    lambda x: (x['width_cm'] * x['length_cm']).sum()
).to_dict()

# --- 3. CLASSES ---

class Shelf:
    def __init__(self, oven_id, shelf_id, width, length, height):
        self.oven_id = oven_id
        self.shelf_id = shelf_id
        self.width = width
        self.length = length
        self.height = height
        self.placed_items = [] 

    def can_fit_geo(self, item, x, y):
        # Boundary Check
        if x + item['w_i'] > self.width: return False
        if y + item['l_i'] > self.length: return False
        if item['h_i'] > self.height: return False

        # Overlap Check
        for p in self.placed_items:
            px, py, pi = p['x'], p['y'], p['item']
            if (x < px + pi['w_i']) and (x + item['w_i'] > px) and \
               (y < py + pi['l_i']) and (y + item['l_i'] > py):
                return False
        return True

    def find_position_bottom_left(self, item):
        candidates = [(0, 0)]
        for p in self.placed_items:
            candidates.append((p['x'] + p['item']['w_i'], p['y']))
            candidates.append((p['x'], p['y'] + p['item']['l_i']))
        candidates.sort(key=lambda pos: (pos[1], pos[0]))
        
        for x, y in candidates:
            if self.can_fit_geo(item, x, y):
                return x, y
        return None

    def add_item(self, item, x, y):
        self.placed_items.append({'item': item, 'x': x, 'y': y})

    def remove_item(self, item_id):
        for i, p in enumerate(self.placed_items):
            if p['item']['item_id'] == item_id:
                del self.placed_items[i]
                return True
        return False

class Batch:
    def __init__(self, batch_id, df_shelves_template):
        self.batch_id = batch_id
        self.oven_configs = {} 
        self.assigned_oven = None # Single Oven Constraint
        self.shelves = []
        for _, row in df_shelves_template.iterrows():
            self.shelves.append(Shelf(row['oven_id'], row['shelf_id'], 
                                      row['width_cm'], row['length_cm'], row['height_clearance_cm']))

    def get_stats(self):
        max_p, max_r = 0, 0
        all_items = []
        for s in self.shelves:
            for p in s.placed_items:
                all_items.append(p['item'])
                if p['item']['p_i'] > max_p: max_p = p['item']['p_i']
                if p['item']['r_i'] > max_r: max_r = p['item']['r_i']
        return max_p, max_r, all_items

    def try_pack(self, new_item):
        curr_max_p, curr_max_r, existing_items = self.get_stats()
        
        # 1. Overcure
        new_duration = max(curr_max_p, new_item['p_i'])
        if new_duration > (new_item['p_i'] + new_item['delta_t']): return False 
        for item in existing_items:
            if new_duration > (item['p_i'] + item['delta_t']): return False

        # 2. Waiting Time
        new_start_time = max(curr_max_r, new_item['r_i'])
        if new_start_time > (new_item['r_i'] + W_MAX): return False
        for item in existing_items:
            if new_start_time > (item['r_i'] + W_MAX): return False

        # 3. Spatial, Recipe & Single Oven Constraint
        for shelf in self.shelves:
            oven = shelf.oven_id
            
            # Single Oven Check
            if self.assigned_oven is not None and oven != self.assigned_oven:
                continue

            # Recipe Check
            if oven in self.oven_configs:
                if self.oven_configs[oven] != new_item['recipe_id']:
                    continue 
            
            # Height Check
            if new_item['h_i'] > shelf.height:
                continue

            pos = shelf.find_position_bottom_left(new_item)
            if pos:
                x, y = pos
                if self.assigned_oven is None:
                    self.assigned_oven = oven
                    self.oven_configs[oven] = new_item['recipe_id']
                elif oven not in self.oven_configs:
                    self.oven_configs[oven] = new_item['recipe_id']
                
                shelf.add_item(new_item, x, y)
                return True
        return False

    def remove_item(self, item_id):
        for shelf in self.shelves:
            if shelf.remove_item(item_id):
                return True
        return False

    def get_all_items(self):
        items = []
        for s in self.shelves:
            for p in s.placed_items:
                items.append(p['item'])
        return items

    def is_empty(self):
        return sum(len(s.placed_items) for s in self.shelves) == 0

# --- 4. GRASP FUNCTIONS ---

def construct_solution(df_items_source, df_shelves_source, alpha_val):
    unassigned = df_items_source.sort_values(by='area', ascending=False)
    batches = []
    
    while not unassigned.empty:
        n_candidates = max(1, int(len(unassigned) * alpha_val))
        candidates = unassigned.iloc[:n_candidates]
        
        selected_idx = random.choice(candidates.index)
        item = unassigned.loc[selected_idx]
        unassigned = unassigned.drop(selected_idx)
        
        packed = False
        for b in batches:
            if b.try_pack(item):
                packed = True
                break
        
        if not packed:
            new_batch = Batch(len(batches) + 1, df_shelves_source)
            if new_batch.try_pack(item):
                batches.append(new_batch)
            
    return batches

def apply_move_operator(batches):
    for i in range(MAX_MOVE_ITERATIONS):
        active_batches = [b for b in batches if not b.is_empty()]
        if not active_batches: break
        
        active_batches.sort(key=lambda b: b.get_stats()[0], reverse=True)
        bottleneck = active_batches[0]
        
        move_made = False
        items_to_move = bottleneck.get_all_items()
        items_to_move.sort(key=lambda x: x['area'], reverse=True)
        
        for item in items_to_move:
            for target in active_batches:
                if target.batch_id == bottleneck.batch_id: continue
                if target.try_pack(item):
                    bottleneck.remove_item(item['item_id'])
                    move_made = True
                    break 
            if move_made: break
        if not move_made: break
            
    final_batches = [b for b in batches if not b.is_empty()]
    for i, b in enumerate(final_batches):
        b.batch_id = i + 1
    return final_batches

# --- 5. MAIN TUNING LOOP ---

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f" MODULE 3: TUNING WITH SINGLE-OVEN & TEMPORAL CONSTRAINTS")
    print(f"{'='*60}")
    print(f"Constraints: W_max={W_MAX} min, Single Oven per Batch=ON")
    
    # 1. Theoretical Lower Bound (Area Only)
    total_area = df_items_raw['area'].sum()
    min_batches = math.ceil(total_area / SYS_CAP)
    
    print(f"Total Item Area:     {total_area:,.0f} cm²")
    print(f"Theoretical Area LB: {min_batches} Full System Cycles (Approx)")
    
    # 2. Tuning Grid Search
    best_overall_batches = float('inf')
    best_overall_avg = float('inf')
    best_config = None
    best_solution_obj = None
    results_log = []
    
    for alpha in ALPHA_VALUES:
        print(f"\n--- Testing Alpha = {alpha} ({ITERATIONS_PER_CONFIG} iterations) ---")
        
        current_best_batches = float('inf')
        best_sol_for_alpha = None
        sum_batches = 0
        
        for i in range(ITERATIONS_PER_CONFIG):
            sol = construct_solution(df_items_raw, df_shelves_raw, alpha)
            final_sol = apply_move_operator(sol)
            
            count = len(final_sol)
            sum_batches += count
            
            if count < current_best_batches:
                current_best_batches = count
                best_sol_for_alpha = final_sol

        avg_batches_count = sum_batches / ITERATIONS_PER_CONFIG
        
        # --- STATISTICS FOR THIS ALPHA ---
        # Note: 'Capacity Used' here assumes sequential usage of full system capacity per batch
        # Since we use Single Oven, this metric is just for relative comparison.
        capacity_used = current_best_batches * SYS_CAP
        global_utilization = (total_area / capacity_used) * 100

        batch_utils = []
        if best_sol_for_alpha:
            for b in best_sol_for_alpha:
                b_used_area = sum(item['area'] for item in b.get_all_items())
                # Use assigned oven capacity if available, else system cap (fallback)
                oven_cap = OVEN_CAPACITIES.get(b.assigned_oven, SYS_CAP) 
                if b.assigned_oven:
                    batch_utils.append((b_used_area / oven_cap) * 100)
        
        avg_batch_util = sum(batch_utils) / len(batch_utils) if batch_utils else 0.0
        
        print(f"   -> Result: Best={current_best_batches}, AvgCount={avg_batches_count:.2f}")
        print(f"      Global Util={global_utilization:.2f}%, Avg Batch Util={avg_batch_util:.2f}%")
        
        # --- DETAILED SHELF UTILIZATION FOR THIS ALPHA ---
        print(f"   -> Detailed Shelf Utilization (Best Sol for Alpha {alpha}):")
        if best_sol_for_alpha:
            for i, b in enumerate(best_sol_for_alpha):
                oven_str = f"Oven {b.assigned_oven}" if b.assigned_oven else "Unassigned"
                # Find matching util in list or 0
                b_util_disp = 0.0
                if i < len(batch_utils):
                    b_util_disp = batch_utils[i]
                
                print(f"      Batch {b.batch_id} [{oven_str}] (Fill: {b_util_disp:.2f}%):")
                
                for s in b.shelves:
                    # Filter for active oven only
                    if b.assigned_oven is not None and s.oven_id != b.assigned_oven:
                        continue
                        
                    shelf_area = s.width * s.length
                    used_area = sum(p['item']['area'] for p in s.placed_items)
                    util = (used_area / shelf_area) * 100
                    status = f"{len(s.placed_items)} items" if len(s.placed_items) > 0 else "EMPTY"
                    print(f"        Shelf {s.shelf_id}: {util:.2f}% Util ({status})")

        # Winner Logic
        is_new_winner = False
        if current_best_batches < best_overall_batches:
            is_new_winner = True
        elif current_best_batches == best_overall_batches:
            if avg_batches_count < best_overall_avg:
                is_new_winner = True
        
        if is_new_winner:
            best_overall_batches = current_best_batches
            best_overall_avg = avg_batches_count
            best_config = alpha
            best_solution_obj = best_sol_for_alpha
            print(f"      >>> NEW LEADER FOUND")
        
        results_log.append({
            'Alpha': alpha, 
            'Best': current_best_batches, 
            'AvgCount': avg_batches_count,
            'GlobalUtil': global_utilization,
            'AvgBatchUtil': avg_batch_util
        })

    # 3. Final Report
    print(f"\n{'='*60}")
    print(f" TUNING RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Alpha':<8} | {'Best':<6} | {'AvgCnt':<8} | {'GlobUtil':<10} | {'AvgBatchUtil':<12}")
    print("-" * 55)
    for res in results_log:
        print(f"{res['Alpha']:<8} | {res['Best']:<6} | {res['AvgCount']:<8.2f} | {res['GlobalUtil']:<10.2f} | {res['AvgBatchUtil']:<12.2f}")
    
    print(f"\nWINNER: Alpha = {best_config}")
    
    # 4. Save Best Solution
    if best_solution_obj:
        results = []
        for b in best_solution_obj:
            for s in b.shelves:
                for p in s.placed_items:
                    results.append({
                        'Batch': b.batch_id,
                        'Oven': s.oven_id,
                        'Shelf': s.shelf_id,
                        'Item_ID': p['item']['item_id'],
                        'Recipe': p['item']['recipe_id'],
                        'p_i': p['item']['p_i'],
                        'r_i': p['item']['r_i'],
                        'delta_t': p['item']['delta_t'],
                        # UPDATED: Included m_i and c_i in output
                        'm_i': p['item'].get('m_i', 0),
                        'c_i': p['item'].get('c_i', 0),
                        'X_Start': p['x'], 'Y_Start': p['y']
                    })
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
        print(f"\n--> Best solution saved to '{OUTPUT_FILE}'")