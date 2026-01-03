import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from sqlalchemy import create_engine
import time

# --- 1. CONFIGURATION ---
DB_USER = 'root'
DB_PASSWORD = ''          # <--- Update with your password
DB_HOST = 'localhost'
DB_NAME = 'grad_project'

# --- GLOBAL OPTIMIZATION SETTINGS ---
MAX_BATCHES = 8             
TIME_LIMIT = 1200           # 20 Minutes Max
MIP_GAP = 0.01              # 1% Gap

# CONSTRAINTS
W_MAX = 225                # Max Waiting Time

# OUTPUT
OUTPUT_CSV = 'mod3_optimal_solution.csv'

# Connect to DB
connection_string = f'mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}'
engine = create_engine(connection_string)

# --- HELPER CLASS FOR SHELF PACKING (POST-PROCESS) ---
class Shelf:
    def __init__(self, oven_id, shelf_id, width, length):
        self.oven_id = oven_id
        self.shelf_id = shelf_id
        self.width = width
        self.length = length
        self.items = []
        self.used_area = 0

    def can_fit(self, item_w, item_l):
        current_area = self.used_area
        item_area = item_w * item_l
        if (current_area + item_area) <= (self.width * self.length):
            return True
        return False

    def add_item(self, item):
        self.items.append(item)
        self.used_area += (item['w_i'] * item['l_i'])

def solve_global():
    print(f"\n{'='*60}")
    print(f" MODULE 3: GLOBAL OPTIMIZATION (Min Batches + Min Duration)")
    print(f"{'='*60}")
    
    # --- LOAD DATA ---
    with engine.connect() as conn:
        df_items = pd.read_sql("SELECT * FROM items", conn)
        df_shelves = pd.read_sql("SELECT * FROM shelves", conn)
        df_recipes = pd.read_sql("SELECT * FROM recipes", conn)

    # Pre-process
    df_items['area'] = df_items['w_i'] * df_items['l_i']
    
    # Fix Dictionary Key Issue
    # Ensure all columns including m_i, c_i are carried over
    all_items = df_items.set_index('item_id').to_dict('index')
    for iid, data in all_items.items():
        data['item_id'] = iid
        # Ensure defaults if columns are missing (though input.py guarantees them)
        if 'm_i' not in data: data['m_i'] = 0.0
        if 'c_i' not in data: data['c_i'] = 0.0
        
    item_ids = df_items['item_id'].tolist()
    
    # Calculate Oven Capacities
    oven_caps = {}
    oven_shelves_dims = {} 
    
    for oven_id, group in df_shelves.groupby('oven_id'):
        total_area = (group['width_cm'] * group['length_cm']).sum()
        oven_caps[oven_id] = total_area
        
        dims = {}
        for _, row in group.iterrows():
            dims[row['shelf_id']] = (row['width_cm'], row['length_cm'])
        oven_shelves_dims[oven_id] = dims
        
    oven_ids = list(oven_caps.keys())
    recipe_ids = df_recipes['recipe_id'].unique().tolist()
    batches = list(range(1, MAX_BATCHES + 1))
    
    print(f"--> Scope: {len(item_ids)} Items into Max {MAX_BATCHES} Batches")

    # --- MODEL SETUP ---
    m = gp.Model("Global_Oven_Scheduling")
    m.setParam('TimeLimit', TIME_LIMIT)
    m.setParam('MIPGap', MIP_GAP)
    m.setParam('OutputFlag', 1)
    
    # --- VARIABLES ---
    x = m.addVars(item_ids, batches, vtype=GRB.BINARY, name="Assign")
    y = m.addVars(batches, vtype=GRB.BINARY, name="BatchUsed")
    z = m.addVars(batches, oven_ids, vtype=GRB.BINARY, name="BatchOven")
    r_var = m.addVars(batches, recipe_ids, vtype=GRB.BINARY, name="BatchRecipe")
    duration = m.addVars(batches, vtype=GRB.CONTINUOUS, lb=0, name="Duration")
    start_time = m.addVars(batches, vtype=GRB.CONTINUOUS, lb=0, name="Start")
    
    # --- OBJECTIVE (UPDATED) ---
    # 1. Minimize Batches (Priority 1 - High Weight)
    batch_cost = gp.quicksum(y[k] * (1000 + (MAX_BATCHES - k)) for k in batches)
    
    # 2. Minimize Duration (Priority 2 - Small Weight)
    # This forces the solver to tighten the duration to the minimum possible (max p_i)
    duration_cost = gp.quicksum(duration[k] for k in batches) * 0.01
    
    m.setObjective(batch_cost + duration_cost, GRB.MINIMIZE)
    
    # --- CONSTRAINTS ---
    for i in item_ids:
        m.addConstr(gp.quicksum(x[i, k] for k in batches) == 1, name=f"Pack_{i}")
        
    for k in batches:
        for i in item_ids:
            m.addConstr(x[i, k] <= y[k])
            
    for k in batches:
        m.addConstr(gp.quicksum(z[k, o] for o in oven_ids) == y[k])
        m.addConstr(gp.quicksum(r_var[k, r] for r in recipe_ids) == y[k])

    for k in batches:
        for i in item_ids:
            target_recipe = all_items[i]['recipe_id']
            m.addConstr(x[i, k] <= r_var[k, target_recipe])
            
    # Area Constraint (Relaxed)
    for k in batches:
        total_item_area = gp.quicksum(x[i, k] * all_items[i]['area'] for i in item_ids)
        assigned_capacity = gp.quicksum(z[k, o] * oven_caps[o] for o in oven_ids)
        m.addConstr(total_item_area <= assigned_capacity * 0.99, name=f"Cap_{k}")

    # Temporal
    M_time = 10000 
    for k in batches:
        for i in item_ids:
            # Duration must be at least the processing time of any item in the batch
            m.addConstr(duration[k] >= all_items[i]['p_i'] * x[i, k])
            
            # Duration must not exceed overcure limit (p_i + delta_t)
            m.addConstr(duration[k] <= all_items[i]['p_i'] + all_items[i]['delta_t'] + M_time*(1 - x[i, k]))
            
            m.addConstr(start_time[k] >= all_items[i]['r_i'] * x[i, k])
            m.addConstr(start_time[k] <= all_items[i]['r_i'] + W_MAX + M_time*(1 - x[i, k]))

    # Symmetry Breaking
    for k in range(1, MAX_BATCHES):
        m.addConstr(y[k] >= y[k+1])

    # --- SOLVE ---
    start_cpu = time.time()
    m.optimize()
    runtime = time.time() - start_cpu
    
    # --- POST-PROCESSING & REPORTING ---
    if m.SolCount > 0:
        print(f"\n{'='*60}")
        print(f" GLOBAL OPTIMIZATION RESULTS (Time: {runtime:.2f}s)")
        print(f"{'='*60}")
        
        final_results = []
        batch_util_stats = []
        
        for k in batches:
            if y[k].X > 0.5:
                # 1. Identify Batch Config
                used_oven = next(o for o in oven_ids if z[k, o].X > 0.5)
                used_recipe = next(r for r in recipe_ids if r_var[k, r].X > 0.5)
                
                # 2. Get Items Assigned
                batch_item_ids = [i for i in item_ids if x[i, k].X > 0.5]
                batch_items_data = [all_items[i] for i in batch_item_ids]
                
                # Sort descending by Area
                batch_items_data.sort(key=lambda x: x['area'], reverse=True)
                
                # 3. Simulate Shelf Packing
                shelves = []
                s_dims = oven_shelves_dims[used_oven]
                sorted_shelf_ids = sorted(s_dims.keys()) 
                
                for s_id in sorted_shelf_ids:
                    w, l = s_dims[s_id]
                    shelves.append(Shelf(used_oven, s_id, w, l))
                
                for item in batch_items_data:
                    placed = False
                    for s in shelves:
                        if s.can_fit(item['w_i'], item['l_i']):
                            s.add_item(item)
                            final_results.append({
                                'Batch': k,
                                'Oven': used_oven,
                                'Shelf': s.shelf_id,
                                'Item_ID': item['item_id'],
                                'Recipe': used_recipe,
                                'p_i': item['p_i'],
                                'r_i': item['r_i'],
                                'delta_t': item['delta_t'],
                                # UPDATED: Include m_i and c_i in output
                                'm_i': item['m_i'],
                                'c_i': item['c_i'],
                                'Area': item['area'],
                                'Batch_Duration': duration[k].X
                            })
                            placed = True
                            break
                    if not placed:
                        print(f"Warning: Item {item['item_id']} overflowed in Batch {k} (Area Relaxation Limit Reached)")
                        shelves[-1].add_item(item) 
                        final_results.append({
                                'Batch': k,
                                'Oven': used_oven,
                                'Shelf': shelves[-1].shelf_id,
                                'Item_ID': item['item_id'],
                                'Recipe': used_recipe,
                                'p_i': item['p_i'],
                                'r_i': item['r_i'],
                                'delta_t': item['delta_t'],
                                # UPDATED: Include m_i and c_i in output
                                'm_i': item['m_i'],
                                'c_i': item['c_i'],
                                'Area': item['area'],
                                'Batch_Duration': duration[k].X
                        })

                # 4. Utilization Stats
                total_batch_area = sum(it['area'] for it in batch_items_data)
                oven_cap = oven_caps[used_oven]
                batch_fill = (total_batch_area / oven_cap) * 100
                batch_util_stats.append(batch_fill)
                
                print(f"\nBatch {k} [Oven {used_oven}] (Fill: {batch_fill:.2f}%, Duration: {duration[k].X:.2f} min):")
                for s in shelves:
                    shelf_area = s.width * s.length
                    s_util = (s.used_area / shelf_area) * 100
                    count = len(s.items)
                    status = f"{count} items" if count > 0 else "EMPTY"
                    print(f"  Shelf {s.shelf_id}: {s_util:6.2f}% Utilized | {status}")

        avg_util = sum(batch_util_stats) / len(batch_util_stats) if batch_util_stats else 0
        total_batches = int(sum(y[k].X for k in batches))
        
        print(f"\nTotal Batches: {total_batches}")
        print(f"Average Batch Utilization: {avg_util:.2f}%")
        
        df = pd.DataFrame(final_results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"--> Solution saved to {OUTPUT_CSV}")
        
    else:
        print("--> No solution found.")

if __name__ == "__main__":
    solve_global()