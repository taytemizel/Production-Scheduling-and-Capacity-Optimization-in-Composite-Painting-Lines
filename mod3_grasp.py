import pandas as pd
import numpy as np
import random
import time
from sqlalchemy import create_engine

# --- 1. CONFIGURATION ---
DB_USER = 'root'
DB_PASSWORD = ''
DB_HOST = 'localhost'
DB_NAME = 'grad_project'

OUTPUT_FILE = 'mod3_grasp_solution.csv'

GRASP_ITERATIONS = 50
ALPHA = 0.10
MAX_MOVE_ITERATIONS = 100

W_MAX = 225  # minutes

# --- DB CONNECTION ---
connection_string = f'mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}'
engine = create_engine(connection_string)

# --- LOAD DATA ---
def load_data():
    print("--> Loading data from MySQL...")
    with engine.connect() as conn:
        items = pd.read_sql("SELECT * FROM items", conn)
        shelves = pd.read_sql("SELECT * FROM shelves", conn)

    cols_to_float = ['w_i', 'l_i', 'h_i', 'p_i', 'r_i', 'delta_t', 'm_i', 'c_i', 'temp']
    for col in cols_to_float:
        if col in items.columns:
            items[col] = items[col].astype(float)

    for col in ['width_cm', 'length_cm', 'height_clearance_cm']:
        shelves[col] = shelves[col].astype(float)

    return items, shelves


df_items_raw, df_shelves_raw = load_data()

required_cols = ['p_i', 'r_i', 'delta_t', 'recipe_id']
missing_cols = [c for c in required_cols if c not in df_items_raw.columns]
if missing_cols:
    raise ValueError(f"Missing columns {missing_cols}")

df_items_raw['area'] = df_items_raw['w_i'] * df_items_raw['l_i']

OVEN_CAPACITIES = df_shelves_raw.groupby('oven_id').apply(
    lambda x: (x['width_cm'] * x['length_cm']).sum()
).to_dict()

# --- CLASSES ---
class Shelf:
    def __init__(self, oven_id, shelf_id, width, length, height):
        self.oven_id = oven_id
        self.shelf_id = shelf_id
        self.width = width
        self.length = length
        self.height = height
        self.placed_items = []

    def can_fit_geo(self, item, x, y):
        if x + item['w_i'] > self.width: return False
        if y + item['l_i'] > self.length: return False
        if item['h_i'] > self.height: return False

        for p in self.placed_items:
            px, py, pi = p['x'], p['y'], p['item']
            if (x < px + pi['w_i'] and x + item['w_i'] > px and
                y < py + pi['l_i'] and y + item['l_i'] > py):
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
        self.assigned_oven = None
        self.oven_configs = {}
        self.shelves = [
            Shelf(row['oven_id'], row['shelf_id'],
                  row['width_cm'], row['length_cm'], row['height_clearance_cm'])
            for _, row in df_shelves_template.iterrows()
        ]

    def get_all_items(self):
        return [p['item'] for s in self.shelves for p in s.placed_items]

    def get_duration(self):
        items = self.get_all_items()
        return max(item['p_i'] for item in items) if items else 0.0

    def try_pack(self, item):
        curr_items = self.get_all_items()
        curr_max_p = max([i['p_i'] for i in curr_items], default=0)
        curr_max_r = max([i['r_i'] for i in curr_items], default=0)

        new_duration = max(curr_max_p, item['p_i'])

        if new_duration > item['p_i'] + item['delta_t']:
            return False

        for i in curr_items:
            if new_duration > i['p_i'] + i['delta_t']:
                return False

        new_start = max(curr_max_r, item['r_i'])
        if new_start > item['r_i'] + W_MAX:
            return False

        for i in curr_items:
            if new_start > i['r_i'] + W_MAX:
                return False

        for s in self.shelves:
            if self.assigned_oven is not None and s.oven_id != self.assigned_oven:
                continue

            if s.oven_id in self.oven_configs:
                if self.oven_configs[s.oven_id] != item['recipe_id']:
                    continue

            pos = s.find_position_bottom_left(item)
            if pos:
                if self.assigned_oven is None:
                    self.assigned_oven = s.oven_id
                self.oven_configs[s.oven_id] = item['recipe_id']
                s.add_item(item, *pos)
                return True

        return False

    def remove_item(self, item_id):
        for s in self.shelves:
            if s.remove_item(item_id):
                return True
        return False

    def is_empty(self):
        return all(len(s.placed_items) == 0 for s in self.shelves)


# --- GRASP FUNCTIONS ---
def construct_solution(df_items, df_shelves):
    unassigned = df_items.sort_values(by='area', ascending=False)
    batches = []

    while not unassigned.empty:
        rcl_size = max(1, int(len(unassigned) * ALPHA))
        candidates = unassigned.iloc[:rcl_size]
        idx = random.choice(candidates.index)
        item = unassigned.loc[idx]
        unassigned = unassigned.drop(idx)

        placed = False
        for b in batches:
            if b.try_pack(item):
                placed = True
                break

        if not placed:
            new_batch = Batch(len(batches) + 1, df_shelves)
            new_batch.try_pack(item)
            batches.append(new_batch)

    return batches


def apply_move_operator(batches):
    for _ in range(MAX_MOVE_ITERATIONS):
        active = [b for b in batches if not b.is_empty()]
        if not active:
            break

        active.sort(key=lambda b: b.get_duration(), reverse=True)
        bottleneck = active[0]

        moved = False
        for item in sorted(bottleneck.get_all_items(), key=lambda x: x['area'], reverse=True):
            for target in active:
                if target.batch_id == bottleneck.batch_id:
                    continue
                if target.try_pack(item):
                    bottleneck.remove_item(item['item_id'])
                    moved = True
                    break
            if moved:
                break
        if not moved:
            break

    final = [b for b in batches if not b.is_empty()]
    for i, b in enumerate(final):
        b.batch_id = i + 1
    return final


# --- MAIN ---
if __name__ == "__main__":
    print(f"\n=== STARTING GRASP METAHEURISTIC ({GRASP_ITERATIONS} Iterations) ===")
    print("Constraints: W_max=225m, Single Oven/Batch, 4 Shelves/Oven")

    start_time = time.time()
    best_solution = None
    best_batch_count = float('inf')

    for i in range(GRASP_ITERATIONS):
        sol = apply_move_operator(construct_solution(df_items_raw, df_shelves_raw))
        if len(sol) < best_batch_count:
            print(f"   [Iter {i+1}] New Best Found: {len(sol)} batches")
            best_batch_count = len(sol)
            best_solution = sol

    total_time = time.time() - start_time

    print("\n=== GRASP COMPLETE ===")
    print(f"Time: {total_time:.2f}s")
    print(f"Final solution saved to '{OUTPUT_FILE}'")
    print(f"Best Batch Count: {best_batch_count}")

    print("\n--- Detailed Shelf Utilization ---")

    for b in best_solution:
        oven = b.assigned_oven
        duration = b.get_duration()
        used_area = sum(i['area'] for i in b.get_all_items())
        util = (used_area / OVEN_CAPACITIES[oven]) * 100

        print(
            f"Batch {b.batch_id} [Oven {oven}] "
            f"(Duration: {duration:.2f} min, Total Oven Fill: {util:.2f}%):"
        )

        for s in b.shelves:
            if s.oven_id != oven:
                continue
            shelf_area = s.width * s.length
            used = sum(p['item']['area'] for p in s.placed_items)
            shelf_util = (used / shelf_area) * 100 if shelf_area else 0
            status = f"{len(s.placed_items)} items" if s.placed_items else "EMPTY"
            print(f"  Shelf {s.shelf_id}: {shelf_util:.2f}% Util ({status})")
