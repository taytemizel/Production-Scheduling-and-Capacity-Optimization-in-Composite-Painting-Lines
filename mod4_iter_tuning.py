import pandas as pd
import numpy as np
import random
import math
import time
import os
import copy

# --- CONFIGURATION ---
INPUT_JOBS = 'mod4_jobs.csv'
INPUT_SETUP = 'mod4_setup.csv'
INPUT_LPT_SOL = 'mod4_lpt_solution.csv'

W_MAX = 225

# --- FIXED PARAMETERS ---
FIXED_LAMBDA = 20      
FIXED_INITIAL_TEMP = 1000 
FIXED_ALPHA = 0.95     # <--- Updated as requested

# --- TUNING SETTINGS ---
ITER_SCENARIOS = [10, 20, 50, 100, 200, 500, 1000]
FINAL_TEMP = 1.0
NUM_TRIALS = 5         # Run each scenario 5 times

def load_data():
    if not os.path.exists(INPUT_JOBS) or not os.path.exists(INPUT_SETUP):
        print(f"CRITICAL ERROR: Input files missing.")
        return None, None

    # Load Jobs
    df_jobs = pd.read_csv(INPUT_JOBS)
    
    # --- SAFETY: Force numeric columns to float ---
    # UPDATED: Added m_j and c_j to the float conversion list to match input schema
    cols_to_float = ['p_j', 'r_j', 'temp_c', 'm_j', 'c_j']
    for col in cols_to_float:
        if col in df_jobs.columns:
            df_jobs[col] = df_jobs[col].astype(float)
            
    df_jobs_active = df_jobs[df_jobs['job_id'] != 0].copy()
    jobs_dict = df_jobs_active.set_index('job_id').to_dict('index')
    
    # Load Setup Matrix
    df_setup = pd.read_csv(INPUT_SETUP)
    setup_lookup = df_setup.set_index(['from_job', 'to_job'])['setup_time'].to_dict()
    
    return jobs_dict, setup_lookup

def load_lpt_solution():
    """Reads the LPT solution CSV and reconstructs the sequence dictionary."""
    if not os.path.exists(INPUT_LPT_SOL):
        print(f"ERROR: LPT Solution file '{INPUT_LPT_SOL}' not found. Run lpt.py first.")
        return None
        
    df = pd.read_csv(INPUT_LPT_SOL)
    
    sol = {1: [], 2: []}
    df_sorted = df.sort_values(by=['assigned_oven', 'start_time'])
    
    for _, row in df_sorted.iterrows():
        oven = int(row['assigned_oven'])
        job = int(row['job_id'])
        if oven in sol:
            sol[oven].append(job)
            
    return sol

def calculate_cost(sol, jobs_dict, setup_lookup):
    """Calculates Makespan + Penalty."""
    makespans = []
    total_penalty = 0
    
    for oven_id, sequence in sol.items():
        time_tracker = 0
        prev_job = 0 
        
        for job_id in sequence:
            job = jobs_dict[job_id]
            setup = setup_lookup.get((prev_job, job_id), 0)
            
            start_time = max(time_tracker + setup, job['r_j'])
            completion = start_time + job['p_j']
            
            wait_time = start_time - job['r_j']
            if wait_time > W_MAX:
                total_penalty += (wait_time - W_MAX) * FIXED_LAMBDA
            
            time_tracker = completion
            prev_job = job_id
            
        makespans.append(time_tracker)
    
    makespan = max(makespans) if makespans else 0
    total_cost = makespan + total_penalty
    return total_cost, makespan, total_penalty, {}

def generate_neighbor(sol):
    """Swap, Move, or Reverse."""
    new_sol = copy.deepcopy(sol)
    ovens = list(new_sol.keys())
    
    move_type = random.random()
    
    if move_type < 0.33: # Move between ovens
        src = random.choice(ovens)
        dest = random.choice(ovens)
        if new_sol[src]:
            job = new_sol[src].pop(random.randint(0, len(new_sol[src])-1))
            new_sol[dest].insert(random.randint(0, len(new_sol[dest])), job)
            
    elif move_type < 0.66: # Swap within oven
        oven = random.choice(ovens)
        if len(new_sol[oven]) >= 2:
            i, j = random.sample(range(len(new_sol[oven])), 2)
            new_sol[oven][i], new_sol[oven][j] = new_sol[oven][j], new_sol[oven][i]
            
    else: # Reverse sub-sequence
        oven = random.choice(ovens)
        if len(new_sol[oven]) >= 2:
            i, j = sorted(random.sample(range(len(new_sol[oven])), 2))
            new_sol[oven][i:j+1] = reversed(new_sol[oven][i:j+1])
            
    return new_sol

if __name__ == "__main__":
    print(f"\n{'='*100}")
    print(f" TUNING ITERATIONS PER TEMP (File Based, Alpha=0.95)")
    print(f"{'='*100}")
    
    start_time_total = time.time()
    
    jobs_dict, setup_lookup = load_data()
    lpt_sol = load_lpt_solution()
    
    if jobs_dict is None or lpt_sol is None:
        exit()

    # --- BASELINE ---
    base_cost, base_mk, base_pen, _ = calculate_cost(lpt_sol, jobs_dict, setup_lookup)
    
    print(f"Baseline (LPT) Cost: {base_cost:.2f} | Makespan: {base_mk:.2f}")
    print("-" * 100)
    print(f"{'Iters':<8} | {'Avg MK':<10} | {'Best MK':<10} | {'Avg Pen':<10} | {'Best Pen':<10} | {'Imp. MK(%)':<12} | {'Avg Time(s)':<12}")
    print("-" * 100)
    
    for iters in ITER_SCENARIOS:
        mk_results = []
        pen_results = []
        time_results = []
        
        for i in range(NUM_TRIALS):
            trial_start = time.time()
            
            # Start from LPT solution
            current_sol = copy.deepcopy(lpt_sol)
            curr_cost, _, _, _ = calculate_cost(current_sol, jobs_dict, setup_lookup)
            
            best_trial_cost = curr_cost
            _, best_trial_mk, best_trial_pen, _ = calculate_cost(current_sol, jobs_dict, setup_lookup)
            
            T = FIXED_INITIAL_TEMP
            
            while T > FINAL_TEMP:
                for _ in range(iters):
                    neighbor = generate_neighbor(current_sol)
                    new_cost, new_mk, new_pen, _ = calculate_cost(neighbor, jobs_dict, setup_lookup)
                    
                    delta = new_cost - curr_cost
                    if delta < 0 or random.random() < math.exp(-delta / T):
                        current_sol = neighbor
                        curr_cost = new_cost
                        
                        if new_cost < best_trial_cost:
                            best_trial_cost = new_cost
                            best_trial_mk = new_mk
                            best_trial_pen = new_pen
                T *= FIXED_ALPHA
            
            elapsed = time.time() - trial_start
            
            mk_results.append(best_trial_mk)
            pen_results.append(best_trial_pen)
            time_results.append(elapsed)

        # --- AGGREGATE STATS ---
        avg_mk = sum(mk_results) / NUM_TRIALS
        best_mk_of_all = min(mk_results)
        
        avg_pen = sum(pen_results) / NUM_TRIALS
        best_pen_of_all = min(pen_results)
        
        avg_time = sum(time_results) / NUM_TRIALS
        
        imp_mk = ((base_mk - best_mk_of_all) / base_mk) * 100
        
        print(f"{iters:<8} | {avg_mk:<10.0f} | {best_mk_of_all:<10.0f} | {avg_pen:<10.0f} | {best_pen_of_all:<10.0f} | {imp_mk:<12.2f} | {avg_time:<12.4f}")
        
    end_time_total = time.time()
    runtime = end_time_total - start_time_total
    
    print("-" * 100)
    print(f"--> Total Run Time: {runtime:.4f} seconds")
    print(f"{'='*100}")