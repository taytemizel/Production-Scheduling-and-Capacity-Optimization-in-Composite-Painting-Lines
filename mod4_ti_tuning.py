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
FIXED_LAMBDA = 20      

TEMP_SCENARIOS = [100, 500, 1000, 1500, 2000, 5000, 10000]
FINAL_TEMP = 1.0
COOLING_RATE = 0.95   
ITERATIONS = 100
NUM_TRIALS = 5         

def load_data():
    if not os.path.exists(INPUT_JOBS) or not os.path.exists(INPUT_SETUP):
        print(f"CRITICAL ERROR: Input files missing.")
        return None, None
        
    df_jobs = pd.read_csv(INPUT_JOBS)
    
    # --- SAFETY: Force numeric columns to float ---
    # UPDATED: Added m_j and c_j to the float conversion list to match input schema
    cols_to_float = ['p_j', 'r_j', 'temp_c', 'm_j', 'c_j']
    for col in cols_to_float:
        if col in df_jobs.columns:
            df_jobs[col] = df_jobs[col].astype(float)
            
    df_jobs_active = df_jobs[df_jobs['job_id'] != 0].copy()
    jobs_dict = df_jobs_active.set_index('job_id').to_dict('index')
    
    df_setup = pd.read_csv(INPUT_SETUP)
    setup_lookup = df_setup.set_index(['from_job', 'to_job'])['setup_time'].to_dict()
    
    return jobs_dict, setup_lookup

def load_lpt_solution():
    if not os.path.exists(INPUT_LPT_SOL):
        print(f"ERROR: LPT Solution file '{INPUT_LPT_SOL}' not found.")
        return None
    df = pd.read_csv(INPUT_LPT_SOL)
    sol = {1: [], 2: []}
    # Sort to ensure we reconstruct the sequence order correctly
    df_sorted = df.sort_values(by=['assigned_oven', 'start_time'])
    for _, row in df_sorted.iterrows():
        oven = int(row['assigned_oven'])
        job = int(row['job_id'])
        if oven in sol: sol[oven].append(job)
    return sol

def calculate_cost(sol, jobs_dict, setup_lookup):
    makespans = []
    total_penalty = 0
    for oven_id, sequence in sol.items():
        time_tracker = 0
        prev_job = 0 
        for job_id in sequence:
            job = jobs_dict[job_id]
            setup = setup_lookup.get((prev_job, job_id), 0)
            
            # Start time is max of (current time + setup) OR (release time)
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
    new_sol = copy.deepcopy(sol)
    ovens = list(new_sol.keys())
    move_type = random.random()
    
    # Move Operator 1: Reinsert (Shift job to another oven)
    if move_type < 0.33: 
        src, dest = random.sample(ovens, 2)
        if new_sol[src]:
            job = new_sol[src].pop(random.randint(0, len(new_sol[src])-1))
            new_sol[dest].insert(random.randint(0, len(new_sol[dest])), job)
            
    # Move Operator 2: Swap (Swap two jobs in same oven)
    elif move_type < 0.66: 
        oven = random.choice(ovens)
        if len(new_sol[oven]) >= 2:
            i, j = random.sample(range(len(new_sol[oven])), 2)
            new_sol[oven][i], new_sol[oven][j] = new_sol[oven][j], new_sol[oven][i]
            
    # Move Operator 3: Reverse (Reverse a sub-segment)
    else: 
        oven = random.choice(ovens)
        if len(new_sol[oven]) >= 2:
            i, j = sorted(random.sample(range(len(new_sol[oven])), 2))
            new_sol[oven][i:j+1] = reversed(new_sol[oven][i:j+1])
            
    return new_sol

if __name__ == "__main__":
    print(f"\n{'='*105}")
    print(f" IMPROVING LPT SOLUTION WITH SIMULATED ANNEALING")
    print(f"{'='*105}")
    
    start_time_total = time.time()
    jobs_dict, setup_lookup = load_data()
    lpt_sol = load_lpt_solution()
    
    if jobs_dict is None or lpt_sol is None: exit()

    base_cost, base_mk, base_pen, _ = calculate_cost(lpt_sol, jobs_dict, setup_lookup)
    
    print(f"Baseline (LPT) Cost: {base_cost:.2f} | Makespan: {base_mk:.2f} | Penalty: {base_pen:.0f}")
    print("-" * 105)
    print(f"{'Temp':<8} | {'Avg MK':<10} | {'Best MK':<10} | {'Avg Pen':<10} | {'Best Pen':<10} | {'Imp. MK(%)':<12} | {'Avg Time(s)':<10}")
    print("-" * 105)
    
    for temp in TEMP_SCENARIOS:
        mk_results = []
        pen_results = []
        trial_durations = [] 
        
        for i in range(NUM_TRIALS):
            trial_start = time.time()
            current_sol = copy.deepcopy(lpt_sol)
            curr_cost, _, _, _ = calculate_cost(current_sol, jobs_dict, setup_lookup)
            
            best_trial_cost = curr_cost
            _, best_trial_mk, best_trial_pen, _ = calculate_cost(current_sol, jobs_dict, setup_lookup)
            
            T = temp
            while T > FINAL_TEMP:
                for _ in range(ITERATIONS):
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
                T *= COOLING_RATE
            
            trial_durations.append(time.time() - trial_start)
            mk_results.append(best_trial_mk)
            pen_results.append(best_trial_pen)

        avg_mk = sum(mk_results) / NUM_TRIALS
        best_mk_of_all = min(mk_results)
        avg_pen = sum(pen_results) / NUM_TRIALS
        best_pen_of_all = min(pen_results)
        avg_trial_time = sum(trial_durations) / NUM_TRIALS
        imp_mk = ((base_mk - best_mk_of_all) / base_mk) * 100
        
        print(f"{temp:<8} | {avg_mk:<10.0f} | {best_mk_of_all:<10.0f} | {avg_pen:<10.0f} | {best_pen_of_all:<10.0f} | {imp_mk:<12.2f} | {avg_trial_time:<10.2f}")
        
    runtime = time.time() - start_time_total
    print("-" * 105)
    print(f"--> Total Tuning Run Time: {runtime:.4f} seconds")
    print(f"{'='*105}")