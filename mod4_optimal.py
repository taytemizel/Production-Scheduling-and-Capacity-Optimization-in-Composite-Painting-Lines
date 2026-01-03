import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import time
import os

# --- CONFIGURATION ---
INPUT_JOBS = 'mod4_jobs.csv'
INPUT_SETUP = 'mod4_setup.csv'
OUTPUT_SOL = 'mod4_optimal_solution.csv'
OUTPUT_PLOT = 'mod4_optimal_plot.png'

# Constraints
W_MAX = 225
LAMBDA_WAIT = 20  
TIME_LIMIT = 600  # 10 Minutes

def solve_and_visualize():
    print(f"\n{'='*60}")
    print(" PHASE 1: GLOBAL OPTIMIZATION (MIP)")
    print(f"{'='*60}")
    
    # --- 1. LOAD DATA ---
    if not os.path.exists(INPUT_JOBS) or not os.path.exists(INPUT_SETUP):
        print("Error: Input CSVs (mod4_jobs.csv, mod4_setup.csv) not found.")
        return

    df_jobs = pd.read_csv(INPUT_JOBS)
    real_jobs = df_jobs[df_jobs['job_id'] != 0]['job_id'].tolist()
    jobs_data = df_jobs.set_index('job_id').to_dict('index')
    
    df_setup = pd.read_csv(INPUT_SETUP)
    setup_time = df_setup.set_index(['from_job', 'to_job'])['setup_time'].to_dict()
    
    nodes = [0] + real_jobs
    ovens = [1, 2]

    # --- 2. BUILD GUROBI MODEL ---
    m = gp.Model("Global_Scheduling")
    m.setParam('TimeLimit', TIME_LIMIT)
    m.setParam('MIPGap', 0.0) 
    m.setParam('OutputFlag', 1) 

    print(f"--> Building model for {len(real_jobs)} batches...")

    # Variables
    # Optimize: Only create x for valid transitions (i != j)
    x_indices = [(i, j, k) for i in nodes for j in nodes for k in ovens if i != j]
    x = m.addVars(x_indices, vtype=GRB.BINARY, name="x")
    
    c = m.addVars(real_jobs, vtype=GRB.CONTINUOUS, lb=0, name="C")
    s_start = m.addVars(real_jobs, vtype=GRB.CONTINUOUS, lb=0, name="S")
    pen = m.addVars(real_jobs, vtype=GRB.CONTINUOUS, lb=0, name="Penalty")
    c_max = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="Makespan")

    # Objective: Min (Makespan + Total Penalty)
    m.setObjective(c_max + gp.quicksum(pen[j] for j in real_jobs), GRB.MINIMIZE)

    # --- Constraints ---

    # 1. Assignment: Each real job visited exactly once
    for j in real_jobs:
        m.addConstr(gp.quicksum(x[i, j, k] for k in ovens for i in nodes if i != j) == 1, name=f"Assign_{j}")

    # 2. Flow Conservation
    for k in ovens:
        # Dummy Start: At most one path starts from 0 per oven
        m.addConstr(gp.quicksum(x[0, j, k] for j in real_jobs) <= 1, name=f"Start_{k}")
        
        # Conservation at each job node
        for j in real_jobs:
            m.addConstr(
                gp.quicksum(x[i, j, k] for i in nodes if i != j) == 
                gp.quicksum(x[j, l, k] for l in nodes if l != j),
                name=f"Flow_{j}_{k}"
            )

    # 3. Sequencing & Timing (Indicator Constraints)
    # Replaced Big-M with Indicator constraints for numerical stability
    for k in ovens:
        for i in nodes:
            for j in real_jobs:
                if i == j: continue
                
                s_ij = setup_time.get((i, j), 0)
                
                if i == 0:
                    # If moving from Start (0) -> Job j
                    # s_start[j] >= 0 + s_0j
                    m.addConstr((x[0, j, k] == 1) >> (s_start[j] >= s_ij), name=f"Seq_0_{j}_{k}")
                else:
                    # If moving from Job i -> Job j
                    # s_start[j] >= Completion[i] + s_ij
                    m.addConstr((x[i, j, k] == 1) >> (s_start[j] >= c[i] + s_ij), name=f"Seq_{i}_{j}_{k}")

    # 4. Release & Completion
    for j in real_jobs:
        r_j = jobs_data[j]['r_j']
        p_j = jobs_data[j]['p_j']
        
        m.addConstr(s_start[j] >= r_j, name=f"Release_{j}")
        m.addConstr(c[j] == s_start[j] + p_j, name=f"Comp_{j}")
        m.addConstr(c_max >= c[j], name=f"Makespan_{j}")
        m.addConstr(pen[j] >= (s_start[j] - r_j - W_MAX) * LAMBDA_WAIT, name=f"Pen_{j}")

    # --- 3. SOLVE ---
    start_clock = time.time()
    m.optimize()
    runtime = time.time() - start_clock

    # --- 4. EXTRACT RESULTS ---
    if m.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and m.SolCount > 0:
        schedule_results = []
        for k in ovens:
            curr = 0
            # Reconstruct path
            while True:
                next_node = None
                for j in real_jobs:
                    if (curr, j, k) in x and x[curr, j, k].X > 0.5:
                        next_node = j
                        break
                
                if next_node is None:
                    break
                
                j = next_node
                schedule_results.append({
                    'job_id': j,
                    'assigned_oven': k,
                    'start_time': s_start[j].X,
                    'end_time': c[j].X,
                    'penalty_cost': pen[j].X
                })
                curr = next_node
                if curr == 0: break # Safety break for TSP loops

        df_res = pd.DataFrame(schedule_results)
        df_res = df_res.sort_values(by=['assigned_oven', 'start_time'])
        df_res.to_csv(OUTPUT_SOL, index=False)
        
        # --- PRINT SCHEDULE ---
        print(f"\n{'='*60}")
        print(f" FINAL SCHEDULE (Makespan: {m.ObjVal:.2f} | Time: {runtime:.2f}s)")
        print(f"{'='*60}")
        print_df = df_res.copy()
        print_df['job_id'] = print_df['job_id'].astype(int)
        print_df['oven'] = print_df['assigned_oven'].astype(int)
        print_df = print_df[['oven', 'job_id', 'start_time', 'end_time', 'penalty_cost']]
        print(print_df.to_string(index=False))
        print(f"\n--> Schedule saved to {OUTPUT_SOL}")

        # --- 5. VISUALIZATION ---
        visualize_chart(df_res, df_jobs)
    else:
        print("No solution found.")

def visualize_chart(df_sol, df_jobs):
    print(f"\n{'='*60}")
    print(" PHASE 2: VISUALIZATION")
    print(f"{'='*60}")

    # Prepare Data
    df_sol['job_id'] = df_sol['job_id'].astype(int)
    df_jobs['job_id'] = df_jobs['job_id'].astype(int)
    df = pd.merge(df_sol, df_jobs[['job_id', 'r_j']], on='job_id', how='left')
    
    df['alias'] = df.apply(lambda r: f"{'A' if r['assigned_oven']==1 else 'B'}{int(r['job_id'])}", axis=1)
    
    makespan = df['end_time'].max()
    
    # Plot
    fig, ax = plt.subplots(figsize=(18, 10))
    colors = plt.cm.tab10.colors 
    ovens = sorted(df['assigned_oven'].unique())
    row_height = 40
    label_width = makespan * 0.05
    
    y_ticks, y_labels = [], []
    
    for i, oven in enumerate(ovens):
        oven_data = df[df['assigned_oven'] == oven].sort_values('start_time')
        y_center = (len(ovens) - 1 - i) * row_height
        y_ticks.append(y_center)
        y_labels.append(f"OVEN {oven}")
        
        ax.axhspan(y_center - 15, y_center + 15, facecolor='whitesmoke', alpha=0.5)
        ax.axhline(y_center, color='gray', linestyle=':', alpha=0.3)
        
        level_tracks = {}
        
        for _, row in oven_data.iterrows():
            start, end = row['start_time'], row['end_time']
            dur, r_j, alias = end - start, row['r_j'], row['alias']
            
            # Bar
            bar_color = colors[int(row['job_id']) % 10]
            ax.barh(y_center, dur, left=start, height=10, align='center', 
                    color=bar_color, edgecolor='black', alpha=0.9)
            ax.text(start + dur/2, y_center, alias, ha='center', va='center', 
                    color='white', fontweight='bold', fontsize=20)
            
            # Marker
            marker_y = y_center - 8
            ax.scatter(r_j, marker_y, color='red', marker='D', s=40, zorder=5, edgecolors='black')
            
            # Label
            lvl = 1
            max_lvl = 10 
            while lvl < max_lvl:
                last_x = level_tracks.get(lvl, -999)
                if r_j > (last_x + label_width/2): break
                lvl += 1
            level_tracks[lvl] = r_j + label_width
            
            ty = marker_y - 2 - (lvl * 4)
            ax.plot([r_j, r_j], [marker_y, ty+1], color='red', lw=0.8, alpha=0.6)
            ax.text(r_j, ty, alias, ha='center', va='center', color='darkred', fontsize=12, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_title(f"Optimal Production Schedule (Makespan: {makespan:.2f} min)", fontsize=16, fontweight='bold', pad=20)
    
    leg_r = mlines.Line2D([], [], color='red', marker='D', linestyle='None', markersize=8, label='Release Time')
    leg_b = mpatches.Patch(facecolor='gray', edgecolor='black', label='Batch')
    ax.legend(handles=[leg_r, leg_b], loc='upper right')
    
    ax.set_xlim(-10, makespan * 1.1)
    ax.axvline(makespan, color='black', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    print(f"--> Visualization saved to {OUTPUT_PLOT}")
    plt.show()

if __name__ == "__main__":
    solve_and_visualize()