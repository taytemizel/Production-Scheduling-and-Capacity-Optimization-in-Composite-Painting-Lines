import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import gurobipy as gp
from gurobipy import GRB
import time
import os

# --- CONFIGURATION ---
INPUT_JOBS = 'mod4_jobs.csv'
INPUT_SETUP = 'mod4_setup.csv'
OUTPUT_SOLUTION = 'mod4_lpt_solution.csv' 
OUTPUT_PLOT = 'mod4_lpt_plot.png'

def run_balanced_scheduler():
    print(f"\n{'='*50}")
    print(" PHASE 1: BALANCED SCHEDULING (LPT HEURISTIC)")
    print(f"{'='*50}")
    start_time = time.time()
    
    # --- 1. LOAD DATA FROM CSV ---
    if not os.path.exists(INPUT_JOBS) or not os.path.exists(INPUT_SETUP):
        print(f"CRITICAL ERROR: Input files '{INPUT_JOBS}' or '{INPUT_SETUP}' missing.")
        return False
        
    df_jobs = pd.read_csv(INPUT_JOBS)
    df_setup = pd.read_csv(INPUT_SETUP)
    
    # --- 2. FORCE 2 OVENS ---
    available_ovens = [1, 2] 
    
    # --- 3. LOAD BALANCING (LPT) ---
    active_jobs = df_jobs[df_jobs['job_id'] != 0].copy()
    if active_jobs.empty:
        print("Error: No jobs found.")
        return False
        
    # Sort Longest Processing Time first to distribute heavy loads across ovens
    active_jobs = active_jobs.sort_values(by='p_j', ascending=False)
    
    oven_loads = {oid: 0.0 for oid in available_ovens}
    job_oven_map = {}
    
    print(f"--> Distributing {len(active_jobs)} batches across 2 Ovens...")
    
    for _, job in active_jobs.iterrows():
        j_id = int(job['job_id'])
        p_j = float(job['p_j'])
        # Greedy assignment to the oven with the least current workload
        chosen_oven = min(oven_loads, key=oven_loads.get)
        job_oven_map[j_id] = chosen_oven
        oven_loads[chosen_oven] += p_j
        
    print(f"   Final Estimated Loads (Minutes): {oven_loads}")

    # --- 4. PREPARE OPTIMIZATION DATA ---
    setup_dict = {}
    for _, row in df_setup.iterrows():
        setup_dict[(int(row['from_job']), int(row['to_job']))] = row['setup_time']

    jobs = df_jobs.set_index('job_id').to_dict('index')
    
    oven_groups = {oid: [] for oid in available_ovens}
    for j_id, oven in job_oven_map.items():
        oven_groups[oven].append(j_id)
        
    final_schedule = []

    # --- 5. SOLVE SEQUENCING PER OVEN ---
    print("\n--> Optimizing Sequences with Gurobi...")
    
    for oven_id, job_ids in oven_groups.items():
        if not job_ids: continue
        print(f"   [Oven {oven_id}] Sequencing {len(job_ids)} batches...")
        
        nodes = [0] + job_ids
        m = gp.Model(f"Seq_Oven_{oven_id}")
        m.setParam('OutputFlag', 0)
        
        # Variables
        x = m.addVars(nodes, nodes, vtype=GRB.BINARY, name="x")
        c = m.addVars(nodes, vtype=GRB.CONTINUOUS, name="C")
        c_max = m.addVar(vtype=GRB.CONTINUOUS, name="C_max")
        
        m.setObjective(c_max, GRB.MINIMIZE)
        
        # Sequencing constraints
        m.addConstr(gp.quicksum(x[0, j] for j in job_ids) == 1)
        for j in job_ids:
            m.addConstr(gp.quicksum(x[i, j] for i in nodes if i != j) == 1)
            m.addConstr(gp.quicksum(x[j, k] for k in job_ids if k != j) <= 1)
            m.addConstr(c_max >= c[j])
            
        M = 100000 # Big-M for temporal linearization
        
        # Start Time Constraints (Linearized to avoid max() errors)
        for j in job_ids:
            s_0j = setup_dict.get((0, j), 0)
            p_j = jobs[j]['p_j']
            r_j = jobs[j]['r_j']
            # If j is the first job in the oven track
            m.addConstr(c[j] >= r_j + p_j - M*(1 - x[0,j]))
            m.addConstr(c[j] >= 0 + s_0j + p_j - M*(1 - x[0,j]))

        for i in job_ids:
            for j in job_ids:
                if i == j: continue
                s_ij = setup_dict.get((i, j), 0)
                p_j = jobs[j]['p_j']
                r_j = jobs[j]['r_j']
                # Sequence constraint: Completion of j depends on i + setup or j's release time
                m.addConstr(c[j] >= c[i] + s_ij + p_j - M*(1 - x[i,j]))
                m.addConstr(c[j] >= r_j + p_j - M*(1 - x[i,j]))

        m.optimize()
        
        if m.Status == GRB.OPTIMAL:
            curr = 0
            while True:
                next_node = None
                for j in job_ids:
                    if x[curr, j].X > 0.5:
                        next_node = j
                        break
                if not next_node: break
                
                final_schedule.append({
                    'job_id': next_node,
                    'assigned_oven': oven_id,
                    'start_time': c[next_node].X - jobs[next_node]['p_j'],
                    'end_time': c[next_node].X,
                    # UPDATED: Preserve m_j and c_j in the output
                    'm_j': jobs[next_node].get('m_j', 0),
                    'c_j': jobs[next_node].get('c_j', 0)
                })
                curr = next_node

    # --- 6. SAVE TO CSV AND LOG RUNTIME ---
    calculation_time = time.time() - start_time
    if final_schedule:
        df_sol = pd.DataFrame(final_schedule)
        df_sol.to_csv(OUTPUT_SOLUTION, index=False)
        makespan = df_sol['end_time'].max()
        print(f"\n--> [SUCCESS] Schedule saved to '{OUTPUT_SOLUTION}'.")
        print(f"--> Total Makespan: {makespan:.2f} minutes")
        print(f"--> Total Calculation Runtime: {calculation_time:.4f} seconds")
        return True
    else:
        print("--> [FAIL] No schedule generated.")
        return False

def visualize_results():
    print(f"\n{'='*50}")
    print(" PHASE 2: VISUALIZATION")
    print(f"{'='*50}")
    
    if not os.path.exists(OUTPUT_SOLUTION):
        print(f"Error: '{OUTPUT_SOLUTION}' not found. Cannot visualize.")
        return

    df_sol = pd.read_csv(OUTPUT_SOLUTION)
    df_jobs = pd.read_csv(INPUT_JOBS)
    
    df = pd.merge(df_sol, df_jobs[['job_id', 'r_j']], on='job_id', how='left')
    df['alias'] = df.apply(lambda r: f"{'A' if r['assigned_oven']==1 else 'B'}{int(r['job_id'])}", axis=1)
    
    makespan = df['end_time'].max()
    fig, ax = plt.subplots(figsize=(18, 10))
    colors = plt.cm.tab10.colors
    ovens = sorted(df['assigned_oven'].unique())
    
    y_ticks, y_labels = [], []
    for i, oven in enumerate(ovens):
        oven_data = df[df['assigned_oven'] == oven].sort_values('start_time')
        y_center = (len(ovens) - 1 - i) * 60  # Increased spacing for stacked labels
        y_ticks.append(y_center)
        y_labels.append(f"OVEN {oven}")
        
        ax.axhspan(y_center - 25, y_center + 25, facecolor='whitesmoke', alpha=0.5)
        
        # Track vertical levels for release time labels to prevent overlapping
        level_tracks = {}
        label_width = makespan * 0.05
        
        for _, row in oven_data.iterrows():
            start, end = row['start_time'], row['end_time']
            r_j = row['r_j']
            dur = end - start
            alias = row['alias']
            
            # 1. Plot Processing Bar
            ax.barh(y_center, dur, left=start, height=12, align='center', 
                    color=colors[int(row['job_id'])%10], edgecolor='black', alpha=0.9)
            ax.text(start + dur/2, y_center, alias, ha='center', va='center', 
                    color='white', fontweight='bold', fontsize=20)
            
            # 2. Plot Release Time Marker
            marker_y = y_center - 12
            ax.scatter(r_j, marker_y, color='red', marker='D', s=50, zorder=5, edgecolors='black')
            
            # 3. Dynamic Label Placement (Anti-collision)
            lvl = 1
            while True:
                last_x = level_tracks.get(lvl, -999)
                if r_j > (last_x + label_width/2): break
                lvl += 1
            level_tracks[lvl] = r_j + label_width
            
            ty = marker_y - 4 - (lvl * 6)
            ax.plot([r_j, r_j], [marker_y, ty+1], color='red', lw=0.8, alpha=0.6)
            
            # Batch Alias Label on Diamond
            ax.text(r_j, ty, alias, ha='center', va='center', color='darkred', fontsize=12, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_title(f"Balanced LPT Schedule (Makespan: {makespan:.2f} min)", fontsize=16, fontweight='bold', pad=15)
    
    # Legend
    leg_r = mlines.Line2D([], [], color='red', marker='D', linestyle='None', markersize=8, label='Release Time (Batch ID)')
    leg_b = mpatches.Patch(facecolor='gray', edgecolor='black', label='Batch Processing')
    ax.legend(handles=[leg_r, leg_b], loc='upper right', shadow=True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    print(f"--> [SUCCESS] Visualization saved to '{OUTPUT_PLOT}'.")
    plt.show()

if __name__ == "__main__":
    if run_balanced_scheduler():
        visualize_results()