import pandas as pd
import numpy as np
import random
import math
import time
import os
import copy
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# --- CONFIGURATION ---
INPUT_JOBS = 'mod4_jobs.csv'
INPUT_SETUP = 'mod4_setup.csv'
INPUT_LPT_SOL = 'mod4_lpt_solution.csv'

OUTPUT_SA_SOL = 'mod4_sa_solution.csv'
OUTPUT_SA_PLOT = 'mod4_sa_plot.png'

W_MAX = 225

# --- FIXED SA PARAMETERS ---
FIXED_LAMBDA = 20
INITIAL_TEMP = 1000
ALPHA = 0.95
ITERATIONS_PER_TEMP = 100
FINAL_TEMP = 1.0


def load_data():
    if not os.path.exists(INPUT_JOBS) or not os.path.exists(INPUT_SETUP):
        print("CRITICAL ERROR: Input files missing.")
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
        print(f"ERROR: LPT solution file '{INPUT_LPT_SOL}' not found.")
        return None

    df = pd.read_csv(INPUT_LPT_SOL)
    sol = {1: [], 2: []}

    df_sorted = df.sort_values(by=['assigned_oven', 'start_time'])
    for _, row in df_sorted.iterrows():
        sol[int(row['assigned_oven'])].append(int(row['job_id']))

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

    if move_type < 0.33:  # Move between ovens
        src = random.choice(ovens)
        dest = random.choice(ovens)
        if new_sol[src]:
            job = new_sol[src].pop(random.randint(0, len(new_sol[src]) - 1))
            new_sol[dest].insert(random.randint(0, len(new_sol[dest])), job)

    elif move_type < 0.66:  # Swap within oven
        oven = random.choice(ovens)
        if len(new_sol[oven]) >= 2:
            i, j = random.sample(range(len(new_sol[oven])), 2)
            new_sol[oven][i], new_sol[oven][j] = new_sol[oven][j], new_sol[oven][i]

    else:  # Reverse subsequence
        oven = random.choice(ovens)
        if len(new_sol[oven]) >= 2:
            i, j = sorted(random.sample(range(len(new_sol[oven])), 2))
            new_sol[oven][i:j+1] = reversed(new_sol[oven][i:j+1])

    return new_sol


def visualize_sa_schedule():
    print("\n" + "=" * 50)
    print(" PHASE 2: VISUALIZATION (SA)")
    print("=" * 50)

    df_sol = pd.read_csv(OUTPUT_SA_SOL)
    df_jobs = pd.read_csv(INPUT_JOBS)

    df = pd.merge(df_sol, df_jobs[['job_id', 'r_j']], on='job_id', how='left')
    df['alias'] = df.apply(
        lambda r: f"{'A' if r['assigned_oven']==1 else 'B'}{int(r['job_id'])}", axis=1
    )

    makespan = df['end_time'].max()
    fig, ax = plt.subplots(figsize=(18, 10))
    colors = plt.cm.tab10.colors
    ovens = sorted(df['assigned_oven'].unique())

    y_ticks, y_labels = [], []

    for i, oven in enumerate(ovens):
        oven_data = df[df['assigned_oven'] == oven].sort_values('start_time')
        y_center = (len(ovens) - 1 - i) * 60
        y_ticks.append(y_center)
        y_labels.append(f"OVEN {oven}")

        ax.axhspan(y_center - 25, y_center + 25, facecolor='whitesmoke', alpha=0.5)

        level_tracks = {}
        label_width = makespan * 0.05

        for _, row in oven_data.iterrows():
            start, end = row['start_time'], row['end_time']
            dur, r_j, alias = end - start, row['r_j'], row['alias']

            ax.barh(
                y_center, dur, left=start, height=12,
                color=colors[int(row['job_id']) % 10],
                edgecolor='black', alpha=0.9
            )
            ax.text(start + dur / 2, y_center, alias,
                    ha='center', va='center',
                    color='white', fontweight='bold', fontsize=20)

            marker_y = y_center - 12
            ax.scatter(r_j, marker_y, color='red', marker='D',
                       s=50, zorder=5, edgecolors='black')

            lvl = 1
            while True:
                last_x = level_tracks.get(lvl, -999)
                if r_j > (last_x + label_width / 2):
                    break
                lvl += 1

            level_tracks[lvl] = r_j + label_width
            ty = marker_y - 4 - (lvl * 6)

            ax.plot([r_j, r_j], [marker_y, ty + 1],
                    color='red', lw=0.8, alpha=0.6)

            ax.text(
                r_j, ty, alias,
                ha='center', va='center',
                color='darkred', fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_title(f"Simulated Annealing Schedule (Makespan: {makespan:.2f} min)",
                 fontsize=16, fontweight='bold', pad=15)

    leg_r = mlines.Line2D([], [], color='red', marker='D',
                          linestyle='None', markersize=8,
                          label='Release Time (Batch ID)')
    leg_b = mpatches.Patch(facecolor='gray', edgecolor='black',
                           label='Batch Processing')

    ax.legend(handles=[leg_r, leg_b], loc='upper right', shadow=True)

    plt.tight_layout()
    plt.savefig(OUTPUT_SA_PLOT)
    print(f"--> [SUCCESS] Visualization saved to '{OUTPUT_SA_PLOT}'")
    plt.show()


if __name__ == "__main__":

    print("=" * 100)
    print(" SINGLE RUN SIMULATED ANNEALING (File Based)")
    print("=" * 100)

    start_time = time.time()

    jobs_dict, setup_lookup = load_data()
    lpt_sol = load_lpt_solution()

    if jobs_dict is None or lpt_sol is None:
        exit()

    base_cost, base_mk, base_pen, _ = calculate_cost(lpt_sol, jobs_dict, setup_lookup)
    print(f"Baseline (LPT) Cost: {base_cost:.2f} | Makespan: {base_mk:.2f} | Penalty: {base_pen}")
    print("-" * 100)

    current_sol = copy.deepcopy(lpt_sol)
    curr_cost, curr_mk, curr_pen, _ = calculate_cost(current_sol, jobs_dict, setup_lookup)

    best_sol = copy.deepcopy(current_sol)
    best_cost, best_mk, best_pen = curr_cost, curr_mk, curr_pen

    T = INITIAL_TEMP

    while T > FINAL_TEMP:
        for _ in range(ITERATIONS_PER_TEMP):
            neighbor = generate_neighbor(current_sol)
            new_cost, new_mk, new_pen, _ = calculate_cost(neighbor, jobs_dict, setup_lookup)

            delta = new_cost - curr_cost
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_sol = neighbor
                curr_cost, curr_mk, curr_pen = new_cost, new_mk, new_pen

                if new_cost < best_cost:
                    best_sol = copy.deepcopy(current_sol)
                    best_cost, best_mk, best_pen = new_cost, new_mk, new_pen

        T *= ALPHA

    runtime = time.time() - start_time

    print(f"Final SA Cost: {best_cost:.2f}")
    print(f"Makespan: {best_mk:.2f} | Penalty: {best_pen}")
    print(f"Run Time: {runtime:.4f} seconds")
    print("=" * 100)

    rows = []
    for oven_id, sequence in best_sol.items():
        time_tracker = 0
        prev_job = 0

        for job_id in sequence:
            job = jobs_dict[job_id]
            setup = setup_lookup.get((prev_job, job_id), 0)

            start_time = max(time_tracker + setup, job['r_j'])
            end_time = start_time + job['p_j']

            rows.append({
                'assigned_oven': oven_id,
                'job_id': job_id,
                'start_time': start_time,
                'end_time': end_time,
                # UPDATED: Preserve m_j and c_j in the output
                'm_j': job.get('m_j', 0),
                'c_j': job.get('c_j', 0)
            })

            time_tracker = end_time
            prev_job = job_id

    pd.DataFrame(rows).to_csv(OUTPUT_SA_SOL, index=False)
    print(f"SA solution saved to '{OUTPUT_SA_SOL}'")

    visualize_sa_schedule()