import matplotlib.pyplot as plt
import numpy as np

def plot_batch_utilization():
    # -------------------------------------------------
    # DATA: Batch fill percentages (50 iterations each)
    # -------------------------------------------------
    alpha_data = {
        0.1: [84.77, 96.65, 95.97, 66.47, 19.32, 21.17],
        0.2: [92.17, 91.95, 88.55, 70.50, 27.43, 13.76],
        0.3: [66.77, 93.36, 85.76, 69.09, 30.21, 39.16],
        0.4: [90.35, 88.80, 10.97, 72.09, 27.17, 94.96],
        0.5: [11.59, 89.53, 80.73, 72.92, 35.24, 94.34],
        0.6: [92.97, 10.99, 79.09, 69.48, 36.89, 94.94]
    }

    # Global Optimal (MIP) batch fill percentages
    global_optimal_data = [98.33, 94.92, 64.11, 94.88, 11.05, 21.06]

    # -------------------------------------------------
    # PLOTTING
    # -------------------------------------------------
    plt.figure(figsize=(15, 8))
    batches = list(range(1, 7))
    markers = ['o', 's', '^', 'D', 'v', 'P']

    # SA alpha curves (lighter)
    for i, (alpha, fills) in enumerate(alpha_data.items()):
        plt.plot(
            batches,
            fills,
            marker=markers[i % len(markers)],
            markersize=7,
            linewidth=2.0,
            alpha=0.45,
            label=f'Alpha = {alpha}'
        )

    # Global Optimal curve (dark & dominant)
    plt.plot(
        batches,
        global_optimal_data,
        marker='*',
        markersize=16,
        linewidth=4.0,
        linestyle='--',
        color='black',
        label='Global Optimal'
    )

    # -------------------------------------------------
    # FORMATTING
    # -------------------------------------------------
    plt.title(
        'Batch Utilization Profile: SA Alpha Tuning vs. Global Optimal',
        fontsize=16,
        fontweight='bold',
        pad=15
    )
    plt.xlabel('Batch Number', fontsize=12, fontweight='bold')
    plt.ylabel('Oven Utilization (%)', fontsize=12, fontweight='bold')

    plt.ylim(0, 105)
    plt.xlim(0.8, 6.2)
    plt.xticks(batches)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.legend(
        title='Configuration',
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        frameon=True
    )

    plt.tight_layout()
    plt.subplots_adjust(right=0.82)

    # -------------------------------------------------
    # SAVE & SHOW
    # -------------------------------------------------
    plt.savefig("mod3_plot.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_batch_utilization()
