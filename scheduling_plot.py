import matplotlib.pyplot as plt

# Updated labels for English
labels = ['Baseline\n(FIFO)', 'Optimized\n(Simulated Annealing)']
values = [19.39, 17.60]
colors = ['#7F7F7F', '#0033A0']

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(labels, values, color=colors, width=0.5)

for bar in bars:
    yval = bar.get_height()
    # Changed 'saat' to 'hours'
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{yval} hours', 
            ha='center', va='bottom', color='black', fontweight='bold', fontsize=12)

# Updated axes and title
ax.set_ylabel('Total Completion Time (Hours)', fontsize=12, fontweight='bold')
ax.set_title('Performance Comparison: Baseline vs. SA Algorithm', fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0, 22)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
