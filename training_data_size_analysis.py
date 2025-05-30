import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

df = pd.read_csv('training_data_size.csv')

# 2. Keep only rows where State == "finished"
df = df[df['State'] == 'finished']

# 3. Simplify the 'algo' column values:
#    replace any occurrence of the long thrifty path with 'thrifty'
long_thrifty = 'src.thrifty.algos.thriftydagger_venv.thrifty'
df['algo'] = df['algo'].replace({long_thrifty: 'thrifty'})

# 4. Select only the columns we need
df = df[['avg_payload_tracking_error', 'expert_queries', 'policy_queries', 'algo', 'completed_all']]
# 4.5. For 'dagger' rows, double the expert and policy queries
mask = df['algo'] == 'dagger'
df.loc[mask, 'expert_queries']   = df.loc[mask, 'expert_queries']   * 2
df.loc[mask, 'policy_queries']   = df.loc[mask, 'policy_queries']   * 2


# 5. Compute derived columns for plotting
df['total_queries'] = df['expert_queries'] + df['policy_queries']
df['expert_prop']    = df['expert_queries'] / df['total_queries']
df['accuracy']       = 1 - df['avg_payload_tracking_error']
df['color']          = df['algo'].map({'thrifty': 'red', 'dagger': 'blue'})

# Define the size range you want, e.g. between 50 and 400 points²
min_size, max_size = 10, 1000

# Linearly scale accuracy into [min_size, max_size]
acc_min = df['accuracy'].min()
acc_max = df['accuracy'].max()

df['marker_size'] = (
    (df['accuracy'] - acc_min)
    / (acc_max - acc_min)
) * (max_size - min_size) + min_size

# 7. Plot with x = total_queries, y = expert_queries
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(
    x = df['total_queries'],
    y = df['expert_queries'],
    s = df['marker_size'],
    c = df['color'],
    alpha = 0.7,
    edgecolors = 'k'
)

# — new bit: overlay a small black dot on completed runs —
completed_mask = df['completed_all'] == True
ax.scatter(
    x = df.loc[completed_mask, 'total_queries'],
    y = df.loc[completed_mask, 'expert_queries'],
    # you can pick a fixed size or scale it; here 80 is a nice medium dot
    s = 80,
    c = 'black',
    marker = 'o',
    label = 'completed_all',
    zorder = 3
)

# 8. Legend for color (algorithm)
alg_legend = [
    Line2D([0], [0], marker='o', color='w', label='thrifty',
           markerfacecolor='red', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='dagger',
           markerfacecolor='blue', markersize=10)
]
color_legend = ax.legend(handles=alg_legend, title='Algorithm', loc='upper left')
ax.add_artist(color_legend)

# 9. Legend for size (accuracy)
acc_vals = [acc_min, (acc_min+acc_max)/2, acc_max]
size_vals = [
    (val - acc_min)/(acc_max - acc_min)*(max_size-min_size) + min_size
    for val in acc_vals
]
size_legend = [
    ax.scatter([], [], s=size, c='grey', alpha=0.7, edgecolors='k',
               label=f'{val:.2f}')
    for size, val in zip(size_vals, acc_vals)
]
size_legend = ax.legend(handles=size_legend, title='Accuracy', loc='upper right')
ax.add_artist(size_legend)

# 10. Legend for completion status
completion_legend = ax.legend(
    handles=[Line2D([0], [0], marker='o', color='w', label='1.0 Success Rate',
                    markerfacecolor='black', markersize=8)],
    title='Completion',
    loc='lower right'
)
ax.add_artist(completion_legend)

# 11. Labels & styling
ax.set_xlabel('Total Queries (Expert + Policy)')
ax.set_ylabel('Expert Queries')
ax.set_title('Expert and Total Queries\n(size ∝ accuracy, color by algorithm)')
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()