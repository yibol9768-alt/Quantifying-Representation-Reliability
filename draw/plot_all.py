"""生成所有实验图表"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['savefig.bbox'] = 'tight'

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 数据
# ============================================================

models_labels = ['1\n(CLIP)', '2\n(+DINO)', '3\n(+MAE)', '4\n(+SigLIP)', '5\n(+ConvNeXt)', '6\n(+Data2Vec)']
n_models = [1, 2, 3, 4, 5, 6]

# Few-shot scaling (Gated, 10-shot) — 来自原始实验
fewshot = {
    'STL10':      [91.44, 95.34, 95.21, 95.71, 94.74, 94.99],
    'GTSRB':      [67.63, 66.47, 63.30, 75.00, 71.82, 70.75],
    'SVHN':       [29.76, 25.68, 26.40, 35.42, 34.05, 32.18],
    'Pets':       [88.99, 94.19, 94.06, 94.71, 95.48, 95.45],
    'EuroSAT':    [83.50, 88.15, 88.11, 86.94, 88.72, 87.65],
    'DTD':        [67.93, 75.85, 75.16, 76.54, 76.70, 76.28],
    'Country211': [29.62, 27.47, 26.92, 27.08, 26.87, 25.92],
}

# Full-data scaling (Gated) — 新实验结果
fulldata = {
    'STL10': [94.93, 97.84, 97.83, 97.93, 97.93, 97.95],
    'GTSRB': [87.81, 88.57, 78.46, 78.62, 91.63, 91.10],
    'SVHN':  [73.33, 66.61, 66.52, 66.36, 75.66, 75.10],
}

# Full-data vs few-shot 方法对比 (6模型)
method_comparison = {
    'STL10':      {'concat': [94.66, 97.96], 'gated': [93.80, 97.85], 'moe': [93.65, 97.80]},
    'Pets':       {'concat': [95.26, 96.40], 'gated': [94.17, 96.13], 'moe': [94.74, 96.29]},
    'EuroSAT':    {'concat': [86.30, 97.83], 'gated': [80.56, 97.63], 'moe': [82.69, 97.54]},
    'DTD':        {'concat': [76.33, 83.88], 'gated': [74.20, 82.61], 'moe': [76.44, 83.14]},
    'GTSRB':      {'concat': [60.93, 91.18], 'gated': [49.61, 90.86], 'moe': [69.18, 90.98]},
    'SVHN':       {'concat': [22.47, 83.05], 'gated': [33.01, 77.97], 'moe': [25.68, 82.47]},
    'Country211': {'concat': [12.47, 26.64], 'gated': [15.25, 27.58], 'moe': [12.08, 26.81]},
}

colors = {'STL10': '#2196F3', 'GTSRB': '#F44336', 'SVHN': '#FF9800',
          'Pets': '#4CAF50', 'EuroSAT': '#9C27B0', 'DTD': '#795548',
          'Country211': '#607D8B'}

# ============================================================
# 图1: Few-shot scaling (所有7个数据集)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
for ds, vals in fewshot.items():
    peak_idx = np.argmax(vals)
    ax.plot(n_models, vals, 'o-', label=ds, color=colors[ds], linewidth=2, markersize=6)
    ax.plot(n_models[peak_idx], vals[peak_idx], '*', color=colors[ds], markersize=15)
ax.set_xlabel('Number of Models')
ax.set_ylabel('Best Test Accuracy (%)')
ax.set_title('Few-shot (10-shot) Scaling: Accuracy vs Model Count\n(Gated Fusion, ★=peak)')
ax.set_xticks(n_models)
ax.set_xticklabels(models_labels)
ax.legend(loc='lower left', fontsize=10)
ax.grid(True, alpha=0.3)
fig.savefig(os.path.join(OUT_DIR, 'fig1_fewshot_scaling.png'))
plt.close()

# ============================================================
# 图2: Full-data vs Few-shot scaling 对比 (3个数据集, 双曲线)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, ds in zip(axes, ['STL10', 'GTSRB', 'SVHN']):
    fs = fewshot[ds]
    fd = fulldata[ds]
    ax.plot(n_models, fs, 's--', label='10-shot', color='#F44336', linewidth=2, markersize=7)
    ax.plot(n_models, fd, 'o-', label='Full data', color='#2196F3', linewidth=2, markersize=7)
    # 标注峰值
    fs_peak = np.argmax(fs)
    fd_peak = np.argmax(fd)
    ax.plot(n_models[fs_peak], fs[fs_peak], '*', color='#F44336', markersize=18)
    ax.plot(n_models[fd_peak], fd[fd_peak], '*', color='#2196F3', markersize=18)
    ax.set_title(ds, fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Models')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(n_models)
    ax.set_xticklabels(models_labels, fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
fig.suptitle('Few-shot vs Full-data Scaling (Gated Fusion, ★=peak)', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig2_fewshot_vs_fulldata_scaling.png'))
plt.close()

# ============================================================
# 图3: 每一步模型增量的贡献 (delta bar chart)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
step_labels = ['+DINO', '+MAE', '+SigLIP', '+ConvNeXt', '+Data2Vec']
for ax, ds in zip(axes, ['STL10', 'GTSRB', 'SVHN']):
    fd = fulldata[ds]
    deltas = [fd[i+1] - fd[i] for i in range(5)]
    bar_colors = ['#4CAF50' if d >= 0 else '#F44336' for d in deltas]
    bars = ax.bar(step_labels, deltas, color=bar_colors, edgecolor='black', linewidth=0.5)
    for bar, d in zip(bars, deltas):
        va = 'bottom' if d >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2, d, f'{d:+.1f}', ha='center', va=va, fontsize=10, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_title(f'{ds} (Full-data)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Δ Accuracy (pp)')
    ax.grid(True, alpha=0.3, axis='y')
fig.suptitle('Per-step Accuracy Change (Full-data, Gated Fusion)', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig3_fulldata_delta_per_step.png'))
plt.close()

# ============================================================
# 图4: Full-data vs Few-shot 方法对比 (grouped bar)
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
datasets_order = ['STL10', 'Pets', 'EuroSAT', 'DTD', 'GTSRB', 'SVHN', 'Country211']
method_names = ['Concat', 'Gated', 'MoE']
method_keys = ['concat', 'gated', 'moe']
bar_colors_fs = ['#FFCDD2', '#C8E6C9', '#BBDEFB']
bar_colors_fd = ['#F44336', '#4CAF50', '#2196F3']

for i, ds in enumerate(datasets_order):
    ax = axes[i]
    x = np.arange(3)
    width = 0.35
    fs_vals = [method_comparison[ds][k][0] for k in method_keys]
    fd_vals = [method_comparison[ds][k][1] for k in method_keys]
    bars1 = ax.bar(x - width/2, fs_vals, width, label='10-shot', color=bar_colors_fs, edgecolor='gray')
    bars2 = ax.bar(x + width/2, fd_vals, width, label='Full data', color=bar_colors_fd, edgecolor='gray')
    ax.set_title(ds, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, fontsize=9)
    ax.set_ylabel('Acc (%)')
    if i == 0:
        ax.legend(fontsize=8)
    # 标注最优值
    best_fs = max(fs_vals)
    best_fd = max(fd_vals)
    for j, (v1, v2) in enumerate(zip(fs_vals, fd_vals)):
        if v1 == best_fs:
            ax.text(j - width/2, v1 + 0.3, f'{v1:.1f}', ha='center', fontsize=7, fontweight='bold')
        if v2 == best_fd:
            ax.text(j + width/2, v2 + 0.3, f'{v2:.1f}', ha='center', fontsize=7, fontweight='bold')

axes[7].axis('off')
fig.suptitle('Few-shot vs Full-data: Method Comparison (6 Models)', fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig4_method_comparison.png'))
plt.close()

# ============================================================
# 图5: 数据量提升幅度 (heatmap)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
ds_list = ['STL10', 'Pets', 'EuroSAT', 'DTD', 'GTSRB', 'SVHN', 'Country211']
methods = ['Concat', 'Gated', 'MoE']
delta_matrix = []
for ds in ds_list:
    row = []
    for k in method_keys:
        fs, fd = method_comparison[ds][k]
        row.append(fd - fs)
    delta_matrix.append(row)
delta_matrix = np.array(delta_matrix)

im = ax.imshow(delta_matrix, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(3))
ax.set_xticklabels(methods)
ax.set_yticks(range(7))
ax.set_yticklabels(ds_list)
for i in range(7):
    for j in range(3):
        ax.text(j, i, f'+{delta_matrix[i,j]:.1f}', ha='center', va='center',
                fontsize=10, fontweight='bold',
                color='white' if delta_matrix[i,j] > 30 else 'black')
plt.colorbar(im, label='Accuracy Gain (Full-data − 10-shot, pp)')
ax.set_title('Data Quantity Impact: Full-data − Few-shot Accuracy Gain')
fig.savefig(os.path.join(OUT_DIR, 'fig5_data_impact_heatmap.png'))
plt.close()

print("All figures saved:")
for i in range(1, 6):
    print(f"  fig{i}_*.png")
