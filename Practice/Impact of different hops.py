import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns

# Customize colors and styles
auc_color = 'purple'
ndcg5_color = 'teal'
ndcg10_color = 'darkorange'
auc_marker = 'd'
ndcg5_marker = 'x'
ndcg10_marker = 'v'

# 数据
hop_labels = ["1-hop", "2-hop", "3-hop"]
auc_values = [68.1, 68.2, 67.9]
ndcg5_values = [37.6, 37.8, 37.5]
ndcg10_values = [42.3, 42.4, 42.1]
runtime_values = [1.0, 2.5, 6.5]  # 归一化后的相对运行时间

fig, axes = plt.subplots(2, 1, figsize=(8, 10))

# ----------- (a) AUC & nDCG vs. Hops -----------
ax1 = axes[0]
ax1.plot(hop_labels, auc_values, marker=auc_marker, linestyle='--', color=auc_color, label='AUC', linewidth=2)
ax1.set_ylabel('AUC', fontsize=12)
ax1.set_xlabel('Hops ($K$)', fontsize=12)
ax1.set_title('Impact of different hops ($K$) (on fixed $M_e = 10$, $M_n = 8$)', fontsize=14)
ax1.set_ylim(67.2, 68.5)
ax1.grid(True, linestyle=':', alpha=0.6)

ax2 = ax1.twinx()
ax2.plot(hop_labels, ndcg5_values, marker=ndcg5_marker, linestyle='-', color=ndcg5_color, label='nDCG@5', linewidth=2)
ax2.plot(hop_labels, ndcg10_values, marker=ndcg10_marker, linestyle='-.', color=ndcg10_color, label='nDCG@10', linewidth=2)
ax2.set_ylabel('nDCG', fontsize=12)
ax2.set_ylim(35, 45)

# 合并图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=11)

# ----------- (b) Runtime vs. Hops -----------
ax3 = axes[1]
ax3.bar(hop_labels, runtime_values, color='tab:blue', alpha=0.7)
ax3.set_ylabel('Run-Time (relative to 1-hop)', fontsize=12)
ax3.set_xlabel('Hops ($K$)', fontsize=12)
ax3.set_title('Impact of different hops ($K$) (on fixed $M_e = 10$, $M_n = 8$)', fontsize=14)
ax3.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
