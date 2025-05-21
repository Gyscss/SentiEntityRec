import matplotlib.pyplot as plt
import numpy as np

# Sample data (Replace these with actual experiment results)
Mn_values = [4, 8, 12, 16, 20]
Me_values = [5, 10, 15, 20, 35]

auc_Mn = [67.8, 68.1, 67.9, 68.2, 67.8]  # AUC values for different Mn
ndcg5_Mn = [37.5, 37.8, 37.6, 37.9, 37.4]  # nDCG@5 values for different Mn
ndcg10_Mn = [42.1, 42.3, 42.2, 42.4, 42.0]  # nDCG@10 values for different Mn

auc_Me = [67.8, 68.2, 68.1, 67.9, 67.7]  # AUC values for different Me
ndcg5_Me = [37.5, 37.9, 37.7, 37.6, 37.3]  # nDCG@5 values for different Me
ndcg10_Me = [42.1, 42.4, 42.3, 42.1, 41.9]  # nDCG@10 values for different Me

# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(8, 10))

# Customize colors and styles
auc_color = 'purple'
ndcg5_color = 'teal'
ndcg10_color = 'darkorange'
auc_marker = 'd'
ndcg5_marker = 'x'
ndcg10_marker = 'v'

# Plot for Mn (Impact of News Neighbors)
ax1 = axes[0]
ax1.plot(Mn_values, auc_Mn, marker=auc_marker, linestyle='--', color=auc_color, linewidth=2, label='AUC')
ax1.set_ylabel('AUC', fontsize=12)
ax1.set_xlabel('$M_n$', fontsize=12)
ax1.set_title('Impact of news neighbors $M_n$ (on fixed $K = 2$, $M_e = 10$)', fontsize=14)
ax1.set_ylim(67.2, 68.5)
ax1.grid(True, linestyle=':', alpha=0.6)

ax2 = ax1.twinx()
ax2.plot(Mn_values, ndcg5_Mn, marker=ndcg5_marker, linestyle='-', color=ndcg5_color, linewidth=2, label='nDCG@5')
ax2.plot(Mn_values, ndcg10_Mn, marker=ndcg10_marker, linestyle='-.', color=ndcg10_color, linewidth=2, label='nDCG@10')
ax2.set_ylabel('nDCG', fontsize=12)
ax2.set_ylim(35, 45)

ax1.legend(loc='lower left', fontsize=11, frameon=True)
ax2.legend(loc='lower right', fontsize=11, frameon=True)

# Plot for Me (Impact of Adjacent Entities)
ax3 = axes[1]
ax3.plot(Me_values, auc_Me, marker=auc_marker, linestyle='--', color=auc_color, linewidth=2, label='AUC')
ax3.set_ylabel('AUC', fontsize=12)
ax3.set_xlabel('$M_e$', fontsize=12)
ax3.set_title('Impact of adjacent entities $M_e$ (on fixed $K = 2$, $M_n = 8$)', fontsize=14)
ax3.set_ylim(67.2, 68.5)
ax3.grid(True, linestyle=':', alpha=0.6)

ax4 = ax3.twinx()
ax4.plot(Me_values, ndcg5_Me, marker=ndcg5_marker, linestyle='-', color=ndcg5_color, linewidth=2, label='nDCG@5')
ax4.plot(Me_values, ndcg10_Me, marker=ndcg10_marker, linestyle='-.', color=ndcg10_color, linewidth=2, label='nDCG@10')
ax4.set_ylabel('nDCG', fontsize=12)
ax4.set_ylim(35, 45)

ax3.legend(loc='lower left', fontsize=11, frameon=True)
ax4.legend(loc='lower right', fontsize=11, frameon=True)

plt.tight_layout()
plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# import os
#
# # 设置论文风格
# plt.style.use('seaborn-v0_8-ticks')  # 选用简约风格
#
# # 伪造数据（请替换为你的实验数据）
# Mn_values = [4, 8, 12, 16, 20]
# auc_Mn = [67.8, 68.3, 67.9, 68.4, 67.8]
# ndcg5_Mn = [37.2, 37.5, 37.1, 37.8, 37.3]
# ndcg10_Mn = [41.9, 42.2, 41.8, 42.5, 42.0]
#
# Me_values = [5, 10, 15, 20, 35]
# auc_Me = [67.7, 68.2, 67.9, 68.0, 67.6]
# ndcg5_Me = [37.1, 37.6, 37.4, 37.2, 37.0]
# ndcg10_Me = [41.8, 42.3, 42.1, 42.0, 41.7]
#
# # 颜色和标记样式
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 选用蓝色、橙色、绿色
# markers = ['D', 's', 'o']  # 钻石形、方形、圆形
# linestyles = ['-', '--', '-.']
#
# # 创建画布
# fig, axes = plt.subplots(2, 1, figsize=(6, 8))
#
# ### 绘制 M_n 影响图
# ax1 = axes[0]
# ax1.plot(Mn_values, auc_Mn, marker=markers[0], linestyle=linestyles[0], color=colors[0], linewidth=2, label='AUC')
# ax1.set_xlabel(r'Number of News Neighbors $M_n$', fontsize=12)
# ax1.set_ylabel('AUC', fontsize=12)
# ax1.set_title(r'Impact of $M_n$', fontsize=14)
# ax1.grid(True, linestyle=':', alpha=0.6)
#
# # 共享 y 轴
# ax1_twin = ax1.twinx()
# ax1_twin.plot(Mn_values, ndcg5_Mn, marker=markers[1], linestyle=linestyles[1], color=colors[1], linewidth=2, label='nDCG@5')
# ax1_twin.plot(Mn_values, ndcg10_Mn, marker=markers[2], linestyle=linestyles[2], color=colors[2], linewidth=2, label='nDCG@10')
# ax1_twin.set_ylabel('nDCG', fontsize=12)
#
# # 图例合并
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines1_twin, labels1_twin = ax1_twin.get_legend_handles_labels()
# ax1.legend(lines1 + lines1_twin, labels1 + labels1_twin, loc='upper left', fontsize=10, frameon=False)
#
# ### 绘制 M_e 影响图
# ax2 = axes[1]
# ax2.plot(Me_values, auc_Me, marker=markers[0], linestyle=linestyles[0], color=colors[0], linewidth=2, label='AUC')
# ax2.set_xlabel(r'Number of Adjacent Entities $M_e$', fontsize=12)
# ax2.set_ylabel('AUC', fontsize=12)
# ax2.set_title(r'Impact of $M_e$', fontsize=14)
# ax2.grid(True, linestyle=':', alpha=0.6)
#
# ax2_twin = ax2.twinx()
# ax2_twin.plot(Me_values, ndcg5_Me, marker=markers[1], linestyle=linestyles[1], color=colors[1], linewidth=2, label='nDCG@5')
# ax2_twin.plot(Me_values, ndcg10_Me, marker=markers[2], linestyle=linestyles[2], color=colors[2], linewidth=2, label='nDCG@10')
# ax2_twin.set_ylabel('nDCG', fontsize=12)
#
# # 图例合并
# lines2, labels2 = ax2.get_legend_handles_labels()
# lines2_twin, labels2_twin = ax2_twin.get_legend_handles_labels()
# ax2.legend(lines2 + lines2_twin, labels2 + labels2_twin, loc='upper left', fontsize=10, frameon=False)
#
# plt.tight_layout()
#
# # 保存图片到桌面
# desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
# save_path = os.path.join(desktop_path, "hyperparameter_analysis.png")
# plt.savefig(save_path, dpi=300, bbox_inches='tight')
# plt.show()




