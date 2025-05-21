import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Use Seaborn style for professional-looking graphs
sns.set_theme(style="whitegrid")

# Sample data (Replace these with actual dataset values)
datasets = ['MIND', 'Adressa']
positive = [45, 40]  # Example % of positive sentiment
negative = [30, 35]  # Example % of negative sentiment
neutral = [25, 25]   # Example % of neutral sentiment

# Define bar width and position
bar_width = 0.25
indices = np.arange(len(datasets))

# Minimalist colors (Shades of Blue & Gray for scientific readability)
colors = ['#4C72B0', '#55A868', '#C44E52']  # Blue, Green, Red
alpha_value = 0.85  # Slight transparency for better contrast

# Create figure
fig, ax = plt.subplots(figsize=(6, 4))

# Draw bars
bars1 = ax.bar(indices - bar_width, positive, width=bar_width, color=colors[0], alpha=alpha_value, label='Positive')
bars2 = ax.bar(indices, negative, width=bar_width, color=colors[1], alpha=alpha_value, label='Negative')
bars3 = ax.bar(indices + bar_width, neutral, width=bar_width, color=colors[2], alpha=alpha_value, label='Neutral')

# Labels & Title (Clear, Journal-Style Font)
ax.set_xlabel('Datasets', fontsize=12, fontweight='medium')
ax.set_ylabel('Proportion (%)', fontsize=12, fontweight='medium')
ax.set_xticks(indices)
ax.set_xticklabels(datasets, fontsize=11)
ax.set_ylim(0, 50)  # Adjust limit for better visibility

# Remove unnecessary spines (Top & Right) for cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add text labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height}%', ha='center', fontsize=10, fontweight='medium')

# Add legend
ax.legend(fontsize=10, loc='upper right', frameon=False)

# Save as high-quality image for publication
plt.tight_layout()
plt.savefig("sentiment_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

