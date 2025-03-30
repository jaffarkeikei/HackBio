import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Create results directory if it doesn't exist
os.makedirs('updated_plots', exist_ok=True)

# Latest model performance metrics
models = ['CNN', 'LSTM', 'GRU', 'Ensemble', 'RandomForest', 'XGBoost', 'TradEnsemble']
r2_scores = [0.3111, -0.0091, 0.0701, 0.2017, 0.33, 0.09, 0.24]
mse_scores = [5.52, 9.12, 8.37, 7.00, 177.55, 160.14, 166.18]

# Create a DataFrame for easy plotting
df = pd.DataFrame({
    'Model': models,
    'R² Score': r2_scores,
    'MSE': mse_scores
})

# Set style
plt.style.use('ggplot')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Define color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

# Plot R² scores
bars1 = ax1.bar(df['Model'], df['R² Score'], color=colors)
ax1.set_title('Model Comparison - R² Score', fontsize=16)
ax1.set_ylabel('R² Score', fontsize=14)
ax1.set_ylim(-0.1, 0.4)  # Set appropriate y-axis limits
ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Add value labels on the bars
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# Create separate figure for MSE to handle the scale difference
fig2, ax3 = plt.subplots(figsize=(10, 8))

# Neural models MSE (CNN, LSTM, GRU, Ensemble)
neural_models = df['Model'][0:4]
neural_mse = df['MSE'][0:4]
bars2 = ax3.bar(neural_models, neural_mse, color=colors[0:4])
ax3.set_title('Neural Models - MSE (Lower is Better)', fontsize=16)
ax3.set_ylabel('MSE', fontsize=14)

# Calculate appropriate y-limit to include labels
max_neural_mse = max(neural_mse)
y_limit = max_neural_mse * 1.15  # Add 15% padding for labels
ax3.set_ylim(0, y_limit)  # Set y-axis limits to include labels

ax3.tick_params(axis='x', rotation=45)

# Add value labels on the bars (positioned inside the chart)
for bar in bars2:
    height = bar.get_height()
    ax3.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, -15),  # Position label just below the top of the bar
                textcoords="offset points",
                ha='center', va='top',
                fontsize=12, fontweight='bold')

# Traditional models MSE (separate plot due to scale differences)
traditional_models = df['Model'][4:7]
traditional_mse = df['MSE'][4:7]
fig3, ax4 = plt.subplots(figsize=(10, 8))
bars3 = ax4.bar(traditional_models, traditional_mse, color=colors[4:7])
ax4.set_title('Traditional Models - MSE (Lower is Better)', fontsize=16)
ax4.set_ylabel('MSE', fontsize=14)

# Calculate appropriate y-limit for traditional models
max_trad_mse = max(traditional_mse)
y_limit_trad = max_trad_mse * 1.15  # Add 15% padding for labels
ax4.set_ylim(0, y_limit_trad)  # Set y-axis limits to include labels

ax4.tick_params(axis='x', rotation=45)

# Add value labels on the bars (positioned inside the chart)
for bar in bars3:
    height = bar.get_height()
    ax4.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, -15),  # Position label just below the top of the bar
                textcoords="offset points",
                ha='center', va='top',
                fontsize=12, fontweight='bold')

# Create a combined comparison chart highlighting both model types
fig4, ax5 = plt.subplots(figsize=(12, 8))

# Create a grouped bar chart
x = np.arange(len(models))
width = 0.35

# Normalize MSE values for visualization (showing inverse to represent "better is higher")
max_mse = max(mse_scores)
normalized_mse = [1 - (mse/max_mse) for mse in mse_scores]

r2_bars = ax5.bar(x - width/2, r2_scores, width, label='R² Score', color='steelblue')
mse_bars = ax5.bar(x + width/2, normalized_mse, width, label='Normalized MSE (inverse)', color='lightcoral')

ax5.set_xlabel('Model', fontsize=14)
ax5.set_title('Model Performance Comparison', fontsize=16)
ax5.set_xticks(x)
ax5.set_xticklabels(models, rotation=45)
ax5.legend()
ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# Add annotations for R² scores
for bar in r2_bars:
    height = bar.get_height()
    ypos = height if height >= 0 else 0
    ax5.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, ypos),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

# Save plots
fig.tight_layout()
fig.savefig('updated_plots/r2_comparison.png')

fig2.tight_layout()
fig2.savefig('updated_plots/neural_mse_comparison.png')

fig3.tight_layout()
fig3.savefig('updated_plots/traditional_mse_comparison.png')

fig4.tight_layout()
fig4.savefig('updated_plots/combined_performance.png')

print("Updated model comparison plots saved to 'updated_plots' directory") 