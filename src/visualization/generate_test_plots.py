import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Create directories for output
os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

print("Generating test visualizations...")

# 1. Molecular Properties Visualization
print("Creating molecular properties visualization...")
n_compounds = 30
molecular_weight = np.random.normal(300, 50, n_compounds)
logp = np.random.normal(3, 1, n_compounds)
tpsa = np.random.normal(70, 15, n_compounds)

plt.figure(figsize=(10, 6))
plt.scatter(molecular_weight, logp, c=tpsa, cmap='viridis', alpha=0.7, s=100)
plt.colorbar(label='TPSA')
plt.xlabel('Molecular Weight (Da)')
plt.ylabel('LogP')
plt.title('Molecular Properties Distribution')
plt.tight_layout()
plt.savefig('plots/molecular_properties.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Correlation Matrix Heatmap
print("Creating correlation heatmap...")
n_properties = 5
n_genes = 10
property_names = ['MolWeight', 'LogP', 'TPSA', 'HBD', 'HBA']
gene_names = [f'Gene_{i}' for i in range(n_genes)]

# Create random correlation matrix
corr_matrix = np.random.uniform(-1, 1, (n_properties, n_genes))

# Plot correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    corr_matrix, 
    cmap='coolwarm', 
    center=0,
    annot=True, 
    fmt='.2f',
    linewidths=0.5,
    xticklabels=gene_names,
    yticklabels=property_names
)
plt.title('Correlation between Molecular Properties and Gene Expression')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Clustered Heatmap
print("Creating clustered heatmap...")
plt.figure(figsize=(14, 10))

# Add cluster boundaries
cluster_sizes = [2, 1, 2]  # Number of properties in each cluster
cluster_boundaries = np.cumsum(cluster_sizes)[:-1]

# Create hierarchical clustered heatmap
sns.clustermap(
    corr_matrix, 
    cmap='coolwarm',
    center=0,
    figsize=(14, 10),
    row_cluster=False,
    col_cluster=True,
    linewidths=0.5,
    xticklabels=gene_names,
    yticklabels=property_names
)
plt.suptitle('Clustered Properties by Gene Expression Effect', y=1.02)
plt.savefig('plots/clustered_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Feature Importance Plot
print("Creating feature importance visualization...")
n_features = 15
feature_names = [f'CellType_{i}' for i in range(5)] + [f'Drug_{i}' for i in range(10)]
importance = np.random.exponential(0.5, n_features)
importance = importance / importance.sum()  # Normalize to sum to 1

# Sort by importance
sorted_indices = np.argsort(importance)[::-1]
sorted_importance = importance[sorted_indices]
sorted_names = [feature_names[i] for i in sorted_indices]

plt.figure(figsize=(10, 8))
plt.barh(range(n_features), sorted_importance, color='skyblue')
plt.yticks(range(n_features), sorted_names)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance for Drug Response Prediction')
plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Prediction Errors Visualization
print("Creating prediction error visualization...")
n_samples = 50
actual_values = np.random.normal(0, 1, n_samples)
predicted_values = actual_values + np.random.normal(0, 0.5, n_samples)  # Add some noise
groups = np.random.choice(['A', 'B', 'C', 'D'], n_samples)

plt.figure(figsize=(10, 8))
for group in np.unique(groups):
    mask = groups == group
    plt.scatter(
        actual_values[mask], 
        predicted_values[mask], 
        label=f'Group {group}',
        alpha=0.7,
        s=80
    )

# Add diagonal line (perfect predictions)
min_val = min(actual_values.min(), predicted_values.min())
max_val = max(actual_values.max(), predicted_values.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.title('Prediction Accuracy for Gene Response')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots/prediction_errors.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Cell Type Performance
print("Creating cell type performance visualization...")
cell_types = ['T Cells', 'B Cells', 'NK Cells', 'Monocytes', 'Dendritic Cells']
r2_scores = np.random.uniform(0.4, 0.9, len(cell_types))
mse_scores = np.random.uniform(0.1, 0.6, len(cell_types))
sample_counts = np.random.randint(50, 200, len(cell_types))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# R² Plot
bars1 = ax1.bar(cell_types, r2_scores, color='skyblue')
ax1.set_ylim(0, 1)
ax1.set_title('R² Score by Cell Type')
ax1.set_xlabel('Cell Type')
ax1.set_ylabel('R² Score')

# Add sample counts
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width()/2,
        height/2,
        f'n={sample_counts[i]}',
        ha='center',
        va='center',
        color='white',
        fontweight='bold'
    )

# MSE Plot
bars2 = ax2.bar(cell_types, mse_scores, color='salmon')
ax2.set_title('Mean Squared Error by Cell Type')
ax2.set_xlabel('Cell Type')
ax2.set_ylabel('MSE')

# Add sample counts
for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width()/2,
        height/2,
        f'n={sample_counts[i]}',
        ha='center',
        va='center',
        color='white',
        fontweight='bold'
    )

plt.tight_layout()
plt.savefig('plots/cell_type_performance.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Drug Performance
print("Creating drug performance visualization...")
drugs = [f'Drug_{i}' for i in range(10)]
drug_r2 = np.random.uniform(0.3, 0.8, len(drugs))
drug_mse = np.random.uniform(0.2, 0.7, len(drugs))
drug_samples = np.random.randint(10, 100, len(drugs))

# Sort by R2
sorted_indices = np.argsort(drug_r2)[::-1]
sorted_drugs = [drugs[i] for i in sorted_indices]
sorted_r2 = drug_r2[sorted_indices]
sorted_mse = drug_mse[sorted_indices]
sorted_samples = drug_samples[sorted_indices]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# R² Plot
bars1 = ax1.barh(sorted_drugs, sorted_r2, color='skyblue')
ax1.set_xlim(0, 1)
ax1.set_title('R² Score by Drug')
ax1.set_xlabel('R² Score')
ax1.set_ylabel('Drug')

# Add sample counts
for i, bar in enumerate(bars1):
    width = bar.get_width()
    ax1.text(
        width/2,
        bar.get_y() + bar.get_height()/2,
        f'n={sorted_samples[i]}',
        ha='center',
        va='center',
        color='white',
        fontweight='bold'
    )

# MSE Plot
bars2 = ax2.barh(sorted_drugs, sorted_mse, color='salmon')
ax2.set_title('Mean Squared Error by Drug')
ax2.set_xlabel('MSE')
ax2.set_ylabel('')  # No need to repeat label

# Add sample counts
for i, bar in enumerate(bars2):
    width = bar.get_width()
    ax2.text(
        width/2,
        bar.get_y() + bar.get_height()/2,
        f'n={sorted_samples[i]}',
        ha='center',
        va='center',
        color='white',
        fontweight='bold'
    )

plt.tight_layout()
plt.savefig('plots/drug_performance.png', dpi=300, bbox_inches='tight')
plt.close()

print("All test visualizations have been generated in the 'plots/' directory.") 