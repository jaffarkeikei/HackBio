import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors

# Set style
plt.style.use('seaborn-v0_8')  # Using the updated seaborn style name

# Load the dataset
print("Loading dataset...")
df = pd.read_parquet('dataset/de_train_split.parquet')

# Create directory for plots
import os
os.makedirs('plots', exist_ok=True)

# 1. Cell Type Distribution
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='cell_type')
plt.xticks(rotation=45)
plt.title('Distribution of Cell Types')
plt.tight_layout()
plt.savefig('plots/cell_type_distribution.png')
plt.close()

# 2. Gene Expression Distribution
metadata_cols = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']
gene_cols = [col for col in df.columns if col not in metadata_cols]

# Sample 1000 random gene expression values
sample_values = df[gene_cols].values.flatten()
np.random.shuffle(sample_values)
sample_values = sample_values[:1000]

plt.figure(figsize=(12, 6))
sns.histplot(sample_values, bins=50)
plt.title('Distribution of Gene Expression Values (Sample)')
plt.xlabel('Expression Value')
plt.ylabel('Count')
plt.savefig('plots/gene_expression_distribution.png')
plt.close()

# 3. Expression Heatmap (top variable genes)
# Calculate variance for each gene
gene_variance = df[gene_cols].var()
top_variable_genes = gene_variance.nlargest(20).index

plt.figure(figsize=(15, 8))
sns.heatmap(df[top_variable_genes].iloc[:20], cmap='viridis', center=0)
plt.title('Expression Heatmap of Top 20 Variable Genes')
plt.xlabel('Genes')
plt.ylabel('Samples')
plt.tight_layout()
plt.savefig('plots/gene_expression_heatmap.png')
plt.close()

# 4. Drug-Cell Type Interaction
drug_cell_counts = df.groupby(['sm_name', 'cell_type']).size().unstack()
plt.figure(figsize=(15, 8))
sns.heatmap(drug_cell_counts.iloc[:20], cmap='YlOrRd')
plt.title('Drug-Cell Type Interaction (Top 20 Drugs)')
plt.xlabel('Cell Type')
plt.ylabel('Drug Name')
plt.tight_layout()
plt.savefig('plots/drug_cell_interaction.png')
plt.close()

# 5. Control vs Treatment Expression
# Select a few random genes
random_genes = np.random.choice(gene_cols, 5)
melted_df = pd.melt(df[['control'] + list(random_genes)], 
                    id_vars=['control'], 
                    var_name='gene', 
                    value_name='expression')

plt.figure(figsize=(12, 6))
sns.boxplot(data=melted_df, x='gene', y='expression', hue='control')
plt.title('Gene Expression in Control vs Treatment (Sample Genes)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/control_vs_treatment.png')
plt.close()

# 6. Molecular Weight Distribution
molecular_weights = []
for smiles in df['SMILES']:
    mol = Chem.MolFromSmiles(smiles)
    mw = Descriptors.ExactMolWt(mol)
    molecular_weights.append(mw)

plt.figure(figsize=(12, 6))
sns.histplot(molecular_weights, bins=30)
plt.title('Distribution of Molecular Weights')
plt.xlabel('Molecular Weight')
plt.ylabel('Count')
plt.savefig('plots/molecular_weight_distribution.png')
plt.close()

print("Visualizations have been saved in the 'plots' directory.")

# Save some additional statistics
stats = {
    'mean_expression': df[gene_cols].mean().mean(),
    'std_expression': df[gene_cols].std().mean(),
    'median_molecular_weight': np.median(molecular_weights),
    'num_high_variable_genes': sum(gene_variance > gene_variance.median()),
}

with open('plots/statistics.txt', 'w') as f:
    for key, value in stats.items():
        f.write(f"{key}: {value}\n") 