import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

# Load the dataset
print("Loading dataset...")
df = pd.read_parquet('dataset/de_train_split.parquet')

# 1. Basic Dataset Information
print("\n=== Basic Dataset Information ===")
print(f"Dataset shape: {df.shape}")
print("\nColumns:", df.columns[:10].tolist(), "... and more")
print("\nMemory usage:", df.memory_usage().sum() / 1024**2, "MB")

# 2. Cell Type Analysis
print("\n=== Cell Type Analysis ===")
print("\nCell type distribution:")
print(df['cell_type'].value_counts())

# 3. Drug Analysis
print("\n=== Drug Analysis ===")
print(f"Number of unique drugs: {df['sm_name'].nunique()}")
print("\nTop 5 most common drugs:")
print(df['sm_name'].value_counts().head())

# 4. Control Sample Analysis
print("\n=== Control Sample Analysis ===")
print("\nControl vs Treatment distribution:")
print(df['control'].value_counts())

# 5. Gene Expression Analysis
metadata_cols = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']
gene_cols = [col for col in df.columns if col not in metadata_cols]

print("\n=== Gene Expression Analysis ===")
print(f"Number of genes: {len(gene_cols)}")

gene_stats = df[gene_cols].agg(['mean', 'std', 'min', 'max'])
print("\nGene expression summary statistics:")
print(gene_stats.describe())

# 6. SMILES Analysis
print("\n=== SMILES Analysis ===")
print("Sample SMILES strings:")
for i, smiles in enumerate(df['SMILES'].head(3)):
    mol = Chem.MolFromSmiles(smiles)
    mw = Descriptors.ExactMolWt(mol)
    print(f"\nMolecule {i+1}:")
    print(f"SMILES: {smiles[:50]}...")
    print(f"Molecular Weight: {mw:.2f}")
    print(f"Number of atoms: {mol.GetNumAtoms()}")

# 7. Save summary statistics
print("\n=== Saving Summary Statistics ===")
summary_stats = {
    'total_samples': len(df),
    'unique_drugs': df['sm_name'].nunique(),
    'unique_cell_types': df['cell_type'].nunique(),
    'num_genes': len(gene_cols),
    'control_samples': df['control'].sum(),
    'treatment_samples': len(df) - df['control'].sum()
}

with open('data_summary.txt', 'w') as f:
    f.write("=== Dataset Summary ===\n")
    for key, value in summary_stats.items():
        f.write(f"{key}: {value}\n") 