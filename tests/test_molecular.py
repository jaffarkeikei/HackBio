import pandas as pd
import matplotlib.pyplot as plt
from molecular_analysis import MolecularAnalyzer

# Load a small sample of the dataset
df = pd.read_parquet('dataset/de_train_split.parquet').head(20)
print(f'Loaded data with shape: {df.shape}')

# Initialize analyzer with the SMILES data
analyzer = MolecularAnalyzer(df['SMILES'])

# Calculate descriptors
try:
    descriptors = analyzer.calculate_descriptors()
    print(f'Generated descriptors for {len(descriptors)} molecules')
    
    # Save descriptors
    descriptors.to_csv('results/sample_descriptors.csv', index=False)
    print('Saved descriptors to results/sample_descriptors.csv')
    
    # Try plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(descriptors['MolecularWeight'], descriptors['LogP'], alpha=0.7)
    plt.xlabel('Molecular Weight')
    plt.ylabel('LogP')
    plt.title('Molecular Weight vs LogP')
    plt.savefig('plots/molecular_properties.png')
    print('Saved plot to plots/molecular_properties.png')
except Exception as e:
    print(f'Error: {e}') 