import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from correlation_analysis import CorrelationAnalyzer

# Create sample test data
print("Creating sample data for correlation analysis...")
np.random.seed(42)

# Create random descriptors
n_compounds = 20
n_properties = 5
descriptors = pd.DataFrame({
    'MolecularWeight': np.random.normal(300, 50, n_compounds),
    'LogP': np.random.normal(3, 1, n_compounds),
    'TPSA': np.random.normal(70, 15, n_compounds),
    'HBondAcceptors': np.random.normal(4, 1, n_compounds),
    'NumRings': np.random.normal(2, 1, n_compounds)
})

# Create fictional drug names
descriptors['Drug'] = [f'Drug_{i}' for i in range(n_compounds)]
descriptors.to_csv('results/test_descriptors.csv', index=False)
print("Created sample descriptors and saved to results/test_descriptors.csv")

# Create random gene expression data
n_genes = 10
gene_cols = [f'Gene_{i}' for i in range(n_genes)]
gene_data = pd.DataFrame(
    np.random.normal(0, 1, (n_compounds, n_genes)),
    columns=gene_cols
)
gene_data['sm_name'] = descriptors['Drug']
gene_data['control'] = 0  # All are treatment samples in this test
gene_data.to_csv('results/test_gene_data.csv', index=False)
print("Created sample gene expression data and saved to results/test_gene_data.csv")

# Test correlation analyzer
try:
    # Initialize with the sample descriptors
    analyzer = CorrelationAnalyzer('results/test_descriptors.csv')
    
    # Load test response data 
    analyzer.load_response_data('results/test_gene_data.csv')
    
    # Calculate mean responses (simulated)
    mean_responses = analyzer.prepare_mean_responses(by='sm_name', control_norm=False)
    print(f"Calculated mean responses with shape: {mean_responses.shape}")
    
    # Calculate correlations
    corr_matrix, p_vals = analyzer.calculate_correlations(mean_responses, method='pearson')
    print("Generated correlation matrix")
    
    # Save results
    corr_matrix.to_csv('results/test_correlations.csv')
    print("Saved correlation matrix to results/test_correlations.csv")
    
    # Plot heatmap
    fig = analyzer.plot_correlation_heatmap(n_top_genes=10)
    fig.savefig('plots/test_correlation_heatmap.png', dpi=200, bbox_inches='tight')
    print("Saved correlation heatmap to plots/test_correlation_heatmap.png")
    
    # Try clustered heatmap
    fig2 = analyzer.plot_clustered_heatmap(n_clusters=3, n_top_genes=10)
    fig2.savefig('plots/test_clustered_heatmap.png', dpi=200, bbox_inches='tight')
    print("Saved clustered heatmap to plots/test_clustered_heatmap.png")
    
except Exception as e:
    print(f"Error in correlation analysis: {e}")
    import traceback
    traceback.print_exc() 