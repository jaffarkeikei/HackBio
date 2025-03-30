import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import pickle
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename='logs/svd_optimization.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_and_preprocess_data():
    """Load the preprocessed data from advanced_results"""
    try:
        # Load the preprocessed data
        X_train = np.load('results/advanced/advanced_results/X_train.npy')
        X_val = np.load('results/advanced/advanced_results/X_val.npy')
        y_train = np.load('results/advanced/advanced_results/y_train.npy')
        y_val = np.load('results/advanced/advanced_results/y_val.npy')
        
        # Combine train and validation for SVD analysis
        X = np.vstack([X_train, X_val])
        y = np.vstack([y_train, y_val])
        
        logging.info(f'Loaded preprocessed data: X shape {X.shape}, y shape {y.shape}')
        return X, y
        
    except Exception as e:
        logging.error(f'Error loading data: {str(e)}')
        return None, None

def analyze_svd_components(X, y, step=10):
    """Analyze different numbers of SVD components"""
    n_features = X.shape[1]
    max_components = n_features
    components_range = range(step, max_components + 1, step)
    results = []
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    for n_components in components_range:
        # Fit SVD
        svd = TruncatedSVD(n_components=n_components)
        X_transformed = svd.fit_transform(X_scaled)
        
        # Calculate explained variance
        explained_variance = svd.explained_variance_ratio_.sum()
        
        results.append({
            'n_components': n_components,
            'explained_variance': explained_variance,
            'X_shape': X_transformed.shape
        })
        
        logging.info(f'Components: {n_components}, Explained Variance: {explained_variance:.4f}')
    
    return results

def plot_results(results):
    """Create visualization of SVD analysis"""
    # Create results directory
    os.makedirs('results/svd/svd_analysis', exist_ok=True)
    
    # Extract data
    components = [r['n_components'] for r in results]
    variances = [r['explained_variance'] for r in results]
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(components, variances, marker='o')
    plt.title('Explained Variance vs Number of SVD Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.grid(True)
    
    # Add value labels
    for i, (comp, var) in enumerate(zip(components, variances)):
        plt.annotate(f'{var:.3f}', 
                    (comp, var),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center')
    
    plt.savefig('results/svd/svd_analysis/variance_curve.png')
    plt.close()
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/svd/svd_analysis/svd_results.csv', index=False)
    
    return results_df

def save_optimal_transformation(X, optimal_components):
    """Save the optimally transformed data"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    svd = TruncatedSVD(n_components=optimal_components)
    X_transformed = svd.fit_transform(X_scaled)
    
    # Save transformation objects
    with open('results/svd/svd_analysis/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('results/svd/svd_analysis/svd_model.pkl', 'wb') as f:
        pickle.dump(svd, f)
    
    # Save transformed data
    np.save('results/svd/svd_analysis/X_transformed.npy', X_transformed)
    
    logging.info(f'Saved optimal transformation with {optimal_components} components')
    return X_transformed

def main():
    # Load data
    X, y = load_and_preprocess_data()
    if X is None or y is None:
        return
    
    # Analyze SVD components
    results = analyze_svd_components(X, y, step=10)
    
    # Plot and save results
    results_df = plot_results(results)
    
    # Find optimal number of components
    # Choose the number of components that explains 98% of variance
    target_variance = 0.98
    optimal_components = None
    
    for result in results:
        if result['explained_variance'] >= target_variance:
            optimal_components = result['n_components']
            break
    
    if optimal_components is None:
        optimal_components = results[-1]['n_components']
        logging.warning(f'Could not reach {target_variance*100}% variance, using maximum {optimal_components} components')
    else:
        logging.info(f'Found optimal components: {optimal_components} (explains {target_variance*100}% variance)')
    
    # Save optimal transformation
    X_transformed = save_optimal_transformation(X, optimal_components)
    
    logging.info('SVD optimization completed')
    
    return optimal_components, X_transformed

if __name__ == '__main__':
    main() 