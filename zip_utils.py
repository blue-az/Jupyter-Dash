#!/usr/bin/env python3
"""
Utility functions for the Zip Digit Classification project.

This module contains commonly used functions for data loading, preprocessing,
and result evaluation that can be imported by other scripts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
from datetime import datetime
import os

def load_zip_data(file_path):
    """
    Load zip data from file
    
    Parameters:
    -----------
    file_path : str
        Path to the zip data file
    
    Returns:
    --------
    X : numpy.ndarray
        Features (pixel values)
    y : numpy.ndarray
        Labels (digit values)
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            values = [float(val) for val in line.strip().split()]
            data.append(values)
    
    data = np.array(data)
    
    # First column is the label, the rest are features
    y = data[:, 0].astype(int)
    X = data[:, 1:]
    
    return X, y

def check_download_data():
    """
    Check if zip code data files exist
    
    Returns:
    --------
    bool: True if data files exist, False otherwise
    """
    return os.path.exists('zip.train') and os.path.exists('zip.test')

def filter_digits(X, y, digits=[3, 5, 8]):
    """
    Filter data to include only specified digits
    
    Parameters:
    -----------
    X : numpy.ndarray
        Features
    y : numpy.ndarray
        Labels
    digits : list
        List of digits to include
    
    Returns:
    --------
    X_filtered : numpy.ndarray
        Filtered features
    y_filtered : numpy.ndarray
        Filtered labels
    """
    mask = np.isin(y, digits)
    return X[mask], y[mask]

def apply_block_averaging(X, block_size=3):
    """
    Apply block averaging to image data
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature data with shape (n_samples, n_features)
        Assuming each sample is a flattened square image
    block_size : int
        Size of the square blocks to average
    
    Returns:
    --------
    X_averaged : numpy.ndarray
        Feature data with block averaging applied
    """
    n_samples = X.shape[0]
    
    # Determine the dimensions of the original image
    img_dim = int(np.sqrt(X.shape[1]))
    
    # Calculate the dimensions of the averaged image
    avg_dim = img_dim // block_size
    
    # Initialize the output array
    X_averaged = np.zeros((n_samples, avg_dim * avg_dim))
    
    # Apply block averaging to each sample
    for i in range(n_samples):
        # Reshape the flattened image to 2D
        img = X[i].reshape(img_dim, img_dim)
        
        # Initialize the averaged image
        avg_img = np.zeros((avg_dim, avg_dim))
        
        # Apply block averaging
        for r in range(avg_dim):
            for c in range(avg_dim):
                # Extract the block
                block = img[r*block_size:(r+1)*block_size, c*block_size:(c+1)*block_size]
                # Compute the average
                avg_img[r, c] = np.mean(block)
        
        # Flatten the averaged image and store it
        X_averaged[i] = avg_img.ravel()
    
    return X_averaged

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', figsize=(8, 6), cmap=plt.cm.Blues):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    cm : numpy.ndarray
        Confusion matrix
    classes : list
        List of class labels
    title : str
        Plot title
    figsize : tuple
        Figure size
    cmap : colormap
        Colormap to use
    """
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.grid(False)
    return plt

def visualize_digit_samples(X, y, n_samples=5, figsize=(12, 8)):
    """
    Visualize sample digit images
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature data
    y : numpy.ndarray
        Label data
    n_samples : int
        Number of samples to display per digit
    figsize : tuple
        Figure size
    """
    digits = np.unique(y)
    fig, axes = plt.subplots(len(digits), n_samples, figsize=figsize)
    
    for i, digit in enumerate(digits):
        # Get indices of this digit
        indices = np.where(y == digit)[0]
        
        # Select random samples
        sample_indices = np.random.choice(indices, size=n_samples, replace=False)
        
        for j, idx in enumerate(sample_indices):
            # Reshape the 256 features to 16x16 image
            img = X[idx].reshape(16, 16)
            
            # Convert from [-1, 1] range to [0, 1] for display
            img = (img + 1) / 2  
            
            # Display the image
            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].axis('off')
            
            # Add digit label to the first column
            if j == 0:
                axes[i, j].set_title(f'Digit {int(digit)}')
    
    plt.tight_layout()
    return plt

def plot_accuracy_comparison(results):
    """
    Plot accuracy comparison between different models
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results from different models
    """
    plt.figure(figsize=(10, 6))
    
    # Extract model names and accuracy values
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    
    # Create prettier labels
    labels = [' '.join(model.split('_')).title() for model in models]
    
    # Create bar plot
    bars = plt.bar(labels, accuracies, color='skyblue')
    
    # Add value labels on top of each bar
    for bar, accuracy in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{accuracy:.4f}',
            ha='center',
            fontsize=10
        )
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison of Classification Methods')
    plt.ylim(0, 1.1)  # Set y-axis limit to [0, 1.1]
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    return plt

def get_per_class_metrics(result):
    """
    Extract per-class metrics from a classification report
    
    Parameters:
    -----------
    result : dict
        Dictionary containing a 'report' key with a classification report string
    
    Returns:
    --------
    metrics : dict
        Dictionary of per-class metrics
    """
    report = result['report']
    lines = report.split('\n')
    metrics = {}
    
    for line in lines:
        if '3' in line.split() or '5' in line.split() or '8' in line.split():
            parts = line.split()
            if len(parts) >= 5:
                cls = parts[0]
                precision = float(parts[1])
                recall = float(parts[2])
                f1 = float(parts[3])
                metrics[cls] = {'precision': precision, 'recall': recall, 'f1': f1}
    
    return metrics

def create_summary_dataframe(results):
    """
    Create a summary DataFrame from results dictionary
    
    Parameters:
    -----------
    results : dict
        Dictionary of results from different methods
    
    Returns:
    --------
    summary_df : pandas.DataFrame
        DataFrame summarizing the results
    """
    summary_data = []
    for name, result in results.items():
        method_name = name.replace('_', ' ').title()
        metrics = get_per_class_metrics(result)
        
        # Determine feature count based on method
        if 'processed_features' in result:
            features = result['processed_features']
        elif name == 'class_pca_lda':
            features = result['n_components'] * len(metrics)
        elif 'pca' in name:
            features = int(name.split('_')[1])  # Extract number from pca_XX_lda
        else:
            features = 256  # Original feature count
        
        row = {
            'Method': method_name,
            'Accuracy': result['accuracy'],
            'Time (s)': result.get('time', 0.0),
            'Features': features,
            'Class 3 F1': metrics.get('3', {}).get('f1', 0),
            'Class 5 F1': metrics.get('5', {}).get('f1', 0),
            'Class 8 F1': metrics.get('8', {}).get('f1', 0),
        }
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Accuracy', ascending=False)
    
    return summary_df

def save_results(results, X_train_filtered, y_train_filtered, output_dir='results'):
    """
    Save results to files
    
    Parameters:
    -----------
    results : dict
        Dictionary of results from different methods
    X_train_filtered : numpy.ndarray
        Training features
    y_train_filtered : numpy.ndarray
        Training labels
    output_dir : str
        Directory to save results
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create and save summary DataFrame
    summary_df = create_summary_dataframe(results)
    summary_df.to_csv(f"{output_dir}/model_comparison.csv", index=False)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Find best method in terms of accuracy
    best_method = max(results.items(), key=lambda x: x[1]['accuracy'])
    best_method_name = best_method[0].replace('_', ' ').title()
    best_accuracy = best_method[1]['accuracy']
    
    # Find fastest method
    fastest_method = min(results.items(), key=lambda x: x[1].get('time', float('inf')))
    fastest_method_name = fastest_method[0].replace('_', ' ').title()
    fastest_time = fastest_method[1].get('time', 0.0)
    
    # Generate summary text
    summary = f"""# Zip Code Digit Classification Results Summary
Generated on: {timestamp}

## Dataset Information
- Task: Classification of handwritten digits (3, 5, and 8) from zip code data
- Feature dimensions: 16×16 pixel images (256 features per sample)

## Methods Compared

"""
    
    # Add method-specific information
    for name, result in results.items():
        method_name = name.replace('_', ' ').title()
        summary += f"### {method_name}\n"
        summary += f"- Accuracy: {result['accuracy']:.4f}\n"
        summary += f"- Execution time: {result.get('time', 0.0):.2f} seconds\n"
        
        if 'processed_features' in result:
            summary += f"- Feature count: {result['processed_features']} (reduced dimensions)\n"
        elif 'explained_variance_ratio' in result:
            summary += f"- Explained variance ratio: {result['explained_variance_ratio']:.4f}\n"
            summary += f"- Feature count: 30 (reduced dimensions)\n"
        else:
            summary += f"- Feature count: 256 (original dimensions)\n"
            
        summary += "\n"
    
    # Add key findings
    summary += f"""## Key Findings

1. **Most Accurate Method**: {best_method_name} (Accuracy: {best_accuracy:.4f})
2. **Fastest Method**: {fastest_method_name} (Time: {fastest_time:.2f} seconds)
"""

    # Save the summary to a file
    with open(f"{output_dir}/results_summary.md", 'w') as f:
        f.write(summary)
    
    print(f"Results saved to {output_dir}/")
    return summary
