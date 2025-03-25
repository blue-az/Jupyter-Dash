#!/usr/bin/env python3
"""
Visualization functions for the Zip Digit Classification project.

This module provides visualization functions for exploring and analyzing
the results of different LDA methods for digit classification.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# Import utility functions
from zip_utils import apply_block_averaging

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
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the visualization
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
            
            # Display the image
            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].axis('off')
            
            # Add digit label to the first column
            if j == 0:
                axes[i, j].set_title(f'Digit {int(digit)}')
    
    plt.tight_layout()
    return fig

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
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the confusion matrix plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    
    # Add text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.tight_layout()
    ax.grid(False)
    return fig

def plot_pca_explained_variance(X, n_components=30, figsize=(10, 6)):
    """
    Plot explained variance ratio of PCA components
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature data
    n_components : int
        Number of components to plot
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the plot
    """
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca.fit(X)
    
    # Plot explained variance ratio
    fig, ax = plt.subplots(figsize=figsize)
    
    # Individual explained variance
    ax.bar(
        range(1, n_components + 1),
        pca.explained_variance_ratio_,
        alpha=0.7,
        align='center',
        label='Individual Explained Variance'
    )
    
    # Cumulative explained variance
    ax.step(
        range(1, n_components + 1),
        np.cumsum(pca.explained_variance_ratio_),
        where='mid',
        label='Cumulative Explained Variance',
        color='red'
    )
    
    # Add cumulative variance values at key points
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    thresholds = [5, 10, 15, 20, 25, 30]
    for t in thresholds:
        if t <= n_components:
            ax.text(
                t, cumsum[t-1] + 0.02,
                f'{cumsum[t-1]:.2f}',
                ha='center',
                fontsize=9
            )
    
    # Add labels and title
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('Explained Variance by PCA Components')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()
    return fig

def visualize_block_averaging(X, block_size=3, n_samples=3, figsize=(15, 10)):
    """
    Visualize the effect of block averaging on digit images
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature data
    block_size : int
        Size of blocks to average
    n_samples : int
        Number of samples to visualize
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the visualization
    """
    # Select random samples
    indices = np.random.choice(X.shape[0], size=n_samples, replace=False)
    X_samples = X[indices]
    
    # Calculate dimensions
    img_dim = int(np.sqrt(X.shape[1]))
    avg_dim = img_dim // block_size
    
    # Set up the figure
    fig, axes = plt.subplots(n_samples, 2, figsize=figsize)
    
    for i, sample in enumerate(X_samples):
        # Original image
        img = sample.reshape(img_dim, img_dim)
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f'Original ({img_dim}×{img_dim})')
        axes[i, 0].axis('off')
        
        # Apply block averaging
        avg_img = np.zeros((avg_dim, avg_dim))
        for r in range(avg_dim):
            for c in range(avg_dim):
                block = img[r*block_size:(r+1)*block_size, c*block_size:(c+1)*block_size]
                avg_img[r, c] = np.mean(block)
        
        # Display block-averaged image
        axes[i, 1].imshow(avg_img, cmap='gray')
        axes[i, 1].set_title(f'Block Averaged ({avg_dim}×{avg_dim})')
        axes[i, 1].axis('off')
    
    fig.tight_layout()
    return fig

def plot_accuracy_comparison(results, figsize=(10, 6)):
    """
    Plot accuracy comparison between different models
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results from different models
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract model names and accuracy values
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    
    # Create prettier labels
    labels = [results[model]['method_name'] for model in models]
    
    # Sort by accuracy (descending)
    sorted_indices = np.argsort(accuracies)[::-1]
    labels = [labels[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    
    # Create bar plot
    bars = ax.bar(labels, accuracies, color='skyblue')
    
    # Add value labels on top of each bar
    for bar, accuracy in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{accuracy:.4f}',
            ha='center',
            fontsize=10
        )
    
    # Add labels and title
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison')
    ax.set_ylim(0, 1.1)  # Set y-axis limit to [0, 1.1]
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    return fig

def plot_time_comparison(results, figsize=(10, 6)):
    """
    Plot execution time comparison between different models
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results from different models
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract model names and times
    models = list(results.keys())
    times = [results[model]['time'] for model in models]
    
    # Create prettier labels
    labels = [results[model]['method_name'] for model in models]
    
    # Sort by time (ascending)
    sorted_indices = np.argsort(times)
    labels = [labels[i] for i in sorted_indices]
    times = [times[i] for i in sorted_indices]
    
    # Create bar plot
    bars = ax.bar(labels, times, color='lightgreen')
    
    # Add value labels on top of each bar
    for bar, time_val in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{time_val:.2f}s',
            ha='center',
            fontsize=10
        )
    
    # Add labels and title
    ax.set_xlabel('Model')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Model Execution Time Comparison')
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    return fig

def visualize_regularization_path(X, y, block_size=3, n_alphas=10, figsize=(12, 8)):
    """
    Visualize the regularization path for logistic regression
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature data
    y : numpy.ndarray
        Label data
    block_size : int
        Size of blocks to average
    n_alphas : int
        Number of regularization parameters to try
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the plot
    """
    # Apply block averaging
    X_avg = apply_block_averaging(X, block_size)
    
    # Set up alphas (equivalent to lambda in glmnet, but inverted)
    alphas = np.logspace(-4, 0, n_alphas)
    C_values = 1 / alphas  # C in sklearn is inverse of alpha
    
    # Track number of non-zero coefficients for each class
    n_classes = len(np.unique(y))
    n_features = X_avg.shape[1]
    
    # Store results
    nonzero_coefs = np.zeros((n_alphas, n_classes))
    accuracies = np.zeros(n_alphas)
    
    # Fit models with different regularization strengths
    for i, C in enumerate(C_values):
        # Fit model
        model = LogisticRegression(
            C=C,
            penalty='l1',
            solver='saga',
            max_iter=10000,
            tol=1e-4,
            random_state=42
        )
        model.fit(X_avg, y)
        
        # Predict
        y_pred = model.predict(X_avg)
        accuracies[i] = np.mean(y_pred == y)
        
        # Count non-zero coefficients
        for j in range(n_classes):
            nonzero_coefs[i, j] = np.sum(np.abs(model.coef_[j]) > 1e-10)
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot coefficient paths
    colors = plt.cm.viridis(np.linspace(0, 1, n_classes))
    for j in range(n_classes):
        ax1.plot(alphas, nonzero_coefs[:, j], '-o', color=colors[j], 
                label=f'Class {np.unique(y)[j]}')
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Alpha (regularization strength)')
    ax1.set_ylabel('Number of non-zero coefficients')
    ax1.set_title('Regularization Path for Logistic Regression')
    ax1.legend(loc='upper left')
    
    # Add accuracy on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(alphas, accuracies, '--', color='red', label='Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='upper right')
    
    # Annotate "end of the path"
    best_alpha_idx = np.argmax(accuracies)
    best_alpha = alphas[best_alpha_idx]
    plt.axvline(x=best_alpha, linestyle='--', color='black', alpha=0.5)
    plt.text(
        best_alpha, 
        0.5, 
        f'End of path: alpha={best_alpha:.5f}',
        rotation=90,
        verticalalignment='center'
    )
    
    plt.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()
    return fig

def compare_class_separation(results, figsize=(15, 10)):
    """
    Visualize the class separation achieved by different methods
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results from different methods
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the plot
    """
    # Calculate number of rows and columns for subplots
    n_methods = len(results)
    n_cols = min(3, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols  # Ceiling division
    
    # Set up the figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each method's separation
    for i, (method_name, result) in enumerate(results.items()):
        if i >= len(axes):
            break
            
        # Get model and data
        model = result['model']
        X_test = result.get('X_test', None)
        y_test = result.get('y_test', None)
        y_pred = result['predictions']
        
        # Skip if no test data available
        if X_test is None or y_test is None:
            axes[i].text(0.5, 0.5, f"{method_name}\n(Visualization not available)",
                      ha='center', va='center', fontsize=12)
            axes[i].set_title(method_name)
            axes[i].axis('off')
            continue
        
        # Transform data for visualization (2D)
        try:
            if isinstance(model, dict) and 'pca' in model and 'lda' in model:
                # For class-specific PCA + LDA
                pca_model = model['pca']
                lda_model = model['lda']
                X_pca = pca_model.transform(X_test, y_test)
                try:
                    X_transformed = lda_model.transform(X_pca)
                except:
                    # If transform not available, use PCA
                    pca = PCA(n_components=2)
                    X_transformed = pca.fit_transform(X_pca)
            elif hasattr(model, 'named_steps') and 'pca' in model.named_steps:
                # For PCA + LDA pipeline
                try:
                    X_transformed = model.transform(X_test)
                except:
                    # Try to get PCA components
                    X_pca = model.named_steps['pca'].transform(X_test)
                    pca_viz = PCA(n_components=2)
                    X_transformed = pca_viz.fit_transform(X_pca)
            else:
                # For standard LDA, regularized LDA
                try:
                    # Try using transform if available
                    X_transformed = model.transform(X_test)
                except:
                    try:
                        # Try using decision function
                        decision_scores = model.decision_function(X_test)
                        pca = PCA(n_components=2)
                        X_transformed = pca.fit_transform(decision_scores)
                    except:
                        # Last resort: PCA on original features
                        pca = PCA(n_components=2)
                        X_transformed = pca.fit_transform(X_test)
        except:
            # Fallback: just use PCA on original features
            pca = PCA(n_components=2)
            X_transformed = pca.fit_transform(X_test)
        
        # Make sure we have at least 2 dimensions
        if X_transformed.shape[1] < 2:
            X_transformed = np.column_stack([X_transformed, np.zeros_like(X_transformed)])
        
        # Plot the transformed data
        title = result.get('method_name', method_name)
        scatter = axes[i].scatter(
            X_transformed[:, 0],
            X_transformed[:, 1],
            c=y_test,
            cmap='viridis',
            alpha=0.7,
            s=30
        )
        
        # Mark misclassified samples
        mask = y_test != y_pred
        if np.any(mask):
            axes[i].scatter(
                X_transformed[mask, 0],
                X_transformed[mask, 1],
                facecolors='none',
                edgecolors='red',
                s=100,
                label='Misclassified'
            )
            
        # Add legend
        legend1 = axes[i].legend(*scatter.legend_elements(), title="Classes")
        axes[i].add_artist(legend1)
        
        if np.any(mask):
            axes[i].legend(['Misclassified'], loc='lower right')
            
        axes[i].set_title(title)
        axes[i].set_xlabel('Component 1')
        axes[i].set_ylabel('Component 2')
        axes[i].grid(True, linestyle='--', alpha=0.7)
        
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
        
    fig.tight_layout()
    return fig
