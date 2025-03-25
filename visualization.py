#!/usr/bin/env python3
"""
Visualization Module for Zip Digit Classification

This module provides visualization functions for exploring and analyzing
the results of different LDA methods for digit classification.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

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

def plot_accuracy_comparison(results, title='Model Comparison', figsize=(10, 6)):
    """
    Plot accuracy comparison between different models
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results from different models
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    
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
    plt.title(title)
    plt.ylim(0, 1.1)  # Set y-axis limit to [0, 1.1]
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    return plt

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
    """
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca.fit(X)
    
    # Plot explained variance ratio
    plt.figure(figsize=figsize)
    
    # Individual explained variance
    plt.bar(
        range(1, n_components + 1),
        pca.explained_variance_ratio_,
        alpha=0.7,
        align='center',
        label='Individual Explained Variance'
    )
    
    # Cumulative explained variance
    plt.step(
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
            plt.text(
                t, cumsum[t-1] + 0.02,
                f'{cumsum[t-1]:.2f}',
                ha='center',
                fontsize=9
            )
    
    # Add labels and title
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by PCA Components')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
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

def visualize_feature_importance(model, feature_names=None, top_n=20, figsize=(12, 8)):
    """
    Visualize feature importance from a linear model like LDA
    
    Parameters:
    -----------
    model : object
        Trained model with coef_ attribute
    feature_names : list
        Names of features
    top_n : int
        Number of top features to display
    figsize : tuple
        Figure size
    """
    # Check if model has the right attributes
    if not hasattr(model, 'coef_'):
        raise ValueError("Model does not have coef_ attribute")
    
    # Get coefficients
    coefs = model.coef_
    
    # Create default feature names if none provided
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(coefs.shape[1])]
    
    # Create a DataFrame for easier manipulation
    coef_df = pd.DataFrame()
    
    for i, class_label in enumerate(model.classes_):
        class_coef = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefs[i],
            'Abs_Coefficient': np.abs(coefs[i]),
            'Class': f'Class {class_label}'
        })
        coef_df = pd.concat([coef_df, class_coef])
    
    # Get top features by absolute coefficient value
    top_features = (coef_df.groupby('Feature')['Abs_Coefficient']
                    .mean()
                    .sort_values(ascending=False)
                    .head(top_n)
                    .index.tolist())
    
    # Filter to only include top features
    plot_df = coef_df[coef_df['Feature'].isin(top_features)]
    
    # Plot
    plt.figure(figsize=figsize)
    sns.barplot(
        data=plot_df,
        x='Coefficient',
        y='Feature',
        hue='Class',
        palette='viridis'
    )
    
    plt.title(f'Top {top_n} Feature Importance by Class')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    return plt

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
    """
    # Import the function to apply block averaging
    from lda_methods import apply_block_averaging
    
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
        scaled_img = (img + 1) / 2  # Scale from [-1, 1] to [0, 1]
        axes[i, 0].imshow(scaled_img, cmap='gray')
        axes[i, 0].set_title(f'Original ({img_dim}×{img_dim})')
        axes[i, 0].axis('off')
        
        # Apply block averaging
        avg_img = np.zeros((avg_dim, avg_dim))
        for r in range(avg_dim):
            for c in range(avg_dim):
                block = img[r*block_size:(r+1)*block_size, c*block_size:(c+1)*block_size]
                avg_img[r, c] = np.mean(block)
        
        # Scale to [0, 1] for display
        scaled_avg_img = (avg_img + 1) / 2  
        
        # Display block-averaged image
        axes[i, 1].imshow(scaled_avg_img, cmap='gray')
        axes[i, 1].set_title(f'Block Averaged ({avg_dim}×{avg_dim})')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    return plt

def visualize_logistic_path(X, y, block_size=3, n_alphas=10, figsize=(12, 8)):
    """
    Visualize the regularization path for logistic regression
    This simulates the behavior of glmnet's regularization path
    
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
    """
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    
    # Apply block averaging
    from lda_methods import apply_block_averaging
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
        ax1.plot(alphas, nonzero_coefs[:, j], '-o', color=colors[j], label=f'Class {model.classes_[j]}')
    
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
    plt.tight_layout()
    return plt

def compare_class_separation(results, figsize=(15, 10)):
    """
    Visualize the class separation achieved by different methods
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results from different methods
    figsize : tuple
        Figure size
    """
    from sklearn.decomposition import PCA
    from lda_methods import apply_block_averaging
    
    # Set up the figure
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    # Plot each method's separation
    for i, (method_name, result) in enumerate(results.items()):
        if i >= len(axes):
            break
            
        # Get model
        model = result['model']
        
        # Extract method-specific information
        if method_name == 'standard_lda':
            # For standard LDA, use transform if available
            X_test = result.get('X_test', None)
            y_test = result.get('y_test', None)
            y_pred = result['predictions']
            
            if X_test is None or y_test is None:
                axes[i].text(0.5, 0.5, f"{method_name}\n(Visualization not available)",
                          ha='center', va='center', fontsize=12)
                axes[i].set_title(method_name.replace('_', ' ').title())
                axes[i].axis('off')
                continue
            
            try:
                # Try to use transform if it's available
                X_transformed = model.transform(X_test)
            except (AttributeError, NotImplementedError):
                # If transform is not available, use PCA on model's decision scores
                decision_scores = model.decision_function(X_test)
                pca = PCA(n_components=2)
                X_transformed = pca.fit_transform(decision_scores)
                
            title = "Standard LDA"
            
        elif method_name == 'regularized_lda':
            # For regularized LDA with lsqr solver, transform may not be available
            X_test = result.get('X_test', None)
            y_test = result.get('y_test', None)
            y_pred = result['predictions']
            
            if X_test is None or y_test is None:
                axes[i].text(0.5, 0.5, f"{method_name}\n(Visualization not available)",
                          ha='center', va='center', fontsize=12)
                axes[i].set_title(method_name.replace('_', ' ').title())
                axes[i].axis('off')
                continue
            
            try:
                # Try to use transform if it's available
                X_transformed = model.transform(X_test)
            except (AttributeError, NotImplementedError):
                # If transform is not available, use PCA on model's decision scores
                try:
                    decision_scores = model.decision_function(X_test)
                    pca = PCA(n_components=2)
                    X_transformed = pca.fit_transform(decision_scores)
                except:
                    # If even decision_function fails, use PCA on the features
                    pca = PCA(n_components=2)
                    X_transformed = pca.fit_transform(X_test)
                    
            title = "Regularized LDA"
            
        elif method_name == 'pca_30_lda':
            # For PCA+LDA, we need to apply the transformation
            X_test = result.get('X_test', None)
            y_test = result.get('y_test', None)
            
            if X_test is None or y_test is None:
                axes[i].text(0.5, 0.5, f"{method_name}\n(Visualization not available)",
                          ha='center', va='center', fontsize=12)
                axes[i].set_title(method_name.replace('_', ' ').title())
                axes[i].axis('off')
                continue
                
            # Get predictions
            y_pred = result['predictions']
            
            # Apply PCA for visualization if transform() is not available
            try:
                X_transformed = model.transform(X_test)
            except (AttributeError, NotImplementedError):
                # Try to get PCA components
                try:
                    X_pca = model.named_steps['pca'].transform(X_test)
                    pca_viz = PCA(n_components=2)
                    X_transformed = pca_viz.fit_transform(X_pca)
                except:
                    # Last resort: PCA on original features
                    pca = PCA(n_components=2)
                    X_transformed = pca.fit_transform(X_test)
            
            title = "PCA (30) + LDA"
            
        elif method_name == 'class_pca_lda':
            # For class-specific PCA+LDA, we need both transformations
            X_test = result.get('X_test', None)
            y_test = result.get('y_test', None)
            
            if X_test is None or y_test is None or not isinstance(model, dict):
                axes[i].text(0.5, 0.5, f"{method_name}\n(Visualization not available)",
                          ha='center', va='center', fontsize=12)
                axes[i].set_title(method_name.replace('_', ' ').title())
                axes[i].axis('off')
                continue
                
            # Get predictions
            y_pred = result['predictions']
            
            try:
                # Apply the class-specific PCA + LDA
                pca_model = model['pca']
                lda_model = model['lda']
                X_pca = pca_model.transform(X_test, y_test)
                
                try:
                    X_transformed = lda_model.transform(X_pca)
                except (AttributeError, NotImplementedError):
                    # If transform is not available, use PCA
                    pca_viz = PCA(n_components=2)
                    X_transformed = pca_viz.fit_transform(X_pca)
            except:
                # Fallback to simple PCA
                pca = PCA(n_components=2)
                X_transformed = pca.fit_transform(X_test)
                
            title = "Class-Specific PCA + LDA"
            
        elif method_name == 'block_avg_lda':
            # For block averaging + LDA
            X_test = result.get('X_test', None)
            y_test = result.get('y_test', None)
            
            if X_test is None or y_test is None:
                axes[i].text(0.5, 0.5, f"{method_name}\n(Visualization not available)",
                          ha='center', va='center', fontsize=12)
                axes[i].set_title(method_name.replace('_', ' ').title())
                axes[i].axis('off')
                continue
                
            # Get predictions
            y_pred = result['predictions']
            
            # Apply block averaging
            X_avg = apply_block_averaging(X_test, block_size=3)
            
            try:
                X_transformed = model.transform(X_avg)
            except (AttributeError, NotImplementedError):
                try:
                    # Try decision function
                    decision_scores = model.decision_function(X_avg)
                    pca = PCA(n_components=2)
                    X_transformed = pca.fit_transform(decision_scores)
                except:
                    # Last resort: PCA on averaged features
                    pca = PCA(n_components=2)
                    X_transformed = pca.fit_transform(X_avg)
                
            title = "Block Averaging + LDA"
            
        elif method_name == 'logistic_regression':
            # For logistic regression, compute probabilities as "transformations"
            X_test = result.get('X_test', None)
            y_test = result.get('y_test', None)
            
            if X_test is None or y_test is None:
                axes[i].text(0.5, 0.5, f"{method_name}\n(Visualization not available)",
                          ha='center', va='center', fontsize=12)
                axes[i].set_title(method_name.replace('_', ' ').title())
                axes[i].axis('off')
                continue
                
            # Get predictions
            y_pred = result['predictions']
            
            # Apply block averaging
            X_avg = apply_block_averaging(X_test, block_size=3)
            
            # Get predicted probabilities
            probs = model.predict_proba(X_avg)
            
            # Use PCA to reduce to 2D for visualization
            pca = PCA(n_components=2)
            X_transformed = pca.fit_transform(probs)
                
            title = "Logistic Regression (Block Averaged)"
        
        else:
            # Skip unknown methods
            continue
            
        # Plot the transformed data
        scatter = axes[i].scatter(
            X_transformed[:, 0],
            X_transformed[:, 1] if X_transformed.shape[1] > 1 else np.zeros_like(X_transformed[:, 0]),
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
                X_transformed[mask, 1] if X_transformed.shape[1] > 1 else np.zeros_like(X_transformed[mask, 0]),
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
        axes[i].set_ylabel('Component 2' if X_transformed.shape[1] > 1 else '')
        axes[i].grid(True, linestyle='--', alpha=0.7)
        
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    return plt
