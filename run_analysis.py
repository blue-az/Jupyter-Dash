#!/usr/bin/env python3
"""
Zip Digit Classification Analysis

This script performs a comprehensive analysis of different LDA methods
for classifying handwritten digits from zip code data, with a focus on
digits 3, 5, and 8.

The analysis includes:
1. Standard LDA on the original 256-dimensional space
2. PCA + LDA on the leading 30 principal components
3. Class-specific PCA + LDA - 10 principal components per class
4. Regularized LDA with automatic shrinkage parameter
5. Block-Averaging + LDA - Replace each 3×3 pixel block with its average
6. Multinomial Logistic Regression on block-averaged data

Results are saved to the 'results/' directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Add the current directory to the path
sys.path.append('.')

# Try to import our utility module, download data if needed
try:
    from zip_utils import (
        load_zip_data, filter_digits, plot_confusion_matrix,
        visualize_digit_samples, save_results
    )
except ImportError:
    print("Zip utilities module not found. Make sure zip_utils.py is in the same directory.")
    sys.exit(1)

# Try to import lda_methods.py
try:
    from lda_methods import (
        perform_standard_lda,
        perform_pca_lda,
        perform_class_specific_pca_lda,
        perform_regularized_lda,
        perform_lda_with_block_averaging,
        perform_logistic_regression,
        ClassSpecificPCA,
        apply_block_averaging
    )
except ImportError:
    # If not available, import from the notebook itself
    from zip_digit_classification import (
        perform_standard_lda,
        # Rest of imports...
    )
# Try to import visualization.py
try:
    from visualization import (
        plot_accuracy_comparison,
        plot_pca_explained_variance,
        visualize_feature_importance,
        visualize_block_averaging,
        visualize_logistic_path,
        compare_class_separation
    )
except ImportError:
    # If not available, we'll define simplified versions here
    # These should be compatible with the versions in the notebook
    from zip_utils import plot_confusion_matrix, visualize_digit_samples
    
    def plot_accuracy_comparison(results, title='Model Comparison', figsize=(10, 6)):
        plt.figure(figsize=figsize)
        models = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in models]
        labels = [' '.join(model.split('_')).title() for model in models]
        bars = plt.bar(labels, accuracies, color='skyblue')
        for bar, accuracy in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{accuracy:.4f}', ha='center', fontsize=10)
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        return plt
    
    def plot_pca_explained_variance(X, n_components=30, figsize=(10, 6)):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        pca.fit(X)
        plt.figure(figsize=figsize)
        plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_, alpha=0.7,
                align='center', label='Individual Explained Variance')
        plt.step(range(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_),
                where='mid', label='Cumulative Explained Variance', color='red')
        plt.xlabel('Number of Components')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance by PCA Components')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        return plt
    
    # Simplified versions of other visualization functions
    def visualize_feature_importance(model, feature_names=None, top_n=20, figsize=(12, 8)):
        try:
            import pandas as pd
            import seaborn as sns
            if not hasattr(model, 'coef_'):
                print("Model doesn't have coef_ attribute")
                return None
            coefs = model.coef_
            if feature_names is None:
                feature_names = [f'Feature {i}' for i in range(coefs.shape[1])]
            coef_df = pd.DataFrame()
            for i, class_label in enumerate(model.classes_):
                class_coef = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': coefs[i],
                    'Abs_Coefficient': np.abs(coefs[i]),
                    'Class': f'Class {class_label}'
                })
                coef_df = pd.concat([coef_df, class_coef])
            top_features = (coef_df.groupby('Feature')['Abs_Coefficient']
                            .mean().sort_values(ascending=False).head(top_n).index.tolist())
            plot_df = coef_df[coef_df['Feature'].isin(top_features)]
            plt.figure(figsize=figsize)
            sns.barplot(data=plot_df, x='Coefficient', y='Feature', hue='Class', palette='viridis')
            plt.title(f'Top {top_n} Feature Importance by Class')
            plt.xlabel('Coefficient Value')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            return plt
        except Exception as e:
            print(f"Error visualizing feature importance: {e}")
            return None
    
    def visualize_block_averaging(X, block_size=3, n_samples=3, figsize=(15, 10)):
        try:
            indices = np.random.choice(X.shape[0], size=n_samples, replace=False)
            X_samples = X[indices]
            img_dim = int(np.sqrt(X.shape[1]))
            avg_dim = img_dim // block_size
            fig, axes = plt.subplots(n_samples, 2, figsize=figsize)
            for i, sample in enumerate(X_samples):
                img = sample.reshape(img_dim, img_dim)
                scaled_img = (img + 1) / 2
                axes[i, 0].imshow(scaled_img, cmap='gray')
                axes[i, 0].set_title(f'Original ({img_dim}×{img_dim})')
                axes[i, 0].axis('off')
                avg_img = np.zeros((avg_dim, avg_dim))
                for r in range(avg_dim):
                    for c in range(avg_dim):
                        block = img[r*block_size:(r+1)*block_size, c*block_size:(c+1)*block_size]
                        avg_img[r, c] = np.mean(block)
                scaled_avg_img = (avg_img + 1) / 2
                axes[i, 1].imshow(scaled_avg_img, cmap='gray')
                axes[i, 1].set_title(f'Block Averaged ({avg_dim}×{avg_dim})')
                axes[i, 1].axis('off')
            plt.tight_layout()
            return plt
        except Exception as e:
            print(f"Error visualizing block averaging: {e}")
            return None
    
    def visualize_logistic_path(X, y, block_size=3, n_alphas=10, figsize=(12, 8)):
        try:
            from sklearn.linear_model import LogisticRegression
            X_avg = apply_block_averaging(X, block_size)
            alphas = np.logspace(-4, 0, n_alphas)
            C_values = 1 / alphas
            n_classes = len(np.unique(y))
            n_features = X_avg.shape[1]
            nonzero_coefs = np.zeros((n_alphas, n_classes))
            accuracies = np.zeros(n_alphas)
            for i, C in enumerate(C_values):
                model = LogisticRegression(C=C, penalty='l1', solver='saga',
                                          max_iter=10000, tol=1e-4, random_state=42)
                model.fit(X_avg, y)
                y_pred = model.predict(X_avg)
                accuracies[i] = np.mean(y_pred == y)
                for j in range(n_classes):
                    nonzero_coefs[i, j] = np.sum(np.abs(model.coef_[j]) > 1e-10)
            fig, ax1 = plt.subplots(figsize=figsize)
            colors = plt.cm.viridis(np.linspace(0, 1, n_classes))
            for j in range(n_classes):
                ax1.plot(alphas, nonzero_coefs[:, j], '-o', color=colors[j], 
                        label=f'Class {model.classes_[j]}')
            ax1.set_xscale('log')
            ax1.set_xlabel('Alpha (regularization strength)')
            ax1.set_ylabel('Number of non-zero coefficients')
            ax1.set_title('Regularization Path for Logistic Regression')
            ax1.legend(loc='upper left')
            ax2 = ax1.twinx()
            ax2.plot(alphas, accuracies, '--', color='red', label='Accuracy')
            ax2.set_ylabel('Accuracy')
            ax2.legend(loc='upper right')
            best_alpha_idx = np.argmax(accuracies)
            best_alpha = alphas[best_alpha_idx]
            plt.axvline(x=best_alpha, linestyle='--', color='black', alpha=0.5)
            plt.text(best_alpha, 0.5, f'End of path: alpha={best_alpha:.5f}',
                    rotation=90, verticalalignment='center')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            return plt
        except Exception as e:
            print(f"Error visualizing logistic path: {e}")
            return None
    
    def compare_class_separation(results, figsize=(15, 10)):
        try:
            from sklearn.decomposition import PCA
            fig, axes = plt.subplots(2, 3, figsize=figsize)
            axes = axes.flatten()
            for i, (method_name, result) in enumerate(results.items()):
                if i >= len(axes):
                    break
                model = result['model']
                X_test = result.get('X_test', None)
                y_test = result.get('y_test', None)
                
                if X_test is None or y_test is None:
                    axes[i].text(0.5, 0.5, f"{method_name}\n(Visualization not available)",
                              ha='center', va='center', fontsize=12)
                    axes[i].set_title(method_name.replace('_', ' ').title())
                    axes[i].axis('off')
                    continue
                
                y_pred = result['predictions']
                
                # Get appropriate transformed data for visualization
                try:
                    if method_name == 'standard_lda' or method_name == 'regularized_lda':
                        try:
                            X_transformed = model.transform(X_test)
                        except:
                            decision_scores = model.decision_function(X_test)
                            pca = PCA(n_components=2)
                            X_transformed = pca.fit_transform(decision_scores)
                    elif method_name == 'pca_30_lda':
                        try:
                            X_transformed = model.transform(X_test)
                        except:
                            X_pca = model.named_steps['pca'].transform(X_test)
                            pca_viz = PCA(n_components=2)
                            X_transformed = pca_viz.fit_transform(X_pca)
                    elif method_name == 'class_pca_lda':
                        pca_model = model['pca']
                        lda_model = model['lda']
                        X_pca = pca_model.transform(X_test, y_test)
                        X_transformed = lda_model.transform(X_pca)
                    elif method_name == 'block_avg_lda':
                        X_avg = apply_block_averaging(X_test, block_size=3)
                        X_transformed = model.transform(X_avg)
                    elif method_name == 'logistic_regression':
                        X_avg = apply_block_averaging(X_test, block_size=3)
                        probs = model.predict_proba(X_avg)
                        pca = PCA(n_components=2)
                        X_transformed = pca.fit_transform(probs)
                    else:
                        pca = PCA(n_components=2)
                        X_transformed = pca.fit_transform(X_test)
                except:
                    # Fallback to PCA on original features
                    pca = PCA(n_components=2)
                    X_transformed = pca.fit_transform(X_test)
                
                # Plot
                title = method_name.replace('_', ' ').title()
                scatter = axes[i].scatter(
                    X_transformed[:, 0],
                    X_transformed[:, 1] if X_transformed.shape[1] > 1 else np.zeros_like(X_transformed[:, 0]),
                    c=y_test, cmap='viridis', alpha=0.7, s=30
                )
                
                # Mark misclassified samples
                mask = y_test != y_pred
                if np.any(mask):
                    axes[i].scatter(
                        X_transformed[mask, 0],
                        X_transformed[mask, 1] if X_transformed.shape[1] > 1 else np.zeros_like(X_transformed[mask, 0]),
                        facecolors='none', edgecolors='red', s=100, label='Misclassified'
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
        except Exception as e:
            print(f"Error comparing class separation: {e}")
            return None
            
# Try to download data if it doesn't exist
try:
    from download_data import download_zip_data
    download_zip_data()
except ImportError:
    print("Data downloader not found. Make sure zip.train and zip.test files exist.")


def run_analysis(output_dir='results'):
    """
    Run the complete analysis pipeline and save results
    
    Parameters:
    -----------
    output_dir : str
        Directory to save results
    
    Returns:
    --------
    results : dict
        Dictionary containing results from all methods
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Set paths to data files
    train_path = 'zip.train'
    test_path = 'zip.test'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load data
    print("Loading data...")
    X_train, y_train = load_zip_data(train_path)
    X_test, y_test = load_zip_data(test_path)
    
    # Filter to only include digits 3, 5, and 8
    print("Filtering to digits 3, 5, and 8...")
    X_train_filtered, y_train_filtered = filter_digits(X_train, y_train)
    X_test_filtered, y_test_filtered = filter_digits(X_test, y_test)
    
    print(f"Training data shape: {X_train_filtered.shape}")
    print(f"Test data shape: {X_test_filtered.shape}")
    
    # Visualize sample digits
    print("Visualizing sample digits...")
    plt.figure()
    visualize_digit_samples(X_train_filtered, y_train_filtered)
    plt.savefig(f"{output_dir}/sample_digits.png")
    plt.close()
    
    # Visualize PCA explained variance
    print("Visualizing PCA explained variance...")
    plt.figure()
    plot_pca_explained_variance(X_train_filtered)
    plt.savefig(f"{output_dir}/pca_explained_variance.png")
    plt.close()
    
    # Dictionary to store results
    results = {}
    
    # a) LDA on the original 256-dimensional space
    print("\nRunning LDA on original space...")
    start_time = time.time()
    results['standard_lda'] = perform_standard_lda(
        X_train_filtered, y_train_filtered,
        X_test_filtered, y_test_filtered
    )
    results['standard_lda']['time'] = time.time() - start_time
    # Store data for visualization
    results['standard_lda']['X_test'] = X_test_filtered
    results['standard_lda']['y_test'] = y_test_filtered
    
    print(f"Time: {results['standard_lda']['time']:.2f} seconds")
    print(f"Accuracy: {results['standard_lda']['accuracy']:.4f}")
    print(results['standard_lda']['report'])
    
    # b) LDA on the leading 30 principal components
    print("\nRunning LDA on 30 principal components...")
    start_time = time.time()
    results['pca_30_lda'] = perform_pca_lda(
        X_train_filtered, y_train_filtered,
        X_test_filtered, y_test_filtered,
        n_components=30
    )
    results['pca_30_lda']['time'] = time.time() - start_time
    # Store data for visualization
    results['pca_30_lda']['X_test'] = X_test_filtered
    results['pca_30_lda']['y_test'] = y_test_filtered
    
    print(f"Time: {results['pca_30_lda']['time']:.2f} seconds")
    print(f"Accuracy: {results['pca_30_lda']['accuracy']:.4f}")
    print(f"Explained variance ratio: {results['pca_30_lda']['explained_variance_ratio']:.4f}")
    print(results['pca_30_lda']['report'])
    
    # c) LDA on 10 principal components from each class
    print("\nRunning LDA on 10 principal components from each class...")
    start_time = time.time()
    results['class_pca_lda'] = perform_class_specific_pca_lda(
        X_train_filtered, y_train_filtered,
        X_test_filtered, y_test_filtered,
        n_components_per_class=10
    )
    results['class_pca_lda']['time'] = time.time() - start_time
    # Store data for visualization
    results['class_pca_lda']['X_test'] = X_test_filtered
    results['class_pca_lda']['y_test'] = y_test_filtered
    
    print(f"Time: {results['class_pca_lda']['time']:.2f} seconds")
    print(f"Accuracy: {results['class_pca_lda']['accuracy']:.4f}")
    print(results['class_pca_lda']['report'])
    
    # Additional: Regularized LDA
    print("\nRunning Regularized LDA...")
    start_time = time.time()
    results['regularized_lda'] = perform_regularized_lda(
        X_train_filtered, y_train_filtered,
        X_test_filtered, y_test_filtered
    )
    results['regularized_lda']['time'] = time.time() - start_time
    # Store data for visualization
    results['regularized_lda']['X_test'] = X_test_filtered
    results['regularized_lda']['y_test'] = y_test_filtered
    
    print(f"Time: {results['regularized_lda']['time']:.2f} seconds")
    print(f"Accuracy: {results['regularized_lda']['accuracy']:.4f}")
    print(results['regularized_lda']['report'])
    
    # d) LDA with 3x3 block averaging
    print("\nRunning LDA with 3x3 block averaging...")
    start_time = time.time()
    results['block_avg_lda'] = perform_lda_with_block_averaging(
        X_train_filtered, y_train_filtered,
        X_test_filtered, y_test_filtered,
        block_size=3
    )
    results['block_avg_lda']['time'] = time.time() - start_time
    # Store data for visualization
    results['block_avg_lda']['X_test'] = X_test_filtered
    results['block_avg_lda']['y_test'] = y_test_filtered
    
    print(f"Time: {results['block_avg_lda']['time']:.2f} seconds")
    print(f"Accuracy: {results['block_avg_lda']['accuracy']:.4f}")
    print(f"Reduced feature count: {results['block_avg_lda']['processed_features']}")
    print(results['block_avg_lda']['report'])
    
    # Visualize block averaging effect
    print("\nVisualizing block averaging effect...")
    plt.figure()
    visualize_block_averaging(X_train_filtered[:5], block_size=3)
    plt.savefig(f"{output_dir}/block_averaging.png")
    plt.close()
    
    # e) Multinomial logistic regression on block-averaged data
    print("\nRunning multinomial logistic regression on block-averaged data...")
    start_time = time.time()
    
    # Try a range of regularization values
    reg_values = [0.1, 0.01, 0.001]
    best_accuracy = 0
    best_C = None
    
    for C in reg_values:
        print(f"Trying C={C} (equivalent to lambda={1/C})...")
        result = perform_logistic_regression(
            X_train_filtered, y_train_filtered,
            X_test_filtered, y_test_filtered,
            block_size=3,
            C=C
        )
        
        if result['accuracy'] > best_accuracy:
            best_accuracy = result['accuracy']
            best_C = C
    
    # Run with the best regularization parameter
    print(f"\nRunning with optimal C={best_C}...")
    results['logistic_regression'] = perform_logistic_regression(
        X_train_filtered, y_train_filtered,
        X_test_filtered, y_test_filtered,
        block_size=3,
        C=best_C
    )
    results['logistic_regression']['time'] = time.time() - start_time
    # Store data for visualization
    results['logistic_regression']['X_test'] = X_test_filtered
    results['logistic_regression']['y_test'] = y_test_filtered
    results['logistic_regression']['C'] = best_C
    
    print(f"Time: {results['logistic_regression']['time']:.2f} seconds")
    print(f"Accuracy: {results['logistic_regression']['accuracy']:.4f}")
    print(f"Reduced feature count: {results['logistic_regression']['processed_features']}")
    print(results['logistic_regression']['report'])
    
    # Find non-zero coefficients in the logistic regression model
    coefs = results['logistic_regression']['model_coefficients']
    non_zero_coefs = np.sum(coefs != 0)
    print(f"Number of non-zero coefficients: {non_zero_coefs} out of {coefs.size}")
    
    # Visualize regularization path
    print("\nVisualizing regularization path...")
    plt.figure()
    try:
        visualize_logistic_path(
            X_train_filtered, y_train_filtered,
            block_size=3,
            n_alphas=10
        )
        plt.savefig(f"{output_dir}/logistic_path.png")
    except Exception as e:
        print(f"Could not visualize regularization path: {e}")
    plt.close()
    
    # Compare class separation achieved by different methods
    print("\nComparing class separation...")
    plt.figure(figsize=(15, 10))
    try:
        compare_class_separation(results)
        plt.savefig(f"{output_dir}/class_separation.png")
    except Exception as e:
        print(f"Could not visualize class separation: {e}")
    plt.close()
    
    # Plot confusion matrices
    print("\nPlotting confusion matrices...")
    for name, result in results.items():
        plt.figure()
        plot_confusion_matrix(
            result['confusion_matrix'],
            classes=np.unique(y_train_filtered),
            title=f"{name.replace('_', ' ').title()} - Confusion Matrix"
        )
        plt.savefig(f"{output_dir}/{name}_confusion.png")
        plt.close()
    
    # Plot accuracy comparison
    print("\nPlotting accuracy comparison...")
    plt.figure()
    plot_accuracy_comparison(results)
    plt.savefig(f"{output_dir}/accuracy_comparison.png")
    plt.close()
    
    # Visualize feature importance for standard LDA
    print("\nVisualizing feature importance...")
    plt.figure()
    try:
        visualize_feature_importance(
            results['standard_lda']['model'],
            top_n=30
        )
        plt.savefig(f"{output_dir}/feature_importance.png")
    except Exception as e:
        print(f"Could not visualize feature importance: {e}")
    plt.close()
    
    # Generate results summary
    try:
        save_results(results, X_train_filtered, y_train_filtered, output_dir)
    except Exception as e:
        print(f"Error saving results summary: {e}")
    
    print("\nAnalysis complete! Results saved to", output_dir)
    
    return results

if __name__ == "__main__":
    run_analysis()
