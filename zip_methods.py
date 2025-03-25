#!/usr/bin/env python3
"""
LDA Methods Module for Zip Code Digit Classification

This module implements various methods for classifying handwritten digits from 
zip code data, with a particular focus on distinguishing between digits 3, 5, and 8.
"""

import numpy as np
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression

# Import utility functions
from zip_utils import apply_block_averaging

class ClassSpecificPCA(BaseEstimator, TransformerMixin):
    """
    Apply PCA to each class separately and concatenate the results
    """
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.pcas = {}
        
    def fit(self, X, y):
        classes = np.unique(y)
        for cls in classes:
            X_cls = X[y == cls]
            pca = PCA(n_components=min(self.n_components, X_cls.shape[0], X_cls.shape[1]))
            pca.fit(X_cls)
            self.pcas[cls] = pca
        return self
    
    def transform(self, X, y=None):
        if y is None:
            raise ValueError("ClassSpecificPCA requires y for transform")
        
        # Initialize an empty array to store the transformed features
        transformed_features = []
        
        # Apply the appropriate PCA transformation based on class
        classes = np.unique(y)
        for cls in classes:
            X_cls = X[y == cls]
            if cls in self.pcas:
                transformed = self.pcas[cls].transform(X_cls)
                transformed_features.append(transformed)
            else:
                # If class wasn't in training set, use the first PCA as fallback
                first_pca = next(iter(self.pcas.values()))
                transformed = first_pca.transform(X_cls)
                transformed_features.append(transformed)
        
        # Concatenate the transformed features
        return np.vstack(transformed_features)
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)

def perform_standard_lda(X_train, y_train, X_test, y_test):
    """
    Perform LDA on the original feature space
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training labels
    X_test : numpy.ndarray
        Test features
    y_test : numpy.ndarray
        Test labels
    
    Returns:
    --------
    results : dict
        Dictionary containing model, predictions, and metrics
    """
    # Initialize LDA
    start_time = time.time()
    lda = LinearDiscriminantAnalysis()
    
    # Fit and predict
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    run_time = time.time() - start_time
    
    return {
        'model': lda,
        'predictions': y_pred,
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'time': run_time,
        'X_test': X_test,
        'y_test': y_test,
        'method_name': 'Standard LDA'
    }

def perform_pca_lda(X_train, y_train, X_test, y_test, n_components=30):
    """
    Perform PCA followed by LDA
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training labels
    X_test : numpy.ndarray
        Test features
    y_test : numpy.ndarray
        Test labels
    n_components : int
        Number of principal components to use
    
    Returns:
    --------
    results : dict
        Dictionary containing model, predictions, and metrics
    """
    # Create pipeline with PCA followed by LDA
    start_time = time.time()
    pipeline = Pipeline([
        ('pca', PCA(n_components=n_components)),
        ('lda', LinearDiscriminantAnalysis())
    ])
    
    # Fit and predict
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    run_time = time.time() - start_time
    
    return {
        'model': pipeline,
        'predictions': y_pred,
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'explained_variance_ratio': pipeline.named_steps['pca'].explained_variance_ratio_.sum(),
        'time': run_time,
        'X_test': X_test,
        'y_test': y_test,
        'method_name': f'PCA ({n_components}) + LDA'
    }

def perform_class_specific_pca_lda(X_train, y_train, X_test, y_test, n_components_per_class=10):
    """
    Perform class-specific PCA followed by LDA
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training labels
    X_test : numpy.ndarray
        Test features
    y_test : numpy.ndarray
        Test labels
    n_components_per_class : int
        Number of principal components to use per class
    
    Returns:
    --------
    results : dict
        Dictionary containing model, predictions, and metrics
    """
    start_time = time.time()
    
    # Transform the training data
    class_pca = ClassSpecificPCA(n_components=n_components_per_class)
    X_train_transformed = class_pca.fit_transform(X_train, y_train)
    
    # Transform the test data
    X_test_transformed = class_pca.transform(X_test, y_test)
    
    # Apply LDA to the transformed data
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_transformed, y_train)
    y_pred = lda.predict(X_test_transformed)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    run_time = time.time() - start_time
    
    return {
        'model': {
            'pca': class_pca,
            'lda': lda
        },
        'predictions': y_pred,
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'time': run_time,
        'X_test': X_test,
        'y_test': y_test,
        'method_name': f'Class-Specific PCA ({n_components_per_class}) + LDA'
    }

def perform_regularized_lda(X_train, y_train, X_test, y_test, shrinkage='auto'):
    """
    Perform regularized LDA with shrinkage
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training labels
    X_test : numpy.ndarray
        Test features
    y_test : numpy.ndarray
        Test labels
    shrinkage : str or float
        Shrinkage parameter for regularized LDA
    
    Returns:
    --------
    results : dict
        Dictionary containing model, predictions, and metrics
    """
    # Initialize regularized LDA
    start_time = time.time()
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=shrinkage)
    
    # Fit and predict
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    run_time = time.time() - start_time
    
    return {
        'model': lda,
        'predictions': y_pred,
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'time': run_time,
        'X_test': X_test,
        'y_test': y_test,
        'method_name': 'Regularized LDA'
    }

def perform_lda_with_block_averaging(X_train, y_train, X_test, y_test, block_size=3):
    """
    Perform LDA on data with block averaging applied
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training labels
    X_test : numpy.ndarray
        Test features
    y_test : numpy.ndarray
        Test labels
    block_size : int
        Size of the blocks to average
    
    Returns:
    --------
    results : dict
        Dictionary containing model, predictions, and metrics
    """
    # Apply block averaging
    start_time = time.time()
    X_train_avg = apply_block_averaging(X_train, block_size)
    X_test_avg = apply_block_averaging(X_test, block_size)
    
    # Initialize LDA
    lda = LinearDiscriminantAnalysis()
    
    # Fit and predict
    lda.fit(X_train_avg, y_train)
    y_pred = lda.predict(X_test_avg)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    run_time = time.time() - start_time
    
    return {
        'model': lda,
        'predictions': y_pred,
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'time': run_time,
        'processed_features': X_train_avg.shape[1],
        'X_test': X_test,
        'y_test': y_test,
        'X_test_avg': X_test_avg,
        'method_name': f'Block Averaging ({block_size}×{block_size}) + LDA'
    }

def perform_logistic_regression(X_train, y_train, X_test, y_test, block_size=3, C=1.0):
    """
    Perform multinomial logistic regression on block-averaged data
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training labels
    X_test : numpy.ndarray
        Test features
    y_test : numpy.ndarray
        Test labels
    block_size : int
        Size of the blocks to average
    C : float
        Inverse of regularization strength (smaller values mean stronger regularization)
        This is equivalent to 1/lambda in glmnet
    
    Returns:
    --------
    results : dict
        Dictionary containing model, predictions, and metrics
    """
    # Apply block averaging
    start_time = time.time()
    X_train_avg = apply_block_averaging(X_train, block_size)
    X_test_avg = apply_block_averaging(X_test, block_size)
    
    # Initialize logistic regression
    # solver='saga' supports L1 penalty which is similar to glmnet
    # max_iter=10000 to ensure convergence
    logistic = LogisticRegression(
        solver='saga',
        C=C,
        penalty='l1',
        max_iter=10000,
        tol=1e-4,
        random_state=42
    )
    
    # Fit and predict
    logistic.fit(X_train_avg, y_train)
    y_pred = logistic.predict(X_test_avg)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    run_time = time.time() - start_time
    
    return {
        'model': logistic,
        'predictions': y_pred,
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'time': run_time,
        'processed_features': X_train_avg.shape[1],
        'model_coefficients': logistic.coef_,
        'X_test': X_test,
        'y_test': y_test,
        'X_test_avg': X_test_avg,
        'method_name': f'Logistic Regression (Block Avg {block_size}×{block_size}, C={C})'
    }

def find_best_logistic_model(X_train, y_train, X_test, y_test, block_size=3, C_values=[0.1, 0.01, 0.001]):
    """
    Find the best logistic regression model by testing different C values
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training labels
    X_test : numpy.ndarray
        Test features
    y_test : numpy.ndarray
        Test labels
    block_size : int
        Size of the blocks to average
    C_values : list
        List of C values to test
    
    Returns:
    --------
    results : dict
        Dictionary containing results from all models and the best model
    """
    results = {}
    best_accuracy = 0
    best_C = None
    best_result = None
    
    for C in C_values:
        result = perform_logistic_regression(
            X_train, y_train, X_test, y_test,
            block_size=block_size, C=C
        )
        
        results[f'logistic_C_{C}'] = result
        
        if result['accuracy'] > best_accuracy:
            best_accuracy = result['accuracy']
            best_C = C
            best_result = result
    
    return {
        'all_results': results,
        'best_C': best_C,
        'best_result': best_result
    }
