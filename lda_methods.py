#!/usr/bin/env python3
"""
LDA Methods Module

This module implements various Linear Discriminant Analysis methods
for classifying handwritten digits from zip code data.
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

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
            pca = PCA(n_components=self.n_components)
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
    
    def get_feature_names_out(self, input_features=None):
        # Create feature names for all components from all classes
        feature_names = []
        for cls in sorted(self.pcas.keys()):
            for i in range(self.n_components):
                feature_names.append(f'class_{cls}_pc_{i}')
        return np.array(feature_names)

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
    lda = LinearDiscriminantAnalysis()
    
    # Fit and predict
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'model': lda,
        'predictions': y_pred,
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm
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
    pipeline = Pipeline([
        ('pca', PCA(n_components=n_components)),
        ('lda', LinearDiscriminantAnalysis())
    ])
    
    # Fit and predict
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'model': pipeline,
        'predictions': y_pred,
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'explained_variance_ratio': pipeline.named_steps['pca'].explained_variance_ratio_.sum()
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
    # Split training data by class for class-specific PCA
    classes = np.unique(y_train)
    
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
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'model': {
            'pca': class_pca,
            'lda': lda
        },
        'predictions': y_pred,
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm
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
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=shrinkage)
    
    # Fit and predict
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'model': lda,
        'predictions': y_pred,
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm
    }

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
    X_train_avg = apply_block_averaging(X_train, block_size)
    X_test_avg = apply_block_averaging(X_test, block_size)
    
    # Initialize LDA
    lda = LinearDiscriminantAnalysis()
    
    # Fit and predict
    lda.fit(X_train_avg, y_train)
    y_pred = lda.predict(X_test_avg)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'model': lda,
        'predictions': y_pred,
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'processed_features': X_train_avg.shape[1]
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
    from sklearn.linear_model import LogisticRegression
    
    # Apply block averaging
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
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'model': logistic,
        'predictions': y_pred,
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'processed_features': X_train_avg.shape[1],
        'model_coefficients': logistic.coef_
    }
