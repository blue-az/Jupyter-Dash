#!/usr/bin/env python3
"""
Data Downloader for Zip Digit Classification Project

This script downloads the zip.train and zip.test files from the ESL 
(Elements of Statistical Learning) dataset repository.
"""

import os
import urllib.request
import sys

def download_zip_data():
    """
    Download the zip.train and zip.test files if they don't exist
    
    Returns:
    --------
    bool: True if successful, False otherwise
    """
    # URLs for the data files
    train_url = "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.train.gz"
    test_url = "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.test.gz"
    
    # Local file paths
    train_gz_path = "zip.train.gz"
    test_gz_path = "zip.test.gz"
    train_path = "zip.train"
    test_path = "zip.test"
    
    try:
        # Check if files already exist
        if os.path.exists(train_path) and os.path.exists(test_path):
            print("Data files already exist. Skipping download.")
            return True
        
        # Download training data
        if not os.path.exists(train_path):
            print(f"Downloading {train_url}...")
            urllib.request.urlretrieve(train_url, train_gz_path)
            
            # Extract without external tools
            import gzip
            import shutil
            with gzip.open(train_gz_path, 'rb') as f_in:
                with open(train_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove the .gz file
            os.remove(train_gz_path)
            print(f"Successfully downloaded and extracted {train_path}")
        
        # Download test data
        if not os.path.exists(test_path):
            print(f"Downloading {test_url}...")
            urllib.request.urlretrieve(test_url, test_gz_path)
            
            # Extract without external tools
            import gzip
            import shutil
            with gzip.open(test_gz_path, 'rb') as f_in:
                with open(test_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove the .gz file
            os.remove(test_gz_path)
            print(f"Successfully downloaded and extracted {test_path}")
        
        return True
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        return False

if __name__ == "__main__":
    success = download_zip_data()
    sys.exit(0 if success else 1)
