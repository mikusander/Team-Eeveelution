"""
This script verifies the preprocessing of the Pokémon battle dataset. It checks
that training and test feature sets have consistent shapes, that all columns are numeric,
that there are no missing values, and that scaling has been applied correctly to sample features.
"""

import pandas as pd
import numpy as np
import os

PREPROCESSED_DIR = 'Preprocessed_Data' 

TRAIN_FILE = os.path.join(PREPROCESSED_DIR, 'train_processed.csv')
TEST_FILE = os.path.join(PREPROCESSED_DIR, 'test_processed.csv')
TARGET_FILE = os.path.join(PREPROCESSED_DIR, 'target_train.csv')

print(f"Loading data from folder: {PREPROCESSED_DIR}...")
try:
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    target_df = pd.read_csv(TARGET_FILE)
except FileNotFoundError:
    print(f"ERROR: Files not found in '{PREPROCESSED_DIR}'.")
    print("Ensure '12_preprocessing.py' has been run first.")
    exit()

print("Data loaded.")

print("\n   Shape Analysis:\n")
print(f"Train shape (X): {train_df.shape}")
print(f"Test shape (X):  {test_df.shape}")
print(f"Target shape (y): {target_df.shape}")

if train_df.shape[1] == test_df.shape[1]:
    print("  \033[92mOK:\033[0m Train columns match Test columns.")
else:
    print(f"  \033[91mERROR:\033[0m Column mismatch! ({train_df.shape[1]} vs {test_df.shape[1]})")

if train_df.shape[0] == target_df.shape[0]:
    print(f"  \033[92mOK:\033[0m Train rows (10000) match Target rows.")
else:
    print(f"  \033[91mERROR:\033[0m Train/Target row mismatch!")

print("\n   Data Types Analysis:\n")
print("Data types found in train_df:")
print(train_df.dtypes.value_counts())

if 'object' in train_df.dtypes.values:
    print("  \033[91mERROR:\033[0m There are still 'object' (text) columns! One-Hot Encoding failed.")
else:
    print("  \033[92mOK:\033[0m All columns are numeric (float64, int64, uint8).")

print("\n   Missing Values Analysis:\n")
nan_count_train = train_df.isnull().sum().sum()
nan_count_test = test_df.isnull().sum().sum()
print(f"Total NaN values in Train (X): {nan_count_train}")
print(f"Total NaN values in Test (X):  {nan_count_test}")

if nan_count_train == 0 and nan_count_test == 0:
    print("  \033[92mOK:\033[0m No missing values.")
else:
    print("  \033[91mWARNING:\033[0m Found NaN values! This may cause errors in modeling.")

print("\n   Scaling Analysis (Example):\n")
feature_da_testare = 'faint_delta' 

if feature_da_testare in train_df.columns:
    desc = train_df[feature_da_testare].describe()
    print(f"Statistics for '{feature_da_testare}':")
    print(f"  Mean: {desc['mean']:.2e}") 
    print(f"  Std:   {desc['std']:.2f}") 
    
    if np.isclose(desc['mean'], 0, atol=1e-2) and np.isclose(desc['std'], 1, atol=1e-2):
        print("  \033[92mOK:\033[0m Feature seems scaled correctly (Mean ~0, Std ~1).")
    else:
        print("  \033[91mWARNING:\033[0m Feature does NOT appear scaled.")
else:
    print(f"  Info: Column '{feature_da_testare}' not found.")

print("\nVerification completed.")
print("\n13_verify_preprocessing.py executed successfully.")