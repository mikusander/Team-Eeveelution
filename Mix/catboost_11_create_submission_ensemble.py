"""
This script trains the final CatBoost model on the full training dataset (100%) and generates
predictions for the Kaggle test set. It loads preprocessed data, applies the best hyperparameters,
computes prediction PROBABILITIES, and saves the submission CSV file for blending.

MODIFIED: Saves prediction probabilities instead of 0/1 classes.
"""

import pandas as pd
import numpy as np
import os
import json
from catboost import CatBoostClassifier

PREPROCESSED_DIR = 'Preprocessed_Data'
MODEL_PARAMS_DIR = 'Model_Params'
SUBMISSION_DIR = 'Submissions'
FEATURES_V5_DIR = 'Features_v5'
os.makedirs(SUBMISSION_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(PREPROCESSED_DIR, 'train_processed_selected.csv')
TARGET_FILE = os.path.join(PREPROCESSED_DIR, 'target_train.csv')
TEST_FILE = os.path.join(PREPROCESSED_DIR, 'test_processed_selected.csv')
PARAMS_FILE = os.path.join(MODEL_PARAMS_DIR, 'best_catboost_params.json')
ITERATION_FILE = os.path.join(MODEL_PARAMS_DIR, 'best_iteration.json')

# MODIFIED: Output file name
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, 'submission_catboost_100pct_PROBA.csv')

print("Starting Final Submission Creation Script (Probability Output)...")

print("Loading 100% Training and Test data...")
# Load full training data and test set
try:
    X_train_full = pd.read_csv(TRAIN_FILE)
    y_train_full = pd.read_csv(TARGET_FILE).values.ravel()
    X_test_kaggle = pd.read_csv(TEST_FILE)

    with open(PARAMS_FILE, 'r') as f:
        best_params = json.load(f)
    print(f"Loaded parameters from {PARAMS_FILE}.")

    with open(ITERATION_FILE, 'r') as f:
        best_iteration = json.load(f)['best_iteration']
    print(f"Loaded optimal number of iterations: {best_iteration}")

except FileNotFoundError:
    print("ERROR: Files not found.")
    print("Please ensure 'train_processed.csv', 'test_processed.csv',")
    print("'best_catboost_params.json' and 'best_iteration.json' are available.")
    exit()

print(f"\nTraining final model on 100% of data ({len(X_train_full)} samples)...")

# Train final CatBoost model on full training data
final_params = best_params.copy()
final_params.update({
    'n_estimators': best_iteration,
    'early_stopping_rounds': None,
    'verbose': 200,
    'random_seed': 42
})
final_params.pop('eval_metric', None)
final_params.pop('custom_metric', None)

final_model = CatBoostClassifier(**final_params)
final_model.fit(X_train_full, y_train_full)
print("Final training completed.")

# --- MODIFICATION START ---
print(f"Generating prediction probabilities for the test set ({len(X_test_kaggle)} samples)...")

# Generate prediction probabilities for Kaggle test set
# predict_proba returns an array of shape (n_samples, n_classes)
# We want the probability of the positive class (class 1)
y_pred_proba = final_model.predict_proba(X_test_kaggle)[:, 1]

# y_pred_bool = final_model.predict(X_test_kaggle) # Old line
# y_pred_int = y_pred_bool.astype(int) # Old line
# --- MODIFICATION END ---

try:
    # Use the original test_processed.csv which should have battle_id
    test_ids_df = pd.read_csv(os.path.join(PREPROCESSED_DIR, 'test_processed.csv'))
    test_ids = test_ids_df['battle_id']
    # Ensure IDs match the order of predictions (important!)
    if not X_test_kaggle.index.equals(test_ids_df.index):
        print("WARNING: Index mismatch between selected features and original test file. Attempting to align.")
        # Re-index test_ids based on X_test_kaggle if possible, otherwise error
        test_ids = test_ids_df.loc[X_test_kaggle.index, 'battle_id']

except KeyError:
    print("Warning: 'battle_id' not found in 'test_processed.csv'.")
    v5_test_path = os.path.join(FEATURES_V5_DIR, 'features_expert_test.csv')
    print(f"Loading 'battle_id' from: '{v5_test_path}'...")
    test_ids_df_v5 = pd.read_csv(v5_test_path)
    test_ids = test_ids_df_v5['battle_id']
    if not X_test_kaggle.index.equals(test_ids_df_v5.index):
         print("WARNING: Index mismatch between selected features and v5 test file. Ensure correct alignment.")
         test_ids = test_ids_df_v5.loc[X_test_kaggle.index, 'battle_id']

except Exception as e:
    print(f"ERROR loading battle_ids: {e}")
    exit()

if len(test_ids) != len(y_pred_proba):
    print(f"ERROR: Mismatch between number of test IDs ({len(test_ids)}) and predictions ({len(y_pred_proba)}). Check data alignment.")
    exit()

print(f"Saving submission file with probabilities to {SUBMISSION_FILE}...")

# MODIFIED: Save probabilities with a clear column name
submission_df = pd.DataFrame({
    'battle_id': test_ids,
    'player_won_proba': y_pred_proba # Save the probabilities
})

submission_df.to_csv(SUBMISSION_FILE, index=False)

print("\n   SUBMISSION (PROBABILITIES) READY FOR BLENDING  ")
print(f"File saved at: {SUBMISSION_FILE}")
print("Format: battle_id, player_won_proba (float)")
print(submission_df.head())

print("\n23_create_submission_100.py (PROBA version) executed successfully.")