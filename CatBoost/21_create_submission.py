"""
This script trains the final CatBoost model on the full training dataset (100%) and generates
predictions for the Kaggle test set. It loads preprocessed data, applies the best hyperparameters,
computes predictions (1/0), and saves the submission CSV file with 'battle_id' and 'player_won' columns.
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

SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, 'submission.csv') 

print("Starting Final Submission Creation Script...")

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

print(f"Generating predictions (1/0) for the test set ({len(X_test_kaggle)} samples)...")

# Generate predictions for Kaggle test set
y_pred_bool = final_model.predict(X_test_kaggle)

y_pred_int = y_pred_bool.astype(int)

try:
    test_ids = pd.read_csv(os.path.join(PREPROCESSED_DIR, 'test_processed.csv'))['battle_id']
except KeyError:
    print("Warning: 'battle_id' not found in 'test_processed.csv'.")
    v5_test_path = os.path.join(FEATURES_V5_DIR, 'features_expert_test.csv') 
    print(f"Loading 'battle_id' from: '{v5_test_path}'...")
    test_ids = pd.read_csv(v5_test_path)['battle_id'] 

print(f"Saving submission file to {SUBMISSION_FILE}...")

# Save predictions to submission CSV file
submission_df = pd.DataFrame({
    'battle_id': test_ids,
    'player_won': y_pred_int  
})

submission_df.to_csv(SUBMISSION_FILE, index=False)

print("\n   SUBMISSION READY  ")
print(f"File saved at: {SUBMISSION_FILE}")
print("Format: battle_id, player_won (1/0)")
print(submission_df.head())

print("\n19_create_submission.py executed successfully.")