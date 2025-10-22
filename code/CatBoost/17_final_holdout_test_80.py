"""
This script trains the final CatBoost model on the combined 80% training set (60% Train + 20% Validation)
and evaluates it on the 20% holdout set. It loads preprocessed data and best hyperparameters,
computes performance metrics, and generates confusion matrix and ROC-AUC plots.
"""

import pandas as pd
import numpy as np
import os
import json

from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

SPLIT_DIR = 'Preprocessed_Splits' 
MODEL_PARAMS_DIR = 'Model_Params'
ANALYSIS_DIR = 'Model_Analysis_Holdout_80' 
os.makedirs(ANALYSIS_DIR, exist_ok=True)

X_TRAIN_FILE = os.path.join(SPLIT_DIR, 'train_split_60_X.csv')
Y_TRAIN_FILE = os.path.join(SPLIT_DIR, 'train_split_60_y.csv')
X_VAL_FILE = os.path.join(SPLIT_DIR, 'validation_split_20_X.csv')
Y_VAL_FILE = os.path.join(SPLIT_DIR, 'validation_split_20_y.csv')
X_HOLDOUT_FILE = os.path.join(SPLIT_DIR, 'holdout_split_20_X.csv')
Y_HOLDOUT_FILE = os.path.join(SPLIT_DIR, 'holdout_split_20_y.csv')

PARAMS_FILE = os.path.join(MODEL_PARAMS_DIR, 'best_catboost_params.json') 
ITERATION_FILE = os.path.join(MODEL_PARAMS_DIR, 'best_iteration.json') 

REPORT_TXT_FILE = os.path.join(ANALYSIS_DIR, 'holdout_classification_report.txt')
CM_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'holdout_80_confusion_matrix.png')
ROC_CURVE_FILE = os.path.join(ANALYSIS_DIR, 'holdout_80_roc_auc_curve.png')

print("Starting Final Holdout Test Script  - Training on 80%...")

# Load 60% Train, 20% Validation, and 20% Holdout datasets and parameters
print("Loading 60% Train, 20% Validation, and 20% Holdout data...")
try:
    X_train = pd.read_csv(X_TRAIN_FILE)
    y_train = pd.read_csv(Y_TRAIN_FILE).values.ravel()
    X_val = pd.read_csv(X_VAL_FILE)
    y_val = pd.read_csv(Y_VAL_FILE).values.ravel()
    X_holdout = pd.read_csv(X_HOLDOUT_FILE)
    y_holdout = pd.read_csv(Y_HOLDOUT_FILE).values.ravel()
    
    with open(PARAMS_FILE, 'r') as f:
        best_params = json.load(f)
    print(f"Loaded parameters from {PARAMS_FILE}.")
    
    with open(ITERATION_FILE, 'r') as f:
        best_iteration = json.load(f)['best_iteration']
    print(f"Loaded optimal number of iterations: {best_iteration}")

except FileNotFoundError:
    print(f"ERROR: Files not found.")
    print("Please ensure you have run '16_data_splitter.py' and '17_optimize_and_validate.py'.")
    exit()

# Merge Train and Validation into an 80% training set
print("\nMerging Train (60%) and Validation (20%) into a single 80% training set...")
X_train_80 = pd.concat([X_train, X_val], ignore_index=True)
y_train_80 = np.concatenate([y_train, y_val])
print(f"Total training set size: {len(X_train_80)} samples.")

# Train CatBoost model on 80% training set
print(f"\nTraining model on 80% of data for {best_iteration} iterations...")

final_params_fit = best_params.copy()
final_params_fit.update({
    'n_estimators': best_iteration, 
    'early_stopping_rounds': None, 
    'verbose': 200, 
    'random_seed': 42
})
final_params_fit.pop('eval_metric', None)
final_params_fit.pop('custom_metric', None)


final_model = CatBoostClassifier(**final_params_fit)

final_model.fit(X_train_80, y_train_80)
print("Model trained.")

# Evaluate model on 20% holdout set and compute metrics
print("\nTesting on 20% Holdout:\n")
y_pred_holdout = final_model.predict(X_holdout)
y_pred_proba_holdout = final_model.predict_proba(X_holdout)[:, 1]

holdout_accuracy = accuracy_score(y_holdout, y_pred_holdout)
holdout_auc = roc_auc_score(y_holdout, y_pred_proba_holdout)

print(f"  Accuracy (Holdout 20%): {holdout_accuracy:.4f}")
print(f"  AUC (Holdout 20%): {holdout_auc:.4f}")

print("\nClassification Report (on 20% Holdout)")
print(classification_report(y_holdout, y_pred_holdout, target_names=['False (0)', 'True (1)'], digits=4))
report_dict_holdout = classification_report(y_holdout, y_pred_holdout, target_names=['False (0)', 'True (1)'], output_dict=True, digits=4)

report_text_holdout = classification_report(y_holdout, y_pred_holdout, target_names=['False (0)', 'True (1)'], digits=4)
with open(REPORT_TXT_FILE, 'w') as f:
    f.write("Final Classification Report (on 20% Holdout):\n\n")
    f.write(report_text_holdout)
print(f"Full classification report saved to: {REPORT_TXT_FILE}")

# Generate confusion matrix and ROC-AUC plots
print("Saving final plots (Holdout from 80%)...")

cm_holdout = confusion_matrix(y_holdout, y_pred_holdout)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_holdout, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Predicted False (0)', 'Predicted True (1)'], 
            yticklabels=['Actual False (0)', 'Actual True (1)'])
plt.title("Final Confusion Matrix (Trained on 80%)")
plt.savefig(CM_OUTPUT_FILE)
plt.close()
print(f"CM plot (Holdout) saved to: {CM_OUTPUT_FILE}")

fpr_hold, tpr_hold, _ = roc_curve(y_holdout, y_pred_proba_holdout)
plt.figure(figsize=(10, 8))
plt.plot(fpr_hold, tpr_hold, color='orange', lw=2, label=f'ROC Curve (Holdout AUC = {holdout_auc:.4f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Guess (AUC = 0.50)')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Final ROC-AUC Curve (Trained on 80%)')
plt.legend(loc="lower right"); plt.grid(True)
plt.savefig(ROC_CURVE_FILE)
plt.close()
print(f"ROC-AUC curve plot (Holdout) saved to: {ROC_CURVE_FILE}")

print(f"\nFINAL TEST COMPLETED")
print(f"Your model results are saved in: {ANALYSIS_DIR}")
print("\n18_final_holdout_test_80.py executed successfully.")