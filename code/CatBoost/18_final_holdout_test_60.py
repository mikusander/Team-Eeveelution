"""
This script trains the final CatBoost model on the 60% training split and evaluates
it on the 20% holdout set. It loads the preprocessed datasets, uses the best hyperparameters,
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
ANALYSIS_DIR = 'Model_Analysis_Holdout_60' 
os.makedirs(ANALYSIS_DIR, exist_ok=True)

X_TRAIN_FILE = os.path.join(SPLIT_DIR, 'train_split_60_X.csv')
Y_TRAIN_FILE = os.path.join(SPLIT_DIR, 'train_split_60_y.csv')
X_HOLDOUT_FILE = os.path.join(SPLIT_DIR, 'holdout_split_20_X.csv')
Y_HOLDOUT_FILE = os.path.join(SPLIT_DIR, 'holdout_split_20_y.csv')
PARAMS_FILE = os.path.join(MODEL_PARAMS_DIR, 'best_catboost_params.json')

REPORT_TXT_FILE = os.path.join(ANALYSIS_DIR, 'holdout_classification_report.txt')
CM_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'holdout_confusion_matrix.png')
ROC_CURVE_FILE = os.path.join(ANALYSIS_DIR, 'holdout_roc_auc_curve.png')

print("Starting Final Holdout Test Script (Phase 3)...")

print("Loading 60% Train and 20% Holdout data...")
# Load training, holdout, and parameter data
try:
    X_train = pd.read_csv(X_TRAIN_FILE)
    y_train = pd.read_csv(Y_TRAIN_FILE).values.ravel()
    X_holdout = pd.read_csv(X_HOLDOUT_FILE)
    y_holdout = pd.read_csv(Y_HOLDOUT_FILE).values.ravel()
    
    with open(PARAMS_FILE, 'r') as f:
        best_params_clean = json.load(f)
    print(f"Loaded final parameters from {PARAMS_FILE}.")

except FileNotFoundError:
    print(f"ERROR: Files not found in {SPLIT_DIR} or {PARAMS_FILE}.")
    print("Ensure '16_data_splitter.py' and '17_optimize_and_validate.py' have been run.")
    exit()

print("\nTraining on 60% and Testing on 20% (Holdout):\n")

print("Loading 20% Validation to use as eval_set (for early stopping)...")
X_val = pd.read_csv(os.path.join(SPLIT_DIR, 'validation_split_20_X.csv'))
y_val = pd.read_csv(os.path.join(SPLIT_DIR, 'validation_split_20_y.csv')).values.ravel()

final_params_fit = best_params_clean.copy() 
final_params_fit.update({
    'n_estimators': 2000, 'early_stopping_rounds': 50,
    'eval_metric': 'Logloss', 'custom_metric': ['AUC'],
    'verbose': 0, 
    'random_seed': 42
})

# Train final CatBoost model using validation set for early stopping
print("Training final model (Train 60%, Eval 20%)...")
final_model = CatBoostClassifier(**final_params_fit)
final_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val), 
    plot=False,
    use_best_model=True, 
    verbose=0
)
print("Model trained using best iteration (found on validation set).")

print("\nTesting on 20% Holdout:\n")
# Evaluate model on holdout set and compute metrics
y_pred_holdout = final_model.predict(X_holdout)
y_pred_proba_holdout = final_model.predict_proba(X_holdout)[:, 1]

holdout_accuracy = accuracy_score(y_holdout, y_pred_holdout)
holdout_auc = roc_auc_score(y_holdout, y_pred_proba_holdout)

print(f"  Accuracy (Holdout 20%): {holdout_accuracy:.4f}")
print(f"  AUC (Holdout 20%): {holdout_auc:.4f}")

print("\nFinal Classification Report (on 20% Holdout):\n")
print(classification_report(y_holdout, y_pred_holdout, target_names=['False (0)', 'True (1)'], digits=4))
report_dict_holdout = classification_report(y_holdout, y_pred_holdout, target_names=['False (0)', 'True (1)'], output_dict=True, digits=4)


# Save full classification report as text
report_text_holdout = classification_report(y_holdout, y_pred_holdout, target_names=['False (0)', 'True (1)'], digits=4)
with open(REPORT_TXT_FILE, 'w') as f:
    f.write("Final Classification Report (on 20% Holdout):\n\n")
    f.write(report_text_holdout)
print(f"Full classification report saved to: {REPORT_TXT_FILE}")

# Generate confusion matrix and ROC-AUC plots
print("Saving final plots (Holdout)...")

cm_holdout = confusion_matrix(y_holdout, y_pred_holdout)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_holdout, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Predicted False (0)', 'Predicted True (1)'],
            yticklabels=['Actual False (0)', 'Actual True (1)'])
plt.title("Final Confusion Matrix (20% Holdout Set)")
plt.savefig(CM_OUTPUT_FILE)
plt.close()
print(f"CM plot (Holdout) saved to: {CM_OUTPUT_FILE}")

fpr_hold, tpr_hold, _ = roc_curve(y_holdout, y_pred_proba_holdout)
plt.figure(figsize=(10, 8))
plt.plot(fpr_hold, tpr_hold, color='green', lw=2, label=f'ROC Curve (Holdout AUC = {holdout_auc:.4f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Guess (AUC = 0.50)')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Final ROC-AUC Curve (20% Holdout Set)')
plt.legend(loc="lower right"); plt.grid(True)
plt.savefig(ROC_CURVE_FILE)
plt.close()
print(f"ROC-AUC curve plot (Holdout) saved to: {ROC_CURVE_FILE}")

print(f"\nFINAL TEST COMPLETED")
print(f"The model results are saved in: {ANALYSIS_DIR}")
print("\n17_final_holdout_test_60.py executed successfully.")