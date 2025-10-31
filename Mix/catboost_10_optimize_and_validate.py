"""
This script optimizes and validates a CatBoostClassifier on preprocessed Pokémon battle data.
It optionally performs hyperparameter optimization using GridSearchCV on the training set, trains
the model on the training set, evaluates on the validation set, and saves metrics, plots,
feature importances, and best parameters.
"""

import pandas as pd
import numpy as np
import os
import json
import time

from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

SPLIT_DIR = 'Preprocessed_Splits'
MODEL_PARAMS_DIR = 'Model_Params'
ANALYSIS_DIR = 'Model_Analysis_Validation'
os.makedirs(MODEL_PARAMS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

X_TRAIN_FILE = os.path.join(SPLIT_DIR, 'train_split_60_X.csv')
Y_TRAIN_FILE = os.path.join(SPLIT_DIR, 'train_split_60_y.csv')
X_VAL_FILE = os.path.join(SPLIT_DIR, 'validation_split_20_X.csv')
Y_VAL_FILE = os.path.join(SPLIT_DIR, 'validation_split_20_y.csv')
PARAMS_FILE = os.path.join(MODEL_PARAMS_DIR, 'best_catboost_params.json')

PARAMS_OUTPUT_FILE = os.path.join(MODEL_PARAMS_DIR, 'best_catboost_params.json')
ITERATION_OUTPUT_FILE = os.path.join(MODEL_PARAMS_DIR, 'best_iteration.json')
REPORT_TXT_FILE = os.path.join(ANALYSIS_DIR, 'validation_classification_report.txt')
CM_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'validation_confusion_matrix.png')
IMPORTANCE_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'validation_feature_importance.png')
LOSS_CURVE_FILE = os.path.join(ANALYSIS_DIR, 'validation_loss_curve.png')
AUC_CURVE_FILE = os.path.join(ANALYSIS_DIR, 'validation_auc_learning_curve.png')
ROC_CURVE_FILE = os.path.join(ANALYSIS_DIR, 'validation_roc_auc_curve.png')

ESEGUI_GRID_SEARCH = False # Flag per eseguire o saltare la ricerca

print("Starting Optimization and Validation Script...")

print("Loading 60% Train and 20% Validation data...")
try:
    X_train = pd.read_csv(X_TRAIN_FILE)
    y_train = pd.read_csv(Y_TRAIN_FILE).values.ravel()
    X_val = pd.read_csv(X_VAL_FILE)
    y_val = pd.read_csv(Y_VAL_FILE).values.ravel()

    if(ESEGUI_GRID_SEARCH == False and os.path.exists(PARAMS_FILE)):
        print(f"Loading existing parameters from {PARAMS_FILE}...")
        with open(PARAMS_FILE, 'r') as f:
            best_params_clean = json.load(f)
    elif(ESEGUI_GRID_SEARCH == False):
        print("WARNING: ESEGUI_GRID_SEARCH=False but no parameter file found. Default parameters will be used.")
        best_params_clean = {}
    else:
        print("ESEGUI_GRID_SEARCH=True. Existing parameters will be overwritten.")

except FileNotFoundError:
    print(f"ERROR: Files not found in {SPLIT_DIR}. Run '16_data_splitter.py' first.")
    exit()


if(ESEGUI_GRID_SEARCH == True):
    print(f"\nPhase 1: Starting GridSearchCV (on 60% Train):\n")
    start_time = time.time()

    # --- MODIFICA: Griglia Estesa ---
    # 1. Definisci la griglia dei parametri da testare
    # (Griglia estesa per una ricerca più precisa e più lenta)
    param_grid = {
        'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2],
        'depth': [4, 5, 6, 8, 10],                            
        'l2_leaf_reg': [1.0, 2.0, 2.5, 3.0, 7.0, 10.0],
        # 'learning_rate': [0.015, 0.02, 0.025], per set 11
        # 'depth': [4, 5],                      
        # 'l2_leaf_reg': [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]           
    }
    
    # Calcola il numero di combinazioni
    combinations = 1
    for k in param_grid: combinations *= len(param_grid[k])
    
    print(f"GridSearch parameters to test: {param_grid}")
    print(f"Total combinations: {combinations}. CV Folds: 5. Total fits: {combinations * 5}")
    # --- FINE MODIFICA ---

    # 2. Definisci il modello base (estimator)
    base_model = CatBoostClassifier(
        objective='Logloss',
        eval_metric='AUC', # Metrica per il report interno di CatBoost
        verbose=0,
        random_seed=42,
        n_estimators=1000 # Numero fisso di alberi per la ricerca
    )

    # 3. Definisci la strategia di Cross-Validation
    kf_inner = KFold(n_splits=5, shuffle=True, random_state=42)

    # 4. Configura GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=kf_inner,
        scoring='roc_auc', # Metrica di Scikit-learn per la valutazione
        n_jobs=-1,        # Usa tutti i core disponibili
        verbose=2         # Mostra i progressi
    )

    # 5. Esegui la ricerca
    grid_search.fit(X_train, y_train)

    end_time = time.time()
    print(f"GridSearch completed in {end_time - start_time:.2f} seconds.")

    # 6. Ottieni i risultati migliori
    best_params_clean = grid_search.best_params_
    best_score_clean = grid_search.best_score_

    print(f"\nBest AUC Score (CV on 60%): {best_score_clean:.6f}")
    print(f"Best Parameters found: {best_params_clean}")

    # 7. Salva i parametri
    final_params_to_save = {
        **best_params_clean,
        'objective': 'Logloss', 'eval_metric': 'AUC',
        'verbose': 0, 'random_seed': 42
    }
    with open(PARAMS_OUTPUT_FILE, 'w') as f:
        json.dump(final_params_to_save, f, indent=2)
    print(f"Cleaned parameters saved to: {PARAMS_OUTPUT_FILE}")
    best_params_clean = final_params_to_save

else:
    print(f"\nPhase 1: GridSearchCV optimization skipped (ESEGUI_GRID_SEARCH=False):\n")


print("\nPhase 2: Training on 60% and Diagnostics on 20% (Validation):\n")

final_params_fit = best_params_clean.copy()
final_params_fit.update({
    'n_estimators': 2000, 'early_stopping_rounds': 50,
    'eval_metric': 'Logloss', 'custom_metric': ['AUC'],
    'verbose': 1000, 'random_seed': 42
})

print("Training model: Train (60%), Eval (20%)...")
model = CatBoostClassifier(**final_params_fit)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    plot=False,
    use_best_model=True,
    verbose=1000
)

best_iteration = model.get_best_iteration()
print(f"\nOptimal number of iterations (trees) found: {best_iteration}")
with open(ITERATION_OUTPUT_FILE, 'w') as f:
    json.dump({'best_iteration': best_iteration}, f, indent=2)
print(f"Best iteration saved in: {ITERATION_OUTPUT_FILE}")

print("\nPhase 3: Performance Metrics (on 20% Validation):\n")
y_pred_val = model.predict(X_val)
y_pred_proba_val = model.predict_proba(X_val)[:, 1]

train_accuracy = accuracy_score(y_train, model.predict(X_train))
val_accuracy = accuracy_score(y_val, y_pred_val)
val_auc = roc_auc_score(y_val, y_pred_proba_val)

print(f"  Accuracy (Training 60%): {train_accuracy:.4f}")
print(f"  Accuracy (Validation 20%): {val_accuracy:.4f}")
print(f"  AUC (Validation 20%): {val_auc:.4f}")

if train_accuracy > (val_accuracy + 0.05):
    print("\033[93m  WARNING: Possible Overfitting! (Delta > 5%)\033[0m")
else:
    print("\033[92m  OK: No evident overfitting.\033[0m")

print("\nClassification Report (on 20% Validation):\n")
print(classification_report(y_val, y_pred_val, target_names=['Falso (0)', 'Vero (1)'], digits=4))
report_dict_val = classification_report(y_val, y_pred_val, target_names=['Falso (0)', 'Vero (1)'], output_dict=True, digits=4)

# Save full classification report as text file
print(f"Saving metrics (Validation) to: {REPORT_TXT_FILE}")
report_text_val = classification_report(y_val, y_pred_val, target_names=['Falso (0)', 'Vero (1)'], digits=4)
with open(REPORT_TXT_FILE, 'w') as f:
    f.write("Classification Report (on 20% Validation):\n\n")
    f.write(report_text_val)
print(f"Full classification report saved to: {REPORT_TXT_FILE}")

print("\nPhase 4: Saving Diagnostic Plots (Validation):\n")

cm_val = confusion_matrix(y_val, y_pred_val)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted False (0)', 'Predicted True (1)'],
            yticklabels=['Actual False (0)', 'Actual True (1)'])
plt.title("Confusion Matrix (20% Validation Set)")
plt.savefig(CM_OUTPUT_FILE)
plt.close()
print(f"CM plot (Validation) saved to: {CM_OUTPUT_FILE}")

importances = model.get_feature_importance()
feature_names = X_train.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
top_20_features = importance_df.sort_values(by='Importance', ascending=False).head(20)
print("\nTop 20 Most Important Features:")
print(top_20_features.to_string(index=False))
plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=top_20_features, palette='viridis')
plt.title('Top 20 Feature Importance (CatBoost)')
plt.tight_layout()
plt.savefig(IMPORTANCE_OUTPUT_FILE)
plt.close()
print(f"Feature Importance plot saved to: {IMPORTANCE_OUTPUT_FILE}")

results = model.get_evals_result()
train_set_name = 'validation_0'
valid_set_name = 'validation_1'
if 'Logloss' in results[train_set_name]:
    plt.figure(figsize=(10, 6))
    plt.plot(results[train_set_name]['Logloss'], label='Training Loss (60%)')
    plt.plot(results[valid_set_name]['Logloss'], label='Validation Loss (20%)')
    plt.title('Learning Curve (Logloss)')
    plt.xlabel('Iterations'); plt.ylabel('Logloss'); plt.legend(); plt.grid(True)
    plt.savefig(LOSS_CURVE_FILE)
    plt.close()
    print(f"Loss curve plot saved to: {LOSS_CURVE_FILE}")
if 'AUC' in results[train_set_name]:
    plt.figure(figsize=(10, 6))
    plt.plot(results[train_set_name]['AUC'], label='Training AUC (60%)')
    plt.plot(results[valid_set_name]['AUC'], label='Validation AUC (20%)')
    plt.title('Learning Curve (AUC)')
    plt.xlabel('Iterations'); plt.ylabel('AUC'); plt.legend(); plt.grid(True)
    plt.savefig(AUC_CURVE_FILE)
    plt.close()
    print(f"AUC learning curve plot saved to: {AUC_CURVE_FILE}")

fpr_val, tpr_val, _ = roc_curve(y_val, y_pred_proba_val)
plt.figure(figsize=(10, 8))
plt.plot(fpr_val, tpr_val, color='blue', lw=2, label=f'ROC Curve (Validation AUC = {val_auc:.4f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Guess (AUC = 0.50)')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve (20% Validation Set)')
plt.legend(loc="lower right"); plt.grid(True)
plt.savefig(ROC_CURVE_FILE)
plt.close()
print(f"ROC-AUC curve plot (Validation) saved to: {ROC_CURVE_FILE}")

print("\nTuning cycle completed")
print(f"Check results in {ANALYSIS_DIR}.")
print("\n19_optimize_and_validate.py executed successfully.")