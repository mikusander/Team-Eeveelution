import pandas as pd
import numpy as np
import os
import json
import optuna
import time

from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Configurazione ---
PREPROCESSED_DIR = 'preprocessed_data'
MODEL_PARAMS_DIR = 'model_params'
# NUOVA CARTELLA per questa analisi
ANALYSIS_DIR = 'Model_Analysis_Output_602020' 
os.makedirs(MODEL_PARAMS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Input
TRAIN_FILE = os.path.join(PREPROCESSED_DIR, 'train_processed.csv')
TARGET_FILE = os.path.join(PREPROCESSED_DIR, 'target_train.csv')
PARAMS_FILE = os.path.join(MODEL_PARAMS_DIR, 'best_catboost_params_final.json')

# Output
PARAMS_OUTPUT_FILE = os.path.join(MODEL_PARAMS_DIR, 'best_catboost_params_final.json')
# Grafici di training
IMPORTANCE_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'feature_importance.png')
LOSS_CURVE_FILE = os.path.join(ANALYSIS_DIR, 'learning_curve_loss.png')
AUC_CURVE_FILE = os.path.join(ANALYSIS_DIR, 'learning_curve_auc.png')

# Output per il Validation set (20%)
VALIDATION_METRICS_FILE = os.path.join(ANALYSIS_DIR, 'validation_metrics_report.csv')
VALIDATION_CM_FILE = os.path.join(ANALYSIS_DIR, 'validation_confusion_matrix.png')
VALIDATION_ROC_FILE = os.path.join(ANALYSIS_DIR, 'validation_roc_auc_curve.png')

# Output per il Holdout set (20%)
HOLDOUT_METRICS_FILE = os.path.join(ANALYSIS_DIR, 'holdout_metrics_report.csv')
HOLDOUT_CM_FILE = os.path.join(ANALYSIS_DIR, 'holdout_confusion_matrix.png')
HOLDOUT_ROC_FILE = os.path.join(ANALYSIS_DIR, 'holdout_roc_auc_curve.png')


# --- Impostazioni Esecuzione ---
OPTUNA = True 
N_TRIALS_OPTIMIZATION = 100 

print("Avvio Script Validazione 60/20/20 (Train/Validation/Holdout)...")

# --- 2. Caricamento Dati Completi ---
print("Caricamento dati...")
try:
    X = pd.read_csv(TRAIN_FILE)
    y = pd.read_csv(TARGET_FILE).values.ravel()
    
    if(OPTUNA == False):
        print(f"Caricamento parametri esistenti da {PARAMS_FILE}...")
        with open(PARAMS_FILE, 'r') as f:
            best_params_clean = json.load(f)
except FileNotFoundError:
    print("ERRORE: File non trovati. Assicurati di aver eseguito lo script 12.")
    exit()

# --- 3. Split 60/20/20 (Train / Validation / Holdout) ---
print("Divisione 60/20/20 (Train/Validation/Holdout)...")

# 1. Prima dividiamo 100% -> 80% (Train+Val) e 20% (Holdout)
# Usiamo random_state=42 per garantire che questo split sia SEMPRE lo stesso
X_temp, X_holdout, y_temp, y_holdout = train_test_split(
    X, y, 
    test_size=0.20, # 20% per l'Holdout
    random_state=42, 
    stratify=y
)

# 2. Ora dividiamo l'80% -> 75% (Train) e 25% (Validation)
#    (75% di 80% = 60% del totale; 25% di 80% = 20% del totale)
# Usiamo un random_state diverso (es. 84) per questo secondo split
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25, # 25% di 80% è 20%
    random_state=84, 
    stratify=y_temp
)

print(f"  Training set (60%): {X_train.shape[0]} campioni")
print(f"  Validation set (20%): {X_val.shape[0]} campioni")
print(f"  Holdout set (20%): {X_holdout.shape[0]} campioni")


# --- 4. Fase 1: Ottimizzazione (Eseguita solo su 60% Train) ---
if(OPTUNA == True):
    print(f"\n--- Fase 1: Avvio studio Optuna per {N_TRIALS_OPTIMIZATION} tentativi (sul 60% Train) ---")
    start_time = time.time()

    def objective(trial, X_data, y_data):
        kf_inner = KFold(n_splits=5, shuffle=True, random_state=42)
        params = {
            'objective': 'Logloss', 'eval_metric': 'AUC', 'verbose': 0,
            'random_seed': 42, 'n_estimators': 1000, 'early_stopping_rounds': 50,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
        }
        model = CatBoostClassifier(**params)
        # CV interna solo sul 60%
        auc_scores = cross_val_score(model, X_data, y_data, cv=kf_inner, scoring='roc_auc', n_jobs=-1)
        return auc_scores.mean()

    study = optuna.create_study(direction='maximize')
    # Passiamo solo X_train, y_train (60%)
    study.optimize(lambda trial: objective(trial, X_train, y_train), 
                   n_trials=N_TRIALS_OPTIMIZATION)

    end_time = time.time()
    print(f"Ottimizzazione completata in {end_time - start_time:.2f} secondi.")
    best_params_clean = study.best_params
    best_score_clean = study.best_value
    print(f"\nMiglior Punteggio AUC (CV su 60%): {best_score_clean:.6f}")
    
    final_params_to_save = {
        **best_params_clean, 
        'objective': 'Logloss', 'eval_metric': 'AUC',
        'verbose': 0, 'random_seed': 42
    }
    with open(PARAMS_OUTPUT_FILE, 'w') as f:
        json.dump(final_params_to_save, f, indent=2)
    print(f"Parametri 'puliti' salvati in: {PARAMS_OUTPUT_FILE}")
    best_params_clean = final_params_to_save

else:
    print(f"\n--- Fase 1: Ottimizzazione Optuna saltata (OPTUNA=False) ---")


# --- 5. Fase 2: Addestramento e Diagnostica su Validation Set (20%) ---
print("\n--- Fase 2: Addestramento su 60% e Diagnostica su 20% (Validation) ---")

final_params_fit = best_params_clean.copy() 
final_params_fit.update({
    'n_estimators': 2000, 'early_stopping_rounds': 50,
    'eval_metric': 'Logloss', 'custom_metric': ['AUC'],
    'verbose': 1000, 'random_seed': 42
})

print("Addestramento modello: Train (60%), Eval (20%)...")
final_model = CatBoostClassifier(**final_params_fit)

# Addestriamo sul 60% e usiamo il 20% (Validation) come eval_set
final_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)], # [0] è train, [1] è validation
    plot=False,
    use_best_model=True, 
    verbose=1000
)

print("\n--- 5.1 Metriche di Performance (su 20% Validation) ---")
# Predizioni sul Validation set
y_pred_val = final_model.predict(X_val)
y_pred_proba_val = final_model.predict_proba(X_val)[:, 1]

# Calcoliamo le metriche
train_accuracy = accuracy_score(y_train, final_model.predict(X_train))
val_accuracy = accuracy_score(y_val, y_pred_val)
val_auc = roc_auc_score(y_val, y_pred_proba_val)

print(f"  Accuracy (sul Training 60%): {train_accuracy:.4f}")
print(f"  Accuracy (sul Validation 20%): {val_accuracy:.4f}  <-- Punteggio di tuning")
print(f"  AUC (sul Validation 20%): {val_auc:.4f}      <-- Punteggio di tuning")

if train_accuracy > (val_accuracy + 0.05):
    print("\033[93m  ATTENZIONE: Possibile Overfitting!\033[0m")
else:
    print("\033[92m  OK: Nessun segno evidente di overfitting.\033[0m")

print("\n--- Classification Report (su 20% Validation) ---")
print(classification_report(y_val, y_pred_val, target_names=['Falso (0)', 'Vero (1)']))
report_dict_val = classification_report(y_val, y_pred_val, target_names=['Falso (0)', 'Vero (1)'], output_dict=True)

# Salvataggio report metriche
print(f"Salvataggio metriche (Validation) in: {VALIDATION_METRICS_FILE}")
metrics_data_val = {
    'Metric': ['Accuracy (Training 60%)', 'Accuracy (Validation 20%)', 'AUC (Validation 20%)',
               'Precision (Falso 0)', 'Recall (Falso 0)', 'F1-Score (Falso 0)',
               'Precision (Vero 1)', 'Recall (Vero 1)', 'F1-Score (Vero 1)'],
    'Score': [train_accuracy, val_accuracy, val_auc,
              report_dict_val['Falso (0)']['precision'], report_dict_val['Falso (0)']['recall'], report_dict_val['Falso (0)']['f1-score'],
              report_dict_val['Vero (1)']['precision'], report_dict_val['Vero (1)']['recall'], report_dict_val['Vero (1)']['f1-score']]
}
pd.DataFrame(metrics_data_val).to_csv(VALIDATION_METRICS_FILE, index=False, float_format='%.4f')


print("\n--- 5.2 Salvataggio Grafici Diagnostici (Training & Validation) ---")

# --- Confusion Matrix (Validation) ---
cm_val = confusion_matrix(y_val, y_pred_val)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predetto Falso (0)', 'Predetto Vero (1)'], 
            yticklabels=['Reale Falso (0)', 'Reale Vero (1)'])
plt.title("Confusion Matrix (su 20% Validation Set)")
plt.savefig(VALIDATION_CM_FILE)
plt.close()
print(f"Grafico CM (Validation) salvato in: {VALIDATION_CM_FILE}")

# --- Feature Importance (dal training) ---
importances = final_model.get_feature_importance()
feature_names = X_train.columns # Colonne del 60%
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
top_20_features = importance_df.sort_values(by='Importance', ascending=False).head(20)
print("\nTop 20 Features più importanti:")
print(top_20_features.to_string(index=False))
plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=top_20_features, palette='viridis')
plt.title('Top 20 Feature Importance (CatBoost)')
plt.tight_layout()
plt.savefig(IMPORTANCE_OUTPUT_FILE)
plt.close()
print(f"Grafico Feature Importance salvato in: {IMPORTANCE_OUTPUT_FILE}")

# --- Grafici di Apprendimento (dal training) ---
results = final_model.get_evals_result()
train_set_name = 'validation_0' # CatBoost li nomina così
valid_set_name = 'validation_1'

if 'Logloss' in results[train_set_name]:
    plt.figure(figsize=(10, 6))
    plt.plot(results[train_set_name]['Logloss'], label='Training Loss (60%)')
    plt.plot(results[valid_set_name]['Logloss'], label='Validation Loss (20%)')
    plt.title('Curva di Apprendimento (Logloss)')
    plt.xlabel('Iterazioni'); plt.ylabel('Logloss'); plt.legend(); plt.grid(True)
    plt.savefig(LOSS_CURVE_FILE)
    plt.close()
    print(f"Grafico curva Loss salvato in: {LOSS_CURVE_FILE}")

if 'AUC' in results[train_set_name]:
    plt.figure(figsize=(10, 6))
    plt.plot(results[train_set_name]['AUC'], label='Training AUC (60%)')
    plt.plot(results[valid_set_name]['AUC'], label='Validation AUC (20%)')
    plt.title('Curva di Apprendimento (AUC)')
    plt.xlabel('Iterazioni'); plt.ylabel('AUC'); plt.legend(); plt.grid(True)
    plt.savefig(AUC_CURVE_FILE)
    plt.close()
    print(f"Grafico curva AUC salvato in: {AUC_CURVE_FILE}")

# --- Curva ROC-AUC (Validation) ---
fpr_val, tpr_val, _ = roc_curve(y_val, y_pred_proba_val)
plt.figure(figsize=(10, 8))
plt.plot(fpr_val, tpr_val, color='blue', lw=2, label=f'Curva ROC (Validation AUC = {val_auc:.4f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Caso (AUC = 0.50)')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Curva ROC-AUC (su 20% Validation Set)')
plt.legend(loc="lower right"); plt.grid(True)
plt.savefig(VALIDATION_ROC_FILE)
plt.close()
print(f"Grafico curva ROC-AUC (Validation) salvato in: {VALIDATION_ROC_FILE}")


# --- 6. Fase 3: Validazione Finale su HOLDOUT Set (20% mai visto) ---
print(f"\n\n--- Fase 3: Validazione Finale su 20% HOLDOUT (Dati mai visti) ---")
print("Valutazione del modello finale sul set di holdout...")

# Usiamo il 'final_model' già addestrato per predire sul 'X_holdout'
y_pred_holdout = final_model.predict(X_holdout)
y_pred_proba_holdout = final_model.predict_proba(X_holdout)[:, 1]

# Calcoliamo le metriche "finali"
holdout_accuracy = accuracy_score(y_holdout, y_pred_holdout)
holdout_auc = roc_auc_score(y_holdout, y_pred_proba_holdout)

print(f"\n--- Metriche di Performance 'Finali' (su 20% Holdout) ---")
print(f"  Accuracy (sul Holdout 20%): {holdout_accuracy:.4f}  <-- Punteggio 'VERO'")
print(f"  AUC (sul Holdout 20%): {holdout_auc:.4f}      <-- Punteggio 'VERO'")

print("\n--- Classification Report 'Finale' (su 20% Holdout) ---")
print(classification_report(y_holdout, y_pred_holdout, target_names=['Falso (0)', 'Vero (1)']))
report_dict_holdout = classification_report(y_holdout, y_pred_holdout, target_names=['Falso (0)', 'Vero (1)'], output_dict=True)

# Salvataggio report metriche finali
print(f"Salvataggio metriche (Holdout) in: {HOLDOUT_METRICS_FILE}")
metrics_data_holdout = {
    'Metric': ['Accuracy (Holdout 20%)', 'AUC (Holdout 20%)',
               'Precision (Falso 0)', 'Recall (Falso 0)', 'F1-Score (Falso 0)',
               'Precision (Vero 1)', 'Recall (Vero 1)', 'F1-Score (Vero 1)'],
    'Score': [holdout_accuracy, holdout_auc,
              report_dict_holdout['Falso (0)']['precision'], report_dict_holdout['Falso (0)']['recall'], report_dict_holdout['Falso (0)']['f1-score'],
              report_dict_holdout['Vero (1)']['precision'], report_dict_holdout['Vero (1)']['recall'], report_dict_holdout['Vero (1)']['f1-score']]
}
pd.DataFrame(metrics_data_holdout).to_csv(HOLDOUT_METRICS_FILE, index=False, float_format='%.4f')

# --- Confusion Matrix (Holdout) ---
cm_holdout = confusion_matrix(y_holdout, y_pred_holdout)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_holdout, annot=True, fmt='d', cmap='Greens', # Colore diverso
            xticklabels=['Predetto Falso (0)', 'Predetto Vero (1)'], 
            yticklabels=['Reale Falso (0)', 'Reale Vero (1)'])
plt.title("Confusion Matrix 'Finale' (su 20% Holdout Set)")
plt.savefig(HOLDOUT_CM_FILE)
plt.close()
print(f"Grafico CM (Holdout) salvato in: {HOLDOUT_CM_FILE}")

# --- Curva ROC-AUC (Holdout) ---
fpr_hold, tpr_hold, _ = roc_curve(y_holdout, y_pred_proba_holdout)
plt.figure(figsize=(10, 8))
plt.plot(fpr_hold, tpr_hold, color='green', lw=2, label=f'Curva ROC (Holdout AUC = {holdout_auc:.4f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Caso (AUC = 0.50)')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Curva ROC-AUC \'Finale\' (su 20% Holdout Set)')
plt.legend(loc="lower right"); plt.grid(True)
plt.savefig(HOLDOUT_ROC_FILE)
plt.close()
print(f"Grafico curva ROC-AUC (Holdout) salvato in: {HOLDOUT_ROC_FILE}")


print("\nDiagnostica DEFINITIVA 60/20/20 completata.")
print("Confronta i report 'validation' e 'holdout' per la valutazione finale.")