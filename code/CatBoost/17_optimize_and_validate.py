import pandas as pd
import numpy as np
import os
import json
import optuna
import time

from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Configurazione ---
SPLIT_DIR = 'preprocessed_splits' # Cartella con i dati divisi
MODEL_PARAMS_DIR = 'model_params'
ANALYSIS_DIR = 'Model_Analysis_Validation' # Output di QUESTA fase
os.makedirs(MODEL_PARAMS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Input (Train e Validation)
X_TRAIN_FILE = os.path.join(SPLIT_DIR, 'train_split_60_X.csv')
Y_TRAIN_FILE = os.path.join(SPLIT_DIR, 'train_split_60_y.csv')
X_VAL_FILE = os.path.join(SPLIT_DIR, 'validation_split_20_X.csv')
Y_VAL_FILE = os.path.join(SPLIT_DIR, 'validation_split_20_y.csv')
PARAMS_FILE = os.path.join(MODEL_PARAMS_DIR, 'best_catboost_params.json') # File dei parametri

# Output (Parametri e Grafici di VALIDAZIONE)
PARAMS_OUTPUT_FILE = os.path.join(MODEL_PARAMS_DIR, 'best_catboost_params.json')
ITERATION_OUTPUT_FILE = os.path.join(MODEL_PARAMS_DIR, 'best_iteration.json')
METRICS_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'validation_metrics_report.csv')
CM_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'validation_confusion_matrix.png')
IMPORTANCE_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'validation_feature_importance.png')
LOSS_CURVE_FILE = os.path.join(ANALYSIS_DIR, 'validation_loss_curve.png')
AUC_CURVE_FILE = os.path.join(ANALYSIS_DIR, 'validation_auc_learning_curve.png')
ROC_CURVE_FILE = os.path.join(ANALYSIS_DIR, 'validation_roc_auc_curve.png')

# --- Impostazioni Esecuzione ---
OPTUNA = True # Imposta a True per rieseguire l'ottimizzazione
N_TRIALS_OPTIMIZATION = 100 

print("Avvio Script di Ottimizzazione e VALIDAZIONE (Fase 2)...")

# --- 2. Caricamento Dati (Train 60% e Validation 20%) ---
print("Caricamento dati 60% (Train) e 20% (Validation)...")
try:
    X_train = pd.read_csv(X_TRAIN_FILE)
    y_train = pd.read_csv(Y_TRAIN_FILE).values.ravel()
    X_val = pd.read_csv(X_VAL_FILE)
    y_val = pd.read_csv(Y_VAL_FILE).values.ravel()
    
    if(OPTUNA == False and os.path.exists(PARAMS_FILE)):
        print(f"Caricamento parametri esistenti da {PARAMS_FILE}...")
        with open(PARAMS_FILE, 'r') as f:
            best_params_clean = json.load(f)
    elif(OPTUNA == False):
        print("ATTENZIONE: OPTUNA=False ma nessun file parametri trovato. Verranno usati i parametri di default.")
        best_params_clean = {} # Usa default
    else:
        print("OPTUNA=True. I parametri esistenti verranno sovrascritti.")

except FileNotFoundError:
    print(f"ERRORE: File non trovati in {SPLIT_DIR}. Esegui prima '16_data_splitter.py'.")
    exit()


# --- 3. Fase 1: Ottimizzazione (Eseguita solo su 60% Train) ---
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
        auc_scores = cross_val_score(model, X_data, y_data, cv=kf_inner, scoring='roc_auc', n_jobs=-1)
        return auc_scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), 
                   n_trials=N_TRIALS_OPTIMIZATION)
    end_time = time.time()
    print(f"Ottimizzazione completata in {end_time - start_time:.2f} secondi.")
    
    best_params_clean = study.best_params
    best_score_clean = study.best_value
    print(f"\nMiglior Punteggio AUC (CV su 60%): {best_score_clean:.6f}")
    
    # Prepariamo i parametri completi per il salvataggio
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


# --- 4. Fase 2: Diagnostica su Validation Set (20%) ---
print("\n--- Fase 2: Addestramento su 60% e Diagnostica su 20% (Validation) ---")

final_params_fit = best_params_clean.copy() 
final_params_fit.update({
    'n_estimators': 2000, 'early_stopping_rounds': 50,
    'eval_metric': 'Logloss', 'custom_metric': ['AUC'],
    'verbose': 1000, 'random_seed': 42
})

print("Addestramento modello: Train (60%), Eval (20%)...")
model = CatBoostClassifier(**final_params_fit)

# Addestriamo sul 60% e usiamo il 20% (Validation) come eval_set
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)], # [0] è train, [1] è validation
    plot=False,
    use_best_model=True, 
    verbose=1000
)

best_iteration = model.get_best_iteration()
print(f"\nNumero ottimale di iterazioni (alberi) trovate: {best_iteration}")
with open(ITERATION_OUTPUT_FILE, 'w') as f:
    json.dump({'best_iteration': best_iteration}, f, indent=2)
print(f"Numero iterazioni salvato in: {ITERATION_OUTPUT_FILE}")

print("\n--- 4.1 Metriche di Performance (su 20% Validation) ---")
y_pred_val = model.predict(X_val)
y_pred_proba_val = model.predict_proba(X_val)[:, 1]

# Calcoliamo le metriche
train_accuracy = accuracy_score(y_train, model.predict(X_train))
val_accuracy = accuracy_score(y_val, y_pred_val)
val_auc = roc_auc_score(y_val, y_pred_proba_val)

print(f"  Accuracy (sul Training 60%): {train_accuracy:.4f}")
print(f"  Accuracy (sul Validation 20%): {val_accuracy:.4f}  <-- Punteggio di tuning")
print(f"  AUC (sul Validation 20%): {val_auc:.4f}      <-- Punteggio di tuning")

if train_accuracy > (val_accuracy + 0.05):
    print("\033[93m  ATTENZIONE: Possibile Overfitting! (Delta > 5%)\033[0m")
else:
    print("\033[92m  OK: Nessun segno evidente di overfitting.\033[0m")

print("\n--- Classification Report (su 20% Validation) ---")
print(classification_report(y_val, y_pred_val, target_names=['Falso (0)', 'Vero (1)']))
report_dict_val = classification_report(y_val, y_pred_val, target_names=['Falso (0)', 'Vero (1)'], output_dict=True)

# Salvataggio report metriche
print(f"Salvataggio metriche (Validation) in: {METRICS_OUTPUT_FILE}")
metrics_data_val = {
    'Metric': ['Accuracy (Training 60%)', 'Accuracy (Validation 20%)', 'AUC (Validation 20%)',
               'Precision (Falso 0)', 'Recall (Falso 0)', 'F1-Score (Falso 0)',
               'Precision (Vero 1)', 'Recall (Vero 1)', 'F1-Score (Vero 1)'],
    'Score': [train_accuracy, val_accuracy, val_auc,
              report_dict_val['Falso (0)']['precision'], report_dict_val['Falso (0)']['recall'], report_dict_val['Falso (0)']['f1-score'],
              report_dict_val['Vero (1)']['precision'], report_dict_val['Vero (1)']['recall'], report_dict_val['Vero (1)']['f1-score']]
}
pd.DataFrame(metrics_data_val).to_csv(METRICS_OUTPUT_FILE, index=False, float_format='%.4f')


print("\n--- 4.2 Salvataggio Grafici Diagnostici (Validation) ---")

# --- Confusion Matrix (Validation) ---
cm_val = confusion_matrix(y_val, y_pred_val)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predetto Falso (0)', 'Predetto Vero (1)'], 
            yticklabels=['Reale Falso (0)', 'Reale Vero (1)'])
plt.title("Confusion Matrix (su 20% Validation Set)")
plt.savefig(CM_OUTPUT_FILE)
plt.close()
print(f"Grafico CM (Validation) salvato in: {CM_OUTPUT_FILE}")

# --- Feature Importance ---
importances = model.get_feature_importance()
feature_names = X_train.columns
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

# --- Grafici di Apprendimento ---
results = model.get_evals_result()
train_set_name = 'validation_0'
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
plt.savefig(ROC_CURVE_FILE)
plt.close()
print(f"Grafico curva ROC-AUC (Validation) salvato in: {ROC_CURVE_FILE}")

print("\n--- Ciclo di Tuning Completato ---")
print(f"Controlla i risultati in {ANALYSIS_DIR}.")
print("Se sei soddisfatto, esegui '18_final_holdout_test.py' per il test finale.")