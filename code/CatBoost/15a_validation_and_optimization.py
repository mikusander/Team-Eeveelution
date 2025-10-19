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
ANALYSIS_DIR = 'Model_Analysis_Output_8020'
os.makedirs(MODEL_PARAMS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Input
TRAIN_FILE = os.path.join(PREPROCESSED_DIR, 'train_processed.csv')
TARGET_FILE = os.path.join(PREPROCESSED_DIR, 'target_train.csv')
PARAMS_FILE = os.path.join(MODEL_PARAMS_DIR, 'best_catboost_params_final.json')

# Output
PARAMS_OUTPUT_FILE = os.path.join(MODEL_PARAMS_DIR, 'best_catboost_params_final.json')
METRICS_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'final_metrics_report.csv')
CM_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'final_honest_confusion_matrix.png')
IMPORTANCE_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'final_feature_importance.png')
LOSS_CURVE_FILE = os.path.join(ANALYSIS_DIR, 'final_loss_curve.png')
AUC_CURVE_FILE = os.path.join(ANALYSIS_DIR, 'final_auc_learning_curve.png')
ROC_CURVE_FILE = os.path.join(ANALYSIS_DIR, 'final_roc_auc_curve.png')

# --- Impostazioni Esecuzione ---
# Imposta a True se vuoi ri-eseguire l'ottimizzazione Optuna
OPTUNA = False 
N_TRIALS_OPTIMIZATION = 100 # Numero di tentativi Optuna

print("Avvio Script Unificato di Ottimizzazione e Validazione Completa...")

# --- 2. Caricamento Dati Completi ---
print("Caricamento dati...")
try:
    X = pd.read_csv(TRAIN_FILE)
    y = pd.read_csv(TARGET_FILE).values.ravel()
    
    # Se non eseguiamo Optuna, carichiamo i parametri esistenti
    if(OPTUNA == False):
        print(f"Caricamento parametri esistenti da {PARAMS_FILE}...")
        with open(PARAMS_FILE, 'r') as f:
            best_params_clean = json.load(f)
except FileNotFoundError:
    print("ERRORE: File non trovati. Assicurati di aver eseguito lo script 12.")
    exit()

# --- 3. Split Esterno 80/20 (Train / Validation Hold-out) ---
print("Divisione 80/20 (Train/Validation)...")
# Il 20% (X_val, y_val) sarà il nostro set di validazione 'pulito'
X_train_outer, X_val_outer, y_train_outer, y_val_outer = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=42, 
    stratify=y
)
print(f"  Training set per Optuna/Fit: {X_train_outer.shape[0]} campioni")
print(f"  Validation set 'pulito': {X_val_outer.shape[0]} campioni")


# --- 4. Fase 1: Ottimizzazione (Eseguita solo se OPTUNA == True) ---
if(OPTUNA == True):
    print(f"\n--- Fase 1: Avvio studio Optuna per {N_TRIALS_OPTIMIZATION} tentativi ---")
    start_time = time.time()

    # Definiamo la funzione objective che Optuna ottimizzerà
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
        
        # Eseguiamo la CV interna *solo* sui dati di training (80%)
        auc_scores = cross_val_score(model, X_data, y_data, cv=kf_inner, scoring='roc_auc', n_jobs=-1)
        return auc_scores.mean()

    # Creiamo lo studio e ottimizziamo
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train_outer, y_train_outer), 
                   n_trials=N_TRIALS_OPTIMIZATION)

    end_time = time.time()
    print(f"Ottimizzazione completata in {end_time - start_time:.2f} secondi.")

    # Salvataggio Parametri "Puliti"
    best_params_clean = study.best_params # Questi sono solo i parametri ottimizzati
    best_score_clean = study.best_value

    print("\n--- Risultati Ottimizzazione (sull'80%) ---")
    print(f"Miglior Punteggio AUC (medio, CV interna): {best_score_clean:.6f}")
    
    # Prepariamo i parametri completi per il salvataggio
    final_params_to_save = {
        **best_params_clean, 
        'objective': 'Logloss', 'eval_metric': 'AUC',
        'verbose': 0, 'random_seed': 42
    }
    with open(PARAMS_OUTPUT_FILE, 'w') as f:
        json.dump(final_params_to_save, f, indent=2)
    print(f"Parametri 'puliti' salvati in: {PARAMS_OUTPUT_FILE}")
    
    # Sovrascriviamo best_params_clean con la versione completa per la Fase 2
    best_params_clean = final_params_to_save

else:
    print(f"\n--- Fase 1: Ottimizzazione Optuna saltata (OPTUNA=False) ---")


# --- 5. Fase 2: Diagnostica Finale (sul 20% 'pulito') ---
print("\n--- Fase 2: Validazione Finale e Diagnostica sul 20% 'pulito' ---")

# Definiamo i parametri finali per l'addestramento
# Partiamo dai parametri ottimizzati (o caricati)
final_params_fit = best_params_clean.copy() 
# Aggiungiamo/sovrascriviamo i parametri per il fit finale
final_params_fit.update({
    'n_estimators': 2000,
    'early_stopping_rounds': 50,
    'eval_metric': 'Logloss',     # Metrica per l'early stopping
    'custom_metric': ['AUC'],     # Metrica aggiuntiva da tracciare
    'verbose': 1000,
    'random_seed': 42
})

print("Addestramento modello finale sull'80%, validazione sul 20%...")
final_model = CatBoostClassifier(**final_params_fit)

# Addestriamo sull'80% e usiamo il 20% 'pulito' come eval_set
# Includiamo anche i dati di train nell'eval_set per le curve di apprendimento
final_model.fit(
    X_train_outer, y_train_outer,
    eval_set=[(X_train_outer, y_train_outer), (X_val_outer, y_val_outer)],
    plot=False,
    use_best_model=True, # Carica il modello dal bestIteration (basato su Logloss)
    verbose=1000
)

print("\n--- 5.1 Metriche di Performance 'Oneste' (sul 20%) ---")
y_pred_final = final_model.predict(X_val_outer)
y_pred_proba_final = final_model.predict_proba(X_val_outer)[:, 1]

# Calcoliamo le metriche
train_accuracy = accuracy_score(y_train_outer, final_model.predict(X_train_outer))
val_accuracy = accuracy_score(y_val_outer, y_pred_final)
val_auc = roc_auc_score(y_val_outer, y_pred_proba_final)

print(f"  Accuracy (sul Training 80%): {train_accuracy:.4f}")
print(f"  Accuracy (sul Validation 20% 'pulito'): {val_accuracy:.4f}  <-- Punteggio 'onesto'")
print(f"  AUC (sul Validation 20% 'pulito'): {val_auc:.4f}      <-- Punteggio 'onesto'")

if train_accuracy > (val_accuracy + 0.05):
    print("\033[93m  ATTENZIONE: Possibile Overfitting!\033[0m")
else:
    print("\033[92m  OK: Nessun segno evidente di overfitting.\033[0m")

# Classification Report
print("\n--- Classification Report 'Onesto' (sul 20%) ---")
print(classification_report(y_val_outer, y_pred_final, target_names=['Falso (0)', 'Vero (1)']))
report_dict = classification_report(y_val_outer, y_pred_final, target_names=['Falso (0)', 'Vero (1)'], output_dict=True)

# Salvataggio report metriche
print(f"Salvataggio metriche (tabella pulita) in: {METRICS_OUTPUT_FILE}")
metrics_data = {
    'Metric': ['Accuracy (Training)', 'Accuracy (Validation)', 'AUC (Validation)',
               'Precision (Falso 0)', 'Recall (Falso 0)', 'F1-Score (Falso 0)',
               'Precision (Vero 1)', 'Recall (Vero 1)', 'F1-Score (Vero 1)'],
    'Score': [train_accuracy, val_accuracy, val_auc,
              report_dict['Falso (0)']['precision'], report_dict['Falso (0)']['recall'], report_dict['Falso (0)']['f1-score'],
              report_dict['Vero (1)']['precision'], report_dict['Vero (1)']['recall'], report_dict['Vero (1)']['f1-score']]
}
metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv(METRICS_OUTPUT_FILE, index=False, float_format='%.4f')


print("\n--- 5.2 Salvataggio Grafici Diagnostici ---")

# --- Confusion Matrix ---
cm = confusion_matrix(y_val_outer, y_pred_final)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predetto Falso (0)', 'Predetto Vero (1)'], 
            yticklabels=['Reale Falso (0)', 'Reale Vero (1)'])
plt.title("Confusion Matrix 'Onesta' (su 20% di dati mai visti)")
plt.savefig(CM_OUTPUT_FILE)
plt.close()
print(f"Grafico della Confusion Matrix salvato in: {CM_OUTPUT_FILE}")

# --- Feature Importance ---
importances = final_model.get_feature_importance()
feature_names = X_train_outer.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
top_20_features = importance_df.head(20)

print("Top 20 Features più importanti:")
print(top_20_features.to_string(index=False))

plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=top_20_features, palette='viridis')
plt.title('Top 20 Feature Importance (CatBoost)')
plt.tight_layout()
plt.savefig(IMPORTANCE_OUTPUT_FILE)
plt.close()
print(f"Grafico Feature Importance salvato in: {IMPORTANCE_OUTPUT_FILE}")

# --- Grafici di Apprendimento ---
results = final_model.get_evals_result()
train_set_name = 'learn' if 'learn' in results else 'validation_0'
valid_set_name = 'validation_1'

# --- Curva Logloss ---
if 'Logloss' in results[train_set_name]:
    plt.figure(figsize=(10, 6))
    plt.plot(results[train_set_name]['Logloss'], label='Training Loss')
    plt.plot(results[valid_set_name]['Logloss'], label='Validation Loss')
    plt.title('Curva di Apprendimento (Logloss)')
    plt.xlabel('Iterazioni (Alberi)')
    plt.ylabel('Logloss')
    plt.legend()
    plt.grid(True)
    plt.savefig(LOSS_CURVE_FILE)
    plt.close()
    print(f"Grafico curva Loss salvato in: {LOSS_CURVE_FILE}")
else:
    print("⚠️ Nessuna metrica Logloss trovata nei risultati di addestramento.")

# --- Curva AUC (Aggiunta) ---
if 'AUC' in results[train_set_name]:
    plt.figure(figsize=(10, 6))
    plt.plot(results[train_set_name]['AUC'], label='Training AUC')
    plt.plot(results[valid_set_name]['AUC'], label='Validation AUC')
    plt.title('Curva di Apprendimento (AUC)')
    plt.xlabel('Iterazioni (Alberi)')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)
    plt.savefig(AUC_CURVE_FILE)
    plt.close()
    print(f"Grafico curva AUC salvato in: {AUC_CURVE_FILE}")
else:
    print("⚠️ Nessuna metrica AUC trovata nei risultati di addestramento.")


# --- Curva ROC-AUC ---
fpr, tpr, thresholds = roc_curve(y_val_outer, y_pred_proba_final)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Curva ROC (AUC = {val_auc:.4f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Caso (AUC = 0.50)')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Curva ROC-AUC (sul 20% di validazione)')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(ROC_CURVE_FILE)
plt.close()
print(f"Grafico curva ROC-AUC salvato in: {ROC_CURVE_FILE}")

print("\nDiagnostica DEFINITIVA completata.")
print("Ora siamo pronti per lo script di submission finale.")