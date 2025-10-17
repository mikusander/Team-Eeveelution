import pandas as pd
import numpy as np
import os
import json
import optuna
import time

from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score, KFold

# --- 1. Configurazione ---
PREPROCESSED_DIR = 'Preprocessed_Data'
MODEL_PARAMS_DIR = 'Model_Params' # <-- NUOVA CARTELLA PER L'OUTPUT
os.makedirs(MODEL_PARAMS_DIR, exist_ok=True) # <-- Creiamo la cartella

TRAIN_FILE = os.path.join(PREPROCESSED_DIR, 'train_processed.csv')
TARGET_FILE = os.path.join(PREPROCESSED_DIR, 'target_train.csv')

# File di output per i nostri parametri
PARAMS_OUTPUT_FILE = os.path.join(MODEL_PARAMS_DIR, 'best_catboost_params.json') # <-- Salviamo nella nuova cartella
N_TRIALS = 100 

print("Avvio Ottimizzazione iperparametri per CatBoost con Optuna...")

# --- 2. Caricamento Dati Processati ---
print(f"Caricamento dati da {PREPROCESSED_DIR}...")
try:
    X_train = pd.read_csv(TRAIN_FILE)
    y_train = pd.read_csv(TARGET_FILE).values.ravel() 
except FileNotFoundError:
    print(f"ERRORE: File non trovati in '{PREPROCESSED_DIR}'.")
    exit()

print("Dati 100% numerici caricati.")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# --- 3. Definizione della Funzione "Objective" ---
def objective(trial):
    """
    Definisce lo spazio di ricerca e restituisce il punteggio
    (AUC medio) per una data combinazione di iperparametri.
    """
    
    params = {
        'objective': 'Logloss',
        'eval_metric': 'AUC',
        'verbose': 0, 
        'random_seed': 42,
        'n_estimators': 1000, 
        
        # --- MODIFICA QUI ---
        # Aggiungiamo 'early_stopping_rounds' qui DENTRO.
        # CatBoost userà automaticamente una parte dei dati di
        # training (di ogni fold) per fare l'early stopping.
        'early_stopping_rounds': 50, 
        # --------------------

        # Spazio di ricerca di Optuna
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
    }

    model = CatBoostClassifier(**params)
    
    # --- MODIFICA QUI ---
    # Rimuoviamo l'argomento 'fit_params' che causava l'errore
    auc_scores = cross_val_score(
        model, 
        X_train, 
        y_train, 
        cv=kf, 
        scoring='roc_auc', 
        n_jobs=-1
    )
    # --------------------
    
    return auc_scores.mean()

# --- 4. Esecuzione dello Studio Optuna ---
print(f"Avvio studio Optuna per {N_TRIALS} tentativi...")
start_time = time.time()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=N_TRIALS)

end_time = time.time()
print(f"Ottimizzazione completata in {end_time - start_time:.2f} secondi.")

# --- 5. Salvataggio Parametri Migliori ---
best_params = study.best_params
best_score = study.best_value

print("\n--- Risultati Ottimizzazione ---")
print(f"Miglior Punteggio AUC (medio): {best_score:.6f}")
print("Parametri Migliori Trovati:")
print(json.dumps(best_params, indent=2))

# Aggiungiamo i parametri fissi
final_params_to_save = {
    **best_params, 
    'objective': 'Logloss',
    'eval_metric': 'AUC',
    'verbose': 0,
    'random_seed': 42
}

# Salviamo i parametri in un file JSON
with open(PARAMS_OUTPUT_FILE, 'w') as f:
    json.dump(final_params_to_save, f, indent=2)

print(f"\nParametri migliori salvati in: {PARAMS_OUTPUT_FILE}")
print("Ora puoi usare questo file per addestrare il modello finale.")