import pandas as pd
import numpy as np
import os
import json
import time

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

# --- 1. Configurazione ---
PREPROCESSED_DIR = 'preprocessed_data'
MODEL_PARAMS_DIR = 'model_params'
SUBMISSION_DIR = 'Submissions'
os.makedirs(SUBMISSION_DIR, exist_ok=True) # Cartella per i file di submission

TRAIN_FILE = os.path.join(PREPROCESSED_DIR, 'train_processed.csv')
TARGET_FILE = os.path.join(PREPROCESSED_DIR, 'target_train.csv')
TEST_FILE = os.path.join(PREPROCESSED_DIR, 'test_processed.csv')

# File dei parametri (ASSICURATI CHE I NOMI SIANO CORRETTI!)
PARAMS_FILE_100 = os.path.join(MODEL_PARAMS_DIR, 'best_catboost_params.json') # Da Optuna sul 100%
PARAMS_FILE_80 = os.path.join(MODEL_PARAMS_DIR, 'best_catboost_params_final.json') # Da Optuna sull'80% (validato)

# File di output
SUBMISSION_FILE_100 = os.path.join(SUBMISSION_DIR, 'submission_optuna100.csv')
SUBMISSION_FILE_80 = os.path.join(SUBMISSION_DIR, 'submission_optuna80_validated.csv')

# --- 2. Caricamento Dati ---
print("Caricamento dati processati (Train e Test)...")
try:
    X_train = pd.read_csv(TRAIN_FILE)
    y_train = pd.read_csv(TARGET_FILE).values.ravel()
    X_test = pd.read_csv(TEST_FILE)

    # Carichiamo anche gli ID originali del test set per la submission
    # Dobbiamo rileggerli dal file non processato
    test_ids_df = pd.read_csv(os.path.join('features_expert', 'features_expert_test.csv'))
    test_ids = test_ids_df['battle_id']

except FileNotFoundError as e:
    print(f"ERRORE: File non trovato. Dettagli: {e}")
    exit()

# --- 3. Caricamento Parametri ---
print("Caricamento set di parametri...")
try:
    with open(PARAMS_FILE_100, 'r') as f:
        params_100 = json.load(f)
    print(f"  Parametri da '{PARAMS_FILE_100}' caricati.")

    with open(PARAMS_FILE_80, 'r') as f:
        params_80 = json.load(f)
    print(f"  Parametri da '{PARAMS_FILE_80}' caricati.")
except FileNotFoundError as e:
    print(f"ERRORE: File parametri non trovato. Dettagli: {e}")
    exit()

# --- Funzione Helper per Addestrare e Predire ---
def train_and_predict(params, X_train, y_train, X_test, model_name):
    """Addestra un modello CatBoost e genera predizioni."""
    print(f"\n--- Addestramento Modello: {model_name} ---")
    
    # Decidiamo il numero di alberi (iterations / n_estimators)
    # Usiamo un numero fisso alto MA con early stopping basato su un piccolo split interno
    # O potremmo usare il bestIteration trovato nello script 16 (se salvato)
    # Per semplicità, usiamo n_estimators alto + early stopping
    
    fit_params = {
        **params,
        'n_estimators': 3000, # Numero alto
        'early_stopping_rounds': 50 # Lasciamo che si fermi da solo
    }
    # Rimuoviamo eventuali chiavi duplicate se Optuna le aveva salvate
    fit_params.pop('n_estimators', None)
    fit_params.pop('early_stopping_rounds', None)
    
    model = CatBoostClassifier(**fit_params)
    
    # Addestriamo sul 100% dei dati di training
    # Usiamo un piccolo eval_set interno solo per l'early stopping
    # (Non è una validazione vera, serve solo a fermare l'addestramento)
    X_train_small, X_eval_small, y_train_small, y_eval_small = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    start_time = time.time()
    model.fit(
        X_train_small, y_train_small,
        eval_set=(X_eval_small, y_eval_small),
        verbose=1000,
        plot=False
    )
    end_time = time.time()
    print(f"Addestramento completato in {end_time - start_time:.2f} secondi.")
    print(f"  Migliore iterazione trovata: {model.get_best_iteration()}")
    
    # Predizioni sul VERO test set
    print("Generazione predizioni sul test set...")
    predictions = model.predict(X_test).astype(int) # Assicura 0 o 1
    
    return predictions

# --- 4. Addestramento e Predizione ---

# Modello 1 (Parametri da Optuna sul 100%)
preds_100 = train_and_predict(params_100, X_train, y_train, X_test, "Optuna 100%")

# Modello 2 (Parametri da Optuna sull'80%, validati)
preds_80 = train_and_predict(params_80, X_train, y_train, X_test, "Optuna 80% (Validated)")

# --- 5. Creazione File di Submission ---

# Submission 1
print(f"\nSalvataggio submission 1 in: {SUBMISSION_FILE_100}...")
submission_100 = pd.DataFrame({'battle_id': test_ids, 'player_won': preds_100})
submission_100.to_csv(SUBMISSION_FILE_100, index=False)

# Submission 2
print(f"Salvataggio submission 2 in: {SUBMISSION_FILE_80}...")
submission_80 = pd.DataFrame({'battle_id': test_ids, 'player_won': preds_80})
submission_80.to_csv(SUBMISSION_FILE_80, index=False)

print("\nProcesso completato. Due file di submission sono stati creati.")