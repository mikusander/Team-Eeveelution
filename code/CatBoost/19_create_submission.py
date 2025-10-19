import pandas as pd
import numpy as np
import os
import json
from catboost import CatBoostClassifier

# --- 1. Configurazione ---
PREPROCESSED_DIR = 'preprocessed_data' 
MODEL_PARAMS_DIR = 'model_params'
SUBMISSION_DIR = 'Submissions'
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Input
TRAIN_FILE = os.path.join(PREPROCESSED_DIR, 'train_processed.csv')
TARGET_FILE = os.path.join(PREPROCESSED_DIR, 'target_train.csv')
TEST_FILE = os.path.join(PREPROCESSED_DIR, 'test_processed.csv') 
PARAMS_FILE = os.path.join(MODEL_PARAMS_DIR, 'best_catboost_params.json')
ITERATION_FILE = os.path.join(MODEL_PARAMS_DIR, 'best_iteration.json')

# Output
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, 'submission.csv') # Nome file aggiornato

print("Avvio Script di Creazione Submission Finale (Formato Accuracy 1/0)...")

# --- 2. Caricamento Dati e Parametri ---
print("Caricamento 100% Dati di Training e Dati di Test...")
try:
    X_train_full = pd.read_csv(TRAIN_FILE)
    y_train_full = pd.read_csv(TARGET_FILE).values.ravel()
    X_test_kaggle = pd.read_csv(TEST_FILE)
    
    with open(PARAMS_FILE, 'r') as f:
        best_params = json.load(f)
    print(f"Caricati parametri da {PARAMS_FILE}.")
    
    with open(ITERATION_FILE, 'r') as f:
        best_iteration = json.load(f)['best_iteration']
    print(f"Caricato numero iterazioni ottimale: {best_iteration}")

except FileNotFoundError:
    print("ERRORE: File non trovati.")
    print("Assicurati di avere 'train_processed.csv', 'test_processed.csv',")
    print("'best_catboost_params.json' e 'best_iteration.json'.")
    exit()

# --- 3. Addestramento Modello Finale (su 100% Dati) ---
print(f"\nAddestramento modello finale su 100% dei dati ({len(X_train_full)} campioni)...")

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
print("Addestramento finale completato.")

# --- 4. Creazione Predizioni sul Test Set di Kaggle (MODIFICATO) ---
print(f"Creazione predizioni (1/0) per il test set ({len(X_test_kaggle)} campioni)...")

# Usiamo .predict() che restituisce True/False (dato che abbiamo trainato su True/False)
y_pred_bool = final_model.predict(X_test_kaggle)

# Convertiamo True/False in 1/0 come richiesto da Kaggle
y_pred_int = y_pred_bool.astype(int)

# Recuperiamo i battle_id per la submission
# (Dobbiamo caricare il file originale non processato, o il processato se lo abbiamo salvato)
# Lo script 12_preprocessing.py rimuoveva battle_id dal df processato,
# quindi ricarichiamo il test_processed.csv per sicurezza e prendiamo solo l'id.
# NOTA: Assicurati che 'test_processed.csv' contenga ancora 'battle_id'
# Se non lo contiene, dobbiamo caricare da 'features_expert/features_expert_test.csv'
try:
    test_ids = pd.read_csv(os.path.join(PREPROCESSED_DIR, 'test_processed.csv'))['battle_id']
except KeyError:
    print("Avviso: 'battle_id' non trovato in 'test_processed.csv'.")
    print("Caricamento da 'features_expert/features_expert_test.csv'...")
    test_ids = pd.read_csv(os.path.join('features_expert', 'features_expert_test.csv'))['battle_id']


# --- 5. Salvataggio File di Submission (MODIFICATO) ---
print(f"Salvataggio file di submission in {SUBMISSION_FILE}...")

# Creiamo il DataFrame finale nel formato corretto
submission_df = pd.DataFrame({
    'battle_id': test_ids,
    'player_won': y_pred_int  # Usiamo la colonna di 0 e 1
})

submission_df.to_csv(SUBMISSION_FILE, index=False)

print("\n--- SUBMISSION PRONTA ---")
print(f"File salvato in: {SUBMISSION_FILE}")
print("Formato: battle_id, player_won (con 1/0)")
print(submission_df.head())