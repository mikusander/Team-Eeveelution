"""
(SCRIPT AGGIORNATO - Metodo 5: RFE + CatBoost)
Questo script esegue la selezione delle feature utilizzando RFE
(Recursive Feature Elimination) con uno stimatore CatBoost "leggero".

1. Carica 'train_processed.csv' e 'test_processed.csv'.
2. Definisce un CatBoost leggero come stimatore.
3. Addestra un selettore RFE per trovare le 75 feature migliori.
4. Trasforma i dati e salva 'train_processed_selected.csv' e 'test_processed_selected.csv'.
"""

import pandas as pd
import os
from catboost import CatBoostClassifier # <-- IMPORTIAMO CatBoost
from sklearn.feature_selection import RFE # <-- IMPORTIAMO RFE
# Non servono SelectFromModel o train_test_split

print("Avvio Script 16: Selezione Feature (RFE + CatBoost 'Leggero')...")
print("ATTENZIONE: Questo processo sarà MOLTO LENTO (potrebbe richiedere 30-60+ minuti).")

# --- 1. Definisci le directory e i file di Input ---
PREPROCESSED_DIR_IN = 'Preprocessed_Data' 
TRAIN_FILE_IN = os.path.join(PREPROCESSED_DIR_IN, 'train_processed.csv')
TARGET_FILE_IN = os.path.join(PREPROCESSED_DIR_IN, 'target_train.csv')
TEST_FILE_IN = os.path.join(PREPROCESSED_DIR_IN, 'test_processed.csv') 

# --- 2. Definisci i file di Output ---
PREPROCESSED_DIR_OUT = 'Preprocessed_Data' 
TRAIN_FILE_OUT = os.path.join(PREPROCESSED_DIR_OUT, 'train_processed_selected.csv')
TEST_FILE_OUT = os.path.join(PREPROCESSED_DIR_OUT, 'test_processed_selected.csv')
SELECTED_FEATURES_FILE = os.path.join('Model_Params', 'selected_feature_names.txt')

os.makedirs('Model_Params', exist_ok=True)

# --- 3. Carica i dati ---
try:
    print(f"Caricamento dati da {PREPROCESSED_DIR_IN}...")
    X_train_full = pd.read_csv(TRAIN_FILE_IN)
    y_train_full = pd.read_csv(TARGET_FILE_IN).values.ravel()
    X_test_kaggle = pd.read_csv(TEST_FILE_IN)
    
    original_feature_names = X_train_full.columns
    
    # RFE con CatBoost gestisce bene i DataFrame
    
    print(f"Dati caricati: X_train_full {X_train_full.shape}, X_test_kaggle {X_test_kaggle.shape}")

except FileNotFoundError as e:
    print(f"ERRORE: File processati non trovati in '{PREPROCESSED_DIR_IN}'.")
    print(e)
    print("Esegui prima lo script di preprocessing (es. 14_... o 15_...).")
    exit()

# --- 4. Addestra il Selettore RFE + CatBoost ---
print("Avvio addestramento selettore RFE con CatBoost...")

# 1. Definisci il CatBoost "leggero" che RFE userà ad ogni iterazione
estimator = CatBoostClassifier(
    n_estimators=100, # Basso per velocità (verrà eseguito molte volte)
    depth=5,          # Basso per velocità
    random_seed=42,
    verbose=0,
    thread_count=-1
)

# 2. Definisci il selettore RFE
N_FEATURES_DA_TENERE = 75 # <-- Il nostro obiettivo! Vicino al 77 di Lasso
selector = RFE(
    estimator, 
    n_features_to_select=N_FEATURES_DA_TENERE, 
    step=0.1,  # Rimuove il 10% delle feature peggiori a ogni iterazione
    verbose=1  # Mostra il progresso (es. "Fitting estimator... 315/349 features")
)

# 3. Addestra RFE (Questo è il passaggio LENTO)
selector.fit(X_train_full, y_train_full)
print("Addestramento RFE completato.")

# --- 5. Applica la Selezione ---
print("Applicazione trasformazione RFE...")
X_train_selected = selector.transform(X_train_full)
X_test_selected = selector.transform(X_test_kaggle) 

original_count = X_train_full.shape[1]
selected_count = X_train_selected.shape[1]

print(f"Selezione completata.")
print(f"Feature originali: {original_count}")
print(f"Feature selezionate: {selected_count} (Obiettivo: {N_FEATURES_DA_TENERE})")

# --- 6. Salva i nuovi file .csv ---
selected_mask = selector.get_support()
selected_features = original_feature_names[selected_mask]

# Ricrea i DataFrame con i dati selezionati e i nomi corretti
X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features)
X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features)

X_train_selected_df.to_csv(TRAIN_FILE_OUT, index=False)
X_test_selected_df.to_csv(TEST_FILE_OUT, index=False)

print(f"\nDati selezionati salvati:")
print(f"Training data -> {TRAIN_FILE_OUT} (Shape: {X_train_selected_df.shape})")
print(f"Test data -> {TEST_FILE_OUT} (Shape: {X_test_selected_df.shape})")

pd.Series(selected_features).to_csv(SELECTED_FEATURES_FILE, index=False, header=False)
print(f"Nomi feature salvati in -> {SELECTED_FEATURES_FILE}")

print("\nScript 16_feature_selection.py completato.")

