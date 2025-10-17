import pandas as pd
import numpy as np
import os

# --- 1. Configurazione ---
# Assicurati che questo sia il nome della cartella che hai usato
PREPROCESSED_DIR = 'Preprocessed_Data' 

TRAIN_FILE = os.path.join(PREPROCESSED_DIR, 'train_processed.csv')
TEST_FILE = os.path.join(PREPROCESSED_DIR, 'test_processed.csv')
TARGET_FILE = os.path.join(PREPROCESSED_DIR, 'target_train.csv')

print(f"Caricamento dati dalla cartella: {PREPROCESSED_DIR}...")
try:
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    target_df = pd.read_csv(TARGET_FILE)
except FileNotFoundError:
    print(f"ERRORE: File non trovati in '{PREPROCESSED_DIR}'.")
    print("Assicurati di aver eseguito prima '12_preprocessing.py'.")
    exit()

print("Dati caricati.")

# --- 1. Analisi Dimensioni ---
print("\n--- 1. Analisi Dimensioni ---")
print(f"Forma Train (X): {train_df.shape}")
print(f"Forma Test (X):  {test_df.shape}")
print(f"Forma Target (y): {target_df.shape}")

if train_df.shape[1] == test_df.shape[1]:
    print("  \033[92mOK:\033[0m N. colonne Train == N. colonne Test.")
else:
    print(f"  \033[91mERRORE:\033[0m N. colonne non combaciano! ({train_df.shape[1]} vs {test_df.shape[1]})")

if train_df.shape[0] == target_df.shape[0]:
    print(f"  \033[92mOK:\033[0m N. righe Train (10000) == N. righe Target.")
else:
    print(f"  \033[91mERRORE:\033[0m N. righe Train/Target non combaciano!")

# --- 2. Analisi Tipi Dati (Dtype) ---
print("\n--- 2. Analisi Tipi Dati (Dtype) ---")
print("Tipi di dati trovati in train_df:")
print(train_df.dtypes.value_counts())

if 'object' in train_df.dtypes.values:
    print("  \033[91mERRORE:\033[0m Ci sono ancora colonne 'object' (testo)! Il One-Hot Encoding ha fallito.")
else:
    print("  \033[92mOK:\033[0m Tutte le colonne sono numeriche (float64, int64, uint8).")

# --- 3. Analisi Valori Mancanti (NaN) ---
print("\n--- 3. Analisi Valori Mancanti (NaN) ---")
nan_count_train = train_df.isnull().sum().sum()
nan_count_test = test_df.isnull().sum().sum()
print(f"Valori NaN totali in Train (X): {nan_count_train}")
print(f"Valori NaN totali in Test (X):  {nan_count_test}")

if nan_count_train == 0 and nan_count_test == 0:
    print("  \033[92mOK:\033[0m Nessun valore mancante.")
else:
    print("  \033[91mATTENZIONE:\033[0m Trovati valori NaN! Questo potrebbe causare errori nel modello.")

# --- 4. Analisi Scaling (Esempio) ---
print("\n--- 4. Analisi Scaling (Esempio) ---")
# Verifichiamo una delle nostre feature numeriche originali
# (Quelle create da get_dummies avranno media e std diverse)
feature_da_testare = 'faint_delta' 

if feature_da_testare in train_df.columns:
    desc = train_df[feature_da_testare].describe()
    print(f"Statistiche per '{feature_da_testare}':")
    print(f"  Media: {desc['mean']:.2e}") # Formato scientifico (dovrebbe essere ~0)
    print(f"  Std:   {desc['std']:.2f}") # Deviazione standard (dovrebbe essere 1.0)
    
    # Controlliamo se la media è vicina a 0 e la std è vicina a 1
    if np.isclose(desc['mean'], 0, atol=1e-5) and np.isclose(desc['std'], 1, atol=1e-5):
        print("  \033[92mOK:\033[0m La feature sembra scalata correttamente (Media ~0, Std ~1).")
    else:
        print("  \033[91mATTENZIONE:\033[0m La feature NON sembra scalata.")
else:
    print(f"  Info: La colonna '{feature_da_testare}' non è stata trovata.")

print("\nVerifica rapida completata.")