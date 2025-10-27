# Nome file: 06_data_splitter.py
"""
Divide i dati sequenziali (da Preprocessed_LSTM) e opzionalmente statici
(da Preprocessed_LSTM_Hybrid) in split 60/20/20.
Salva tutti gli split in Preprocessed_LSTM_Splits.
"""

import numpy as np
import os
from sklearn.model_selection import train_test_split
import shutil # Non più necessario per la copia

print("Avvio Script 06: Divisione Dati LSTM 60/20/20...")

# --- Definisci le directory ---
BASE_LSTM_DIR = 'Preprocessed_LSTM'           # Directory per seq numeriche/categoriche e target
HYBRID_DIR = 'Preprocessed_LSTM_Hybrid'       # Directory per feature statiche
SPLIT_DIR_OUT = 'Preprocessed_LSTM_Splits'    # Directory di output per tutti gli split
os.makedirs(SPLIT_DIR_OUT, exist_ok=True)

# --- Controlla esistenza feature statiche ---
STATIC_FEATURES_PATH_CHECK = os.path.join(HYBRID_DIR, 'train_static_features.npy')
STATIC_FEATURES_EXIST = os.path.exists(STATIC_FEATURES_PATH_CHECK)

if STATIC_FEATURES_EXIST:
    print(f"Trovate feature statiche ibride in '{HYBRID_DIR}'.")
    # Percorso input per feature statiche
    TRAIN_STATIC_IN = os.path.join(HYBRID_DIR, 'train_static_features.npy')
else:
    print(f"Feature statiche ibride NON trovate. Splitterà solo i dati sequenziali.")
    TRAIN_STATIC_IN = None

print(f"Dati sequenziali letti da: '{BASE_LSTM_DIR}'")
print(f"Output split salvati in: '{SPLIT_DIR_OUT}'")

# --- Percorsi Input Base (sempre da BASE_LSTM_DIR) ---
TRAIN_NUMERIC_IN = os.path.join(BASE_LSTM_DIR, 'train_numeric_seq.npy')
TRAIN_CATEGORICAL_IN = os.path.join(BASE_LSTM_DIR, 'train_categorical_seq.npy')
TARGET_IN = os.path.join(BASE_LSTM_DIR, 'target.npy') # Legge target dalla base

# --- Carica i dati ---
try:
    print(f"Caricamento dati base da '{BASE_LSTM_DIR}'...")
    X_num = np.load(TRAIN_NUMERIC_IN)
    X_cat = np.load(TRAIN_CATEGORICAL_IN)
    y = np.load(TARGET_IN)

    # Carica statiche solo se esistono (dal percorso HYBRID)
    X_static = np.load(TRAIN_STATIC_IN) if STATIC_FEATURES_EXIST else None

except FileNotFoundError as e:
    print(f"❌ ERRORE: File .npy non trovati nelle directory corrette.")
    print(e)
    print(f"  Assicurati che '{BASE_LSTM_DIR}' contenga num, cat, target.")
    print(f"  E che '{HYBRID_DIR}' contenga static (se 05b è stato eseguito).")
    exit()
except Exception as e:
     print(f"❌ ERRORE durante il caricamento dei file .npy: {e}")
     exit()

print(f"Dati base caricati: {len(X_num)} campioni.")
if X_static is not None:
    print(f"Feature statiche caricate: {X_static.shape}")
    if X_static.shape[0] != len(y):
        print(f"❌ ERRORE: Mismatch campioni! Static ({X_static.shape[0]}) vs Target ({len(y)})")
        exit()

# --- Splitta gli indici ---
print("Divisione indici 60/20/20 (Train/Validation/Holdout)...")
indices = np.arange(len(y))
indices_temp, indices_holdout, y_temp, y_holdout = train_test_split(
    indices, y, test_size=0.20, random_state=42, stratify=y
)
indices_train, indices_val, y_train, y_val = train_test_split(
    indices_temp, y_temp, test_size=0.25, random_state=84, stratify=y_temp
)
print(f"  Training set (60%): {len(indices_train)} campioni")
print(f"  Validation set (20%): {len(indices_val)} campioni")
print(f"  Holdout set (20%): {len(indices_holdout)} campioni")

# --- Funzione helper per salvare (invariata) ---
def save_split(name, indices_split, y_split):
    print(f"Salvataggio split: {name} ({len(indices_split)} campioni)")
    np.save(os.path.join(SPLIT_DIR_OUT, f'{name}_num.npy'), X_num[indices_split])
    np.save(os.path.join(SPLIT_DIR_OUT, f'{name}_cat.npy'), X_cat[indices_split])
    np.save(os.path.join(SPLIT_DIR_OUT, f'{name}_y.npy'), y_split)
    if X_static is not None:
        np.save(os.path.join(SPLIT_DIR_OUT, f'{name}_static.npy'), X_static[indices_split])

# --- Salva tutti gli split ---
save_split('train_60', indices_train, y_train)
save_split('val_20', indices_val, y_val)
save_split('holdout_20', indices_holdout, y_holdout)

# --- Rimossa la logica di copia file ---

print(f"\nSalvataggio completato nella cartella: {SPLIT_DIR_OUT}")
print("\n06_data_splitter.py completato con successo.")