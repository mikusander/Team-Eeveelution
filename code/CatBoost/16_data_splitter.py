import pandas as pd
import os
from sklearn.model_selection import train_test_split

# --- 1. Configurazione ---
PREPROCESSED_DIR = 'preprocessed_data'
SPLIT_DIR = 'Preprocessed_Splits' # Nuova cartella per i dati divisi
os.makedirs(SPLIT_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(PREPROCESSED_DIR, 'train_processed.csv')
TARGET_FILE = os.path.join(PREPROCESSED_DIR, 'target_train.csv')

print("Avvio Script di divisione 60/20/20...")

# --- 2. Caricamento Dati Completi ---
try:
    X = pd.read_csv(TRAIN_FILE)
    y = pd.read_csv(TARGET_FILE) # Carichiamo come DataFrame per facilitare i salvataggi
except FileNotFoundError:
    print(f"ERRORE: File non trovati in {PREPROCESSED_DIR}.")
    exit()

print(f"Dati caricati: {len(X)} campioni.")

# --- 3. Split 60/20/20 ---
print("Divisione 60/20/20 (Train/Validation/Holdout)...")

# 1. Prima dividiamo 100% -> 80% (Train+Val) e 20% (Holdout)
X_temp, X_holdout, y_temp, y_holdout = train_test_split(
    X, y, 
    test_size=0.20, # 20% per l'Holdout
    random_state=42, 
    stratify=y
)

# 2. Ora dividiamo l'80% -> 75% (Train) e 25% (Validation)
#    (75% di 80% = 60% del totale; 25% di 80% = 20% del totale)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25, # 25% di 80% è 20%
    random_state=84, # Stato diverso per questo split
    stratify=y_temp
)

print(f"  Training set (60%): {len(X_train)} campioni")
print(f"  Validation set (20%): {len(X_val)} campioni")
print(f"  Holdout set (20%): {len(X_holdout)} campioni")

# --- 4. Salvataggio su Disco ---
print("Salvataggio degli split su disco...")

# Definiamo i percorsi
X_train_path = os.path.join(SPLIT_DIR, 'train_split_60_X.csv')
y_train_path = os.path.join(SPLIT_DIR, 'train_split_60_y.csv')
X_val_path = os.path.join(SPLIT_DIR, 'validation_split_20_X.csv')
y_val_path = os.path.join(SPLIT_DIR, 'validation_split_20_y.csv')
X_holdout_path = os.path.join(SPLIT_DIR, 'holdout_split_20_X.csv')
y_holdout_path = os.path.join(SPLIT_DIR, 'holdout_split_20_y.csv')

# Salvataggio
X_train.to_csv(X_train_path, index=False)
y_train.to_csv(y_train_path, index=False)
X_val.to_csv(X_val_path, index=False)
y_val.to_csv(y_val_path, index=False)
X_holdout.to_csv(X_holdout_path, index=False)
y_holdout.to_csv(y_holdout_path, index=False)

print(f"Salvataggio completato nella cartella: {SPLIT_DIR}")
print("Ora puoi eseguire '17_optimize_and_validate.py'.")