import pandas as pd
import os
from sklearn.model_selection import train_test_split

PREPROCESSED_DIR = 'Preprocessed_Data'
SPLIT_DIR = 'Preprocessed_Splits' 
os.makedirs(SPLIT_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(PREPROCESSED_DIR, 'train_processed.csv')
TARGET_FILE = os.path.join(PREPROCESSED_DIR, 'target_train.csv')

print("Avvio Script di divisione 60/20/20...")

try:
    X = pd.read_csv(TRAIN_FILE)
    y = pd.read_csv(TARGET_FILE) 
except FileNotFoundError:
    print(f"ERRORE: File non trovati in {PREPROCESSED_DIR}.")
    exit()

print(f"Dati caricati: {len(X)} campioni.")

print("Divisione 60/20/20 (Train/Validation/Holdout)...")

X_temp, X_holdout, y_temp, y_holdout = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=42, 
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25, 
    random_state=84, 
    stratify=y_temp
)

print(f"  Training set (60%): {len(X_train)} campioni")
print(f"  Validation set (20%): {len(X_val)} campioni")
print(f"  Holdout set (20%): {len(X_holdout)} campioni")

print("Salvataggio degli split su disco...")

X_train_path = os.path.join(SPLIT_DIR, 'train_split_60_X.csv')
y_train_path = os.path.join(SPLIT_DIR, 'train_split_60_y.csv')
X_val_path = os.path.join(SPLIT_DIR, 'validation_split_20_X.csv')
y_val_path = os.path.join(SPLIT_DIR, 'validation_split_20_y.csv')
X_holdout_path = os.path.join(SPLIT_DIR, 'holdout_split_20_X.csv')
y_holdout_path = os.path.join(SPLIT_DIR, 'holdout_split_20_y.csv')

X_train.to_csv(X_train_path, index=False)
y_train.to_csv(y_train_path, index=False)
X_val.to_csv(X_val_path, index=False)
y_val.to_csv(y_val_path, index=False)
X_holdout.to_csv(X_holdout_path, index=False)
y_holdout.to_csv(y_holdout_path, index=False)

print(f"Salvataggio completato nella cartella: {SPLIT_DIR}")
print("\n15_data_splitter.py completato con successo.")