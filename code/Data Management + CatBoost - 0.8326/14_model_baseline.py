import pandas as pd
import numpy as np
import os
import time

# Importiamo i modelli
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Importiamo lo strumento di validazione
from sklearn.model_selection import cross_val_score, KFold

# --- 1. Configurazione ---
PREPROCESSED_DIR = 'Preprocessed_Data'
TRAIN_FILE = os.path.join(PREPROCESSED_DIR, 'train_processed.csv')
TARGET_FILE = os.path.join(PREPROCESSED_DIR, 'target_train.csv')

print("Avvio del torneo dei modelli baseline...")

# --- 2. Caricamento Dati Processati ---
print(f"Caricamento dati da {PREPROCESSED_DIR}...")
try:
    X_train = pd.read_csv(TRAIN_FILE)
    # .values.ravel() trasforma il DataFrame (10000, 1) in un array (10000,)
    # che è il formato che i modelli si aspettano per 'y'
    y_train = pd.read_csv(TARGET_FILE).values.ravel() 
except FileNotFoundError:
    print(f"ERRORE: File non trovati in '{PREPROCESSED_DIR}'.")
    print("Assicurati di aver eseguito prima '12_preprocessing.py'.")
    exit()

print("Dati 100% numerici caricati. (X_train, y_train)")

# --- 3. Definizione dei Concorrenti ---

# Per CatBoost, 'verbose=0' è fondamentale per non inondare
# la console di log durante la cross-validation
models = {
    "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(random_state=42, n_jobs=-1, verbosity=-1),
    "CatBoost": CatBoostClassifier(random_state=42, verbose=0)
}

# --- 4. Avvio della Gara (Cross-Validation) ---

# Definiamo la strategia di cross-validation (5 "fold")
# 'shuffle=True' mischia i dati prima di dividerli, è una best practice
kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = []

print("\nInizio gara... Valutazione 4 modelli con 5-fold cross-validation.")

for model_name, model in models.items():
    print(f"  Addestramento {model_name}...")
    start_time = time.time()
    
    # Calcoliamo l'Accuracy
    acc_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy', n_jobs=-1)
    
    # Calcoliamo l'AUC-ROC (metrica migliore per classificazione binaria)
    auc_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='roc_auc', n_jobs=-1)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Salviamo i risultati
    results.append({
        "Modello": model_name,
        "Accuracy Media": acc_scores.mean(),
        "AUC Medio": auc_scores.mean(),
        "Tempo (sec)": elapsed_time
    })

print("Gara completata.")

# --- 5. Pubblicazione dei Risultati ---

print("\n--- Risultati del Torneo (Baseline) ---")
results_df = pd.DataFrame(results)

# Ordiniamo i risultati per la metrica migliore (AUC)
results_df = results_df.sort_values(by="AUC Medio", ascending=False)

# Formattiamo i numeri per una migliore leggibilità
results_df["Accuracy Media"] = results_df["Accuracy Media"].map('{:.4f}'.format)
results_df["AUC Medio"] = results_df["AUC Medio"].map('{:.4f}'.format)
results_df["Tempo (sec)"] = results_df["Tempo (sec)"].map('{:.2f}s'.format)

print(results_df.to_string(index=False))

print("\nAnalisi: Il modello con l'AUC Medio più alto è il nostro vincitore 'baseline'.")