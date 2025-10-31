"""
(SCRIPT AGGIORNATO - Metodo 7: RFECV + CatBoost 'SERIO')
Questo script esegue la selezione delle feature utilizzando RFECV
per trovare il NUMERO OTTIMALE di feature in automatico, 
utilizzando una cross-validation ad ogni step.
"""

import pandas as pd
import os
from catboost import CatBoostClassifier
# <--- MODIFICA: Importiamo RFECV invece di RFE
from sklearn.feature_selection import RFECV
# <--- MODIFICA: Importiamo KFold per la cross-validation
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt # <--- MODIFICA: Per il grafico
import numpy as np

# <--- MODIFICA: Corretto il numero dello script (era 16 nel testo)
print("Avvio Script 17: Selezione Feature (RFECV + CatBoost 'SERIO')...")

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

# <--- MODIFICA: Aggiungiamo un path per il grafico
ANALYSIS_DIR = 'Model_Analysis_Validation' 
RFECV_PLOT_FILE = os.path.join(ANALYSIS_DIR, 'rfecv_performance_curve.png')

os.makedirs('Model_Params', exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True) # <--- MODIFICA: Creiamo la cartella

# --- 3. Carica i dati ---
try:
    print(f"Caricamento dati da {PREPROCESSED_DIR_IN}...")
    X_train_full = pd.read_csv(TRAIN_FILE_IN)
    y_train_full = pd.read_csv(TARGET_FILE_IN).values.ravel()
    X_test_kaggle = pd.read_csv(TEST_FILE_IN)

    original_feature_names = X_train_full.columns

    print(f"Dati caricati: X_train_full {X_train_full.shape}, X_test_kaggle {X_test_kaggle.shape}")

except FileNotFoundError as e:
    print(f"ERRORE: File processati non trovati in '{PREPROCESSED_DIR_IN}'.")
    print(e)
    print("Esegui prima lo script di preprocessing.")
    exit()

# --- 4. Addestra il Selettore RFECV + CatBoost 'SERIO' ---
print("Avvio addestramento selettore RFECV con CatBoost...")

# 1. Definisci il CatBoost "SERIO" che RFECV userà
print("Stimatore: n_estimators=500, depth=7.")
estimator = CatBoostClassifier(
    n_estimators=500,
    depth=7,
    random_seed=42,
    verbose=0,
    thread_count=-1
)

# 2. Definisci la strategia di Cross-Validation
# Useremo 5 fold. RFECV addestrerà il modello 5 volte per ogni step!
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
print("Strategia CV: KFold a 5 split.")

# 3. Definisci il selettore RFECV
# <--- MODIFICA PRINCIPALE: da RFE a RFECV ---
print("Avvio di RFECV.fit()... Questo processo sarà LUNGO.")
selector = RFECV(
    estimator,
    cv=cv_strategy,           # <--- USA LA CROSS-VALIDATION
    scoring='roc_auc',        # <--- Metrica per decidere (molto meglio di 'accuracy')
    step=10,                  # <--- Rimuove 10 feature alla volta (più granulare di 0.1)
    min_features_to_select=50,# <--- Non scendere sotto le 50 feature
    n_jobs=-1,                # <--- USA TUTTI I CORE
    verbose=2                 # <--- Mostra il progresso
)
# Nota: N_FEATURES_DA_TENERE non serve più!

# 4. Addestra RFECV (Questo è il passaggio LENTO)
selector.fit(X_train_full, y_train_full)
print("Addestramento RFECV completato.")

# --- 5. Applica la Selezione ---
print("Applicazione trasformazione RFECV...")
X_train_selected = selector.transform(X_train_full)
X_test_selected = selector.transform(X_test_kaggle)

original_count = X_train_full.shape[1]
# <--- MODIFICA: Il numero ottimale è ora un *risultato*
optimal_count = selector.n_features_

print(f"Selezione completata.")
print(f"Feature originali: {original_count}")
print(f"NUMERO OTTIMALE DI FEATURE IDENTIFICATO: {optimal_count}")

# --- 6. Salva i nuovi file .csv ---
selected_mask = selector.get_support()
selected_features = original_feature_names[selected_mask]

X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features)
X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features)

X_train_selected_df.to_csv(TRAIN_FILE_OUT, index=False)
X_test_selected_df.to_csv(TEST_FILE_OUT, index=False)

print(f"\nDati selezionati salvati:")
print(f"Training data -> {TRAIN_FILE_OUT} (Shape: {X_train_selected_df.shape})")
print(f"Test data -> {TEST_FILE_OUT} (Shape: {X_test_selected_df.shape})")

pd.Series(selected_features).to_csv(SELECTED_FEATURES_FILE, index=False, header=False)
print(f"Nomi feature salvati in -> {SELECTED_FEATURES_FILE}")

# --- 7. Salva il grafico della performance ---
# <--- MODIFICA: Aggiunta FASE 7 per visualizzare i risultati
print("\nSalvataggio grafico performance RFECV...")
try:
    # 'cv_results_' è il dizionario che contiene i punteggi (disponibile in sklearn > 1.0)
    scores = selector.cv_results_['mean_test_score']
    n_features_tested = selector.cv_results_['n_features']
    
    plt.figure(figsize=(12, 7))
    plt.plot(n_features_tested, scores, marker='o')
    
    # Evidenzia il punto migliore
    best_score = np.max(scores)
    best_n_features = n_features_tested[np.argmax(scores)]
    
    plt.axvline(x=best_n_features, color='red', linestyle='--', 
                label=f'Ottimo: {best_n_features} features (AUC: {best_score:.4f})')
    
    plt.title('Performance RFECV vs. Numero di Feature')
    plt.xlabel('Numero di Feature Selezionate')
    plt.ylabel(f'Punteggio CV (roc_auc)')
    plt.legend()
    plt.grid(True)
    plt.gca().invert_xaxis() # Mostra da "tutte le feature" a "poche feature"
    plt.savefig(RFECV_PLOT_FILE)
    plt.close()
    
    print(f"Grafico performance RFECV salvato in: {RFECV_PLOT_FILE}")

except AttributeError:
    print("Attenzione: 'cv_results_' non trovato. Potrebbe essere una versione vecchia di sklearn.")
    print("Prova a usare 'selector.grid_scores_' e 'selector.n_features_' se il plot fallisce.")
except Exception as e:
    print(f"Errore durante la creazione del grafico: {e}")

print(f"\nScript 17_feature_selection.py completato.")