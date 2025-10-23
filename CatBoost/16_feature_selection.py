"""
Questo script esegue la selezione delle feature utilizzando L1 (Lasso).
Si posiziona dopo il preprocessing (es. 14, 15) e prima dello splitter (17).

1. Carica 'train_processed.csv' e 'test_processed.csv' da 'Preprocessed_Data'.
2. Addestra un selettore LogisticRegression (Lasso) sull'intero set di training.
3. Trasforma sia i dati di training che quelli di test.
4. Salva 'train_processed_selected.csv' e 'test_processed_selected.csv' in 'Preprocessed_Data'.
"""

import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

print("Avvio Script 16: Selezione Feature (Lasso L1)...") # Messaggio corretto

# --- 1. Definisci le directory e i file di Input ---
# CORRETTO: Questa è la cartella che usi negli altri script
PREPROCESSED_DIR_IN = 'Preprocessed_Data' 
TRAIN_FILE_IN = os.path.join(PREPROCESSED_DIR_IN, 'train_processed.csv')
TARGET_FILE_IN = os.path.join(PREPROCESSED_DIR_IN, 'target_train.csv')
TEST_FILE_IN = os.path.join(PREPROCESSED_DIR_IN, 'test_processed.csv') 

# --- 2. Definisci i file di Output ---
# CORRETTO: Salviamo nella stessa cartella
PREPROCESSED_DIR_OUT = 'Preprocessed_Data' 
TRAIN_FILE_OUT = os.path.join(PREPROCESSED_DIR_OUT, 'train_processed_selected.csv')
TEST_FILE_OUT = os.path.join(PREPROCESSED_DIR_OUT, 'test_processed_selected.csv')
SELECTED_FEATURES_FILE = os.path.join('Model_Params', 'selected_feature_names.txt')

# Assicurati che la cartella Model_Params esista
os.makedirs('Model_Params', exist_ok=True)

# --- 3. Carica i dati ---
try:
    print(f"Caricamento dati da {PREPROCESSED_DIR_IN}...")
    X_train_full = pd.read_csv(TRAIN_FILE_IN)
    y_train_full = pd.read_csv(TARGET_FILE_IN).values.ravel()
    X_test_kaggle = pd.read_csv(TEST_FILE_IN)
    
    # Conserviamo i nomi delle colonne originali PRIMA di convertirli in numpy
    original_feature_names = X_train_full.columns
    
    # Convertiamo in array NumPy per evitare i warning di scikit-learn
    X_train_full_np = X_train_full.values
    X_test_kaggle_np = X_test_kaggle.values
    
    print(f"Dati caricati: X_train_full {X_train_full.shape}, X_test_kaggle {X_test_kaggle.shape}")

except FileNotFoundError as e:
    print(f"ERRORE: File processati non trovati in '{PREPROCESSED_DIR_IN}'.") # Messaggio di errore più chiaro
    print(e)
    print("Esegui prima lo script di preprocessing (es. 14_... o 15_...).")
    exit()

# --- 4. Addestra il Selettore Lasso (L1) ---
print("Avvio addestramento selettore Lasso (L1)...")

lasso_selector_model = LogisticRegression(
    penalty='l1',
    C=0.1,          
    solver='saga',  
    random_state=42,
    max_iter=5000,  # <-- MODIFICA: Aumentato per garantire la convergenza
    n_jobs=-1       
)

# Addestra sui valori NumPy
lasso_selector_model.fit(X_train_full_np, y_train_full)
print("Addestramento selettore completato.")

# --- 5. Applica la Selezione ---
print("Creazione del trasformatore SelectFromModel...")
selector = SelectFromModel(lasso_selector_model, prefit=True, threshold=1e-5)

# Trasforma i valori NumPy
X_train_selected_np = selector.transform(X_train_full_np)
X_test_selected_np = selector.transform(X_test_kaggle_np) 

original_count = X_train_full.shape[1]
selected_count = X_train_selected_np.shape[1]

print(f"Selezione completata.")
print(f"Feature originali: {original_count}")
print(f"Feature selezionate: {selected_count}")

# --- 6. Salva i nuovi file .csv ---

# Questa logica non cambia e funziona ancora
selected_mask = selector.get_support()
selected_features = original_feature_names[selected_mask]

# Ricrea i DataFrame con i nomi delle colonne selezionate
X_train_selected_df = pd.DataFrame(X_train_selected_np, columns=selected_features)
X_test_selected_df = pd.DataFrame(X_test_selected_np, columns=selected_features)

X_train_selected_df.to_csv(TRAIN_FILE_OUT, index=False)
X_test_selected_df.to_csv(TEST_FILE_OUT, index=False)

print(f"\nDati selezionati salvati:")
print(f"Training data -> {TRAIN_FILE_OUT} (Shape: {X_train_selected_df.shape})")
print(f"Test data -> {TEST_FILE_OUT} (Shape: {X_test_selected_df.shape})")

pd.Series(selected_features).to_csv(SELECTED_FEATURES_FILE, index=False, header=False)
print(f"Nomi feature salvati in -> {SELECTED_FEATURES_FILE}")

print("\nScript 16_feature_selection.py completato.")

