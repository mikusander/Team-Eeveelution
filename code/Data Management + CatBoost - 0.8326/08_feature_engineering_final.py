import pandas as pd
import numpy as np
import os

# --- 1. Configurazione dei Percorsi ---
CSV_DIR = 'Output_CSVs'       # Per le timeline originali
FEATURES_V3_DIR = 'Features_v3' # Il nostro ultimo set di features
FEATURES_FINAL_DIR = 'Features_final' # L'output definitivo
os.makedirs(FEATURES_FINAL_DIR, exist_ok=True)

print(f"I file di features finali (FINAL) verranno salvati in: {FEATURES_FINAL_DIR}")

# Elenco degli stati che consideriamo "negativi"
# Escludiamo 'fnt' (faint) perché lo contiamo già.
NEGATIVE_STATUSES = ['par', 'slp', 'frz', 'psn', 'brn']

# --- 2. Funzione per calcolare le Features di Stato ---
def create_status_features(timeline_df):
    """
    Calcola il numero totale di turni passati con uno stato negativo.
    """
    print("Inizio calcolo features sugli stati alterati...")

    # Creiamo flag booleane (vere/false) per ogni turno
    # .isin() controlla se lo stato è nella nostra lista di stati negativi
    timeline_df['p1_has_status_turn'] = timeline_df['p1_pokemon_state.status'].isin(NEGATIVE_STATUSES)
    timeline_df['p2_has_status_turn'] = timeline_df['p2_pokemon_state.status'].isin(NEGATIVE_STATUSES)

    # Aggreghiamo per battle_id
    aggregations = {
        'p1_has_status_turn': 'sum', # Sommiamo i turni (True=1, False=0)
        'p2_has_status_turn': 'sum'
    }
    
    status_features_df = timeline_df.groupby('battle_id').agg(aggregations)
    
    # Rinominiamo le colonne
    status_features_df = status_features_df.rename(columns={
        'p1_has_status_turn': 'p1_total_status_turns',
        'p2_has_status_turn': 'p2_total_status_turns'
    })
    
    # Creiamo la feature "delta" (la più importante)
    # Un valore negativo è buono per P1 (P1 ha passato meno turni con stati negativi)
    status_features_df['status_turns_delta'] = (
        status_features_df['p1_total_status_turns'] - status_features_df['p2_total_status_turns']
    )
    
    print("Calcolo features di stato completato.")
    return status_features_df

# --- 3. Esecuzione del Processo ---

# --- TRAIN ---
print("\n--- Processo di TRAIN ---")
print("Caricamento timeline di TRAIN...")
timeline_train = pd.read_csv(os.path.join(CSV_DIR, 'timelines_train_dynamic.csv'))
status_features_train = create_status_features(timeline_train)

print("Caricamento features_v3 di TRAIN...")
features_train_v3 = pd.read_csv(os.path.join(FEATURES_V3_DIR, 'features_train_v3.csv'))
features_train_final = pd.merge(features_train_v3, status_features_train, on='battle_id', how='left')
features_train_final.to_csv(os.path.join(FEATURES_FINAL_DIR, 'features_final_train.csv'), index=False)
print(f"File finale di TRAIN salvato con {features_train_final.shape[1]} colonne.")

# --- TEST ---
print("\n--- Processo di TEST ---")
print("Caricamento timeline di TEST...")
timeline_test = pd.read_csv(os.path.join(CSV_DIR, 'timelines_test_dynamic.csv'))
status_features_test = create_status_features(timeline_test)

print("Caricamento features_v3 di TEST...")
features_test_v3 = pd.read_csv(os.path.join(FEATURES_V3_DIR, 'features_test_v3.csv'))
features_test_final = pd.merge(features_test_v3, status_features_test, on='battle_id', how='left')
features_test_final.to_csv(os.path.join(FEATURES_FINAL_DIR, 'features_final_test.csv'), index=False)
print(f"File finale di TEST salvato con {features_test_final.shape[1]} colonne.")

print("\nProcesso di feature engineering completato.")
print("Abbiamo i nostri dataset finali: features_final_train.csv e features_final_test.csv")