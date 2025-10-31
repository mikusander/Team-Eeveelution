# Nome file: 12b_blender_3models.py
"""
Script per fare il 'blending' (media) delle predizioni (probabilità)
provenienti da tre modelli diversi: LSTM, CatBoost e XGBoost.
"""

import pandas as pd
import os
import numpy as np

print("Avvio Blending (3 Modelli)...")

# --- 1. Configurazione ---
SUBMISSION_DIR = 'Submissions'
# NUOVO: Cartella OOF/Test Preds per .npy (sebbene useremo i CSV da Submissions)
OOF_PREDS_DIR = 'OOF_Predictions'
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(OOF_PREDS_DIR, exist_ok=True)


# File di input (assicurati che i nomi siano corretti!)
LSTM_PROBA_FILE = os.path.join(SUBMISSION_DIR, 'submission_lstm_100pct_PROBA.csv')
CATBOOST_PROBA_FILE = os.path.join(SUBMISSION_DIR, 'submission_catboost_100pct_PROBA.csv')

# NUOVO: Aggiungi il file di probabilità di XGBoost
# Assicurati di aver creato questo file (eseguendo 16_create_xgboost_stack_files.py
# e poi salvando le test_preds_xgboost_proba.npy come CSV)
# PER SEMPLICITA', assumiamo che tu abbia creato un file CSV come gli altri:
XGBOOST_PROBA_FILE = os.path.join(SUBMISSION_DIR, 'submission_xgboost_100pct_PROBA.csv')
# Se hai solo il file .npy da 16_create_xgboost_stack_files.py, esegui prima uno script
# per convertirlo in CSV con 'battle_id' e 'player_won_proba'.

# File di output
BLENDED_SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, 'submission_blended_3models.csv')

# Pesi per il blending (equilibrato)
LSTM_WEIGHT = 1/3
CATBOOST_WEIGHT = 1/3
XGBOOST_WEIGHT = 1/3

# --- 2. Caricamento ---
try:
    print(f"Caricamento LSTM proba da: {LSTM_PROBA_FILE}")
    lstm_df = pd.read_csv(LSTM_PROBA_FILE)
    print(f"Caricamento CatBoost proba da: {CATBOOST_PROBA_FILE}")
    catboost_df = pd.read_csv(CATBOOST_PROBA_FILE)
    print(f"Caricamento XGBoost proba da: {XGBOOST_PROBA_FILE}") # NUOVO
    xgboost_df = pd.read_csv(XGBOOST_PROBA_FILE) # NUOVO
except FileNotFoundError as e:
    print(f"❌ ERRORE: File non trovato: {e}")
    print("Assicurati di aver generato TUTTI E TRE i file .csv con le probabilità.")
    print(f"(Potrebbe mancare '{XGBOOST_PROBA_FILE}')")
    exit()
except Exception as e:
    print(f"❌ ERRORE durante il caricamento: {e}")
    exit()

# --- 3. Unione e Blending ---
print("Unione dei DataFrame per battle_id...")
try:
    # Rinomina le colonne delle probabilità per chiarezza
    lstm_df = lstm_df.rename(columns={'player_won_proba': 'proba_lstm'})
    catboost_df = catboost_df.rename(columns={'player_won_proba': 'proba_catboost'})
    xgboost_df = xgboost_df.rename(columns={'player_won_proba': 'proba_xgboost'}) # NUOVO

    # Unisci usando battle_id
    blended_df = pd.merge(lstm_df[['battle_id', 'proba_lstm']],
                          catboost_df[['battle_id', 'proba_catboost']],
                          on='battle_id')
    blended_df = pd.merge(blended_df,
                          xgboost_df[['battle_id', 'proba_xgboost']],
                          on='battle_id') # NUOVO
    
    if len(blended_df) != len(lstm_df):
        print(f"⚠️ ATTENZIONE: Mismatch nelle lunghezze dopo il merge! ({len(blended_df)} vs {len(lstm_df)})")

except KeyError as e:
    print(f"❌ ERRORE: Manca una colonna: {e}")
    print("Assicurati che tutti i CSV contengano 'battle_id' e una colonna di probabilità ('player_won_proba').")
    exit()
except Exception as e:
    print(f"❌ ERRORE durante l'unione: {e}")
    exit()

print("Calcolo media pesata (3 modelli)...")
# Calcola la probabilità finale
blended_df['final_proba'] = (LSTM_WEIGHT * blended_df['proba_lstm']) + \
                            (CATBOOST_WEIGHT * blended_df['proba_catboost']) + \
                            (XGBOOST_WEIGHT * blended_df['proba_xgboost']) # NUOVO

# Converti la probabilità finale in 0 o 1
blended_df['player_won'] = (blended_df['final_proba'] > 0.5).astype(int)

# --- 4. Salvataggio Submission Finale ---
final_submission_df = blended_df[['battle_id', 'player_won']]

try:
    final_submission_df.to_csv(BLENDED_SUBMISSION_FILE, index=False)
    print("\n--- BLENDING (3 MODELLI) COMPLETATO ---")
    print(f"Submission finale salvata in: {BLENDED_SUBMISSION_FILE}")
    print(final_submission_df.head())
except Exception as e:
    print(f"❌ ERRORE durante il salvataggio: {e}")