# Nome file: 14_compare_submissions.py
"""
Script per confrontare le predizioni binarie (0/1) di due file
di submission (generati dalle probabilità) e del loro blend,
per vedere quanto sono in disaccordo.
"""

import pandas as pd
import os
import numpy as np

print("Avvio Script di Confronto Submission (con Blend)...")

# --- 1. Configurazione ---
SUBMISSION_DIR = 'Submissions'

# File di input (quelli con le probabilità)
LSTM_PROBA_FILE = os.path.join(SUBMISSION_DIR, 'submission_lstm_100pct_PROBA.csv')
CATBOOST_PROBA_FILE = os.path.join(SUBMISSION_DIR, 'submission_catboost_100pct_PROBA.csv')

# File di input (quello blended finale 0/1)
# Assicurati che il nome corrisponda a quello generato da 12_blender.py
BLENDED_FINAL_FILE = os.path.join(SUBMISSION_DIR, 'submission_blended_50_50.csv')

# Soglia per convertire probabilità in 0/1
THRESHOLD = 0.5

# --- 2. Caricamento ---
try:
    print(f"Caricamento LSTM proba da: {LSTM_PROBA_FILE}")
    lstm_df = pd.read_csv(LSTM_PROBA_FILE)
    print(f"Caricamento CatBoost proba da: {CATBOOST_PROBA_FILE}")
    catboost_df = pd.read_csv(CATBOOST_PROBA_FILE)
    print(f"Caricamento Blended finale da: {BLENDED_FINAL_FILE}")
    blended_df = pd.read_csv(BLENDED_FINAL_FILE)
except FileNotFoundError as e:
    print(f"❌ ERRORE: File non trovato: {e}")
    print("Assicurati di aver generato TUTTI E TRE i file .csv prima di eseguire.")
    exit()
except Exception as e:
    print(f"❌ ERRORE durante il caricamento: {e}")
    exit()

# --- 3. Unione e Conversione ---
print("Unione dei DataFrame per battle_id...")
try:
    # Rinomina colonne probabilità e predizioni finali
    lstm_df = lstm_df.rename(columns={'player_won_proba': 'proba_lstm'})
    catboost_df = catboost_df.rename(columns={'player_won_proba': 'proba_catboost'})
    # La colonna nel file blended si chiama già 'player_won'
    blended_df = blended_df.rename(columns={'player_won': 'pred_blended'})

    # Unisci LSTM e CatBoost
    compare_df = pd.merge(lstm_df[['battle_id', 'proba_lstm']],
                          catboost_df[['battle_id', 'proba_catboost']],
                          on='battle_id')
    # Unisci il risultato con il Blended
    compare_df = pd.merge(compare_df,
                          blended_df[['battle_id', 'pred_blended']],
                          on='battle_id')


    if len(compare_df) != len(lstm_df):
        print(f"⚠️ ATTENZIONE: Mismatch nelle lunghezze dopo il merge!")

    # Converti probabilità in predizioni binarie
    compare_df['pred_lstm'] = (compare_df['proba_lstm'] > THRESHOLD).astype(int)
    compare_df['pred_catboost'] = (compare_df['proba_catboost'] > THRESHOLD).astype(int)

except KeyError as e:
    print(f"❌ ERRORE: Manca una colonna: {e}")
    print("Assicurati che i CSV contengano 'battle_id' e le colonne corrette ('player_won_proba' o 'player_won').")
    exit()
except Exception as e:
    print(f"❌ ERRORE durante l'unione o la conversione: {e}")
    exit()

# --- 4. Calcolo Disaccordo ---
print("\nCalcolo dei tassi di disaccordo...")

total_predictions = len(compare_df)

if total_predictions > 0:
    # Funzione helper per calcolare e stampare il disaccordo
    def calculate_disagreement(df, model1_pred_col, model2_pred_col, model1_name, model2_name):
        disagreements = df[df[model1_pred_col] != df[model2_pred_col]]
        num_disagreements = len(disagreements)
        disagreement_rate = (num_disagreements / total_predictions) * 100
        print(f"\n--- {model1_name} vs {model2_name} ---")
        print(f"  Numero di predizioni in disaccordo: {num_disagreements}")
        print(f"  Tasso di disaccordo: {disagreement_rate:.2f}%")
        return disagreements

    # Calcola i disaccordi
    dis_lstm_cat = calculate_disagreement(compare_df, 'pred_lstm', 'pred_catboost', 'LSTM', 'CatBoost')
    dis_lstm_blend = calculate_disagreement(compare_df, 'pred_lstm', 'pred_blended', 'LSTM', 'Blend')
    dis_cat_blend = calculate_disagreement(compare_df, 'pred_catboost', 'pred_blended', 'CatBoost', 'Blend')

    # Mostra alcuni esempi di disaccordo tra LSTM e CatBoost (come prima)
    if not dis_lstm_cat.empty:
        print("\nEsempi di disaccordo LSTM vs CatBoost (prime 5):")
        print(dis_lstm_cat[['battle_id', 'proba_lstm', 'pred_lstm', 'proba_catboost', 'pred_catboost', 'pred_blended']].head(5))

else:
    print("Nessuna predizione da confrontare.")

print("\n--- Confronto Completato ---")