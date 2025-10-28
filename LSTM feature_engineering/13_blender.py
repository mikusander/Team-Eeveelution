"""
Script per fare il 'blending' (media) delle predizioni (probabilità)
provenienti da modelli diversi (es. LSTM e CatBoost).
"""

import pandas as pd
import os

print("Avvio Blending...")

# --- 1. Configurazione ---
SUBMISSION_DIR = 'Submissions'
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# File di input (assicurati che i nomi siano corretti!)
# (Il file LSTM sarà generato dallo script 11 modificato)
LSTM_PROBA_FILE = os.path.join(SUBMISSION_DIR, 'submission_lstm_100pct_PROBA.csv')
# (Questo è il file che devi generare tu con il tuo modello CatBoost)
CATBOOST_PROBA_FILE = os.path.join(SUBMISSION_DIR, 'submission_catboost_100pct_PROBA.csv')

# File di output
BLENDED_SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, 'submission_blended_50_50.csv')

# Pesi per il blending (iniziamo con 50/50)
LSTM_WEIGHT = 0.5
CATBOOST_WEIGHT = 0.5

# --- 2. Caricamento ---
try:
    print(f"Caricamento LSTM proba da: {LSTM_PROBA_FILE}")
    lstm_df = pd.read_csv(LSTM_PROBA_FILE)
    print(f"Caricamento CatBoost proba da: {CATBOOST_PROBA_FILE}")
    catboost_df = pd.read_csv(CATBOOST_PROBA_FILE)
except FileNotFoundError as e:
    print(f"❌ ERRORE: File non trovato: {e}")
    print("Assicurati di aver generato entrambi i file .csv con le probabilità prima di eseguire questo script.")
    exit()
except Exception as e:
    print(f"❌ ERRORE durante il caricamento: {e}")
    exit()

# --- 3. Unione e Blending ---
print("Unione dei DataFrame per battle_id...")
try:
    # Rinomina le colonne delle probabilità per chiarezza
    lstm_df = lstm_df.rename(columns={'player_won_proba': 'proba_lstm'})
    # Assicurati che il tuo file CatBoost abbia una colonna 'player_won_proba'
    catboost_df = catboost_df.rename(columns={'player_won_proba': 'proba_catboost'})

    # Unisci usando battle_id
    blended_df = pd.merge(lstm_df, catboost_df, on='battle_id')
    
    if len(blended_df) != len(lstm_df):
        print(f"⚠️ ATTENZIONE: Mismatch nelle lunghezze dopo il merge! ({len(blended_df)} vs {len(lstm_df)})")
        print("Controlla che i file di submission abbiano gli stessi battle_id.")

except KeyError as e:
    print(f"❌ ERRORE: Manca una colonna: {e}")
    print("Assicurati che entrambi i CSV contengano 'battle_id' e una colonna di probabilità ('player_won_proba').")
    exit()
except Exception as e:
    print(f"❌ ERRORE durante l'unione: {e}")
    exit()

print("Calcolo media pesata...")
# Calcola la probabilità finale
blended_df['final_proba'] = (LSTM_WEIGHT * blended_df['proba_lstm']) + \
                            (CATBOOST_WEIGHT * blended_df['proba_catboost'])

# Converti la probabilità finale in 0 o 1
blended_df['player_won'] = (blended_df['final_proba'] > 0.5).astype(int)

# --- 4. Salvataggio Submission Finale ---
final_submission_df = blended_df[['battle_id', 'player_won']]

try:
    final_submission_df.to_csv(BLENDED_SUBMISSION_FILE, index=False)
    print("\n--- BLENDING COMPLETATO ---")
    print(f"Submission finale salvata in: {BLENDED_SUBMISSION_FILE}")
    print(final_submission_df.head())
except Exception as e:
    print(f"❌ ERRORE durante il salvataggio: {e}")