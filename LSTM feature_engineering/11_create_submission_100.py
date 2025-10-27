"""
Addestra il modello finale LSTM Avanzato sul 100% DEI DATI (Train+Val+Holdout)
usando gli iperparametri ottimali per epoche ottimali (lette da JSON) e crea la submission.
Senza MLflow.
(Versione SUPER-AVANZATA con 6 input categorici)
"""

import numpy as np
import pandas as pd
import os
import json
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

# --- Set Random Seeds ---
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
# ------------------------

# --- 1. Configurazione ---
LSTM_DATA_DIR = 'Preprocessed_LSTM'
SPLIT_DIR = 'Preprocessed_LSTM_Splits'
MODEL_PARAMS_DIR = 'Model_Params_LSTM_Advanced'
SUBMISSION_DIR = 'Submissions'
ENCODERS_FILE = os.path.join('Preprocessed_LSTM', 'encoders.json')
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# --- File di Input ---
# Dati di Test (per la predizione finale)
TEST_NUMERIC_IN = os.path.join(LSTM_DATA_DIR, 'test_numeric_seq.npy')
TEST_CATEGORICAL_IN = os.path.join(LSTM_DATA_DIR, 'test_categorical_seq.npy')

# Dati di Training (100%)
TRAIN_NUM_IN = os.path.join(SPLIT_DIR, 'train_60_num.npy')
TRAIN_CAT_IN = os.path.join(SPLIT_DIR, 'train_60_cat.npy')
TRAIN_Y_IN = os.path.join(SPLIT_DIR, 'train_60_y.npy')
VAL_NUM_IN = os.path.join(SPLIT_DIR, 'val_20_num.npy')
VAL_CAT_IN = os.path.join(SPLIT_DIR, 'val_20_cat.npy')
VAL_Y_IN = os.path.join(SPLIT_DIR, 'val_20_y.npy')
HOLD_NUM_IN = os.path.join(SPLIT_DIR, 'holdout_20_num.npy')
HOLD_CAT_IN = os.path.join(SPLIT_DIR, 'holdout_20_cat.npy')
HOLD_Y_IN = os.path.join(SPLIT_DIR, 'holdout_20_y.npy')

PARAMS_FILE = os.path.join(MODEL_PARAMS_DIR, 'best_lstm_params_advanced.json') # Contiene epoca ottimale

TIMELINE_TEST_IN = os.path.join('Output_CSVs', 'timelines_test_dynamic.csv') # Per battle_id
SUBMISSION_FILE_OUT = os.path.join(SUBMISSION_DIR, 'submission_lstm_advanced_100pct.csv')

MAX_TURNS = 30

# --- 2. Copia della funzione build_model (AGGIORNATA) ---
print("Caricamento encoders e definizione build_model (Super-Avanzata)...")
try:
    with open(ENCODERS_FILE, 'r') as f:
        encoders = json.load(f)
except Exception as e:
    print(f"❌ ERRORE: Impossibile caricare encoders.json: {e}")
    exit()

# Determina dimensioni input
try:
    _temp_train_num = np.load(TRAIN_NUM_IN)
    _temp_train_cat = np.load(TRAIN_CAT_IN)
    NUM_NUMERIC_FEATURES = _temp_train_num.shape[2]
    NUM_CATEGORICAL_FEATURES = _temp_train_cat.shape[2] # Dovrebbe essere 6
    del _temp_train_num, _temp_train_cat
    print(f"Rilevate {NUM_NUMERIC_FEATURES} feature numeriche, {NUM_CATEGORICAL_FEATURES} categoriche.")
    if NUM_CATEGORICAL_FEATURES != 6:
        print(f"⚠️ ATTENZIONE: Rilevate {NUM_CATEGORICAL_FEATURES} feat. categoriche, ma 6 attese. Verifica '05_...py'.")
except Exception as e:
     print(f"❌ ERRORE: Impossibile determinare dimensioni input dai file .npy: {e}")
     exit()

def build_model(hp_dict):
    """Costruisce il modello Keras dagli iperparametri salvati (Versione Super-Avanzata)."""
    
    # --- Ramo 1: Dati Numerici (Invariato) ---
    input_numeric = layers.Input(shape=(MAX_TURNS, NUM_NUMERIC_FEATURES), name='input_numeric')
    lstm_numeric_out = layers.LSTM(hp_dict['lstm_units_num'], return_sequences=False)(input_numeric)
    
    # --- Ramo 2: Dati Categorici (AGGIORNATO per 6 features) ---
    input_categorical = layers.Input(shape=(MAX_TURNS, NUM_CATEGORICAL_FEATURES), name='input_categorical', dtype='int32')
    
    # Estrai i 6 canali
    p1_name_in = layers.Lambda(lambda x: x[:, :, 0])(input_categorical)
    p2_name_in = layers.Lambda(lambda x: x[:, :, 1])(input_categorical)
    p1_move_in = layers.Lambda(lambda x: x[:, :, 2])(input_categorical)
    p2_move_in = layers.Lambda(lambda x: x[:, :, 3])(input_categorical)
    p1_type_in = layers.Lambda(lambda x: x[:, :, 4])(input_categorical) # NUOVO
    p2_type_in = layers.Lambda(lambda x: x[:, :, 5])(input_categorical) # NUOVO

    # Embedding (senza mask_zero)
    embed_p1_name = layers.Embedding(input_dim=encoders['p1_pokemon_state.name']['vocab_size'], output_dim=hp_dict['embed_dim_name'], mask_zero=False)(p1_name_in)
    embed_p2_name = layers.Embedding(input_dim=encoders['p2_pokemon_state.name']['vocab_size'], output_dim=hp_dict['embed_dim_name'], mask_zero=False)(p2_name_in)
    embed_p1_move = layers.Embedding(input_dim=encoders['p1_move_details.name']['vocab_size'], output_dim=hp_dict['embed_dim_move'], mask_zero=False)(p1_move_in)
    embed_p2_move = layers.Embedding(input_dim=encoders['p2_move_details.name']['vocab_size'], output_dim=hp_dict['embed_dim_move'], mask_zero=False)(p2_move_in)
    
    # NUOVI Embedding per Tipi Mosse
    embed_p1_type = layers.Embedding(input_dim=encoders['p1_move_details.type']['vocab_size'], output_dim=hp_dict.get('embed_dim_type', 8), mask_zero=False)(p1_type_in)
    embed_p2_type = layers.Embedding(input_dim=encoders['p2_move_details.type']['vocab_size'], output_dim=hp_dict.get('embed_dim_type', 8), mask_zero=False)(p2_type_in)

    # Concatena TUTTI (6) gli embedding
    concatenated_embeddings = layers.Concatenate()([
        embed_p1_name, embed_p2_name, 
        embed_p1_move, embed_p2_move,
        embed_p1_type, embed_p2_type # NUOVI
    ])
    
    lstm_categorical_out = layers.LSTM(hp_dict['lstm_units_cat'], return_sequences=False)(concatenated_embeddings)
    
    # --- Fine: Unione dei Rami ---
    concatenated_features = layers.Concatenate()([lstm_numeric_out, lstm_categorical_out])
    
    # --- Testa del Modello ---
    x = layers.Dense(hp_dict['dense_units'], activation='relu')(concatenated_features)
    x = layers.Dropout(hp_dict['dropout'])(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=[input_numeric, input_categorical], outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_dict['learning_rate']), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

# --- 3. Carica il 100% dei Dati di Training ---
print("Caricamento 100% dati di training (Train+Val+Holdout)...")
try:
    X_train_num_100 = np.concatenate([ np.load(TRAIN_NUM_IN), np.load(VAL_NUM_IN), np.load(HOLD_NUM_IN) ])
    X_train_cat_100 = np.concatenate([ np.load(TRAIN_CAT_IN).astype(np.int32), np.load(VAL_CAT_IN).astype(np.int32), np.load(HOLD_CAT_IN).astype(np.int32) ])
    y_train_100 = np.concatenate([ np.load(TRAIN_Y_IN), np.load(VAL_Y_IN), np.load(HOLD_Y_IN) ])
except Exception as e:
    print(f"❌ ERRORE: Impossibile caricare tutti i dati di training: {e}")
    exit()

X_train_100_inputs = { 'input_numeric': X_train_num_100, 'input_categorical': X_train_cat_100 }
print(f"Dati di training totali caricati: {len(y_train_100)} campioni.")

# --- 4. Costruisci e Addestra il Modello Finale ---
print(f"Caricamento iperparametri da {PARAMS_FILE}...")
try:
    # ✅ Carica parametri E l'epoca ottimale
    with open(PARAMS_FILE, 'r') as f:
        best_hps_dict = json.load(f)
    OPTIMAL_EPOCHS = best_hps_dict.get('optimal_epoch', None)
    if OPTIMAL_EPOCHS is None:
        raise ValueError("'optimal_epoch' non trovato nel file JSON dei parametri.")
    print(f"Parametri caricati. Epoca ottimale: {OPTIMAL_EPOCHS}")
except Exception as e:
    print(f"❌ ERRORE: Impossibile caricare il file dei parametri: {e}")
    print("Assicurati di aver eseguito '07_optimize_and_validate_ADVANCED.py' (versione senza MLflow).")
    exit()

print("Costruzione modello finale...")
final_model = build_model(best_hps_dict)

print(f"Addestramento modello finale su 100% dei dati per {OPTIMAL_EPOCHS} epoche...")
# NOTA: L'output qui mostrerà accuracy/loss sul training set, che potrebbe essere molto alto (overfitting atteso).
final_model.fit(
    X_train_100_inputs,
    y_train_100,
    epochs=OPTIMAL_EPOCHS, # ✅ Usa epoca da JSON
    batch_size=64,
    verbose=1
)
print("Addestramento finale completato.")

# --- 5. Carica Dati di Test e Genera Predizioni ---
print("Caricamento dati di Test Kaggle...")
try:
    X_test_num = np.load(TEST_NUMERIC_IN)
    X_test_cat = np.load(TEST_CATEGORICAL_IN).astype(np.int32) # Assicura int32
except Exception as e:
    print(f"❌ ERRORE: Impossibile caricare i dati di test Kaggle: {e}")
    exit()

X_test_inputs = { 'input_numeric': X_test_num, 'input_categorical': X_test_cat }

print(f"Generazione predizioni Kaggle per {len(X_test_num)} campioni...")
y_pred_proba = final_model.predict(X_test_inputs)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# --- 6. Crea il File di Submission ---
print("Caricamento battle_id per la submission...")
try:
    test_df = pd.read_csv(TIMELINE_TEST_IN)
    test_ids = test_df['battle_id'].unique()
except Exception as e:
    print(f"❌ ERRORE: Impossibile caricare battle_ids da {TIMELINE_TEST_IN}: {e}")
    exit()

if len(test_ids) != len(y_pred):
    print(f"❌ ERRORE: Mismatch tra ID ({len(test_ids)}) e Predizioni ({len(y_pred)})")
    exit()

submission_df = pd.DataFrame({ 'battle_id': test_ids, 'player_won': y_pred })
submission_df.to_csv(SUBMISSION_FILE_OUT, index=False)

print("\n--- SUBMISSION LSTM (100% TRAIN) PRONTA ---")
print(f"File salvato in: {SUBMISSION_FILE_OUT}")
print(submission_df.head())

# --- Rimossi log MLflow ---

print(f"\n--- Script 23 (LSTM Submission 100%, Senza MLflow) Completato ---")