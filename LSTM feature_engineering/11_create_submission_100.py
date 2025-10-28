"""
MODIFICATO (v5 - Ibrido):
Addestra il modello finale LSTM Avanzato (potenzialmente ibrido)
sul 100% DEI DATI (Train+Val+Holdout)
usando gli iperparametri ottimali per epoche ottimali (lette da JSON) e crea la submission.
(Versione SUPER-AVANZATA v2 con 8 input categorici + statici opzionali)
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
HYBRID_DATA_DIR = 'Preprocessed_LSTM_Hybrid' # NUOVO
SPLIT_DIR = 'Preprocessed_LSTM_Splits'
MODEL_PARAMS_DIR = 'Model_Params_LSTM_Advanced'
SUBMISSION_DIR = 'Submissions'
ENCODERS_FILE = os.path.join('Preprocessed_LSTM', 'encoders.json')
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# --- File di Input ---
# Dati di Test (per la predizione finale)
TEST_NUMERIC_IN = os.path.join(LSTM_DATA_DIR, 'test_numeric_seq.npy')
TEST_CATEGORICAL_IN = os.path.join(LSTM_DATA_DIR, 'test_categorical_seq.npy')
TEST_STATIC_IN = os.path.join(HYBRID_DATA_DIR, 'test_static_features.npy') # NUOVO

# Dati di Training (100%)
TRAIN_NUM_IN = os.path.join(SPLIT_DIR, 'train_60_num.npy')
TRAIN_CAT_IN = os.path.join(SPLIT_DIR, 'train_60_cat.npy')
TRAIN_STATIC_IN = os.path.join(SPLIT_DIR, 'train_60_static.npy') # NUOVO
TRAIN_Y_IN = os.path.join(SPLIT_DIR, 'train_60_y.npy')

VAL_NUM_IN = os.path.join(SPLIT_DIR, 'val_20_num.npy')
VAL_CAT_IN = os.path.join(SPLIT_DIR, 'val_20_cat.npy')
VAL_STATIC_IN = os.path.join(SPLIT_DIR, 'val_20_static.npy') # NUOVO
VAL_Y_IN = os.path.join(SPLIT_DIR, 'val_20_y.npy')

HOLD_NUM_IN = os.path.join(SPLIT_DIR, 'holdout_20_num.npy')
HOLD_CAT_IN = os.path.join(SPLIT_DIR, 'holdout_20_cat.npy')
HOLD_STATIC_IN = os.path.join(SPLIT_DIR, 'holdout_20_static.npy') # NUOVO
HOLD_Y_IN = os.path.join(SPLIT_DIR, 'holdout_20_y.npy')

PARAMS_FILE = os.path.join(MODEL_PARAMS_DIR, 'best_lstm_params_advanced.json') # Contiene epoca ottimale

TIMELINE_TEST_IN = os.path.join('Output_CSVs', 'timelines_test_dynamic.csv') # Per battle_id
SUBMISSION_FILE_OUT = os.path.join(SUBMISSION_DIR, 'submission_lstm_advanced_100pct.csv')

MAX_TURNS = 30

# --- Globals per la build_model ---
STATIC_FEATURES_EXIST = False
NUM_STATIC_FEATURES = 0
# ---------------------------------

# --- 2. Copia della funzione build_model (AGGIORNATA v5 Ibrida) ---
print("Caricamento encoders e definizione build_model (Super-Avanzata v5 Ibrida)...")
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
    NUM_CATEGORICAL_FEATURES = _temp_train_cat.shape[2] # Dovrebbe essere 8
    del _temp_train_num, _temp_train_cat
    print(f"Rilevate {NUM_NUMERIC_FEATURES} feature numeriche, {NUM_CATEGORICAL_FEATURES} categoriche.")
    if NUM_CATEGORICAL_FEATURES != 8:
        print(f"⚠️ ATTENZIONE: Rilevate {NUM_CATEGORICAL_FEATURES} feat. categoriche, ma 8 attese. Verifica '05_...py'.")
        
    # Controlla statiche (sia train che test per sicurezza)
    STATIC_FEATURES_EXIST = os.path.exists(TRAIN_STATIC_IN) and os.path.exists(TEST_STATIC_IN)
    if STATIC_FEATURES_EXIST:
        _temp_static = np.load(TRAIN_STATIC_IN)
        NUM_STATIC_FEATURES = _temp_static.shape[1]
        del _temp_static
        print(f"Rilevate {NUM_STATIC_FEATURES} feature statiche. Modalità Ibrida ATTIVA.")
    else:
        print("Feature statiche non trovate. Modalità Ibrida DISATTIVATA.")

except Exception as e:
     print(f"❌ ERRORE: Impossibile determinare dimensioni input dai file .npy: {e}")
     exit()

def build_model(hp_dict):
    """Costruisce il modello Keras ibrido (v5) dagli iperparametri salvati."""
    
    model_inputs = []
    branches_to_concat = []
    
    # --- Ramo 1: Dati Numerici (Invariato) ---
    input_numeric = layers.Input(shape=(MAX_TURNS, NUM_NUMERIC_FEATURES), name='input_numeric')
    model_inputs.append(input_numeric)
    lstm_numeric_out = layers.LSTM(hp_dict['lstm_units_num'], return_sequences=False)(input_numeric)
    branches_to_concat.append(lstm_numeric_out)
    
    # --- Ramo 2: Dati Categorici (8 features, invariato) ---
    input_categorical = layers.Input(shape=(MAX_TURNS, NUM_CATEGORICAL_FEATURES), name='input_categorical', dtype='int32')
    model_inputs.append(input_categorical)
    
    p1_name_in = layers.Lambda(lambda x: x[:, :, 0])(input_categorical)
    p2_name_in = layers.Lambda(lambda x: x[:, :, 1])(input_categorical)
    p1_move_in = layers.Lambda(lambda x: x[:, :, 2])(input_categorical)
    p2_move_in = layers.Lambda(lambda x: x[:, :, 3])(input_categorical)
    p1_type_in = layers.Lambda(lambda x: x[:, :, 4])(input_categorical) 
    p2_type_in = layers.Lambda(lambda x: x[:, :, 5])(input_categorical)
    p1_category_in = layers.Lambda(lambda x: x[:, :, 6])(input_categorical) 
    p2_category_in = layers.Lambda(lambda x: x[:, :, 7])(input_categorical) 

    embed_p1_name = layers.Embedding(input_dim=encoders['p1_pokemon_state.name']['vocab_size'], output_dim=hp_dict['embed_dim_name'], mask_zero=False)(p1_name_in)
    embed_p2_name = layers.Embedding(input_dim=encoders['p2_pokemon_state.name']['vocab_size'], output_dim=hp_dict['embed_dim_name'], mask_zero=False)(p2_name_in)
    embed_p1_move = layers.Embedding(input_dim=encoders['p1_move_details.name']['vocab_size'], output_dim=hp_dict['embed_dim_move'], mask_zero=False)(p1_move_in)
    embed_p2_move = layers.Embedding(input_dim=encoders['p2_move_details.name']['vocab_size'], output_dim=hp_dict['embed_dim_move'], mask_zero=False)(p2_move_in)
    embed_p1_type = layers.Embedding(input_dim=encoders['p1_move_details.type']['vocab_size'], output_dim=hp_dict.get('embed_dim_type', 8), mask_zero=False)(p1_type_in)
    embed_p2_type = layers.Embedding(input_dim=encoders['p2_move_details.type']['vocab_size'], output_dim=hp_dict.get('embed_dim_type', 8), mask_zero=False)(p2_type_in)
    embed_p1_category = layers.Embedding(input_dim=encoders['p1_move_details.category']['vocab_size'], output_dim=hp_dict.get('embed_dim_category', 8), mask_zero=False)(p1_category_in)
    embed_p2_category = layers.Embedding(input_dim=encoders['p2_move_details.category']['vocab_size'], output_dim=hp_dict.get('embed_dim_category', 8), mask_zero=False)(p2_category_in)

    concatenated_embeddings = layers.Concatenate()([
        embed_p1_name, embed_p2_name, 
        embed_p1_move, embed_p2_move,
        embed_p1_type, embed_p2_type,
        embed_p1_category, embed_p2_category
    ])
    
    lstm_categorical_out = layers.LSTM(hp_dict['lstm_units_cat'], return_sequences=False)(concatenated_embeddings)
    branches_to_concat.append(lstm_categorical_out)
    
    # --- NUOVO: Ramo 3: Dati Statici (Opzionale) ---
    if STATIC_FEATURES_EXIST:
        # Usa .get() per default sicuri se il JSON non è ibrido
        static_dense = hp_dict.get('static_dense_units', 64)
        static_dropout = hp_dict.get('static_dropout', 0.3)
        
        input_static = layers.Input(shape=(NUM_STATIC_FEATURES,), name='input_static')
        model_inputs.append(input_static)
        
        static_branch = layers.Dense(static_dense, activation='relu')(input_static)
        static_branch = layers.Dropout(static_dropout)(static_branch)
        branches_to_concat.append(static_branch)
    # ------------------------------------------------
    
    # --- Fine: Unione dei Rami ---
    if len(branches_to_concat) > 1:
        concatenated_features = layers.Concatenate()(branches_to_concat)
    else:
        concatenated_features = branches_to_concat[0]
    
    # --- Testa del Modello ---
    x = layers.Dense(hp_dict['dense_units'], activation='relu')(concatenated_features)
    x = layers.Dropout(hp_dict['dropout'])(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=model_inputs, outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_dict['learning_rate']), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

# --- 3. Carica il 100% dei Dati di Training ---
print("Caricamento 100% dati di training (Train+Val+Holdout)...")
try:
    X_train_num_100 = np.concatenate([ np.load(TRAIN_NUM_IN), np.load(VAL_NUM_IN), np.load(HOLD_NUM_IN) ])
    X_train_cat_100 = np.concatenate([ np.load(TRAIN_CAT_IN).astype(np.int32), np.load(VAL_CAT_IN).astype(np.int32), np.load(HOLD_CAT_IN).astype(np.int32) ])
    y_train_100 = np.concatenate([ np.load(TRAIN_Y_IN), np.load(VAL_Y_IN), np.load(HOLD_Y_IN) ])
    
    X_train_100_inputs = { 'input_numeric': X_train_num_100, 'input_categorical': X_train_cat_100 }

    # --- NUOVO: Carica e concatena statiche se esistono ---
    if STATIC_FEATURES_EXIST:
        print("Caricamento e concatenazione dati statici (100%)...")
        X_train_100_static = np.concatenate([
            np.load(TRAIN_STATIC_IN),
            np.load(VAL_STATIC_IN),
            np.load(HOLD_STATIC_IN)
        ])
        X_train_100_inputs['input_static'] = X_train_100_static
    # ---------------------------------------------------
    
except Exception as e:
    print(f"❌ ERRORE: Impossibile caricare tutti i dati di training: {e}")
    exit()

print(f"Dati di training totali caricati: {len(y_train_100)} campioni.")

# --- 4. Costruisci e Addestra il Modello Finale ---
print(f"Caricamento iperparametri da {PARAMS_FILE}...")
try:
    with open(PARAMS_FILE, 'r') as f:
        best_hps_dict = json.load(f)
    OPTIMAL_EPOCHS = best_hps_dict.get('optimal_epoch', None)
    if OPTIMAL_EPOCHS is None:
        raise ValueError("'optimal_epoch' non trovato nel file JSON dei parametri.")
    print(f"Parametri caricati. Epoca ottimale: {OPTIMAL_EPOCHS}")
except Exception as e:
    print(f"❌ ERRORE: Impossibile caricare il file dei parametri: {e}")
    print("Assicurati di aver eseguito '07_optimize_and_validate_ADVANCED.py'.")
    exit()

print("Costruzione modello finale...")
final_model = build_model(best_hps_dict)

print(f"Addestramento modello finale su 100% dei dati per {OPTIMAL_EPOCHS} epoche...")
final_model.fit(
    X_train_100_inputs,
    y_train_100,
    epochs=OPTIMAL_EPOCHS, 
    batch_size=64,
    verbose=1
)
print("Addestramento finale completato.")

# --- 5. Carica Dati di Test e Genera Predizioni ---
print("Caricamento dati di Test Kaggle...")
try:
    X_test_num = np.load(TEST_NUMERIC_IN)
    X_test_cat = np.load(TEST_CATEGORICAL_IN).astype(np.int32) # Assicura int32
    X_test_inputs = { 'input_numeric': X_test_num, 'input_categorical': X_test_cat }

    if STATIC_FEATURES_EXIST:
        print("Caricamento dati statici di Test Kaggle...")
        X_test_static = np.load(TEST_STATIC_IN)
        X_test_inputs['input_static'] = X_test_static

except Exception as e:
    print(f"❌ ERRORE: Impossibile caricare i dati di test Kaggle: {e}")
    exit()

print(f"Generazione predizioni Kaggle per {len(X_test_num)} campioni...")
y_pred_proba = final_model.predict(X_test_inputs)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# --- 6. Crea il File di Submission ---
print("Caricamento battle_id per la submission...")
try:
    # MODIFICA: Carica ID da file .npy per coerenza (salvato da 05)
    test_ids = np.load(os.path.join(LSTM_DATA_DIR, 'test_ids.npy'), allow_pickle=True)
    if len(test_ids) != len(y_pred):
        # Fallback su CSV se il .npy non è allineato (non dovrebbe succedere)
        print(f"⚠️ Warning: Mismatch ID .npy ({len(test_ids)}) vs Preds ({len(y_pred)}). Tento fallback su CSV...")
        test_df = pd.read_csv(TIMELINE_TEST_IN)
        test_ids = test_df['battle_id'].unique()
        
except Exception as e:
    print(f"❌ ERRORE: Impossibile caricare battle_ids: {e}")
    exit()

if len(test_ids) != len(y_pred):
    print(f"❌ ERRORE: Mismatch tra ID ({len(test_ids)}) e Predizioni ({len(y_pred)})")
    exit()

submission_df = pd.DataFrame({ 'battle_id': test_ids, 'player_won': y_pred })
submission_df.to_csv(SUBMISSION_FILE_OUT, index=False)

print(f"\n--- SUBMISSION LSTM (100% TRAIN, Ibrido: {STATIC_FEATURES_EXIST}) PRONTA ---")
print(f"File salvato in: {SUBMISSION_FILE_OUT}")
print(submission_df.head())

print(f"\n--- Script 11 (LSTM Submission 100% v5 Ibrido) Completato ---")