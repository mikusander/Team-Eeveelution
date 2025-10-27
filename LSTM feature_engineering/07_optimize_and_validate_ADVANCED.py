# Nome file: 07_optimize_and_validate_ADVANCED.py
"""
Versione AVANZATA (per feature sequenziali SUPER-AVANZATE)
con opzione per saltare il tuning KerasTuner.
Usa architettura LSTM Two-Branch (Num + Cat potenziato).
Usa BayesianOptimization.
Salva i risultati in un file di testo.
AGGIUNTO controllo overfitting (Train vs Val Accuracy).
"""

import numpy as np
import os
import json
import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import time
import pprint

# --- Flag di Controllo ---
RUN_KERAS_TUNER = True # MODIFICA QUI SE NECESSARIO
# -------------------------

# --- Set Random Seeds ---
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
# ------------------------

# --- 1. Configurazione ---
SPLIT_DIR = 'Preprocessed_LSTM_Splits' # Legge sempre da qui
# Legge encoders e pokedex dalla cartella base LSTM (dove li salva 05)
BASE_LSTM_DIR = 'Preprocessed_LSTM'
ENCODERS_FILE = os.path.join(BASE_LSTM_DIR, 'encoders.json')
POKEDEX_FILE = os.path.join(BASE_LSTM_DIR, 'pokedex_with_stats.json') # Serve per info dimensioni

MODEL_PARAMS_DIR = 'Model_Params_LSTM_Advanced' # Cartella per HP e modello
ANALYSIS_DIR = 'Model_Analysis_LSTM_Advanced' # Cartella per summary
os.makedirs(MODEL_PARAMS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

SUMMARY_FILE = os.path.join(ANALYSIS_DIR, 'training_summary_advanced.txt')
PARAMS_FILE = os.path.join(MODEL_PARAMS_DIR, 'best_lstm_params_advanced.json')
MODEL_FILE = os.path.join(MODEL_PARAMS_DIR, 'best_lstm_model_advanced.keras')

# --- 2. Carica i Dati e Determina Dimensioni ---
print("Caricamento dati (60% Train, 20% Validation)...")
try:
    X_train_num = np.load(os.path.join(SPLIT_DIR, 'train_60_num.npy'))
    X_train_cat = np.load(os.path.join(SPLIT_DIR, 'train_60_cat.npy'))
    y_train = np.load(os.path.join(SPLIT_DIR, 'train_60_y.npy'))

    X_val_num = np.load(os.path.join(SPLIT_DIR, 'val_20_num.npy'))
    X_val_cat = np.load(os.path.join(SPLIT_DIR, 'val_20_cat.npy'))
    y_val = np.load(os.path.join(SPLIT_DIR, 'val_20_y.npy'))

    with open(ENCODERS_FILE, 'r') as f:
        encoders = json.load(f)
    # Pokedex non serve qui direttamente, ma carichiamo encoders

except Exception as e:
    print(f"❌ ERRORE durante il caricamento dei dati o encoders: {e}")
    exit()

# --- Correzione Tipi di Dati ---
print(f"Tipo dati categorici (prima): {X_train_cat.dtype}")
X_train_cat = X_train_cat.astype(np.int32)
X_val_cat = X_val_cat.astype(np.int32)
print(f"Tipo dati categorici (dopo): {X_train_cat.dtype}")

# Ottieni dimensioni input DAI FILE CARICATI
MAX_TURNS = X_train_num.shape[1]
try:
    NUM_NUMERIC_FEATURES = X_train_num.shape[2]
    NUM_CATEGORICAL_FEATURES = X_train_cat.shape[2] # Ora dovrebbe essere 6 (nome1, nome2, mossa1, mossa2, tipo1, tipo2)
    print(f"Dimensioni input RILEVATE: MAX_TURNS={MAX_TURNS}, NUM_NUMERIC={NUM_NUMERIC_FEATURES}, NUM_CATEGORICAL={NUM_CATEGORICAL_FEATURES}")
    if NUM_CATEGORICAL_FEATURES != 6:
         print(f"⚠️ ATTENZIONE: Numero feature categoriche ({NUM_CATEGORICAL_FEATURES}) diverso da 6. Verifica lo script 05.")
except Exception as e:
    print(f"❌ ERRORE: Impossibile determinare le dimensioni: {e}")
    exit()

X_train_inputs = { 'input_numeric': X_train_num, 'input_categorical': X_train_cat }
X_val_inputs = { 'input_numeric': X_val_num, 'input_categorical': X_val_cat }

print("Dati caricati e pronti per Keras (Architettura Two-Branch Potenziata).")

# --- 3. Definizione Modello (AGGIORNATO per 6 Input Categorici) ---
def build_model(hp):
    """Costruisce il modello Keras con ramo numerico + ramo categorico potenziato."""
    # HP (potremmo espandere un po' i range dato l'input più ricco)
    hp_lstm_units_num = hp.Choice('lstm_units_num', values=[32, 64, 96]) # Aggiunto 96
    hp_lstm_units_cat = hp.Choice('lstm_units_cat', values=[64, 128, 192]) # Aggiunto 192
    hp_embed_dim_name = hp.Choice('embed_dim_name', values=[10, 20, 32])
    hp_embed_dim_move = hp.Choice('embed_dim_move', values=[10, 20, 32])
    hp_embed_dim_type = hp.Choice('embed_dim_type', values=[8, 16]) # Nuovo HP per Embedding Tipi
    hp_dense_units = hp.Int('dense_units', 64, 256, step=64) # Range più ampio
    hp_dropout = hp.Float('dropout', min_value=0.3, max_value=0.7, step=0.1) # Range più ampio
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])

    # --- Architettura ---
    model_inputs = []
    branches_to_concat = []

    # --- Ramo 1: Dati Numerici (Invariato) ---
    input_numeric = layers.Input(shape=(MAX_TURNS, NUM_NUMERIC_FEATURES), name='input_numeric')
    model_inputs.append(input_numeric)
    lstm_numeric_out = layers.LSTM(hp_lstm_units_num, return_sequences=False)(input_numeric)
    branches_to_concat.append(lstm_numeric_out)

    # --- Ramo 2: Dati Categorici (AGGIORNATO) ---
    input_categorical = layers.Input(shape=(MAX_TURNS, NUM_CATEGORICAL_FEATURES), name='input_categorical', dtype='int32')
    model_inputs.append(input_categorical)

    # Estrai i 6 canali usando Lambda
    p1_name_in = layers.Lambda(lambda x: x[:, :, 0])(input_categorical)
    p2_name_in = layers.Lambda(lambda x: x[:, :, 1])(input_categorical)
    p1_move_in = layers.Lambda(lambda x: x[:, :, 2])(input_categorical)
    p2_move_in = layers.Lambda(lambda x: x[:, :, 3])(input_categorical)
    p1_type_in = layers.Lambda(lambda x: x[:, :, 4])(input_categorical) # NUOVO
    p2_type_in = layers.Lambda(lambda x: x[:, :, 5])(input_categorical) # NUOVO

    # Embedding (senza mask_zero)
    embed_p1_name = layers.Embedding(input_dim=encoders['p1_pokemon_state.name']['vocab_size'], output_dim=hp_embed_dim_name, mask_zero=False)(p1_name_in)
    embed_p2_name = layers.Embedding(input_dim=encoders['p2_pokemon_state.name']['vocab_size'], output_dim=hp_embed_dim_name, mask_zero=False)(p2_name_in)
    embed_p1_move = layers.Embedding(input_dim=encoders['p1_move_details.name']['vocab_size'], output_dim=hp_embed_dim_move, mask_zero=False)(p1_move_in)
    embed_p2_move = layers.Embedding(input_dim=encoders['p2_move_details.name']['vocab_size'], output_dim=hp_embed_dim_move, mask_zero=False)(p2_move_in)
    # NUOVI Embedding per Tipi Mosse
    embed_p1_type = layers.Embedding(input_dim=encoders['p1_move_details.type']['vocab_size'], output_dim=hp_embed_dim_type, mask_zero=False)(p1_type_in)
    embed_p2_type = layers.Embedding(input_dim=encoders['p2_move_details.type']['vocab_size'], output_dim=hp_embed_dim_type, mask_zero=False)(p2_type_in)

    # Concatena TUTTI gli embedding
    concatenated_embeddings = layers.Concatenate()([
        embed_p1_name, embed_p2_name,
        embed_p1_move, embed_p2_move,
        embed_p1_type, embed_p2_type # NUOVI
    ])

    lstm_categorical_out = layers.LSTM(hp_lstm_units_cat, return_sequences=False)(concatenated_embeddings)
    branches_to_concat.append(lstm_categorical_out)

    # --- Fine: Unione dei Rami ---
    concatenated_features = layers.Concatenate()(branches_to_concat)

    # --- Testa del Modello ---
    x = layers.Dense(hp_dense_units, activation='relu')(concatenated_features)
    x = layers.Dropout(hp_dropout)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=model_inputs, outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

# --- Variabili per contenere i risultati ---
# ... (invariate) ...
best_hps_values = None
optimal_epoch = None
best_model = None
history = None

# --- 4. ESEGUI O SALTA TUNING ---
if RUN_KERAS_TUNER:
    print("\n[MODALITÀ TUNING ATTIVA]")
    print("Avvio Ottimizzazione (SUPER-AVANZATA) con KerasTuner (BayesianOptimization)...")
    tuner = kt.BayesianOptimization(
        build_model, objective=kt.Objective("val_auc", direction="max"),
        max_trials=30, # Aumentiamo leggermente i tentativi
        executions_per_trial=1, directory=MODEL_PARAMS_DIR,
        project_name='Pokemon_BayesianOpt_SuperAdv', # Nuovo nome progetto
        overwrite=True
    )
    stop_early_tuner = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8) # Pazienza leggermente aumentata
    tuner.search(
        X_train_inputs, y_train, epochs=120, # Max epoche leggermente aumentate
        validation_data=(X_val_inputs, y_val),
        callbacks=[stop_early_tuner], batch_size=64
    )
    print("\nRicerca KerasTuner (Bayesiana) completata.")
    try:
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_hps_values = best_hps.values
        print("\nIperparametri Ottimali Trovati (Super-Avanzati):")
        print(pprint.pformat(best_hps_values, indent=2))
    except Exception as e: print(f"❌ ERRORE iperparametri: {e}"); exit()

    print("\nAddestramento modello migliore (dopo tuning)...")
    best_model = tuner.hypermodel.build(best_hps)
    final_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True) # Pazienza aumentata
    history = best_model.fit(
        X_train_inputs, y_train, epochs=200, # Max epoche aumentate
        validation_data=(X_val_inputs, y_val),
        callbacks=[final_early_stopping], batch_size=64, verbose=1
    )
    optimal_epoch = final_early_stopping.stopped_epoch - final_early_stopping.patience + 1
    if optimal_epoch <= 0: optimal_epoch = len(history.history['val_loss'])
    print(f"\nTraining fermato. Epoca ottimale (basata su val_loss): {optimal_epoch}")

    print(f"Salvataggio modello in: {MODEL_FILE}")
    best_model.save(MODEL_FILE)
    print(f"Salvataggio parametri (inclusa epoca ottimale) in: {PARAMS_FILE}")
    best_hps_to_save = best_hps_values.copy()
    best_hps_to_save['optimal_epoch'] = optimal_epoch
    with open(PARAMS_FILE, 'w') as f: json.dump(best_hps_to_save, f, indent=4)

else: # --- SALTA TUNING ---
    print("\n[MODALITÀ CARICAMENTO ATTIVA (Tuning Saltato)]")
    # ... (Blocco else invariato, carica PARAMS_FILE e MODEL_FILE) ...
    print(f"Caricamento parametri da: {PARAMS_FILE}")
    try:
        with open(PARAMS_FILE, 'r') as f: best_hps_loaded = json.load(f)
        optimal_epoch = best_hps_loaded.get('optimal_epoch')
        if optimal_epoch is None: raise ValueError("'optimal_epoch' non trovato.")
        best_hps_values = best_hps_loaded.copy()
        print(f"Parametri caricati. Epoca ottimale: {optimal_epoch}")
        print("Iperparametri Caricati:\n", pprint.pformat(best_hps_values, indent=2))
    except Exception as e: print(f"❌ ERRORE parametri: {e}"); exit()
    print(f"Caricamento modello da: {MODEL_FILE}")
    try:
        best_model = keras.models.load_model(MODEL_FILE, safe_mode=False) # Necessario per Lambda
        print("Modello caricato.")
    except Exception as e: print(f"❌ ERRORE caricamento modello: {e}"); exit()
    history = None


# --- 5. Valutazione ---
if best_model is None: print("❌ ERRORE: Modello non disponibile."); exit()

print("\nCalcolo predizioni su Train (60%) e Validation (20%)...")
y_pred_proba_train = best_model.predict(X_train_inputs)
y_pred_train = (y_pred_proba_train > 0.5).astype(int)
train_accuracy = accuracy_score(y_train, y_pred_train)

y_pred_proba_val = best_model.predict(X_val_inputs)
y_pred_val = (y_pred_proba_val > 0.5).astype(int)
val_accuracy = accuracy_score(y_val, y_pred_val)
val_auc = roc_auc_score(y_val, y_pred_proba_val)

report_dict = classification_report(y_val, y_pred_val, output_dict=True, zero_division=0)
report_text = classification_report(y_val, y_pred_val, digits=4, zero_division=0)

print("\n--- Performance Metrics ---")
print(f"  Accuracy (Training 60%):   {train_accuracy:.4f}")
print(f"  Accuracy (Validation 20%): {val_accuracy:.4f}")
print(f"  AUC (Validation 20%):      {val_auc:.4f}")
overfitting_diff = train_accuracy - val_accuracy; overfitting_perc = overfitting_diff * 100
overfitting_msg = f"\033[92m  OK: No overfitting evidente. (Delta: {overfitting_perc:.2f}%)\033[0m"
if overfitting_diff > 0.07: # Soglia leggermente aumentata a 7%
    overfitting_msg = f"\033[93m  WARNING: Possibile Overfitting! (Delta: {overfitting_perc:.2f}%)\033[0m"
print(overfitting_msg)
print("---------------------------------------------")
print("\n--- Classification Report (Validation 20%) ---")
print(report_text)
print("---------------------------------------------")


# --- 6. Crea e salva il file di riepilogo ---
print(f"\nSalvataggio riepilogo in: {SUMMARY_FILE}...")
try:
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(" RIEPILOGO TRAINING MODELLO LSTM (SUPER-AVANZATO) \n")
        f.write("="*60 + "\n\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Modalità Esecuzione: {'TUNING + TRAINING' if RUN_KERAS_TUNER else 'CARICAMENTO MODELLO'}\n\n")
        f.write("--- Iperparametri Utilizzati ---\n")
        f.write(pprint.pformat(best_hps_values, indent=2)); f.write("\n\n")
        f.write("--- Performance Metrics ---\n")
        f.write(f"Epoca ottimale utilizzata/trovata: {optimal_epoch}\n")
        f.write(f"Accuracy (Training 60%):   {train_accuracy:.4f}\n")
        f.write(f"Accuracy (Validation 20%): {val_accuracy:.4f}\n")
        f.write(f"AUC (Validation 20%):      {val_auc:.4f}\n")
        overfitting_msg_clean = overfitting_msg.replace('\033[93m','').replace('\033[92m','').replace('\033[0m','')
        f.write(f"Overfitting Check: {overfitting_msg_clean}\n\n")
        f.write("--- Classification Report (Validation 20%) ---\n"); f.write(report_text); f.write("\n\n")
        f.write("--- Percorsi File ---\n")
        f.write(f"Modello utilizzato/salvato: {MODEL_FILE}\n")
        f.write(f"Parametri utilizzati/salvati: {PARAMS_FILE}\n")
        if history and RUN_KERAS_TUNER:
             f.write("\n--- Dettagli Training (ultima esecuzione) ---\n")
             f.write(f"Epoche eseguite: {len(history.history['loss'])}\n")
             try:
                 opt_idx = optimal_epoch - 1
                 if 0 <= opt_idx < len(history.history['loss']):
                      f.write(f"Metriche all'epoca ottimale ({optimal_epoch}):\n")
                      f.write(f"  Loss(T): {history.history['loss'][opt_idx]:.4f} | Loss(V): {history.history['val_loss'][opt_idx]:.4f}\n")
                      f.write(f"  AUC (T): {history.history['auc'][opt_idx]:.4f} | AUC (V): {history.history['val_auc'][opt_idx]:.4f}\n")
                 else: raise IndexError()
             except Exception:
                 f.write(" (Impossibile recuperare metriche all'epoca ottimale)\n")
                 f.write(f"Metriche finali (epoca {len(history.history.get('loss', []))}):\n")
                 f.write(f"  Loss(T): {history.history.get('loss', [np.nan])[-1]:.4f} | Loss(V): {history.history.get('val_loss', [np.nan])[-1]:.4f}\n")
                 f.write(f"  AUC (T): {history.history.get('auc', [np.nan])[-1]:.4f} | AUC (V): {history.history.get('val_auc', [np.nan])[-1]:.4f}\n")
    print("File di riepilogo creato.")
except Exception as e: print(f"❌ ERRORE riepilogo: {e}")

print(f"\n--- Completato (Super-Avanzato, {'Tuning Eseguito' if RUN_KERAS_TUNER else 'Modello Caricato'}) ---")
print(f"Modello/parametri in {MODEL_PARAMS_DIR}")
print(f"Riepilogo in {ANALYSIS_DIR}")