"""
MODIFICATO: Addestra il modello LSTM Avanzato sul 60% Train
per un numero fisso di epoche letto dal file JSON e valuta sul 20% Holdout.
Genera metriche e grafici.
(Versione SUPER-AVANZATA con 6 input categorici)
"""

import numpy as np
import pandas as pd
import os
import json
# import mlflow -> RIMOSSO
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

# --- Set Random Seeds ---
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
# ------------------------

# --- 1. Configurazione ---
SPLIT_DIR = 'Preprocessed_LSTM_Splits'
ENCODERS_FILE = os.path.join('Preprocessed_LSTM', 'encoders.json')
MODEL_PARAMS_DIR = 'Model_Params_LSTM_Advanced' # Carica da qui
ANALYSIS_DIR = 'Model_Analysis_LSTM_Holdout_60' # Salva qui
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# File di Input
X_TRAIN_NUM_FILE = os.path.join(SPLIT_DIR, 'train_60_num.npy')
X_TRAIN_CAT_FILE = os.path.join(SPLIT_DIR, 'train_60_cat.npy')
Y_TRAIN_FILE = os.path.join(SPLIT_DIR, 'train_60_y.npy')

X_HOLDOUT_NUM_FILE = os.path.join(SPLIT_DIR, 'holdout_20_num.npy')
X_HOLDOUT_CAT_FILE = os.path.join(SPLIT_DIR, 'holdout_20_cat.npy')
Y_HOLDOUT_FILE = os.path.join(SPLIT_DIR, 'holdout_20_y.npy')

PARAMS_FILE = os.path.join(MODEL_PARAMS_DIR, 'best_lstm_params_advanced.json') # Contiene epoca ottimale

# File di Output
REPORT_TXT_FILE = os.path.join(ANALYSIS_DIR, 'holdout_60_classification_report.txt')
CM_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'holdout_60_confusion_matrix.png')
ROC_CURVE_FILE = os.path.join(ANALYSIS_DIR, 'holdout_60_roc_auc_curve.png')

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
    _temp_train_num = np.load(X_TRAIN_NUM_FILE)
    _temp_train_cat = np.load(X_TRAIN_CAT_FILE)
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
    # Assicurati che 'embed_dim_type' sia nel tuo file JSON (dovrebbe esserci se hai eseguito il nuovo '07')
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

# --- 3. Carica Dati (60% Train, 20% Holdout) e Parametri ---
print("Caricamento dati (Train 60%, Holdout 20%)...")
try:
    X_train_num = np.load(X_TRAIN_NUM_FILE)
    X_train_cat = np.load(X_TRAIN_CAT_FILE).astype(np.int32)
    y_train = np.load(Y_TRAIN_FILE)

    X_holdout_num = np.load(X_HOLDOUT_NUM_FILE)
    X_holdout_cat = np.load(X_HOLDOUT_CAT_FILE).astype(np.int32)
    y_holdout = np.load(Y_HOLDOUT_FILE)

    # ✅ Carica parametri E l'epoca ottimale
    with open(PARAMS_FILE, 'r') as f:
        best_hps_dict = json.load(f)
    OPTIMAL_EPOCHS = best_hps_dict.get('optimal_epoch', None)
    if OPTIMAL_EPOCHS is None:
        raise ValueError("'optimal_epoch' non trovato nel file JSON dei parametri.")
    print(f"Parametri caricati da {PARAMS_FILE}. Epoca ottimale: {OPTIMAL_EPOCHS}")

except Exception as e:
    print(f"❌ ERRORE: Impossibile caricare dati o parametri: {e}")
    exit()

X_train_inputs = {'input_numeric': X_train_num, 'input_categorical': X_train_cat}
X_holdout_inputs = {'input_numeric': X_holdout_num, 'input_categorical': X_holdout_cat}
print(f"Dimensione set addestramento 60%: {len(y_train)} campioni.")

# --- 4. Addestra Modello per Epoche Fisse ---
print(f"\nCostruzione e Addestramento modello su 60% per {OPTIMAL_EPOCHS} epoche...")
final_model = build_model(best_hps_dict)

# Addestra solo sul 60% train, senza validation_data
history = final_model.fit(
    X_train_inputs,
    y_train,
    epochs=OPTIMAL_EPOCHS, # ✅ Usa l'epoca letta dal JSON
    batch_size=64,
    verbose=1 # Mostra progresso
)
print("Addestramento completato.")

# --- 5. Valuta su Holdout ---
print("\nValutazione finale su 20% Holdout:\n")
y_pred_proba_holdout = final_model.predict(X_holdout_inputs)
y_pred_holdout = (y_pred_proba_holdout > 0.5).astype(int)

holdout_accuracy = accuracy_score(y_holdout, y_pred_holdout)
holdout_auc = roc_auc_score(y_holdout, y_pred_proba_holdout)

print(f"  Accuracy (Holdout 20%): {holdout_accuracy:.4f}")
print(f"  AUC (Holdout 20%): {holdout_auc:.4f}")

print("\nClassification Report (su 20% Holdout)")
report_text_holdout = classification_report(y_holdout, y_pred_holdout, target_names=['False (0)', 'True (1)'], digits=4, zero_division=0)
print(report_text_holdout)

# Salva report
with open(REPORT_TXT_FILE, 'w', encoding='utf-8') as f:
    f.write(f"Final Classification Report (Train 60% for {OPTIMAL_EPOCHS} epochs, Test Holdout 20%):\n\n")
    f.write(report_text_holdout)
print(f"Report salvato in: {REPORT_TXT_FILE}")

# --- 6. Genera Grafici ---
print("\nGenerazione grafici (Holdout)...")

# Confusion Matrix
cm_holdout = confusion_matrix(y_holdout, y_pred_holdout)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_holdout, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Predicted False (0)', 'Predicted True (1)'],
            yticklabels=['Actual False (0)', 'Actual True (1)'])
plt.title("LSTM Confusion Matrix (Train 60%, Test Holdout 20%)")
plt.savefig(CM_OUTPUT_FILE)
plt.close()
print(f"Confusion Matrix salvata in: {CM_OUTPUT_FILE}")

# ROC Curve
fpr_hold, tpr_hold, _ = roc_curve(y_holdout, y_pred_proba_holdout)
plt.figure(figsize=(10, 8))
plt.plot(fpr_hold, tpr_hold, color='green', lw=2, label=f'LSTM ROC Curve (Holdout AUC = {holdout_auc:.4f})')
plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--', label='Random Guess (AUC = 0.50)')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('LSTM ROC-AUC Curve (Train 60%, Test Holdout 20%)')
plt.legend(loc="lower right"); plt.grid(True)
plt.savefig(ROC_CURVE_FILE)
plt.close()
print(f"ROC Curve salvata in: {ROC_CURVE_FILE}")

print(f"\n--- Script 20 (LSTM Holdout 60% Fixed Epochs, Senza MLflow) Completato ---")
print(f"Risultati salvati in: {ANALYSIS_DIR}")