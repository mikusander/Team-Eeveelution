# Nome file: 05_lstm_preprocessing.py
"""
Prepara i dati per un modello sequenziale (LSTM) - VERSIONE SUPER-AVANZATA.
Aggiunge numerose feature sequenziali: tipo mossa, efficacia, status,
flags mosse, turno, stats attive, delta relativi, flag switch.
SALVA ANCHE GLI ID delle battaglie nell'ordine corretto.

MODIFICATO (v3):
- Aggiunti tutti i 5 boosts (atk, def, spa, spd, spe)
- Aggiunte 5 statistiche base attive (atk, def, spa, spd, spe)
- RIMOSSA 'basePower' (non presente nei dati di origine)
- Aggiunta 'category' della mossa (categorica)
- Aggiunti 'delta' relativi per tutte le 5 statistiche base
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Assicurati di usare il path corretto per Keras
# from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences


print("Avvio Preprocessing LSTM (SUPER-AVANZATO v3)...")

# --- 1. Configurazione ---
CSV_DIR = 'Output_CSVs'
TIMELINE_TRAIN_IN = os.path.join(CSV_DIR, 'timelines_train_dynamic.csv')
TIMELINE_TEST_IN = os.path.join(CSV_DIR, 'timelines_test_dynamic.csv')
TARGET_IN = os.path.join(CSV_DIR, 'battles_train_static.csv')
BATTLES_TRAIN_STATIC_IN = os.path.join(CSV_DIR, 'battles_train_static.csv') # Per Pokedex e stats

LSTM_DATA_DIR = 'Preprocessed_LSTM' # Sovrascrive i file precedenti
os.makedirs(LSTM_DATA_DIR, exist_ok=True)

# File di Output
TRAIN_NUMERIC_OUT = os.path.join(LSTM_DATA_DIR, 'train_numeric_seq.npy')
TRAIN_CATEGORICAL_OUT = os.path.join(LSTM_DATA_DIR, 'train_categorical_seq.npy')
TARGET_OUT = os.path.join(LSTM_DATA_DIR, 'target.npy')
TRAIN_IDS_OUT = os.path.join(LSTM_DATA_DIR, 'train_ids.npy')
TEST_NUMERIC_OUT = os.path.join(LSTM_DATA_DIR, 'test_numeric_seq.npy')
TEST_CATEGORICAL_OUT = os.path.join(LSTM_DATA_DIR, 'test_categorical_seq.npy')
TEST_IDS_OUT = os.path.join(LSTM_DATA_DIR, 'test_ids.npy')
ENCODERS_OUT = os.path.join(LSTM_DATA_DIR, 'encoders.json')
# NUOVO: Salva anche il Pokedex con stats per dopo
POKEDEX_STATS_OUT = os.path.join(LSTM_DATA_DIR, 'pokedex_with_stats.json')

MAX_TURNS = 30 # Manteniamo 30 per ora

# Metadati
try:
    from move_effects_final import MOVE_EFFECTS_DETAILED
    STATUS_MOVES = [m for m, e in MOVE_EFFECTS_DETAILED.items() if 'opponent_status' in e]
    HEALING_MOVES = [m for m, e in MOVE_EFFECTS_DETAILED.items() if 'healing' in e]
    print("Caricati metadati mosse.")
except ImportError: print("❌ ERRORE: File 'move_effects_final.py' non trovato."); exit()
try:
    try: from unique_statuses import STATUSES as UNIQUE_STATUSES_SET
    except ImportError: from unique_statuses import NEGATIVE_STATUSES as UNIQUE_STATUSES_SET
    NEGATIVE_STATUSES = list(UNIQUE_STATUSES_SET)
    print(f"Caricati {len(NEGATIVE_STATUSES)} status negativi.")
except ImportError: print("❌ ERRORE: File 'unique_statuses.py' non trovato."); exit()

# Costanti Tipi (Concesso dall'utente)
TYPE_EFFECTIVENESS = {
    'normal': {'rock': 0.5, 'ghost': 0.0}, 'fire': {'fire': 0.5, 'water': 0.5, 'grass': 2.0, 'ice': 2.0, 'bug': 2.0, 'rock': 0.5, 'dragon': 0.5},
    'water': {'fire': 2.0, 'water': 0.5, 'grass': 0.5, 'ground': 2.0, 'rock': 2.0, 'dragon': 0.5}, 'electric': {'water': 2.0, 'electric': 0.5, 'grass': 0.5, 'ground': 0.0, 'flying': 2.0, 'dragon': 0.5},
    'grass': {'fire': 0.5, 'water': 2.0, 'electric': 0.5, 'grass': 0.5, 'poison': 0.5, 'ground': 2.0, 'flying': 0.5, 'bug': 0.5, 'rock': 2.0, 'dragon': 0.5}, 'ice': {'fire': 0.5, 'water': 0.5, 'grass': 2.0, 'ice': 0.5, 'ground': 2.0, 'flying': 2.0, 'dragon': 2.0},
    'fighting': {'normal': 2.0, 'ice': 2.0, 'poison': 0.5, 'flying': 0.5, 'psychic': 0.5, 'bug': 0.5, 'rock': 2.0, 'ghost': 0.0}, 'poison': {'grass': 2.0, 'poison': 0.5, 'ground': 0.5, 'bug': 2.0, 'rock': 0.5, 'ghost': 0.5},
    'ground': {'fire': 2.0, 'electric': 2.0, 'grass': 0.5, 'poison': 2.0, 'flying': 0.0, 'bug': 0.5, 'rock': 2.0}, 'flying': {'electric': 0.5, 'grass': 2.0, 'fighting': 2.0, 'bug': 2.0, 'rock': 0.5},
    'psychic': {'fighting': 2.0, 'poison': 2.0, 'psychic': 0.5, 'ghost': 0.0}, 'bug': {'fire': 0.5, 'grass': 2.0, 'fighting': 0.5, 'poison': 2.0, 'flying': 0.5, 'psychic': 2.0},
    'rock': {'fire': 2.0, 'ice': 2.0, 'fighting': 0.5, 'ground': 0.5, 'flying': 2.0, 'bug': 2.0}, 'ghost': {'normal': 0.0, 'psychic': 0.0, 'ghost': 2.0},
    'dragon': {'dragon': 2.0}, 'notype': {}, 'none': {}
}

# --- 2. Definizione Feature (SUPER-AVANZATA v3) ---

# Aggiunti tutti i 5 boost
BASE_NUMERIC_FEATURES = [
    'p1_pokemon_state.hp_pct', 'p2_pokemon_state.hp_pct',
    'p1_pokemon_state.boosts.atk', 'p1_pokemon_state.boosts.def', 'p1_pokemon_state.boosts.spa', 'p1_pokemon_state.boosts.spd', 'p1_pokemon_state.boosts.spe',
    'p2_pokemon_state.boosts.atk', 'p2_pokemon_state.boosts.def', 'p2_pokemon_state.boosts.spa', 'p2_pokemon_state.boosts.spd', 'p2_pokemon_state.boosts.spe',
]
# Aggiunte stats attive, delta stats (RIMOSSO basePower)
NEW_NUMERIC_FEATURES = [
    'turn_norm', 
    'p1_move_effectiveness', 'p2_move_effectiveness',
    # 'p1_move_details.basePower', 'p2_move_details.basePower', # RIMOSSO - Causa KeyError
    'p1_has_negative_status', 'p2_has_negative_status',
    'p1_is_stab', 'p2_is_stab',
    'p1_is_status_move', 'p2_is_status_move',
    'p1_is_healing_move', 'p2_is_healing_move',
    'p1_active_base_atk', 'p1_active_base_def', 'p1_active_base_spa', 'p1_active_base_spd', 'p1_active_base_spe', # NUOVO (5 stats)
    'p2_active_base_atk', 'p2_active_base_def', 'p2_active_base_spa', 'p2_active_base_spd', 'p2_active_base_spe', # NUOVO (5 stats)
    'relative_hp_delta', 'relative_boost_delta',
    'relative_atk_delta', 'relative_def_delta', 'relative_spa_delta', 'relative_spd_delta', 'relative_spe_delta', # NUOVO (5 delta)
    'p1_switched', 'p2_switched'
]
NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + NEW_NUMERIC_FEATURES

BASE_CATEGORICAL_FEATURES = [
    'p1_pokemon_state.name', 'p2_pokemon_state.name',
    'p1_move_details.name', 'p2_move_details.name',
]
# Aggiunta categoria mossa (MANTENUTO)
NEW_CATEGORICAL_FEATURES = [
    'p1_move_details.type', 'p2_move_details.type',
    'p1_move_details.category', 'p2_move_details.category' # NUOVO (Totale 8 features cat)
]
CATEGORICAL_FEATURES = BASE_CATEGORICAL_FEATURES + NEW_CATEGORICAL_FEATURES

# --- 3. Funzioni Helper ---
def build_pokedex_with_stats(static_battles_df_path):
    """Costruisce un Pokédex {nome: {'types': [t1, t2], 'stats': {hp:x, ...}}}."""
    print("Building Pokédex with Stats...")
    try:
        static_battles_df = pd.read_csv(static_battles_df_path)
    except FileNotFoundError: return None
    pokedex = {}
    stat_keys = ['hp', 'atk', 'def', 'spa', 'spd', 'spe']

    # Team P1
    for i in range(6):
        prefix = f'p1_team.{i}.'
        name_col = prefix + 'name'
        if name_col in static_battles_df.columns:
            cols_to_load = [name_col, prefix+'type1', prefix+'type2'] + [prefix+'base_'+s for s in stat_keys]
            # Assicurati che tutte le colonne esistano prima di caricarle
            cols_exist = [c for c in cols_to_load if c in static_battles_df.columns]
            if len(cols_exist) < 3: continue # Salta se mancano nome o tipi

            pkmn_data = static_battles_df[cols_exist].dropna(subset=[name_col]).drop_duplicates(subset=[name_col])
            for _, row in pkmn_data.iterrows():
                name = row[name_col]
                if name not in pokedex:
                    stats = {s: row.get(prefix+'base_'+s, 0) for s in stat_keys} # Default a 0 se manca
                    # Correggi NaN potenziali nelle stats
                    stats = {k: (0 if pd.isna(v) else int(v)) for k, v in stats.items()}
                    pokedex[name] = {
                        'types': [str(row.get(prefix+'type1','none')).lower(), str(row.get(prefix+'type2','none')).lower()],
                        'stats': stats
                    }
    # Lead P2
    prefix = 'p2_lead.'
    name_col = prefix + 'name'
    if name_col in static_battles_df.columns:
        cols_to_load = [name_col, prefix+'type1', prefix+'type2'] + [prefix+'base_'+s for s in stat_keys]
        cols_exist = [c for c in cols_to_load if c in static_battles_df.columns]
        if len(cols_exist) >= 3:
             p2_lead_data = static_battles_df[cols_exist].dropna(subset=[name_col]).drop_duplicates(subset=[name_col])
             for _, row in p2_lead_data.iterrows():
                 name = row[name_col]
                 if name not in pokedex:
                     stats = {s: row.get(prefix+'base_'+s, 0) for s in stat_keys}
                     stats = {k: (0 if pd.isna(v) else int(v)) for k, v in stats.items()}
                     pokedex[name] = {
                         'types': [str(row.get(prefix+'type1','none')).lower(), str(row.get(prefix+'type2','none')).lower()],
                         'stats': stats
                     }
    # Aggiungi fallback per 'none'
    if 'none' not in pokedex:
        pokedex['none'] = {'types': ['none', 'none'], 'stats': {s: 0 for s in stat_keys}}

    print(f"Pokédex with Stats built ({len(pokedex)} entries).")
    return pokedex

def get_effectiveness(move_type, target_types):
    move_type = str(move_type).lower()
    if move_type in ['notype', 'none', 'nan']: return 1.0
    effectiveness_map = TYPE_EFFECTIVENESS.get(move_type, {})
    multiplier = 1.0
    valid_target_types_found = False
    for target_type in target_types:
        target_type = str(target_type).lower()
        if target_type not in ['notype', 'none', 'nan']:
            multiplier *= effectiveness_map.get(target_type, 1.0)
            valid_target_types_found = True
    return multiplier if valid_target_types_found else 1.0

# --- 4. Funzione di Processing Principale (SUPER-AVANZATA v3) ---
def process_timelines_advanced(filepath, pokedex_stats, is_train=True, encoders=None):
    print(f"Caricamento e Processing Avanzato v3 {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError: return None, None, None, None

    # Ordina per sicurezza
    df = df.sort_values(by=['battle_id', 'turn']).reset_index(drop=True)

    # --- Calcolo Feature Turno-per-Turno ---
    print("Calcolo feature sequenziali avanzate v3...")

    # Aggiungi tipi attivi e STATS attive
    df['p1_active_data'] = df['p1_pokemon_state.name'].fillna('none').apply(lambda x: pokedex_stats.get(x, pokedex_stats['none']))
    df['p2_active_data'] = df['p2_pokemon_state.name'].fillna('none').apply(lambda x: pokedex_stats.get(x, pokedex_stats['none']))
    df['p1_active_types'] = df['p1_active_data'].apply(lambda x: x['types'])
    df['p2_active_types'] = df['p2_active_data'].apply(lambda x: x['types'])
    # Estrai tutte le 5 stats (escluso hp)
    for stat in ['atk', 'def', 'spa', 'spd', 'spe']:
        df[f'p1_active_base_{stat}'] = df['p1_active_data'].apply(lambda x: x['stats'][stat])
        df[f'p2_active_base_{stat}'] = df['p2_active_data'].apply(lambda x: x['stats'][stat])

    # Pulisci nomi/tipi/categorie mosse
    df['p1_move_name_str'] = df['p1_move_details.name'].fillna('none').astype(str).str.lower()
    df['p2_move_name_str'] = df['p2_move_details.name'].fillna('none').astype(str).str.lower()
    df['p1_move_details.type'] = df['p1_move_details.type'].fillna('none').astype(str).str.lower() 
    df['p2_move_details.type'] = df['p2_move_details.type'].fillna('none').astype(str).str.lower()
    df['p1_move_details.category'] = df['p1_move_details.category'].fillna('none').astype(str).str.lower() # NUOVO
    df['p2_move_details.category'] = df['p2_move_details.category'].fillna('none').astype(str).str.lower() # NUOVO
    # df['p1_move_details.basePower'] = df['p1_move_details.basePower'].fillna(0.0) # RIMOSSO
    # df['p2_move_details.basePower'] = df['p2_move_details.basePower'].fillna(0.0) # RIMOSSO

    # Calcola flag mosse
    df['p1_is_stab'] = df.apply(lambda r: 1 if (r['p1_move_details.type'] in r['p1_active_types'] and r['p1_move_details.type'] != 'none') else 0, axis=1)
    df['p2_is_stab'] = df.apply(lambda r: 1 if (r['p2_move_details.type'] in r['p2_active_types'] and r['p2_move_details.type'] != 'none') else 0, axis=1)
    df['p1_is_status_move'] = df['p1_move_name_str'].isin(STATUS_MOVES).astype(int)
    df['p2_is_status_move'] = df['p2_move_name_str'].isin(STATUS_MOVES).astype(int)
    df['p1_is_healing_move'] = df['p1_move_name_str'].isin(HEALING_MOVES).astype(int)
    df['p2_is_healing_move'] = df['p2_move_name_str'].isin(HEALING_MOVES).astype(int)

    # Calcola efficacia dinamica
    df['p1_move_effectiveness'] = df.apply(lambda r: get_effectiveness(r['p1_move_details.type'], r['p2_active_types']), axis=1)
    df['p2_move_effectiveness'] = df.apply(lambda r: get_effectiveness(r['p2_move_details.type'], r['p1_active_types']), axis=1)

    # Flag Status Negativo
    df['p1_pokemon_state.status'] = df['p1_pokemon_state.status'].fillna('nostatus') # Gestisci NaN in status
    df['p2_pokemon_state.status'] = df['p2_pokemon_state.status'].fillna('nostatus')
    df['p1_has_negative_status'] = df['p1_pokemon_state.status'].isin(NEGATIVE_STATUSES).astype(int)
    df['p2_has_negative_status'] = df['p2_pokemon_state.status'].isin(NEGATIVE_STATUSES).astype(int)

    # Turno Normalizzato (usa transform per applicarlo per gruppo)
    df['turn_norm'] = df.groupby('battle_id')['turn'].transform(lambda x: x / x.max() if x.max() > 0 else 0)

    # Delta Relativi HP e Boost (ora con 5 stats)
    df['p1_pokemon_state.hp_pct'] = df['p1_pokemon_state.hp_pct'].fillna(0.0) # Gestisci NaN in HP
    df['p2_pokemon_state.hp_pct'] = df['p2_pokemon_state.hp_pct'].fillna(0.0)
    df['relative_hp_delta'] = df['p1_pokemon_state.hp_pct'] - df['p2_pokemon_state.hp_pct']

    boost_cols_p1 = [f'p1_pokemon_state.boosts.{s}' for s in ['atk', 'def', 'spa', 'spd', 'spe']] # AGGIORNATO
    boost_cols_p2 = [f'p2_pokemon_state.boosts.{s}' for s in ['atk', 'def', 'spa', 'spd', 'spe']] # AGGIORNATO
    df['p1_boost_sum'] = df[boost_cols_p1].fillna(0).sum(axis=1)
    df['p2_boost_sum'] = df[boost_cols_p2].fillna(0).sum(axis=1)
    df['relative_boost_delta'] = df['p1_boost_sum'] - df['p2_boost_sum']
    
    # NUOVO: Delta Statistici Base Relativi
    for stat in ['atk', 'def', 'spa', 'spd', 'spe']:
        df[f'relative_{stat}_delta'] = df[f'p1_active_base_{stat}'] - df[f'p2_active_base_{stat}']

    # Flag Switch (verifica cambio nome rispetto al turno precedente nel gruppo)
    df['p1_prev_name'] = df.groupby('battle_id')['p1_pokemon_state.name'].shift(1).fillna('none')
    df['p2_prev_name'] = df.groupby('battle_id')['p2_pokemon_state.name'].shift(1).fillna('none')
    df['p1_switched'] = (df['p1_pokemon_state.name'] != df['p1_prev_name']).astype(int)
    df['p2_switched'] = (df['p2_pokemon_state.name'] != df['p2_prev_name']).astype(int)
    # Correggi il primo turno (non è uno switch)
    df.loc[df['turn'] == 0, ['p1_switched', 'p2_switched']] = 0


    # --- Pulisci colonne non necessarie prima di encoding ---
    cols_to_drop_before_encode = ['p1_active_data', 'p2_active_data', 'p1_active_types', 'p2_active_types',
                                  'p1_move_name_str', 'p2_move_name_str', 'p1_boost_sum', 'p2_boost_sum',
                                  'p1_prev_name', 'p2_prev_name', 'p1_pokemon_state.status', 'p2_pokemon_state.status']
    df = df.drop(columns=[col for col in cols_to_drop_before_encode if col in df.columns])

    # --- Aggiungi colonne mancanti (di nuovo, per sicurezza) e gestisci NaN Finali ---
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        if col not in df.columns:
            # Determina tipo in base alla lista
            if col in NUMERIC_FEATURES:
                df[col] = 0.0 # Default numerico
            else:
                df[col] = 'none' # Default categorico
    
    # Fillna finale per le numeriche
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(0.0)
    # Fillna finale per le categoriche e cast a stringa
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].fillna('none').astype(str)


    # --- Encoding Categorico ---
    if is_train:
        encoders = {}
        for col in CATEGORICAL_FEATURES:
            le = LabelEncoder()
            # Gestisci il caso in cui una colonna (es. category) non esista nei dati caricati
            if col not in df.columns:
                print(f"⚠️ ATTENZIONE: Colonna '{col}' non trovata. Creazione encoder fittizio.")
                df[col] = 'none' # Crea colonna fittizia
            
            unique_classes = pd.unique(df[col]) # Usa solo classi presenti
            le.fit(unique_classes)
            df[col] = le.transform(df[col]) + 1 # Shift +1 for padding=0
            encoders[col] = {'classes': list(le.classes_), 'vocab_size': len(le.classes_) + 1}
        print("Encoder (LabelEncoder) creato.")
    else: # Applicazione encoder su Test
        if encoders is None: return None, None, None, None # Errore
        print("Applicazione Encoder (LabelEncoder) su Test set...")
        for col in CATEGORICAL_FEATURES:
            if col not in encoders:
                 print(f"❌ ERRORE: Chiave '{col}' mancante negli encoders per Test.")
                 df[col] = 1 # Mappa tutto a 'none' (indice 1)
                 continue 
            classes = encoders[col]['classes']
            mapping = {cls: i+1 for i, cls in enumerate(classes)}
            df[col] = df[col].astype(str).apply(lambda x: mapping.get(x, 1))
        print("Encoder applicato.")

    # Raggruppa per battaglia
    print("Raggruppamento finale per battle_id...")
    grouped = df.groupby('battle_id')
    numeric_sequences, categorical_sequences, battle_ids = [], [], []
    for battle_id, group in grouped:
        numeric_sequences.append(group[NUMERIC_FEATURES].values) # Usa la lista aggiornata
        categorical_sequences.append(group[CATEGORICAL_FEATURES].values) # Usa la lista aggiornata
        battle_ids.append(battle_id)

    # Padding / Truncating
    print(f"Applicazione Padding/Truncating a {MAX_TURNS} turni...")
    numeric_padded = pad_sequences(numeric_sequences, maxlen=MAX_TURNS, dtype='float32', padding='post', truncating='post', value=0.0)
    categorical_padded = pad_sequences(categorical_sequences, maxlen=MAX_TURNS, dtype='int32', padding='post', truncating='post', value=0)

    return numeric_padded, categorical_padded, np.array(battle_ids), encoders

# --- 5. Esecuzione ---
# Costruisci Pokedex (una sola volta, usa train static)
pokedex_with_stats = build_pokedex_with_stats(BATTLES_TRAIN_STATIC_IN)
if pokedex_with_stats is None: exit()

# Salva Pokedex per riutilizzarlo (es. nell'inferenza)
print(f"Salvataggio Pokedex con stats in {POKEDEX_STATS_OUT}...")
try:
    with open(POKEDEX_STATS_OUT, 'w') as f:
        json.dump(pokedex_with_stats, f, indent=2)
except Exception as e:
    print(f"⚠️ WARNING: Impossibile salvare Pokedex: {e}")


# Processa il Training Set
train_num_seq, train_cat_seq, train_ids, encoders = process_timelines_advanced(
    TIMELINE_TRAIN_IN, pokedex_with_stats, is_train=True
)
if train_num_seq is None: exit()

# Carica e allinea il target
print(f"Caricamento e allineamento target da {TARGET_IN}...")
try:
    target_df_full = pd.read_csv(TARGET_IN)
    target_df = target_df_full.set_index('battle_id')
    aligned_target = target_df.loc[train_ids]['player_won'].values.astype(int)
except Exception as e: print(f"❌ ERRORE caricamento/allineamento target: {e}"); exit()

# Scaling Numerico Train
print("Applicazione StandardScaler (Train)...")
nsamples, nsteps, nfeatures = train_num_seq.shape
if nsamples * nsteps == 0: print("❌ ERRORE: Dati numerici train vuoti."); exit()
train_num_2d = train_num_seq.reshape((nsamples * nsteps, nfeatures))
scaler = StandardScaler()
train_num_2d_scaled = scaler.fit_transform(train_num_2d)
train_num_scaled = train_num_2d_scaled.reshape((nsamples, nsteps, nfeatures))
print("Scaling Train completato.")

# Processa il Test Set
test_num_seq, test_cat_seq, test_ids, _ = process_timelines_advanced(
    TIMELINE_TEST_IN, pokedex_with_stats, is_train=False, encoders=encoders
)
if test_num_seq is None: exit()

# Scaling Numerico Test
print("Applicazione StandardScaler (Test)...")
nsamples_test, nsteps_test, nfeatures_test = test_num_seq.shape
if nsamples_test * nsteps_test == 0: print("❌ ERRORE: Dati numerici test vuoti."); exit()
# Assicurati che il numero di feature sia lo stesso (potrebbe fallire se una colonna era tutta NaN nel train)
if nfeatures_test != nfeatures:
    print(f"❌ ERRORE: Mismatch numero feature tra Train ({nfeatures}) e Test ({nfeatures_test}) prima dello scaling!")
    exit()
test_num_2d = test_num_seq.reshape((nsamples_test * nsteps_test, nfeatures_test))
try:
    test_num_2d_scaled = scaler.transform(test_num_2d) # Usa scaler fittato
except ValueError as ve:
     print(f"❌ ERRORE durante lo scaling del Test set: {ve}")
     print("   Questo può accadere se il numero di feature non corrisponde esattamente.")
     exit()

test_num_scaled = test_num_2d_scaled.reshape((nsamples_test, nsteps_test, nfeatures_test))
print("Scaling Test Set completato.")

# --- 6. Salvataggio ---
print(f"Salvataggio file finali in {LSTM_DATA_DIR}...")
try:
    np.save(TRAIN_NUMERIC_OUT, train_num_scaled)
    np.save(TRAIN_CATEGORICAL_OUT, train_cat_seq)
    np.save(TARGET_OUT, aligned_target)
    np.save(TRAIN_IDS_OUT, train_ids)
    np.save(TEST_NUMERIC_OUT, test_num_scaled)
    np.save(TEST_CATEGORICAL_OUT, test_cat_seq)
    np.save(TEST_IDS_OUT, test_ids)
    with open(ENCODERS_OUT, 'w') as f:
        encoders_serializable = {}
        for key, value in encoders.items():
            encoders_serializable[key] = {'classes': [str(cls) for cls in value['classes']], 'vocab_size': value['vocab_size']}
        json.dump(encoders_serializable, f, indent=2)
    print("Salvataggio completato.")
except Exception as e: print(f"❌ ERRORE durante il salvataggio: {e}"); exit()

print("\nPreprocessing LSTM (SUPER-AVANZATO v3) completato con successo.")
print(f"Train Numeric Shape: {train_num_scaled.shape} (Features: {train_num_scaled.shape[2]})")
print(f"Train Categorical Shape: {train_cat_seq.shape} (Features: {train_cat_seq.shape[2]})")
print(f"Target Shape: {aligned_target.shape}")
print(f"Train IDs Shape: {train_ids.shape}")
print(f"Test Numeric Shape: {test_num_scaled.shape}")
print(f"Test Categorical Shape: {test_cat_seq.shape}")
print(f"Test IDs Shape: {test_ids.shape}")
print(f"\nFile .npy, encoders.json e pokedex_with_stats.json salvati in '{LSTM_DATA_DIR}'.")