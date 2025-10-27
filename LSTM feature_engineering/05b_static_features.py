# Nome file: 05b_lstm_add_static_features.py
"""
Script OPZIONALE (versione autonoma) per CALCOLARE e aggiungere
feature statiche/aggregate ai dati preprocessati per LSTM.
Legge i dati RAW (battles_static, timelines_dynamic) e i metadati,
calcola le feature selezionate, le allinea con l'ordine delle sequenze LSTM,
le scala e le salva come .npy.
"""

import pandas as pd
import numpy as np
import os
import json
import joblib # Per salvare scaler/imputer
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

print("Avvio Script 05b (Autonomo): Calcolo Feature Statiche per LSTM...")

# --- 1. Configurazione ---
# Input RAW CSVs
RAW_CSV_DIR = 'Output_CSVs'
BATTLES_TRAIN_STATIC_IN = os.path.join(RAW_CSV_DIR, 'battles_train_static.csv')
TIMELINES_TRAIN_DYNAMIC_IN = os.path.join(RAW_CSV_DIR, 'timelines_train_dynamic.csv')
BATTLES_TEST_STATIC_IN = os.path.join(RAW_CSV_DIR, 'battles_test_static.csv')
TIMELINES_TEST_DYNAMIC_IN = os.path.join(RAW_CSV_DIR, 'timelines_test_dynamic.csv')

# Input Metadati (da script 01, 02, 03)
try:
    from move_effects_final import MOVE_EFFECTS_DETAILED
    print("Caricato MOVE_EFFECTS_DETAILED.")
except ImportError:
    print("❌ ERRORE: File 'move_effects_final.py' non trovato.")
    print("Esegui prima gli script '01_counter.py' e '02_filter.py'.")
    exit()
try:
    # Cerchiamo 'STATUSES' ma gestiamo se si chiama ancora 'NEGATIVE_STATUSES'
    try:
         from unique_statuses import STATUSES as UNIQUE_STATUSES_SET
    except ImportError:
         from unique_statuses import NEGATIVE_STATUSES as UNIQUE_STATUSES_SET # Fallback
    NEGATIVE_STATUSES = list(UNIQUE_STATUSES_SET) # Converti in lista per isin()
    print(f"Caricati {len(NEGATIVE_STATUSES)} NEGATIVE_STATUSES.")
except ImportError:
    print("❌ ERRORE: File 'unique_statuses.py' non trovato.")
    print("Esegui prima lo script '03_process_status.py'.")
    exit()

# Input da preprocessing LSTM (per ordine ID)
LSTM_PREPROCESSED_DIR_IN = 'Preprocessed_LSTM'
LSTM_TRAIN_IDS_FILE = os.path.join(LSTM_PREPROCESSED_DIR_IN, 'train_ids.npy')
LSTM_TEST_IDS_FILE = os.path.join(LSTM_PREPROCESSED_DIR_IN, 'test_ids.npy')
LSTM_TARGET_IN_FILE = os.path.join(LSTM_PREPROCESSED_DIR_IN, 'target.npy') # Per copiarlo

# Output (nuova cartella per i dati ibridi)
LSTM_HYBRID_DIR_OUT = 'Preprocessed_LSTM_Hybrid'
os.makedirs(LSTM_HYBRID_DIR_OUT, exist_ok=True)

# File di output statici .npy
OUTPUT_TRAIN_STATIC_FILE = os.path.join(LSTM_HYBRID_DIR_OUT, 'train_static_features.npy')
OUTPUT_TEST_STATIC_FILE = os.path.join(LSTM_HYBRID_DIR_OUT, 'test_static_features.npy')
# Copieremo anche target e IDs qui
OUTPUT_TARGET_FILE = os.path.join(LSTM_HYBRID_DIR_OUT, 'target.npy')
OUTPUT_TRAIN_IDS_FILE = os.path.join(LSTM_HYBRID_DIR_OUT, 'train_ids.npy')
OUTPUT_TEST_IDS_FILE = os.path.join(LSTM_HYBRID_DIR_OUT, 'test_ids.npy')

SCALER_FILE = os.path.join(LSTM_HYBRID_DIR_OUT, 'static_features_scaler.joblib')
IMPUTER_FILE = os.path.join(LSTM_HYBRID_DIR_OUT, 'static_features_imputer.joblib')

# Costanti (da script 07 e 14)
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

# Liste mosse (da script 13)
STATUS_MOVES = [m for m, e in MOVE_EFFECTS_DETAILED.items() if 'opponent_status' in e]
HEALING_MOVES = [m for m, e in MOVE_EFFECTS_DETAILED.items() if 'healing' in e]

# --- 2. Funzioni Helper ---

# Da 07_feature_engineering_02.py e 14_feature_engineering_06.py
def get_effectiveness(move_type, target_types):
    """Calcola l'efficacia di un tipo contro uno o due tipi difensivi."""
    move_type = str(move_type).lower()
    if move_type in ['notype', 'none', 'nan']:
        # Se la mossa non ha tipo, l'efficacia è neutra (o NaN se lo preferisci per il calcolo dinamico)
        return 1.0 # Ritorniamo 1.0 per coerenza con calcoli statici
        # return np.nan # Usare NaN per calcoli dinamici (script 09)

    effectiveness_map = TYPE_EFFECTIVENESS.get(move_type, {})
    multiplier = 1.0
    valid_target_types_found = False
    for target_type in target_types:
        target_type = str(target_type).lower()
        if target_type not in ['notype', 'none', 'nan']:
            multiplier *= effectiveness_map.get(target_type, 1.0)
            valid_target_types_found = True

    # Se il target non ha tipi validi (es. solo 'none'), l'efficacia è neutra
    return multiplier if valid_target_types_found else 1.0

# Da 09_feature_engineering_03.py e 13_feature_engineering_05.py
def build_pokedex(static_battles_df):
    """Costruisce un Pokédex {nome: [tipo1, tipo2]} dal DataFrame statico."""
    print("Building the Pokédex...")
    pokedex = {}
    # Team P1
    for i in range(6):
        name_col, type1_col, type2_col = f'p1_team.{i}.name', f'p1_team.{i}.type1', f'p1_team.{i}.type2'
        if name_col in static_battles_df.columns:
            pkmn_data = static_battles_df[[name_col, type1_col, type2_col]].dropna(subset=[name_col]).drop_duplicates()
            for _, row in pkmn_data.iterrows():
                name = row[name_col]
                if name not in pokedex:
                    pokedex[name] = [str(row[type1_col]).lower(), str(row[type2_col]).lower()]
    # Lead P2
    if 'p2_lead.name' in static_battles_df.columns:
        p2_lead_data = static_battles_df[['p2_lead.name', 'p2_lead.type1', 'p2_lead.type2']].dropna(subset=['p2_lead.name']).drop_duplicates()
        for _, row in p2_lead_data.iterrows():
            name = row['p2_lead.name']
            if name not in pokedex:
                pokedex[name] = [str(row['p2_lead.type1']).lower(), str(row['p2_lead.type2']).lower()]
    print(f"Pokédex built. Contains {len(pokedex)} unique Pokémon.")
    return pokedex

# --- 3. Funzione Principale di Calcolo Feature ---

def calculate_static_features(battles_static_df, timelines_dynamic_df):
    """
    Calcola TUTTE le feature statiche/aggregate selezionate
    partendo dai DataFrame raw. Restituisce un DataFrame indicizzato per battle_id.
    """
    print("\nInizio Calcolo Feature Statiche Complessivo...")

    # --- A. Preparazione Dati ---
    # Costruisci Pokedex (necessario per STAB, Efficacia Dinamica)
    pokedex = build_pokedex(battles_static_df)

    # Prepara timeline per aggregazioni
    timelines = timelines_dynamic_df.sort_values(by=['battle_id', 'turn']).copy()
    # Aggiungi tipi attivi usando il pokedex per STAB ed efficacia
    timelines['p1_active_types'] = timelines['p1_pokemon_state.name'].apply(lambda x: pokedex.get(x, [None, None]))
    timelines['p2_active_types'] = timelines['p2_pokemon_state.name'].apply(lambda x: pokedex.get(x, [None, None]))

    # --- B. Calcolo Feature Aggregate Dinamiche (per battaglia) ---
    print("Calcolo aggregati dinamici (faint, status, moves, effectiveness)...")
    aggregated_features = defaultdict(dict) # {battle_id: {feature_name: value}}

    # Calcoli pre-raggruppamento (status, stab, move types)
    timelines['p1_has_status_turn'] = timelines['p1_pokemon_state.status'].isin(NEGATIVE_STATUSES).astype(int)
    timelines['p2_has_status_turn'] = timelines['p2_pokemon_state.status'].isin(NEGATIVE_STATUSES).astype(int)

    timelines['p1_move_type_str'] = timelines['p1_move_details.type'].fillna('none').astype(str).str.lower()
    timelines['p2_move_type_str'] = timelines['p2_move_details.type'].fillna('none').astype(str).str.lower()
    timelines['p1_move_name_str'] = timelines['p1_move_details.name'].fillna('none').astype(str).str.lower()
    timelines['p2_move_name_str'] = timelines['p2_move_details.name'].fillna('none').astype(str).str.lower()

    timelines['p1_is_stab'] = timelines.apply(lambda r: 1 if (r['p1_move_type_str'] in r['p1_active_types'] and r['p1_move_type_str'] != 'none') else 0, axis=1)
    timelines['p2_is_stab'] = timelines.apply(lambda r: 1 if (r['p2_move_type_str'] in r['p2_active_types'] and r['p2_move_type_str'] != 'none') else 0, axis=1)
    timelines['p1_is_status_move'] = timelines['p1_move_name_str'].isin(STATUS_MOVES).astype(int)
    timelines['p2_is_status_move'] = timelines['p2_move_name_str'].isin(STATUS_MOVES).astype(int)
    timelines['p1_is_healing_move'] = timelines['p1_move_name_str'].isin(HEALING_MOVES).astype(int)
    timelines['p2_is_healing_move'] = timelines['p2_move_name_str'].isin(HEALING_MOVES).astype(int)

    # Calcolo efficacia dinamica (come da script 09)
    # Usiamo NaN qui per la media, ma fillna(1.0) dopo
    timelines['p1_effectiveness_raw'] = timelines.apply(lambda r: get_effectiveness(r['p1_move_details.type'], r['p2_active_types']) if pd.notna(r['p1_move_details.type']) else np.nan, axis=1)
    timelines['p2_effectiveness_raw'] = timelines.apply(lambda r: get_effectiveness(r['p2_move_details.type'], r['p1_active_types']) if pd.notna(r['p2_move_details.type']) else np.nan, axis=1)

    # Funzione per aggregare una singola battaglia (adattata da script 05 e altri)
    def aggregate_single_battle(group):
        battle_results = {}
        # Faint Count (robusto, conta nomi unici che vanno KO)
        p1_fainted_names = set(group.loc[group['p1_pokemon_state.status'] == 'fnt', 'p1_pokemon_state.name'].dropna().unique())
        p2_fainted_names = set(group.loc[group['p2_pokemon_state.status'] == 'fnt', 'p2_pokemon_state.name'].dropna().unique())
        battle_results['p1_fainted_count'] = len(p1_fainted_names)
        battle_results['p2_fainted_count'] = len(p2_fainted_names)
        battle_results['faint_delta'] = battle_results['p1_fainted_count'] - battle_results['p2_fainted_count']

        # Status Turns
        battle_results['p1_total_status_turns'] = group['p1_has_status_turn'].sum()
        battle_results['p2_total_status_turns'] = group['p2_has_status_turn'].sum()
        battle_results['status_turns_delta'] = battle_results['p1_total_status_turns'] - battle_results['p2_total_status_turns']

        # Move Counts
        battle_results['p1_stab_move_count'] = group['p1_is_stab'].sum()
        battle_results['p2_stab_move_count'] = group['p2_is_stab'].sum()
        battle_results['stab_delta'] = battle_results['p1_stab_move_count'] - battle_results['p2_stab_move_count']

        battle_results['p1_status_move_count'] = group['p1_is_status_move'].sum()
        battle_results['p2_status_move_count'] = group['p2_is_status_move'].sum()
        battle_results['status_move_delta'] = battle_results['p1_status_move_count'] - battle_results['p2_status_move_count']

        battle_results['p1_healing_move_count'] = group['p1_is_healing_move'].sum()
        battle_results['p2_healing_move_count'] = group['p2_is_healing_move'].sum()
        battle_results['healing_move_delta'] = battle_results['p1_healing_move_count'] - battle_results['p2_healing_move_count']

        # Effectiveness
        battle_results['p1_avg_effectiveness'] = group['p1_effectiveness_raw'].mean() # Media include NaN
        battle_results['p2_avg_effectiveness'] = group['p2_effectiveness_raw'].mean()
        battle_results['p1_super_effective_hits'] = (group['p1_effectiveness_raw'] > 1.0).sum()
        battle_results['p2_super_effective_hits'] = (group['p2_effectiveness_raw'] > 1.0).sum()
        # FillNa DOPO aver calcolato le somme
        battle_results['p1_avg_effectiveness'] = battle_results['p1_avg_effectiveness'] if pd.notna(battle_results['p1_avg_effectiveness']) else 1.0
        battle_results['p2_avg_effectiveness'] = battle_results['p2_avg_effectiveness'] if pd.notna(battle_results['p2_avg_effectiveness']) else 1.0
        # Calcola Delta Efficacia
        battle_results['effectiveness_delta'] = battle_results['p1_avg_effectiveness'] - battle_results['p2_avg_effectiveness']
        battle_results['super_effective_delta'] = battle_results['p1_super_effective_hits'] - battle_results['p2_super_effective_hits']

        # Comeback KOs (Logica complessa da script 05)
        p1_damage = 0.0
        p2_damage = 0.0
        last_p1_hp = {}
        last_p2_hp = {}
        prev_p1_fainted = 0
        prev_p2_fainted = 0
        p1_comeback_kos = 0
        p2_comeback_kos = 0
        current_p1_fainted_count = 0
        current_p2_fainted_count = 0
        p1_fainted_names_iter = set()
        p2_fainted_names_iter = set()

        for _, turn in group.iterrows():
            # Conteggio KO istantaneo
            p1_name, p1_hp, p1_status = turn['p1_pokemon_state.name'], turn['p1_pokemon_state.hp_pct'], turn['p1_pokemon_state.status']
            p2_name, p2_hp, p2_status = turn['p2_pokemon_state.name'], turn['p2_pokemon_state.hp_pct'], turn['p2_pokemon_state.status']

            if p1_status == 'fnt' and p1_name not in p1_fainted_names_iter:
                current_p1_fainted_count += 1
                p1_fainted_names_iter.add(p1_name)
            if p2_status == 'fnt' and p2_name not in p2_fainted_names_iter:
                current_p2_fainted_count += 1
                p2_fainted_names_iter.add(p2_name)

            # Calcolo Danno
            if pd.notna(p2_name) and pd.notna(p2_hp):
                prev_hp = last_p2_hp.get(p2_name)
                if prev_hp is not None: p1_damage += max(0.0, prev_hp - p2_hp)
                last_p2_hp[p2_name] = p2_hp
            if pd.notna(p1_name) and pd.notna(p1_hp):
                prev_hp = last_p1_hp.get(p1_name)
                if prev_hp is not None: p2_damage += max(0.0, prev_hp - p1_hp)
                last_p1_hp[p1_name] = p1_hp

            # Calcolo Comeback
            damage_diff_so_far = p1_damage - p2_damage
            if current_p2_fainted_count > prev_p2_fainted and damage_diff_so_far < -1.0: p1_comeback_kos += 1
            if current_p1_fainted_count > prev_p1_fainted and damage_diff_so_far > 1.0: p2_comeback_kos += 1

            prev_p1_fainted = current_p1_fainted_count
            prev_p2_fainted = current_p2_fainted_count

        battle_results['p1_comeback_kos'] = p1_comeback_kos
        battle_results['p2_comeback_kos'] = p2_comeback_kos
        battle_results['comeback_kos_delta'] = p1_comeback_kos - p2_comeback_kos

        return pd.Series(battle_results)

    # Applica l'aggregazione a tutte le battaglie
    # Aggiunto include_groups=False per silenziare il warning e adottare il comportamento futuro
    dynamic_aggregates_df = timelines.groupby('battle_id').apply(aggregate_single_battle, include_groups=False)
    print("Aggregati dinamici calcolati.")

    # --- C. Calcolo Feature Statiche (Matchup e Team Aggregates) ---
    print("Calcolo feature statiche (matchup, team aggregates)...")
    static_features_list = []
    for battle_id, row in battles_static_df.set_index('battle_id').iterrows():
        static_calcs = {}
        # 1. Lead Matchup (da script 07)
        p1_lead_types = [row.get('p1_team.0.type1'), row.get('p1_team.0.type2')]
        p2_lead_types = [row.get('p2_lead.type1'), row.get('p2_lead.type2')]
        p1_offense_score = max(get_effectiveness(p1_lead_types[0], p2_lead_types), get_effectiveness(p1_lead_types[1], p2_lead_types))
        p2_offense_score = max(get_effectiveness(p2_lead_types[0], p1_lead_types), get_effectiveness(p2_lead_types[1], p1_lead_types))
        static_calcs['lead_offense_delta'] = p1_offense_score - p2_offense_score

        # 2. Team Counters vs Lead (da script 07)
        team_counters = 0
        for i in range(6):
            p1_pkmn_types = [row.get(f'p1_team.{i}.type1'), row.get(f'p1_team.{i}.type2')]
            if pd.notna(p1_pkmn_types[0]): # Solo se il Pokémon esiste
                pkmn_offense_score = max(get_effectiveness(p1_pkmn_types[0], p2_lead_types), get_effectiveness(p1_pkmn_types[1], p2_lead_types))
                if pkmn_offense_score > 1.0: team_counters += 1
        static_calcs['team_counters_vs_lead'] = team_counters

        # 3. Team Aggregates (da script 14)
        team_stats = []
        team_types_set = set()
        for i in range(6):
            stats = {'hp': row.get(f'p1_team.{i}.base_hp'), 'atk': row.get(f'p1_team.{i}.base_atk'), 'def': row.get(f'p1_team.{i}.base_def'),
                     'spa': row.get(f'p1_team.{i}.base_spa'), 'spd': row.get(f'p1_team.{i}.base_spd'), 'spe': row.get(f'p1_team.{i}.base_spe'),
                     'type1': row.get(f'p1_team.{i}.type1'), 'type2': row.get(f'p1_team.{i}.type2')}
            if pd.notna(stats['hp']) and pd.notna(stats['spe']): # Assicurati che le stats esistano
                team_stats.append(stats)
                if pd.notna(stats['type1']): team_types_set.add(stats['type1'])
                if pd.notna(stats['type2']) and stats['type2'] != 'none': team_types_set.add(stats['type2'])

        if team_stats:
            static_calcs['p1_avg_team_speed'] = np.mean([s['spe'] for s in team_stats])
            static_calcs['p1_total_team_hp'] = np.sum([s['hp'] for s in team_stats])
            static_calcs['p1_avg_team_atk'] = np.mean([max(s.get('atk', 0), s.get('spa', 0)) for s in team_stats]) # Usa get con default 0
            static_calcs['p1_avg_team_def'] = np.mean([(s.get('def', 0) + s.get('spd', 0)) / 2 for s in team_stats])
            team_types_set.discard('none'); team_types_set.discard(np.nan)
            static_calcs['p1_type_diversity'] = len(team_types_set)

            p1_team_weaknesses_count = 0
            attacking_types = [t for t in TYPE_EFFECTIVENESS.keys() if t not in ['notype', 'none']]
            for att_type in attacking_types:
                weak_pokemon_count = 0
                for pkmn in team_stats:
                    def_types = [pkmn['type1'], pkmn.get('type2', 'none')]
                    if get_effectiveness(att_type, def_types) > 1.0: weak_pokemon_count += 1
                if weak_pokemon_count >= 3: p1_team_weaknesses_count += 1
            static_calcs['p1_team_weaknesses_count'] = p1_team_weaknesses_count
        else: # Fallback se non ci sono dati team
             static_calcs.update({'p1_avg_team_speed': np.nan, 'p1_total_team_hp': np.nan,'p1_avg_team_atk': np.nan,
                                  'p1_avg_team_def': np.nan, 'p1_type_diversity': 0, 'p1_team_weaknesses_count': 0})

        static_features_list.append({'battle_id': battle_id, **static_calcs})

    static_features_df = pd.DataFrame(static_features_list).set_index('battle_id')
    print("Feature statiche calcolate.")

    # --- D. Unione Feature Dinamiche Aggregate e Statiche ---
    print("Unione di tutte le feature calcolate...")
    # Usa pd.concat invece di merge per unire per indice (battle_id)
    final_features_df = pd.concat([dynamic_aggregates_df, static_features_df], axis=1, join='inner') # join='inner' per sicurezza

    # Seleziona solo le colonne finali desiderate per il modello LSTM statico
    final_columns = [
        'faint_delta', 'comeback_kos_delta', 'lead_offense_delta', 'team_counters_vs_lead',
        'effectiveness_delta', 'super_effective_delta', 'status_turns_delta', 'stab_delta',
        'status_move_delta', 'healing_move_delta', 'p1_avg_team_speed', 'p1_total_team_hp',
        'p1_avg_team_atk', 'p1_avg_team_def', 'p1_type_diversity', 'p1_team_weaknesses_count'
    ]
    # Mantieni solo le colonne che esistono effettivamente dopo i calcoli
    final_columns_present = [col for col in final_columns if col in final_features_df.columns]
    final_features_df = final_features_df[final_columns_present]

    print(f"DataFrame finale creato con {final_features_df.shape[1]} feature statiche.")
    print(f"Colonne finali: {final_features_df.columns.tolist()}")

    return final_features_df


# --- 4. Funzione di Allineamento, Scaling e Salvataggio ---
def align_scale_save(static_features_df, lstm_ids_npy_path, output_npy_path, is_train=True, imputer=None, scaler=None):
    """Allinea, imputa, scala e salva le feature statiche calcolate."""
    print(f"\nAllineamento, Scaling e Salvataggio per {os.path.basename(output_npy_path)}...")
    try:
        lstm_ids = np.load(lstm_ids_npy_path, allow_pickle=True)
    except Exception as e:
        print(f"❌ ERRORE: Impossibile caricare il file IDs: {e}")
        return None, None, None

    # Allinea le righe all'ordine degli ID LSTM
    try:
        df_static_aligned = static_features_df.loc[lstm_ids]
    except KeyError:
        missing_ids = set(lstm_ids) - set(static_features_df.index)
        print(f"❌ ERRORE: {len(missing_ids)} battle_id dagli ID LSTM non trovati nelle feature calcolate.")
        print(f"   (Esempio mancanti: {list(missing_ids)[:5]})")
        print("   Questo potrebbe indicare un problema nel calcolo o nei file di input.")
        return None, None, None
    except Exception as e:
         print(f"❌ ERRORE durante l'allineamento: {e}")
         return None, None, None

    # Imputazione e Scaling
    if is_train:
        print("Applicazione SimpleImputer (median) e StandardScaler (fit_transform)...")
        imputer = SimpleImputer(strategy='median')
        data_imputed = imputer.fit_transform(df_static_aligned)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_imputed)
    else:
        if imputer is None or scaler is None:
             print("❌ ERRORE: Imputer o Scaler non forniti per il set di test.")
             return None, None, None
        print("Applicazione SimpleImputer (transform) e StandardScaler (transform)...")
        data_imputed = imputer.transform(df_static_aligned)
        data_scaled = scaler.transform(data_imputed)

    # Verifica NaN
    nan_count = np.isnan(data_scaled).sum()
    if nan_count > 0:
        print(f"⚠️ WARNING: Trovati {nan_count} NaN dopo scaling! Controlla imputazione.")
        # Potresti volerli sostituire con 0 per sicurezza
        # data_scaled = np.nan_to_num(data_scaled, nan=0.0)

    # Salva
    np.save(output_npy_path, data_scaled.astype(np.float32))
    print(f"Feature statiche processate salvate in: {output_npy_path} (Shape: {data_scaled.shape})")

    return data_scaled, imputer, scaler


# --- 5. Esecuzione Principale ---
print("\n--- Processing Training Data ---")
# Carica dati raw
try:
    battles_train_df = pd.read_csv(BATTLES_TRAIN_STATIC_IN)
    timelines_train_df = pd.read_csv(TIMELINES_TRAIN_DYNAMIC_IN)
except Exception as e:
    print(f"❌ ERRORE caricamento file raw di TRAIN: {e}")
    exit()
# Calcola feature statiche
train_static_features = calculate_static_features(battles_train_df, timelines_train_df)
# Allinea, scala, salva
train_static_scaled, fitted_imputer, fitted_scaler = align_scale_save(
    train_static_features, LSTM_TRAIN_IDS_FILE, OUTPUT_TRAIN_STATIC_FILE, is_train=True
)

if train_static_scaled is None: exit() # Esci se fallito

# Salva imputer e scaler
joblib.dump(fitted_imputer, IMPUTER_FILE)
joblib.dump(fitted_scaler, SCALER_FILE)
print(f"Imputer salvato in: {IMPUTER_FILE}")
print(f"Scaler salvato in: {SCALER_FILE}")


print("\n--- Processing Test Data ---")
# Carica dati raw
try:
    battles_test_df = pd.read_csv(BATTLES_TEST_STATIC_IN)
    timelines_test_df = pd.read_csv(TIMELINES_TEST_DYNAMIC_IN)
except Exception as e:
    print(f"❌ ERRORE caricamento file raw di TEST: {e}")
    exit()
# Calcola feature statiche
test_static_features = calculate_static_features(battles_test_df, timelines_test_df)
# Allinea, scala, salva
test_static_scaled, _, _ = align_scale_save(
    test_static_features, LSTM_TEST_IDS_FILE, OUTPUT_TEST_STATIC_FILE,
    is_train=False, imputer=fitted_imputer, scaler=fitted_scaler
)

if test_static_scaled is None: exit() # Esci se fallito


# --- 6. Copia file Target e IDs ---
print("\nCopia dei file target.npy e ids.npy nella cartella Hybrid...")
try:
    import shutil
    shutil.copyfile(LSTM_TARGET_IN_FILE, OUTPUT_TARGET_FILE)
    shutil.copyfile(LSTM_TRAIN_IDS_FILE, OUTPUT_TRAIN_IDS_FILE)
    shutil.copyfile(LSTM_TEST_IDS_FILE, OUTPUT_TEST_IDS_FILE)
    print("File copiati con successo.")
except Exception as e:
    print(f"⚠️ WARNING: Errore durante la copia dei file: {e}")

print("\n--- Script 05b (Autonomo) Completato ---")
print(f"Le feature statiche processate per LSTM si trovano in: {LSTM_HYBRID_DIR_OUT}")