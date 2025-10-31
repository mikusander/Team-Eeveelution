"""
This script performs feature engineering for Pokémon battle data. It creates dynamic features
by aggregating battle timelines, combines them with static battle features, and exports the
resulting training and test feature sets as CSV files for model training.
"""

import pandas as pd
import numpy as np
import os

INPUT_DIR = 'Output_CSVs'  
OUTPUT_DIR = 'Features_v1'    
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Output directory confirmed: {OUTPUT_DIR}")

BATTLES_TRAIN_STATIC_IN = os.path.join(INPUT_DIR, 'battles_train_static.csv')
TIMELINES_TRAIN_DYNAMIC_IN = os.path.join(INPUT_DIR, 'timelines_train_dynamic.csv')
BATTLES_TEST_STATIC_IN = os.path.join(INPUT_DIR, 'battles_test_static.csv')
TIMELINES_TEST_DYNAMIC_IN = os.path.join(INPUT_DIR, 'timelines_test_dynamic.csv')

FEATURES_TRAIN_OUT = os.path.join(OUTPUT_DIR, 'features_train.csv')
FEATURES_TEST_OUT = os.path.join(OUTPUT_DIR, 'features_test.csv')

# --- NUOVA FUNZIONE HELPER ---
# Questa funzione processa i turni di UNA singola battaglia
# e calcola tutte le feature aggregate.
def aggregate_battle_features(battle_turns_df):
    
    # --- Inizializzazione per feature 'comeback_kos' ---
    p1_damage = 0.0
    p2_damage = 0.0
    last_p1_hp = {}
    last_p2_hp = {}
    p1_fainted_names = set() # Per contare i KO in modo robusto
    p2_fainted_names = set()
    p1_fainted_count = 0
    p2_fainted_count = 0
    p1_comeback_kos = 0
    p2_comeback_kos = 0

    # --- Inizializzazione per feature ESISTENTI ---
    p1_names_used = set()
    p2_names_revealed = set()
    p1_hp_list = []
    p2_hp_list = []
    p1_boosts_total = 0
    p2_boosts_total = 0
    
    p1_last_hp = np.nan
    p2_last_hp = np.nan
    
    # Itera su ogni turno della battaglia
    for _, turn in battle_turns_df.iterrows():
        # Salva i conteggi KO del turno precedente
        prev_p1_fainted = p1_fainted_count
        prev_p2_fainted = p2_fainted_count

        # --- Logica per feature ESISTENTI ---
        p1_name = turn['p1_pokemon_state.name']
        p2_name = turn['p2_pokemon_state.name']
        p1_hp = turn['p1_pokemon_state.hp_pct']
        p2_hp = turn['p2_pokemon_state.hp_pct']
        
        if pd.notna(p1_name): p1_names_used.add(p1_name)
        if pd.notna(p2_name): p2_names_revealed.add(p2_name)
        if pd.notna(p1_hp): p1_hp_list.append(p1_hp)
        if pd.notna(p2_hp): p2_hp_list.append(p2_hp)
        
        p1_boosts_total += turn['p1_boosts_sum_turn']
        p2_boosts_total += turn['p2_boosts_sum_turn']
        
        # Salva l'HP per la feature 'last_hp'
        if pd.notna(p1_hp): p1_last_hp = p1_hp 
        if pd.notna(p2_hp): p2_last_hp = p2_hp

        # --- Logica per feature NUOVE (Danno, KO, Comeback) ---
        p1_status = turn['p1_pokemon_state.status']
        p2_status = turn['p2_pokemon_state.status']

        # Conteggio KO (controlla 'fnt' e che non sia già stato contato)
        if p1_status == 'fnt' and p1_name not in p1_fainted_names:
            p1_fainted_count += 1
            p1_fainted_names.add(p1_name)
        if p2_status == 'fnt' and p2_name not in p2_fainted_names:
            p2_fainted_count += 1
            p2_fainted_names.add(p2_name)

        # Calcolo Danno (basato su HP perso dal turno precedente *per quel Pokémon*)
        if pd.notna(p2_name) and pd.notna(p2_hp):
            prev_hp = last_p2_hp.get(p2_name)
            if prev_hp is not None:
                p1_damage += max(0.0, prev_hp - p2_hp)
            last_p2_hp[p2_name] = p2_hp

        if pd.notna(p1_name) and pd.notna(p1_hp):
            prev_hp = last_p1_hp.get(p1_name)
            if prev_hp is not None:
                p2_damage += max(0.0, prev_hp - p1_hp)
            last_p1_hp[p1_name] = p1_hp
            
        # Calcolo Logica 'Comeback'
        damage_diff_so_far = p1_damage - p2_damage
        if p2_fainted_count > prev_p2_fainted and damage_diff_so_far < -1.0: # P1 fa un KO mentre è in svantaggio di danno
            p1_comeback_kos += 1
        if p1_fainted_count > prev_p1_fainted and damage_diff_so_far > 1.0: # P2 fa un KO mentre è in svantaggio di danno
            p2_comeback_kos += 1
            
    # --- Compila i risultati per questa battaglia ---
    results = {
        'p1_pokemon_used_count': len(p1_names_used),
        'p2_pokemon_revealed_count': len(p2_names_revealed),
        'p1_fainted_count': p1_fainted_count,
        'p2_fainted_count': p2_fainted_count,
        'p1_avg_hp_pct': np.mean(p1_hp_list) if p1_hp_list else np.nan,
        'p2_avg_hp_pct': np.mean(p2_hp_list) if p2_hp_list else np.nan,
        'p1_hp_at_turn_30': p1_last_hp,
        'p2_hp_at_turn_30': p2_last_hp,
        'p1_total_boosts': p1_boosts_total,
        'p2_total_boosts': p2_boosts_total,
        # --- NUOVE FEATURE ---
        'p1_comeback_kos': p1_comeback_kos,
        'p2_comeback_kos': p2_comeback_kos
    }
    return pd.Series(results)


# --- FUNZIONE PRINCIPALE MODIFICATA ---
# Sostituisce la vecchia funzione create_dynamic_features
def create_dynamic_features(timelines_df):
    timelines_df = timelines_df.sort_values(by=['battle_id', 'turn'])
    
    # Calcoli pre-aggregazione (come prima)
    boost_cols_p1 = [col for col in timelines_df.columns if 'p1_pokemon_state.boosts' in col]
    boost_cols_p2 = [col for col in timelines_df.columns if 'p2_pokemon_state.boosts' in col]
    
    timelines_df['p1_boosts_sum_turn'] = timelines_df[boost_cols_p1].sum(axis=1)
    timelines_df['p2_boosts_sum_turn'] = timelines_df[boost_cols_p2].sum(axis=1)

    # --- NUOVA Logica di Aggregazione ---
    print("Starting advanced aggregation of dynamic features (incl. comeback_kos)...")
    # Applica la funzione helper a ogni gruppo (battle_id)
    dynamic_features_df = timelines_df.groupby('battle_id').apply(aggregate_battle_features)
    print("Advanced aggregation completed.")

    # Calcolo dei delta (come prima, più il nuovo delta)
    dynamic_features_df['faint_delta'] = dynamic_features_df['p1_fainted_count'] - dynamic_features_df['p2_fainted_count']
    dynamic_features_df['hp_avg_delta'] = dynamic_features_df['p1_avg_hp_pct'] - dynamic_features_df['p2_avg_hp_pct']
    dynamic_features_df['final_hp_delta'] = dynamic_features_df['p1_hp_at_turn_30'] - dynamic_features_df['p2_hp_at_turn_30']
    dynamic_features_df['total_boosts_delta'] = dynamic_features_df['p1_total_boosts'] - dynamic_features_df['p2_total_boosts']
    # --- NUOVO DELTA ---
    dynamic_features_df['comeback_kos_delta'] = dynamic_features_df['p1_comeback_kos'] - dynamic_features_df['p2_comeback_kos']

    print("Dynamic feature creation completed.")
    return dynamic_features_df

# Load training data and generate dynamic features
print("\n   TRAINING PROCESS:\n")
print(f"Loading training static data from {BATTLES_TRAIN_STATIC_IN}...")
battles_train_df = pd.read_csv(BATTLES_TRAIN_STATIC_IN)
print(f"Loading training dynamic timelines from {TIMELINES_TRAIN_DYNAMIC_IN}...")
timelines_train_df = pd.read_csv(TIMELINES_TRAIN_DYNAMIC_IN)

dynamic_features_train = create_dynamic_features(timelines_train_df)

# Merge static and dynamic features for training set
print("Merging static and dynamic features for TRAINING...")
features_train_df = pd.merge(
    battles_train_df, 
    dynamic_features_train, 
    on='battle_id', 
    how='left'
)
print(f"Final TRAINING DataFrame created with {features_train_df.shape[1]} columns.")

# Load test data and generate dynamic features
print("\n   TEST PROCESS:\n")
print(f"Loading test static data from {BATTLES_TEST_STATIC_IN}...")
battles_test_df = pd.read_csv(BATTLES_TEST_STATIC_IN)
print(f"Loading test dynamic timelines from {TIMELINES_TEST_DYNAMIC_IN}...")
timelines_test_df = pd.read_csv(TIMELINES_TEST_DYNAMIC_IN)

dynamic_features_test = create_dynamic_features(timelines_test_df)

# Merge static and dynamic features for test set
print("Merging static and dynamic features for TESTING...")
features_test_df = pd.merge(
    battles_test_df, 
    dynamic_features_test, 
    on='battle_id', 
    how='left'
)
print(f"Final TEST DataFrame created with {features_test_df.shape[1]} columns.")

# Save final feature sets to output directory
print("\nSaving final feature DataFrames...")
features_train_df.to_csv(FEATURES_TRAIN_OUT, index=False)
features_test_df.to_csv(FEATURES_TEST_OUT, index=False)

print("Save completed.\nThe following files have been created:")
print(f"  {FEATURES_TRAIN_OUT}")
print(f"  {FEATURES_TEST_OUT}")

print("\n02_feature_engineering_01.py executed successfully.")