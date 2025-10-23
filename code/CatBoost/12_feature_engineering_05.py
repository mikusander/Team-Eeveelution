"""
This script adds expert-level features to the Pokémon battle dataset, including STAB moves, 
status moves, and healing moves. It calculates these features per turn, aggregates them per battle,
and merges them with the final feature sets to produce training and test expert feature datasets.
"""

import pandas as pd
import numpy as np
import os

CSV_DIR = 'Output_CSVs'       
FEATURES_FINAL_DIR = 'Features_v4' 
FEATURES_EXPERT_DIR = 'Features_v5' 
os.makedirs(FEATURES_EXPERT_DIR, exist_ok=True)

print(f"Expert feature files will be saved in: {FEATURES_EXPERT_DIR}")

# STATUS_MOVES = ['thunderwave', 'sleeppowder', 'stunspore', 'sing', 'lovelykiss', 'spore']
# HEALING_MOVES = ['recover', 'softboiled', 'rest']

try:
    from move_effects_final import MOVE_EFFECTS_DETAILED
except ImportError:
    print("ERRORE: File 'move_effects_final.py' non trovato.")
    print("Assicurati di aver eseguito prima gli script 'counter.py' e 'filter.py'.")
    MOVE_EFFECTS_DETAILED = {}

# 2. Inizializza le liste vuote
STATUS_MOVES = []
HEALING_MOVES = []

# 3. Scorre il dizionario e popola le liste
for move_name, effects_list in MOVE_EFFECTS_DETAILED.items():
    
    # Se 'opponent_status' è negli effetti, aggiungilo alla lista STATUS
    if 'opponent_status' in effects_list:
        STATUS_MOVES.append(move_name)
        
    # Se 'healing' è negli effetti, aggiungilo alla lista HEALING
    if 'healing' in effects_list:
        HEALING_MOVES.append(move_name)

print(STATUS_MOVES)
print(HEALING_MOVES)

# Function to build a Pokédex dictionary from static battle data
def build_pokedex():
    print("Building the Pokédex...")
    static_train_df = pd.read_csv(os.path.join(CSV_DIR, 'battles_train_static.csv'))
    pokedex = {}
    
    for i in range(6):
        name_col = f'p1_team.{i}.name'
        type1_col = f'p1_team.{i}.type1'
        type2_col = f'p1_team.{i}.type2'
        
        pkmn_data = static_train_df[[name_col, type1_col, type2_col]].drop_duplicates()
        for _, row in pkmn_data.iterrows():
            name = row[name_col]
            if name not in pokedex:
                pokedex[name] = [str(row[type1_col]).lower(), str(row[type2_col]).lower()]
                
    p2_lead_data = static_train_df[['p2_lead.name', 'p2_lead.type1', 'p2_lead.type2']].drop_duplicates()
    for _, row in p2_lead_data.iterrows():
        name = row['p2_lead.name']
        if name not in pokedex:
            pokedex[name] = [str(row['p2_lead.type1']).lower(), str(row['p2_lead.type2']).lower()]

    print(f"Pokédex built. Contains {len(pokedex)} unique Pokémon.")
    return pokedex

# Function to compute per-turn and aggregated expert features
def process_timeline_expert_features(timeline_df, pokedex):
    print("Starting calculation of expert features...")
    
    timeline_df['p1_active_types'] = timeline_df['p1_pokemon_state.name'].apply(lambda x: pokedex.get(x, [None, None]))
    timeline_df['p2_active_types'] = timeline_df['p2_pokemon_state.name'].apply(lambda x: pokedex.get(x, [None, None]))
        
    def calculate_turn_features(row):
        move_name_p1 = str(row['p1_move_details.name']).lower()
        move_type_p1 = str(row['p1_move_details.type']).lower()
        active_types_p1 = row['p1_active_types']
        
        move_name_p2 = str(row['p2_move_details.name']).lower()
        move_type_p2 = str(row['p2_move_details.type']).lower()
        active_types_p2 = row['p2_active_types']

        p1_is_stab = 1 if (move_type_p1 in active_types_p1) and (move_type_p1 not in ['nan', 'none']) else 0
        p2_is_stab = 1 if (move_type_p2 in active_types_p2) and (move_type_p2 not in ['nan', 'none']) else 0
        
        p1_is_status_move = 1 if move_name_p1 in STATUS_MOVES else 0
        p1_is_healing_move = 1 if move_name_p1 in HEALING_MOVES else 0
        
        p2_is_status_move = 1 if move_name_p2 in STATUS_MOVES else 0
        p2_is_healing_move = 1 if move_name_p2 in HEALING_MOVES else 0
        
        return pd.Series([
            p1_is_stab, p2_is_stab,
            p1_is_status_move, p1_is_healing_move,
            p2_is_status_move, p2_is_healing_move
        ])

    print("... applying per-turn calculations...")
    new_features_per_turn = timeline_df.apply(calculate_turn_features, axis=1)
    new_features_per_turn.columns = [
        'p1_is_stab', 'p2_is_stab',
        'p1_is_status_move', 'p1_is_healing_move',
        'p2_is_status_move', 'p2_is_healing_move'
    ]
    
    timeline_df = pd.concat([timeline_df, new_features_per_turn], axis=1)
    
    print("Per-turn calculation completed. Starting aggregation...")
    
    aggregations = {
        'p1_is_stab': 'sum',
        'p2_is_stab': 'sum',
        'p1_is_status_move': 'sum',
        'p1_is_healing_move': 'sum',
        'p2_is_status_move': 'sum',
        'p2_is_healing_move': 'sum'
    }
    
    expert_features_df = timeline_df.groupby('battle_id').agg(aggregations)
    
    expert_features_df = expert_features_df.rename(columns={
        'p1_is_stab': 'p1_stab_move_count',
        'p2_is_stab': 'p2_stab_move_count',
        'p1_is_status_move': 'p1_status_move_count',
        'p1_is_healing_move': 'p1_healing_move_count',
        'p2_is_status_move': 'p2_status_move_count',
        'p2_is_healing_move': 'p2_healing_move_count'
    })

    expert_features_df['stab_delta'] = expert_features_df['p1_stab_move_count'] - expert_features_df['p2_stab_move_count']
    expert_features_df['status_move_delta'] = expert_features_df['p1_status_move_count'] - expert_features_df['p2_status_move_count']
    expert_features_df['healing_move_delta'] = expert_features_df['p1_healing_move_count'] - expert_features_df['p2_healing_move_count']

    print("Expert feature aggregation completed.")
    return expert_features_df

# Process TRAIN and TEST datasets, merge with final features, and save expert feature CSVs
pokedex = build_pokedex()

print("\n   TRAINING PROCESS:\n")
print("Loading TRAIN timeline...")
timeline_train = pd.read_csv(os.path.join(CSV_DIR, 'timelines_train_dynamic.csv'))
expert_features_train = process_timeline_expert_features(timeline_train, pokedex)

print("Loading TRAIN final features...")
features_train_final = pd.read_csv(os.path.join(FEATURES_FINAL_DIR, 'features_final_train.csv'))
features_train_expert = pd.merge(features_train_final, expert_features_train, on='battle_id', how='left')
features_train_expert.to_csv(os.path.join(FEATURES_EXPERT_DIR, 'features_expert_train.csv'), index=False)
print(f"Final TRAIN expert file saved with {features_train_expert.shape[1]} columns.")

print("\n   TEST PROCESS:\n")
print("Loading TEST timeline...")
timeline_test = pd.read_csv(os.path.join(CSV_DIR, 'timelines_test_dynamic.csv'))
expert_features_test = process_timeline_expert_features(timeline_test, pokedex)

print("Loading TEST final features...")
features_test_final = pd.read_csv(os.path.join(FEATURES_FINAL_DIR, 'features_final_test.csv'))
features_test_expert = pd.merge(features_test_final, expert_features_test, on='battle_id', how='left')
features_test_expert.to_csv(os.path.join(FEATURES_EXPERT_DIR, 'features_expert_test.csv'), index=False)
print(f"Final TEST expert file saved with {features_test_expert.shape[1]} columns.")

print("\nFeature engineering process completed.")
print("\n10_feature_engineering_05.py executed successfully.")