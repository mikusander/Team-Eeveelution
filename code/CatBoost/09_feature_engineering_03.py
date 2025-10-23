"""
This script adds dynamic type-effectiveness features to Pokémon battle data (v3). 
It calculates turn-by-turn move effectiveness for each Pokémon based on their types,
aggregates these metrics per battle, and merges them with existing feature sets (v2) 
to produce final feature sets for training and testing.
"""

import pandas as pd
import numpy as np
import os

CSV_DIR = 'Output_CSVs'      
FEATURES_V2_DIR = 'Features_v2' 
FEATURES_V3_DIR = 'Features_v3' 
os.makedirs(FEATURES_V3_DIR, exist_ok=True)

print(f"Final v3 feature files will be saved in: {FEATURES_V3_DIR}")

TYPE_EFFECTIVENESS = {
    'normal': {'rock': 0.5, 'ghost': 0.0},
    'fire': {'fire': 0.5, 'water': 0.5, 'grass': 2.0, 'ice': 2.0, 'bug': 2.0, 'rock': 0.5, 'dragon': 0.5},
    'water': {'fire': 2.0, 'water': 0.5, 'grass': 0.5, 'ground': 2.0, 'rock': 2.0, 'dragon': 0.5},
    'electric': {'water': 2.0, 'electric': 0.5, 'grass': 0.5, 'ground': 0.0, 'flying': 2.0, 'dragon': 0.5},
    'grass': {'fire': 0.5, 'water': 2.0, 'electric': 0.5, 'grass': 0.5, 'poison': 0.5, 'ground': 2.0, 'flying': 0.5, 'bug': 0.5, 'rock': 2.0, 'dragon': 0.5},
    'ice': {'fire': 0.5, 'water': 0.5, 'grass': 2.0, 'ice': 0.5, 'ground': 2.0, 'flying': 2.0, 'dragon': 2.0},
    'fighting': {'normal': 2.0, 'ice': 2.0, 'poison': 0.5, 'flying': 0.5, 'psychic': 0.5, 'bug': 0.5, 'rock': 2.0, 'ghost': 0.0},
    'poison': {'grass': 2.0, 'poison': 0.5, 'ground': 0.5, 'bug': 2.0, 'rock': 0.5, 'ghost': 0.5},
    'ground': {'fire': 2.0, 'electric': 2.0, 'grass': 0.5, 'poison': 2.0, 'flying': 0.0, 'bug': 0.5, 'rock': 2.0},
    'flying': {'electric': 0.5, 'grass': 2.0, 'fighting': 2.0, 'bug': 2.0, 'rock': 0.5},
    'psychic': {'fighting': 2.0, 'poison': 2.0, 'psychic': 0.5, 'ghost': 0.0},
    'bug': {'fire': 0.5, 'grass': 2.0, 'fighting': 0.5, 'poison': 2.0, 'flying': 0.5, 'psychic': 2.0},
    'rock': {'fire': 2.0, 'ice': 2.0, 'fighting': 0.5, 'ground': 0.5, 'flying': 2.0, 'bug': 2.0},
    'ghost': {'normal': 0.0, 'psychic': 0.0, 'ghost': 2.0},
    'dragon': {'dragon': 2.0},
    'notype': {}, 'none': {}
} 

# Function to compute type effectiveness multiplier
def get_effectiveness(move_type, target_types):
    if move_type is None or pd.isna(move_type) or move_type in ['notype', 'none']:
        return np.nan 
    
    effectiveness_map = TYPE_EFFECTIVENESS.get(str(move_type).lower(), {})
    
    multiplier = 1.0
    for target_type in target_types:
        if target_type is not None and not pd.isna(target_type) and target_type not in ['notype', 'none']:
            multiplier *= effectiveness_map.get(str(target_type).lower(), 1.0)
            
    return multiplier

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
                pokedex[name] = [row[type1_col], row[type2_col]]
                
    p2_lead_data = static_train_df[['p2_lead.name', 'p2_lead.type1', 'p2_lead.type2']].drop_duplicates()
    for _, row in p2_lead_data.iterrows():
        name = row['p2_lead.name']
        if name not in pokedex:
            pokedex[name] = [row['p2_lead.type1'], row['p2_lead.type2']]

    print(f"Pokédex built. Contains {len(pokedex)} unique Pokémon.")
    return pokedex

# Function to compute dynamic type effectiveness features per battle
def process_timeline_effectiveness(timeline_df, pokedex):
    print("Starting turn-by-turn dynamic effectiveness calculation...")
    
    timeline_df['p1_active_types'] = timeline_df['p1_pokemon_state.name'].apply(lambda x: pokedex.get(x, [None, None]))
    timeline_df['p2_active_types'] = timeline_df['p2_pokemon_state.name'].apply(lambda x: pokedex.get(x, [None, None]))
    
    print("... calculating p1_move_effectiveness")
    timeline_df['p1_move_effectiveness'] = timeline_df.apply(
        lambda row: get_effectiveness(row['p1_move_details.type'], row['p2_active_types']),
        axis=1
    )
    
    print("... calculating p2_move_effectiveness")
    timeline_df['p2_move_effectiveness'] = timeline_df.apply(
        lambda row: get_effectiveness(row['p2_move_details.type'], row['p1_active_types']),
        axis=1
    )
    
    print("Effectiveness calculation completed. Starting aggregation...")
    
    aggregations = {
        'p1_move_effectiveness': ['mean', lambda x: (x > 1.0).sum(), lambda x: (x < 1.0).sum()],
        'p2_move_effectiveness': ['mean', lambda x: (x > 1.0).sum(), lambda x: (x < 1.0).sum()],
    }
    
    dynamic_type_features = timeline_df.groupby('battle_id').agg(aggregations)
    
    dynamic_type_features.columns = [
        'p1_avg_effectiveness', 'p1_super_effective_hits', 'p1_resisted_hits',
        'p2_avg_effectiveness', 'p2_super_effective_hits', 'p2_resisted_hits'
    ]
    
    dynamic_type_features['p1_avg_effectiveness'] = dynamic_type_features['p1_avg_effectiveness'].fillna(1.0)
    dynamic_type_features['p2_avg_effectiveness'] = dynamic_type_features['p2_avg_effectiveness'].fillna(1.0)
    dynamic_type_features = dynamic_type_features.fillna(0) # For hit counts

    print("Dynamic type feature aggregation completed.")
    return dynamic_type_features

pokedex = build_pokedex()

# Process TRAIN and TEST datasets
print("\n   TRAINING PROCESS:\n")
print("Loading TRAIN timeline...")
timeline_train = pd.read_csv(os.path.join(CSV_DIR, 'timelines_train_dynamic.csv'))
dynamic_type_features_train = process_timeline_effectiveness(timeline_train, pokedex)

print("Loading TRAIN features_v2...")
features_train_v2 = pd.read_csv(os.path.join(FEATURES_V2_DIR, 'features_train_v2.csv'))
features_train_v3 = pd.merge(features_train_v2, dynamic_type_features_train, on='battle_id', how='left')
features_train_v3.to_csv(os.path.join(FEATURES_V3_DIR, 'features_train_v3.csv'), index=False)
print(f"Final TRAIN v3 file saved with {features_train_v3.shape[1]} columns.")

print("\n   TEST PROCESS:\n")
print("Loading TEST timeline...")
timeline_test = pd.read_csv(os.path.join(CSV_DIR, 'timelines_test_dynamic.csv'))
dynamic_type_features_test = process_timeline_effectiveness(timeline_test, pokedex)

print("Loading TEST features_v2...")
features_test_v2 = pd.read_csv(os.path.join(FEATURES_V2_DIR, 'features_test_v2.csv'))
features_test_v3 = pd.merge(features_test_v2, dynamic_type_features_test, on='battle_id', how='left')
features_test_v3.to_csv(os.path.join(FEATURES_V3_DIR, 'features_test_v3.csv'), index=False)
print(f"Final TEST v3 file saved with {features_test_v3.shape[1]} columns.")

print("\n06_feature_engineering_03.py executed successfully.")