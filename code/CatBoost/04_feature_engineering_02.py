"""
This script adds static type matchup features to Pokémon battle data. 
It calculates offensive type advantages for lead Pokémon and counts team members
that counter the opponent's lead, then saves the enhanced feature sets for training and testing.
"""
import pandas as pd
import numpy as np
import os

INPUT_DIR = 'Features'
OUTPUT_DIR = 'Features_v2' 
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES_TRAIN_IN = os.path.join(INPUT_DIR, 'features_train.csv')
FEATURES_TEST_IN = os.path.join(INPUT_DIR, 'features_test.csv')

FEATURES_TRAIN_OUT = os.path.join(OUTPUT_DIR, 'features_train_v2.csv')
FEATURES_TEST_OUT = os.path.join(OUTPUT_DIR, 'features_test_v2.csv')

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
    'notype': {},
    'none': {}
}

# Function to compute type effectiveness multiplier
def get_effectiveness(move_type, target_types):
    if move_type is None or pd.isna(move_type) or move_type in ['notype', 'none']:
        return 1.0
    
    effectiveness_map = TYPE_EFFECTIVENESS.get(move_type, {})
    
    multiplier = 1.0
    for target_type in target_types:
        if target_type is not None and not pd.isna(target_type) and target_type not in ['notype', 'none']:
            multiplier *= effectiveness_map.get(target_type, 1.0)
            
    return multiplier

# Function to calculate static matchup features for each battle row
def create_static_matchup_features(row):
    p1_lead_types = [row.get('p1_team.0.type1'), row.get('p1_team.0.type2')]
    p2_lead_types = [row.get('p2_lead.type1'), row.get('p2_lead.type2')]
    
    p1_offense_score = max(
        get_effectiveness(p1_lead_types[0], p2_lead_types),
        get_effectiveness(p1_lead_types[1], p2_lead_types)
    )
    
    p2_offense_score = max(
        get_effectiveness(p2_lead_types[0], p1_lead_types),
        get_effectiveness(p2_lead_types[1], p1_lead_types)
    )
    
    lead_offense_delta = p1_offense_score - p2_offense_score
    
    team_counters = 0
    for i in range(6): 
        p1_pkmn_types = [row.get(f'p1_team.{i}.type1'), row.get(f'p1_team.{i}.type2')]
        
        pkmn_offense_score = max(
            get_effectiveness(p1_pkmn_types[0], p2_lead_types),
            get_effectiveness(p1_pkmn_types[1], p2_lead_types)
        )
        
        if pkmn_offense_score > 1.0:
            team_counters += 1
            
    return pd.Series([lead_offense_delta, team_counters], index=['lead_offense_delta', 'team_counters_vs_lead'])

# Function to load data, compute static features, and save enhanced DataFrame
def process_dataframe(input_path, output_path):
    print(f"\nLoading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File not found. Ensure '02_feature_engineering_01.py' has been run first.")
        return
    
    print("Starting calculation of static type advantage features...")

    new_features = df.apply(create_static_matchup_features, axis=1)
    
    df_final = pd.concat([df, new_features], axis=1)
    
    print(f"Features added. Total number of columns: {df_final.shape[1]}")
    
    df_final.to_csv(output_path, index=False)
    print(f"Feature file saved to: {output_path}")

# Process training and test datasets
process_dataframe(FEATURES_TRAIN_IN, FEATURES_TRAIN_OUT)
process_dataframe(FEATURES_TEST_IN, FEATURES_TEST_OUT)

print("\n04_feature_engineering_02.py executed successfully.")