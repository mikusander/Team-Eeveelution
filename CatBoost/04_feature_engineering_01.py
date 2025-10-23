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

# Function to create aggregated dynamic features from battle timelines
def create_dynamic_features(timelines_df):
    timelines_df = timelines_df.sort_values(by=['battle_id', 'turn'])
    
    boost_cols_p1 = [col for col in timelines_df.columns if 'p1_pokemon_state.boosts' in col]
    boost_cols_p2 = [col for col in timelines_df.columns if 'p2_pokemon_state.boosts' in col]
    
    timelines_df['p1_boosts_sum_turn'] = timelines_df[boost_cols_p1].sum(axis=1)
    timelines_df['p2_boosts_sum_turn'] = timelines_df[boost_cols_p2].sum(axis=1)

    timelines_df['p1_fainted_turn'] = (timelines_df['p1_pokemon_state.status'] == 'fnt').astype(int)
    timelines_df['p2_fainted_turn'] = (timelines_df['p2_pokemon_state.status'] == 'fnt').astype(int)

    # Aggregate dynamic features across battle timelines
    print("Starting aggregation of dynamic features...")
    
    aggregations = {
        'p1_pokemon_state.name': 'nunique',
        'p2_pokemon_state.name': 'nunique',
        'p1_fainted_turn': 'sum',
        'p2_fainted_turn': 'sum',
        
        'p1_pokemon_state.hp_pct': ['mean', 'last'],
        'p2_pokemon_state.hp_pct': ['mean', 'last'],
        
        'p1_boosts_sum_turn': 'sum',
        'p2_boosts_sum_turn': 'sum',
    }
    
    dynamic_features_df = timelines_df.groupby('battle_id').agg(aggregations)

    new_cols = []
    for col in dynamic_features_df.columns:
        if col[1] == '':
             new_cols.append(col[0])
        elif col[1] == 'nunique':
            if col[0] == 'p1_pokemon_state.name':
                new_cols.append('p1_pokemon_used_count')
            else:
                new_cols.append('p2_pokemon_revealed_count')
        else:
             new_cols.append(f'{col[0]}_{col[1]}')
            
    dynamic_features_df.columns = new_cols
    
    dynamic_features_df = dynamic_features_df.rename(columns={
        'p1_fainted_turn_sum': 'p1_fainted_count',
        'p2_fainted_turn_sum': 'p2_fainted_count',
        'p1_pokemon_state.hp_pct_mean': 'p1_avg_hp_pct',
        'p2_pokemon_state.hp_pct_mean': 'p2_avg_hp_pct',
        'p1_pokemon_state.hp_pct_last': 'p1_hp_at_turn_30',
        'p2_pokemon_state.hp_pct_last': 'p2_hp_at_turn_30',
        'p1_boosts_sum_turn_sum': 'p1_total_boosts',
        'p2_boosts_sum_turn_sum': 'p2_total_boosts',
    })

    dynamic_features_df['faint_delta'] = dynamic_features_df['p1_fainted_count'] - dynamic_features_df['p2_fainted_count']
    dynamic_features_df['hp_avg_delta'] = dynamic_features_df['p1_avg_hp_pct'] - dynamic_features_df['p2_avg_hp_pct']
    dynamic_features_df['final_hp_delta'] = dynamic_features_df['p1_hp_at_turn_30'] - dynamic_features_df['p2_hp_at_turn_30']
    dynamic_features_df['total_boosts_delta'] = dynamic_features_df['p1_total_boosts'] - dynamic_features_df['p2_total_boosts']

    print("Dynamic feature aggregation completed.")
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