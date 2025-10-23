"""
This script adds features related to Pokémon status conditions to the existing v3 feature sets. 
It calculates the number of turns each Pokémon is affected by negative status conditions per battle,
computes deltas between players, and merges these features into the final training and test datasets.
"""

import pandas as pd
import numpy as np
import os

CSV_DIR = 'Output_CSVs'       
FEATURES_V3_DIR = 'Features_v3' 
FEATURES_FINAL_DIR = 'Features_v4' 
os.makedirs(FEATURES_FINAL_DIR, exist_ok=True)

print(f"Final feature files will be saved in: {FEATURES_FINAL_DIR}")

try:
    from unique_statuses import STATUSES
except ImportError:
    print("ERRORE: File 'move_effects_final.py' non trovato.")
    print("Assicurati di aver eseguito prima gli script 'counter.py' e 'filter.py'.")
    STATUSES = {}

NEGATIVE_STATUSES = []

for x in STATUSES:
    NEGATIVE_STATUSES.append(x)

print(NEGATIVE_STATUSES)

# Function to calculate per-battle status condition features
def create_status_features(timeline_df):
    print("Starting calculation of status condition features...")

    timeline_df['p1_has_status_turn'] = timeline_df['p1_pokemon_state.status'].isin(NEGATIVE_STATUSES)
    timeline_df['p2_has_status_turn'] = timeline_df['p2_pokemon_state.status'].isin(NEGATIVE_STATUSES)

    aggregations = {
        'p1_has_status_turn': 'sum', 
        'p2_has_status_turn': 'sum'
    }
    
    status_features_df = timeline_df.groupby('battle_id').agg(aggregations)
    
    status_features_df = status_features_df.rename(columns={
        'p1_has_status_turn': 'p1_total_status_turns',
        'p2_has_status_turn': 'p2_total_status_turns'
    })

    status_features_df['status_turns_delta'] = (
        status_features_df['p1_total_status_turns'] - status_features_df['p2_total_status_turns']
    )
    
    print("Status condition feature calculation completed.")
    return status_features_df

# Process TRAIN and TEST datasets, merge with v3 features, and save final CSVs
print("\n    TRAINING PROCESS:\n")
print("Loading TRAIN timeline...")
timeline_train = pd.read_csv(os.path.join(CSV_DIR, 'timelines_train_dynamic.csv'))
status_features_train = create_status_features(timeline_train)

print("Loading TRAIN features_v3...")
features_train_v3 = pd.read_csv(os.path.join(FEATURES_V3_DIR, 'features_train_v3.csv'))
features_train_final = pd.merge(features_train_v3, status_features_train, on='battle_id', how='left')
features_train_final.to_csv(os.path.join(FEATURES_FINAL_DIR, 'features_final_train.csv'), index=False)
print(f"Final TRAIN file saved with {features_train_final.shape[1]} columns.")

print("\n    TEST PROCESS:\n")
print("Loading TEST timeline...")
timeline_test = pd.read_csv(os.path.join(CSV_DIR, 'timelines_test_dynamic.csv'))
status_features_test = create_status_features(timeline_test)

print("Loading TEST features_v3...")
features_test_v3 = pd.read_csv(os.path.join(FEATURES_V3_DIR, 'features_test_v3.csv'))
features_test_final = pd.merge(features_test_v3, status_features_test, on='battle_id', how='left')
features_test_final.to_csv(os.path.join(FEATURES_FINAL_DIR, 'features_final_test.csv'), index=False)
print(f"Final TEST file saved with {features_test_final.shape[1]} columns.")

print("\nFeature engineering process completed.")
print("08_feature_engineering_04.py executed successfully.")