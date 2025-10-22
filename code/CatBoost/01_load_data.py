"""
This script loads Pokémon battle data from JSONL files, processes both static and dynamic features, 
and exports the resulting DataFrames as CSV files for model training and testing.
"""

import pandas as pd
import json
import os

INPUT_DIR = 'Input'
OUTPUT_DIR = 'Output_CSVs' 
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory confirmed: {OUTPUT_DIR}")

TRAIN_FILE = os.path.join(INPUT_DIR, 'train.jsonl')
TEST_FILE = os.path.join(INPUT_DIR, 'test.jsonl')

# Load training and test datasets from JSONL files
def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Error decoding a line in file: {filepath}")
    return data

print(f"Loading training data from {TRAIN_FILE}...")
train_data_raw = load_jsonl(TRAIN_FILE)
print(f"Loading test data from {TEST_FILE}...")
test_data_raw = load_jsonl(TEST_FILE)
print(f"Loaded {len(train_data_raw)} training battles.")
print(f"Loaded {len(test_data_raw)} test battles.")

# Function to flatten and process static battle data
def process_battles_data(raw_data):
    processed_list = []
    for battle in raw_data:
        flat_battle = {
            'battle_id': battle.get('battle_id'),
            'player_won': battle.get('player_won') 
        }
        
        if 'p2_lead_details' in battle and battle['p2_lead_details']:
            for key, value in battle['p2_lead_details'].items():
                if key == 'types':
                    flat_battle['p2_lead.type1'] = value[0] if len(value) > 0 else 'none'
                    flat_battle['p2_lead.type2'] = value[1] if len(value) > 1 else 'none'                
                else:
                    flat_battle[f'p2_lead.{key}'] = value
        
        if 'p1_team_details' in battle and battle['p1_team_details']:
            for i, pokemon in enumerate(battle['p1_team_details']):
                if pokemon: 
                    for key, value in pokemon.items():
                        if key == 'types':
                            flat_battle[f'p1_team.{i}.type1'] = value[0] if len(value) > 0 else 'none'
                            flat_battle[f'p1_team.{i}.type2'] = value[1] if len(value) > 1 else 'none'                        
                        else:
                            flat_battle[f'p1_team.{i}.{key}'] = value
        
        processed_list.append(flat_battle)
    
    return pd.DataFrame(processed_list)

print("\nCreating 'battles_df' (static data) with type handling...")

battles_train_df = process_battles_data(train_data_raw)
battles_test_df = process_battles_data(test_data_raw)

# Extract and flatten dynamic timeline data
print("\nCreating 'timelines_df' (dynamic data)...")

timelines_train_df = pd.json_normalize(
    train_data_raw,
    record_path='battle_timeline',
    meta=['battle_id'],
    errors='ignore'
)
timelines_test_df = pd.json_normalize(
    test_data_raw,
    record_path='battle_timeline',
    meta=['battle_id'],
    errors='ignore'
)

# Drop unnecessary nested columns (move details)
cols_to_drop = ['p1_move_details', 'p2_move_details']
timelines_train_df = timelines_train_df.drop(columns=cols_to_drop, errors='ignore')
timelines_test_df = timelines_test_df.drop(columns=cols_to_drop, errors='ignore')

# Save processed DataFrames to output directory
print("\nSaving DataFrames to CSV files...")

battles_train_path = os.path.join(OUTPUT_DIR, 'battles_train_static.csv')
timelines_train_path = os.path.join(OUTPUT_DIR, 'timelines_train_dynamic.csv')
battles_test_path = os.path.join(OUTPUT_DIR, 'battles_test_static.csv')
timelines_test_path = os.path.join(OUTPUT_DIR, 'timelines_test_dynamic.csv')

battles_train_df.to_csv(battles_train_path, index=False)
timelines_train_df.to_csv(timelines_train_path, index=False)
battles_test_df.to_csv(battles_test_path, index=False)
timelines_test_df.to_csv(timelines_test_path, index=False)

print("Save completed.\nThe following files have been created:")
print(f"  {battles_train_path}")
print(f"  {timelines_train_path}")
print(f"  {battles_test_path}")
print(f"  {timelines_test_path}")

print("\n01_load_data.py executed successfully.")