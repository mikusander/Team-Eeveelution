"""
Utility library for the Pokémon CatBoost pipeline.

This module contains all necessary functions for the entire pipeline:
- Global constants and paths
- Helper functions for parsing, feature engineering and training
- Pipeline functions for each phase (00-11)

"""

# --- 1. CONSOLIDATED IMPORTS ---
import pandas as pd
import numpy as np
import os
import json
import time
import traceback
from pathlib import Path
from collections import defaultdict

# Imports from SKLearn and CatBoost
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from catboost import CatBoostClassifier

# Imports for plots
import seaborn as sns
import matplotlib.pyplot as plt

# --- 2. GLOBAL CONFIGURATION AND PATHS ---

# --- Main Folder Structure ---
BASE_DIR = Path(__file__).resolve().parent

# Input (Original JSONL)
INPUT_JSONL_DIR = BASE_DIR / 'Input'
TRAIN_JSONL_FILE = INPUT_JSONL_DIR / 'train.jsonl'
TEST_JSONL_FILE = INPUT_JSONL_DIR / 'test.jsonl'

# Output PHASE 0-4 (All processed data)
DATA_PIPELINE_DIR = BASE_DIR / 'CatBoost_Data_Pipeline'

# Output PHASE 3 and 5 (Analysis, plots, feature names)
MODEL_OUTPUT_DIR = BASE_DIR / 'CatBoost_Model_Outputs'

# Output PHASE 6 (Final submission)
SUBMISSION_DIR = BASE_DIR / 'Submissions'
OOF_DIR = BASE_DIR / 'OOF_Predictions'

# --- Intermediate and Final File Paths ---
# PHASE 0
BATTLES_TRAIN_STATIC_CSV = DATA_PIPELINE_DIR / 'battles_train_static.csv'
TIMELINES_TRAIN_DYNAMIC_CSV = DATA_PIPELINE_DIR / 'timelines_train_dynamic.csv'
BATTLES_TEST_STATIC_CSV = DATA_PIPELINE_DIR / 'battles_test_static.csv'
TIMELINES_TEST_DYNAMIC_CSV = DATA_PIPELINE_DIR / 'timelines_test_dynamic.csv'

# PHASE 1
TRAIN_FEATURES_FINAL_CSV = DATA_PIPELINE_DIR / 'features_final_train.csv'
TEST_FEATURES_FINAL_CSV = DATA_PIPELINE_DIR / 'features_final_test.csv'

# PHASE 2
TRAIN_PROCESSED_CSV = DATA_PIPELINE_DIR / 'train_processed.csv'
TEST_PROCESSED_CSV = DATA_PIPELINE_DIR / 'test_processed.csv'
TARGET_TRAIN_CSV = DATA_PIPELINE_DIR / 'target_train.csv'

# PHASE 3
TRAIN_PROCESSED_SELECTED_CSV = DATA_PIPELINE_DIR / 'train_processed_selected.csv'
TEST_PROCESSED_SELECTED_CSV = DATA_PIPELINE_DIR / 'test_processed_selected.csv'
SELECTED_FEATURES_FILE = MODEL_OUTPUT_DIR / 'selected_feature_names.txt'
RFECV_PLOT_FILE = MODEL_OUTPUT_DIR / 'rfecv_performance_curve.png'

# PHASE 4
X_TRAIN_SPLIT_FILE = DATA_PIPELINE_DIR / 'train_split_60_X.csv'
Y_TRAIN_SPLIT_FILE = DATA_PIPELINE_DIR / 'train_split_60_y.csv'
X_VAL_SPLIT_FILE = DATA_PIPELINE_DIR / 'validation_split_20_X.csv'
Y_VAL_SPLIT_FILE = DATA_PIPELINE_DIR / 'validation_split_20_y.csv'
X_HOLDOUT_SPLIT_FILE = DATA_PIPELINE_DIR / 'holdout_split_20_X.csv'
Y_HOLDOUT_SPLIT_FILE = DATA_PIPELINE_DIR / 'holdout_split_20_y.csv'

# PHASE 5
PARAMS_OUTPUT_FILE = MODEL_OUTPUT_DIR / 'best_catboost_params.json'
ITERATION_OUTPUT_FILE = MODEL_OUTPUT_DIR / 'best_catboost_iteration.json'
REPORT_TXT_FILE = MODEL_OUTPUT_DIR / 'validation_classification_report.txt'
CM_OUTPUT_FILE = MODEL_OUTPUT_DIR / 'validation_confusion_matrix.png'
IMPORTANCE_OUTPUT_FILE = MODEL_OUTPUT_DIR / 'validation_feature_importance.png'
LOSS_CURVE_FILE = MODEL_OUTPUT_DIR / 'validation_loss_curve.png'
AUC_CURVE_FILE = MODEL_OUTPUT_DIR / 'validation_auc_learning_curve.png'
ROC_CURVE_FILE = MODEL_OUTPUT_DIR / 'validation_roc_auc_curve.png'

# PHASE 6
SUBMISSION_FILE_CSV = SUBMISSION_DIR / 'submission_catboost_100pct_PROBA.csv'
OOF_FILE_NPY = OOF_DIR / 'oof_catboost_proba.npy'
TEST_PREDS_NPY_FILE = OOF_DIR / 'test_preds_catboost_proba.npy'


# Game Constants
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


# --- 3. HELPER FUNCTIONS (GENERAL) ---

def ensure_directories():
    """Creates all necessary output folders if they don't exist."""
    print("Verifying existence of output folders...")
    dirs_to_create = [
        INPUT_JSONL_DIR, DATA_PIPELINE_DIR, MODEL_OUTPUT_DIR,
        SUBMISSION_DIR, OOF_DIR
    ]
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
    print("Folders verified.")

def get_effectiveness(move_type, target_types):
    """Calculates effectiveness for V3 (dynamic aggregates). Returns NAN for unknowns."""
    move_type_str = str(move_type).lower()
    if move_type_str in ['notype', 'none', 'nan'] or pd.isna(move_type):
        return np.nan 
    
    effectiveness_map = TYPE_EFFECTIVENESS.get(move_type_str, {})
    
    multiplier = 1.0
    for target_type in target_types:
        target_type_str = str(target_type).lower()
        if target_type_str not in ['notype', 'none', 'nan'] and not pd.isna(target_type):
            multiplier *= effectiveness_map.get(target_type_str, 1.0)
            
    return multiplier

def get_effectiveness_static(move_type, target_types):
    """Calculates effectiveness for V2 and V6 (static). Returns 1.0 for unknowns."""
    move_type_str = str(move_type).lower()
    if move_type_str in ['notype', 'none', 'nan'] or pd.isna(move_type):
        return 1.0 
    
    effectiveness_map = TYPE_EFFECTIVENESS.get(move_type_str, {})
    
    multiplier = 1.0
    for target_type in target_types:
        target_type_str = str(target_type).lower()
        if target_type_str not in ['notype', 'none', 'nan'] and not pd.isna(target_type):
            multiplier *= effectiveness_map.get(target_type_str, 1.0)
            
    return multiplier

def build_pokedex():
    """Builds a dictionary {name: [type1, type2]} from static data."""
    print("Building Pokédex...")
    try:
        static_train_df = pd.read_csv(BATTLES_TRAIN_STATIC_CSV)
    except FileNotFoundError:
        print(f"ERROR: '{BATTLES_TRAIN_STATIC_CSV.name}' not found.")
        return None
        
    pokedex = {}
    
    for i in range(6):
        name_col = f'p1_team.{i}.name'
        type1_col = f'p1_team.{i}.type1'
        type2_col = f'p1_team.{i}.type2'
        
        pkmn_data = static_train_df[[name_col, type1_col, type2_col]].drop_duplicates()
        for _, row in pkmn_data.iterrows():
            name = row[name_col]
            if pd.notna(name) and name not in pokedex:
                pokedex[name] = [str(row[type1_col]).lower(), str(row[type2_col]).lower()]
                
    p2_lead_data = static_train_df[['p2_lead.name', 'p2_lead.type1', 'p2_lead.type2']].drop_duplicates()
    for _, row in p2_lead_data.iterrows():
        name = row['p2_lead.name']
        if pd.notna(name) and name not in pokedex:
            pokedex[name] = [str(row['p2_lead.type1']).lower(), str(row['p2_lead.type2']).lower()]

    print(f"Pokédex built. Contains {len(pokedex)} unique Pokémon.")
    return pokedex

# --- 4. HELPER FUNCTIONS (FEATURE ENGINEERING) ---

def aggregate_battle_features(battle_turns_df):
    """Helper for FE v1: Aggregates dynamic features for a single battle."""
    p1_damage = 0.0
    p2_damage = 0.0
    last_p1_hp = {}
    last_p2_hp = {}
    p1_fainted_names = set()
    p2_fainted_names = set()
    p1_fainted_count = 0
    p2_fainted_count = 0
    p1_comeback_kos = 0
    p2_comeback_kos = 0
    p1_names_used = set()
    p2_names_revealed = set()
    p1_hp_list = []
    p2_hp_list = []
    p1_boosts_total = 0
    p2_boosts_total = 0
    p1_last_hp = np.nan
    p2_last_hp = np.nan
    
    for _, turn in battle_turns_df.iterrows():
        prev_p1_fainted = p1_fainted_count
        prev_p2_fainted = p2_fainted_count
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
        
        if pd.notna(p1_hp): p1_last_hp = p1_hp 
        if pd.notna(p2_hp): p2_last_hp = p2_hp
        p1_status = turn['p1_pokemon_state.status']
        p2_status = turn['p2_pokemon_state.status']

        if p1_status == 'fnt' and p1_name not in p1_fainted_names:
            p1_fainted_count += 1
            p1_fainted_names.add(p1_name)
        if p2_status == 'fnt' and p2_name not in p2_fainted_names:
            p2_fainted_count += 1
            p2_fainted_names.add(p2_name)

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
            
        damage_diff_so_far = p1_damage - p2_damage
        if p2_fainted_count > prev_p2_fainted and damage_diff_so_far < -1.0:
            p1_comeback_kos += 1
        if p1_fainted_count > prev_p1_fainted and damage_diff_so_far > 1.0:
            p2_comeback_kos += 1
            
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
        'p1_comeback_kos': p1_comeback_kos,
        'p2_comeback_kos': p2_comeback_kos
    }
    return pd.Series(results)

def create_static_matchup_features(row):
    """Helper for FE v2: Calculates static lead-vs-lead type advantage."""
    p1_lead_types = [row.get('p1_team.0.type1'), row.get('p1_team.0.type2')]
    p2_lead_types = [row.get('p2_lead.type1'), row.get('p2_lead.type2')]
    
    p1_offense_score = max(
        get_effectiveness_static(p1_lead_types[0], p2_lead_types),
        get_effectiveness_static(p1_lead_types[1], p2_lead_types)
    )
    
    p2_offense_score = max(
        get_effectiveness_static(p2_lead_types[0], p1_lead_types),
        get_effectiveness_static(p2_lead_types[1], p1_lead_types)
    )
    
    lead_offense_delta = p1_offense_score - p2_offense_score
    
    team_counters = 0
    for i in range(6): 
        p1_pkmn_types = [row.get(f'p1_team.{i}.type1'), row.get(f'p1_team.{i}.type2')]
        
        pkmn_offense_score = max(
            get_effectiveness_static(p1_pkmn_types[0], p2_lead_types),
            get_effectiveness_static(p1_pkmn_types[1], p2_lead_types)
        )
        
        if pkmn_offense_score > 1.0:
            team_counters += 1
            
    return pd.Series([lead_offense_delta, team_counters], index=['lead_offense_delta', 'team_counters_vs_lead'])

def create_team_aggregate_features(row):
    """Helper for FE v6: Calculates aggregated p1 team statistics."""
    team_stats = []
    team_types = set()
    
    for i in range(6):
        stats = {
            'hp': row.get(f'p1_team.{i}.base_hp'),
            'atk': row.get(f'p1_team.{i}.base_atk'),
            'def': row.get(f'p1_team.{i}.base_def'),
            'spa': row.get(f'p1_team.{i}.base_spa'),
            'spd': row.get(f'p1_team.{i}.base_spd'),
            'spe': row.get(f'p1_team.{i}.base_spe'),
            'type1': row.get(f'p1_team.{i}.type1'),
            'type2': row.get(f'p1_team.{i}.type2')
        }
        
        if pd.notna(stats['hp']) and pd.notna(stats['spe']):
            team_stats.append(stats)
            team_types.add(stats['type1'])
            if pd.notna(stats['type2']) and stats['type2'] != 'none':
                team_types.add(stats['type2'])

    if not team_stats:
        return pd.Series({
            'p1_avg_team_speed': np.nan, 'p1_total_team_hp': np.nan,
            'p1_avg_team_atk': np.nan, 'p1_avg_team_def': np.nan,
            'p1_type_diversity': 0, 'p1_team_weaknesses_count': 0
        })

    p1_avg_team_speed = np.mean([s['spe'] for s in team_stats])
    p1_total_team_hp = np.sum([s['hp'] for s in team_stats])
    p1_avg_team_atk = np.mean([max(s['atk'], s['spa']) for s in team_stats])
    p1_avg_team_def = np.mean([(s['def'] + s['spd']) / 2 for s in team_stats])
    
    team_types.discard('none')
    team_types.discard(np.nan)
    p1_type_diversity = len(team_types)

    p1_team_weaknesses_count = 0
    attacking_types = [t for t in TYPE_EFFECTIVENESS.keys() if t not in ['notype', 'none']]
    
    for att_type in attacking_types:
        weak_pokemon_count = 0
        for pkmn in team_stats:
            def_types = [pkmn['type1'], pkmn.get('type2', 'none')]
            effectiveness = get_effectiveness_static(att_type, def_types)
            if effectiveness > 1.0:
                weak_pokemon_count += 1
        
        if weak_pokemon_count >= 3:
            p1_team_weaknesses_count += 1

    return pd.Series({
        'p1_avg_team_speed': p1_avg_team_speed,
        'p1_total_team_hp': p1_total_team_hp,
        'p1_avg_team_atk': p1_avg_team_atk,
        'p1_avg_team_def': p1_avg_team_def,
        'p1_type_diversity': p1_type_diversity,
        'p1_team_weaknesses_count': p1_team_weaknesses_count
    })


# --- 5. PIPELINE FUNCTIONS (PHASE 0 - PRE-ANALYSIS) ---

def run_00a_load_data():
    """PHASE 0a: Loads JSONL and converts them to static and dynamic CSVs."""
    print("\n--- Starting 00a_load_data ---")
    
    def load_jsonl(filepath):
        data = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"JSONL decode error: {filepath}")
        except FileNotFoundError:
            print(f"ERROR: File not found: {filepath}")
            return None
        return data

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

    # Load Train
    print(f"Loading train data from {TRAIN_JSONL_FILE}...")
    train_data_raw = load_jsonl(TRAIN_JSONL_FILE)
    if train_data_raw is None: return False
    print(f"Loaded {len(train_data_raw)} training battles.")
    
    # Load Test
    print(f"Loading test data from {TEST_JSONL_FILE}...")
    test_data_raw = load_jsonl(TEST_JSONL_FILE)
    if test_data_raw is None: return False
    print(f"Loaded {len(test_data_raw)} test battles.")

    # Process static
    print("Creating 'battles_df' (static data)...")
    battles_train_df = process_battles_data(train_data_raw)
    battles_test_df = process_battles_data(test_data_raw)

    # Process dynamic
    print("Creating 'timelines_df' (dynamic data)...")
    timelines_train_df = pd.json_normalize(train_data_raw, record_path='battle_timeline', meta=['battle_id'], errors='ignore')
    timelines_test_df = pd.json_normalize(test_data_raw, record_path='battle_timeline', meta=['battle_id'], errors='ignore')

    # Save CSV
    print("\nSaving DataFrames to CSV...")
    battles_train_df.to_csv(BATTLES_TRAIN_STATIC_CSV, index=False)
    timelines_train_df.to_csv(TIMELINES_TRAIN_DYNAMIC_CSV, index=False)
    battles_test_df.to_csv(BATTLES_TEST_STATIC_CSV, index=False)
    timelines_test_df.to_csv(TIMELINES_TEST_DYNAMIC_CSV, index=False)

    print("Save complete.")
    print(f"--- Completed 00a_load_data. Files saved in {DATA_PIPELINE_DIR} ---")
    return True

def run_00b_analyze_moves():
    """PHASE 0b: Analyzes moves and returns a clean dict."""
    print("\n--- Starting 00b_analyze_moves ---")
    
    print("Analyzing move usage and categories...")
    move_stats = defaultdict(lambda: {'total_uses': 0, 'effects': defaultdict(int)})
    move_categories = {}
    
    try:
        with open(TRAIN_JSONL_FILE, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                try:
                    battle_data = json.loads(line)
                    timeline = battle_data.get("battle_timeline", [])
                    if len(timeline) < 2: continue 

                    for i in range(1, len(timeline)):
                        curr_turn = timeline[i]
                        prev_turn = timeline[i-1]
                        p1_curr, p2_curr = curr_turn["p1_pokemon_state"], curr_turn["p2_pokemon_state"]
                        p1_prev, p2_prev = prev_turn["p1_pokemon_state"], prev_turn["p2_pokemon_state"]
                        p1_move, p2_move = curr_turn.get("p1_move_details"), curr_turn.get("p2_move_details")
                        
                        if p1_move and p1_move["name"] not in move_categories:
                            move_categories[p1_move["name"]] = p1_move.get("category", "UNKNOWN")
                        if p2_move and p2_move["name"] not in move_categories:
                            move_categories[p2_move["name"]] = p2_move.get("category", "UNKNOWN")

                        p1_did_switch = p1_curr["name"] != p1_prev["name"]
                        p2_did_switch = p2_curr["name"] != p2_prev["name"]

                        if p1_move and not p1_did_switch: move_stats[p1_move["name"]]['total_uses'] += 1
                        if p2_move and not p2_did_switch: move_stats[p2_move["name"]]['total_uses'] += 1

                        if not p1_did_switch:
                            if p1_curr["hp_pct"] < p1_prev["hp_pct"] and p2_move: move_stats[p2_move["name"]]['effects']['damage'] += 1
                            if p1_curr["hp_pct"] > p1_prev["hp_pct"] and p1_move: move_stats[p1_move["name"]]['effects']['healing'] += 1
                            if (p1_curr["boosts"] != p1_prev["boosts"] and any(p1_curr["boosts"][s] > p1_prev["boosts"][s] for s in p1_curr["boosts"])) and p1_move: move_stats[p1_move["name"]]['effects']['user_boost'] += 1
                            if p1_curr["effects"] != p1_prev["effects"] and "noeffect" not in p1_curr["effects"] and p1_move: move_stats[p1_move["name"]]['effects']['user_boost'] += 1
                            if p1_curr["status"] != p1_prev["status"] and p1_curr["status"] not in ["nostatus", "fnt"]:
                                if p1_move and p1_move["name"] == "rest" and p1_curr["status"] == "slp": move_stats[p1_move["name"]]['effects']['user_status'] += 1
                                elif p2_move: move_stats[p2_move["name"]]['effects']['opponent_status'] += 1
                            if (p1_curr["boosts"] != p1_prev["boosts"] and any(p1_curr["boosts"][s] < p1_prev["boosts"][s] for s in p1_curr["boosts"])) and p2_move: move_stats[p2_move["name"]]['effects']['opponent_debuff'] += 1

                        if not p2_did_switch:
                            if p2_curr["hp_pct"] < p2_prev["hp_pct"] and p1_move: move_stats[p1_move["name"]]['effects']['damage'] += 1
                            if p2_curr["hp_pct"] > p2_prev["hp_pct"] and p2_move: move_stats[p2_move["name"]]['effects']['healing'] += 1
                            if (p2_curr["boosts"] != p2_prev["boosts"] and any(p2_curr["boosts"][s] > p2_prev["boosts"][s] for s in p2_curr["boosts"])) and p2_move: move_stats[p2_move["name"]]['effects']['user_boost'] += 1
                            if p2_curr["effects"] != p2_prev["effects"] and "noeffect" not in p2_curr["effects"] and p2_move: move_stats[p2_move["name"]]['effects']['user_boost'] += 1
                            if p2_curr["status"] != p2_prev["status"] and p2_curr["status"] not in ["nostatus", "fnt"]:
                                if p2_move and p2_move["name"] == "rest" and p2_curr["status"] == "slp": move_stats[p2_move["name"]]['effects']['user_status'] += 1
                                elif p1_move: move_stats[p1_move["name"]]['effects']['opponent_status'] += 1
                            if (p2_curr["boosts"] != p2_prev["boosts"] and any(p2_curr["boosts"][s] < p2_prev["boosts"][s] for s in p2_curr["boosts"])) and p1_move: move_stats[p1_move["name"]]['effects']['opponent_debuff'] += 1
                
                except json.JSONDecodeError: continue
                except Exception as e: print(f"Error parsing line {line_number} in 00b: {e}")

    except FileNotFoundError:
        print(f"ERROR: File not found: {TRAIN_JSONL_FILE}")
        return None
    except Exception as e:
        print(f"ERROR: Unexpected error in 00b: {e}")
        traceback.print_exc()
        return None
    
    # Clean up raw counts for filtering
    MOVE_STATS_RAW = {}
    for move, data in move_stats.items():
        MOVE_STATS_RAW[move] = {
            'total_uses': data['total_uses'],
            'effects': dict(data['effects'])  
        }
    MOVE_CATEGORIES = move_categories
    print("Move count analysis complete.")

    # Filter effects
    print("Filtering move effects...")
    PRIMARY_THRESHOLD = 0.30 
    SECONDARY_THRESHOLD = 0.01 
    MIN_USES = 1
    final_move_effects = {}

    for move_name, data in sorted(MOVE_STATS_RAW.items()):
        total_uses = data['total_uses']
        if total_uses < MIN_USES: continue 

        category = MOVE_CATEGORIES.get(move_name, "UNKNOWN")
        effects = data['effects']
        final_effects = set()
        
        if category in ["PHYSICAL", "SPECIAL"]:
            if 'damage' in effects and effects['damage'] > 0: final_effects.add('damage')
            if 'healing' in effects and (effects['healing'] / total_uses) >= PRIMARY_THRESHOLD: final_effects.add('healing')
            if 'opponent_status' in effects and (effects['opponent_status'] / total_uses) >= SECONDARY_THRESHOLD: final_effects.add('opponent_status')
            if 'opponent_debuff' in effects and (effects['opponent_debuff'] / total_uses) >= SECONDARY_THRESHOLD: final_effects.add('opponent_debuff')
                
        elif category == "STATUS":
            if 'healing' in effects and (effects['healing'] / total_uses) >= PRIMARY_THRESHOLD: final_effects.add('healing')
            if 'user_boost' in effects and (effects['user_boost'] / total_uses) >= PRIMARY_THRESHOLD: final_effects.add('user_boost')
            if 'user_status' in effects and (effects['user_status'] / total_uses) >= PRIMARY_THRESHOLD: final_effects.add('user_status')
            if 'opponent_status' in effects and (effects['opponent_status'] / total_uses) >= PRIMARY_THRESHOLD: final_effects.add('opponent_status')
            if 'opponent_debuff' in effects and (effects['opponent_debuff'] / total_uses) >= PRIMARY_THRESHOLD: final_effects.add('opponent_debuff')

        if not final_effects and effects: 
            possible_effects = {}
            if category == "STATUS":
                for effect, count in effects.items():
                    if effect != 'damage': possible_effects[effect] = count
            elif category in ["PHYSICAL", "SPECIAL"]:
                for effect, count in effects.items():
                    if effect not in ['user_boost', 'user_status']: possible_effects[effect] = count
            
            if possible_effects:
                best_effect = max(possible_effects, key=possible_effects.get)
                final_effects.add(best_effect)
                
        if final_effects:
            final_move_effects[move_name] = sorted(list(final_effects))
    
    print(f"Move filtering complete. {len(final_move_effects)} moves processed.")
    print("--- Completed 00b_analyze_moves ---")
    return final_move_effects

def run_00c_analyze_statuses():
    """PHASE 0c: Analyses and returns a set of negative statuses."""
    print("\n--- Starting 00c_analyze_statuses ---")
    NEGATIVE_STATUSES = set()
    print(f"Starting status analysis from: {TRAIN_JSONL_FILE}")

    try:
        with open(TRAIN_JSONL_FILE, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f):
                try:
                    data = json.loads(line)
                    timeline = data.get("battle_timeline")
                    if not timeline: continue

                    for turn in timeline:
                        p1_state = turn.get("p1_pokemon_state")
                        if p1_state and p1_state.get("status"):
                            if p1_state.get("status") not in ['fnt', 'nostatus']:
                                NEGATIVE_STATUSES.add(p1_state["status"])
                        p2_state = turn.get("p2_pokemon_state")
                        if p2_state and p2_state.get("status"):
                            if p2_state.get("status") not in ['fnt', 'nostatus']:
                                NEGATIVE_STATUSES.add(p2_state["status"])
                except json.JSONDecodeError: continue
                except Exception as e: print(f"Error processing line {line_number+1} in 00c: {e}")
    
        print(f"Status analysis complete. Found {len(NEGATIVE_STATUSES)} negative statuses.")
        print(f"Statuses found: {NEGATIVE_STATUSES}")
        print("--- Completed 00c_analyze_statuses ---")
        return list(NEGATIVE_STATUSES) 

    except FileNotFoundError:
        print(f"ERROR: File not found: '{TRAIN_JSONL_FILE}'")
        return None
    except Exception as e:
        print(f"ERROR: General error in 00c: {e}")
        return None


# --- 6. PIPELINE FUNCTIONS (PHASE 1 - FEATURE ENGINEERING) ---

def run_01_feature_engineering_v1(battles_df, timelines_df):
    """PHASE 1a: Creates basic dynamic features (v1)."""
    print("\n--- Starting 01_feature_engineering_v1 ---")
    
    timelines_df = timelines_df.sort_values(by=['battle_id', 'turn'])
    
    boost_cols_p1 = [col for col in timelines_df.columns if 'p1_pokemon_state.boosts' in col]
    boost_cols_p2 = [col for col in timelines_df.columns if 'p2_pokemon_state.boosts' in col]
    
    timelines_df['p1_boosts_sum_turn'] = timelines_df[boost_cols_p1].sum(axis=1)
    timelines_df['p2_boosts_sum_turn'] = timelines_df[boost_cols_p2].sum(axis=1)

    print("Starting aggregation of dynamic features (v1)...")
    dynamic_features_df = timelines_df.groupby('battle_id').apply(aggregate_battle_features)
    print("Aggregation v1 complete.")

    dynamic_features_df['faint_delta'] = dynamic_features_df['p1_fainted_count'] - dynamic_features_df['p2_fainted_count']
    dynamic_features_df['hp_avg_delta'] = dynamic_features_df['p1_avg_hp_pct'] - dynamic_features_df['p2_avg_hp_pct']
    dynamic_features_df['final_hp_delta'] = dynamic_features_df['p1_hp_at_turn_30'] - dynamic_features_df['p2_hp_at_turn_30']
    dynamic_features_df['total_boosts_delta'] = dynamic_features_df['p1_total_boosts'] - dynamic_features_df['p2_total_boosts']
    dynamic_features_df['comeback_kos_delta'] = dynamic_features_df['p1_comeback_kos'] - dynamic_features_df['p2_comeback_kos']

    print("Merging static and dynamic data (v1)...")
    features_df = pd.merge(
        battles_df, 
        dynamic_features_df, 
        on='battle_id', 
        how='left'
    )
    print("--- Completed 01_feature_engineering_v1 ---")
    return features_df

def run_02_feature_engineering_v2(features_df):
    """PHASE 1b: Adds static matchups (v2)."""
    print("\n--- Starting 02_feature_engineering_v2 ---")
    print("Starting calculation of static type advantage features (v2)...")
    
    new_features = features_df.apply(create_static_matchup_features, axis=1)
    features_df_v2 = pd.concat([features_df, new_features], axis=1)
    
    print(f"Features v2 added. Total columns: {features_df_v2.shape[1]}")
    print("--- Completed 02_feature_engineering_v2 ---")
    return features_df_v2

def run_03_feature_engineering_v3(features_df_v2, timelines_df, pokedex):
    """PHASE 1c: Adds dynamic type effectiveness (v3)."""
    print("\n--- Starting 03_feature_engineering_v3 ---")
    if pokedex is None:
        print("ERROR (v3): Pokédex not provided. Skipping phase.")
        return features_df_v2
        
    print("Starting dynamic effectiveness calculation (v3)...")
    
    timelines_df['p1_active_types'] = timelines_df['p1_pokemon_state.name'].apply(lambda x: pokedex.get(x, [None, None]))
    timelines_df['p2_active_types'] = timelines_df['p2_pokemon_state.name'].apply(lambda x: pokedex.get(x, [None, None]))
    
    if 'p1_move_details.type' in timelines_df.columns:
        timelines_df['p1_move_effectiveness'] = timelines_df.apply(
            lambda row: get_effectiveness(row['p1_move_details.type'], row['p2_active_types']), axis=1
        )
    else:
        print("WARNING: Column 'p1_move_details.type' not found. Skipping p1_move_effectiveness.")
        timelines_df['p1_move_effectiveness'] = np.nan

    if 'p2_move_details.type' in timelines_df.columns:
        timelines_df['p2_move_effectiveness'] = timelines_df.apply(
            lambda row: get_effectiveness(row['p2_move_details.type'], row['p1_active_types']), axis=1
        )
    else:
        print("WARNING: Column 'p2_move_details.type' not found. Skipping p2_move_effectiveness.")
        timelines_df['p2_move_effectiveness'] = np.nan
    
    print("Effectiveness calculation complete. Starting aggregation (v3)...")
    
    aggregations = {
        'p1_move_effectiveness': ['mean', lambda x: (x > 1.0).sum(), lambda x: (x < 1.0).sum()],
        'p2_move_effectiveness': ['mean', lambda x: (x > 1.0).sum(), lambda x: (x < 1.0).sum()],
    }
    dynamic_type_features = timelines_df.groupby('battle_id').agg(aggregations)
    
    dynamic_type_features.columns = [
        'p1_avg_effectiveness', 'p1_super_effective_hits', 'p1_resisted_hits',
        'p2_avg_effectiveness', 'p2_super_effective_hits', 'p2_resisted_hits'
    ]
    
    dynamic_type_features['p1_avg_effectiveness'] = dynamic_type_features['p1_avg_effectiveness'].fillna(1.0)
    dynamic_type_features['p2_avg_effectiveness'] = dynamic_type_features['p2_avg_effectiveness'].fillna(1.0)
    dynamic_type_features = dynamic_type_features.fillna(0)

    print("Merging v2 with v3 features...")
    features_df_v3 = pd.merge(features_df_v2, dynamic_type_features, on='battle_id', how='left')
    print(f"Features v3 added. Total columns: {features_df_v3.shape[1]}")
    print("--- Completed 03_feature_engineering_v3 ---")
    return features_df_v3

def run_04_feature_engineering_v4(features_df_v3, timelines_df, statuses: list):
    """PHASE 1d: Adds status features (v4)."""
    print("\n--- Starting 04_feature_engineering_v4 ---")
    
    if not statuses:
        print("WARNING (v4): 'statuses' list is empty. Status features will be 0.")
        
    print("Starting status feature calculation (v4)...")
    
    NEGATIVE_STATUSES = statuses
    timeline_df = timelines_df.copy() 
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

    print("Merging v3 with v4 features...")
    features_df_v4 = pd.merge(features_df_v3, status_features_df, on='battle_id', how='left')
    print(f"Features v4 added. Total columns: {features_df_v4.shape[1]}")
    print("--- Completed 04_feature_engineering_v4 ---")
    return features_df_v4

def run_05_feature_engineering_v5(features_df_v4, timelines_df, pokedex, move_effects: dict):
    """PHASE 1e: Adds expert features (STAB, etc.) (v5)."""
    print("\n--- Starting 05_feature_engineering_v5 ---")
    
    STATUS_MOVES = []
    HEALING_MOVES = []

    if not move_effects:
        print("WARNING (v5): 'move_effects' is empty. STAB/Healing features will be 0.")
    else:
        for move_name, effects_list in move_effects.items():
            if 'opponent_status' in effects_list:
                STATUS_MOVES.append(move_name)
            if 'healing' in effects_list:
                HEALING_MOVES.append(move_name)

    print("Starting expert feature calculation (v5)...")
    timeline_df = timelines_df.copy()
    timeline_df['p1_active_types'] = timeline_df['p1_pokemon_state.name'].apply(lambda x: pokedex.get(x, [None, None]))
    timeline_df['p2_active_types'] = timeline_df['p2_pokemon_state.name'].apply(lambda x: pokedex.get(x, [None, None]))
        
    def calculate_turn_features(row):
        move_name_p1 = str(row.get('p1_move_details.name')).lower()
        move_type_p1 = str(row.get('p1_move_details.type')).lower()
        active_types_p1 = row['p1_active_types']
        
        move_name_p2 = str(row.get('p2_move_details.name')).lower()
        move_type_p2 = str(row.get('p2_move_details.type')).lower()
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

    print("... applying per-turn calculations (v5)...")
    new_features_per_turn = timeline_df.apply(calculate_turn_features, axis=1)
    new_features_per_turn.columns = [
        'p1_is_stab', 'p2_is_stab',
        'p1_is_status_move', 'p1_is_healing_move',
        'p2_is_status_move', 'p2_is_healing_move'
    ]
    timeline_df = pd.concat([timeline_df, new_features_per_turn], axis=1)
    
    print("Per-turn calculation complete. Starting aggregation (v5)...")
    aggregations = {
        'p1_is_stab': 'sum', 'p2_is_stab': 'sum',
        'p1_is_status_move': 'sum', 'p1_is_healing_move': 'sum',
        'p2_is_status_move': 'sum', 'p2_is_healing_move': 'sum'
    }
    expert_features_df = timeline_df.groupby('battle_id').agg(aggregations)
    
    expert_features_df = expert_features_df.rename(columns={
        'p1_is_stab': 'p1_stab_move_count', 'p2_is_stab': 'p2_stab_move_count',
        'p1_is_status_move': 'p1_status_move_count', 'p1_is_healing_move': 'p1_healing_move_count',
        'p2_is_status_move': 'p2_status_move_count', 'p2_is_healing_move': 'p2_healing_move_count'
    })

    expert_features_df['stab_delta'] = expert_features_df['p1_stab_move_count'] - expert_features_df['p2_stab_move_count']
    expert_features_df['status_move_delta'] = expert_features_df['p1_status_move_count'] - expert_features_df['p2_status_move_count']
    expert_features_df['healing_move_delta'] = expert_features_df['p1_healing_move_count'] - expert_features_df['p2_healing_move_count']

    print("Merging v4 with v5 features...")
    features_df_v5 = pd.merge(features_df_v4, expert_features_df, on='battle_id', how='left')
    print(f"Features v5 added. Total columns: {features_df_v5.shape[1]}")
    print("--- Completed 05_feature_engineering_v5 ---")
    return features_df_v5

def run_06_feature_engineering_v6(features_df_v5):
    """PHASE 1f: Adds aggregated team statistics (v6)."""
    print("\n--- Starting 06_feature_engineering_v6 ---")
    print("Starting calculation of static team features (v6)...")
    
    new_features = features_df_v5.apply(create_team_aggregate_features, axis=1)
    features_df_v6 = pd.concat([features_df_v5, new_features], axis=1)
    
    print(f"Features v6 added. Total columns: {features_df_v6.shape[1]}")
    print("--- Completed 06_feature_engineering_v6 ---")
    return features_df_v6

# --- 7. PIPELINE FUNCTIONS (PHASE 2, 3, 4 - PREPROCESSING AND SPLIT) ---

def run_07_preprocessing():
    """PHASE 2: Performs OHE, Imputing, Scaling."""
    print("\n--- Starting 07_preprocessing ---")

    try:
        train_df = pd.read_csv(TRAIN_FEATURES_FINAL_CSV)
        test_df = pd.read_csv(TEST_FEATURES_FINAL_CSV)
        print("Data loaded.")
    except FileNotFoundError as e:
        print(f"ERROR: Feature files not found: {e}")
        print("Ensure you have run PHASE 1.")
        return False

    y_train = train_df['player_won']
    y_train.to_csv(TARGET_TRAIN_CSV, index=False, header=True)

    train_df = train_df.drop(columns=['player_won', 'battle_id']) 
    test_df = test_df.drop(columns=['battle_id', 'player_won'])
    train_df['is_train'] = 1
    test_df['is_train'] = 0

    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    categorical_features = ['p2_lead.name', 'p2_lead.type1', 'p2_lead.type2']
    for i in range(6):
        categorical_features.append(f'p1_team.{i}.name')
        categorical_features.append(f'p1_team.{i}.type1')
        categorical_features.append(f'p1_team.{i}.type2')

    categorical_features = [col for col in categorical_features if col in combined_df.columns]
    numeric_features = [col for col in combined_df.columns if col not in categorical_features and col != 'is_train']
    
    print(f"Found {len(numeric_features)} numeric features.")
    print(f"Found {len(categorical_features)} categorical features.")

    print("Executing One-Hot Encoding ...")
    processed_df = pd.get_dummies(
        combined_df, 
        columns=categorical_features, 
        dummy_na=False, 
        drop_first=False
    )
    print(f"DataFrame transformed to {processed_df.shape[1]} total columns.")

    print("Applying SimpleImputer and StandardScaler to numeric features...")
    numeric_imputer = SimpleImputer(strategy='median')
    processed_df[numeric_features] = numeric_imputer.fit_transform(processed_df[numeric_features])

    scaler = StandardScaler()
    processed_df[numeric_features] = scaler.fit_transform(processed_df[numeric_features])
    print("Imputation and scaling complete.")

    print("Separating processed Train and Test sets...")
    X_train_processed = processed_df[processed_df['is_train'] == 1].drop(columns=['is_train'])
    X_test_processed = processed_df[processed_df['is_train'] == 0].drop(columns=['is_train'])

    print(f"Saving {TRAIN_PROCESSED_CSV.name}...")
    X_train_processed.to_csv(TRAIN_PROCESSED_CSV, index=False)
    print(f"Saving {TEST_PROCESSED_CSV.name}...")
    X_test_processed.to_csv(TEST_PROCESSED_CSV, index=False)

    print("--- Completed 07_preprocessing ---")
    return True

def run_08_feature_selection(run_rfecv: bool):
    """
    PHASE 3: Runs RFECV (if run_rfecv=True) or loads results (if False).
    
    Args:
        run_rfecv (bool): Flag from main script to enable/disable RFECV execution.
    """
    
    if not run_rfecv:
        print("\n--- Starting 08_feature_selection (Mode: Load) ---")
        print("run_rfecv is False. Skipping RFECV.")
        print("Checking for existence of selected feature files...")
        
        required_files = [
            TRAIN_PROCESSED_SELECTED_CSV,
            TEST_PROCESSED_SELECTED_CSV,
            SELECTED_FEATURES_FILE
        ]
        
        all_files_found = True
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"\033[91mERROR: Required file not found: {file_path}\033[0m")
                all_files_found = False
        
        if all_files_found:
            print("\033[92mSuccess: All selected feature files are present.\033[0m")
            print(f"Data will be read from: {TRAIN_PROCESSED_SELECTED_CSV.name}")
            print("--- Completed 08_feature_selection (Load) ---")
            return True
        else:
            print("\nERROR: To skip RFECV, the '..._selected.csv' and '...names.txt' files must exist.")
            print(f"Run the pipeline once with RUN_FEATURE_SELECTION=True in your main script")
            print(f"or place the files manually in {DATA_PIPELINE_DIR} and {MODEL_OUTPUT_DIR}.")
            return False
    
    # --- ELSE: RUN RFECV ---
    print("\n--- Starting 08_feature_selection (Mode: Run RFECV) ---")
    print("run_rfecv is True. Starting RFECV process...")
    
    try:
        print(f"Loading data from {DATA_PIPELINE_DIR}...")
        X_train_full = pd.read_csv(TRAIN_PROCESSED_CSV)
        y_train_full = pd.read_csv(TARGET_TRAIN_CSV).values.ravel()
        X_test_kaggle = pd.read_csv(TEST_PROCESSED_CSV)
        original_feature_names = X_train_full.columns
        print(f"Data loaded: X_train_full {X_train_full.shape}, X_test_kaggle {X_test_kaggle.shape}")
    except FileNotFoundError as e:
        print(f"ERROR: Processed files not found in '{DATA_PIPELINE_DIR}'.")
        print(e)
        return False

    print("Starting RFECV selector training with CatBoost...")
    estimator = CatBoostClassifier(
        n_estimators=500, depth=7, random_seed=42, verbose=0, thread_count=-1
    )
    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

    selector = RFECV(
        estimator, cv=cv_strategy, scoring='roc_auc', step=10, 
        min_features_to_select=50, n_jobs=-1, verbose=2
    )
    
    selector.fit(X_train_full, y_train_full)
    print("RFECV training complete.")

    print("Applying RFECV transformation...")
    X_train_selected = selector.transform(X_train_full)
    X_test_selected = selector.transform(X_test_kaggle)
    optimal_count = selector.n_features_
    print(f"Selection complete. OPTIMAL NUMBER OF FEATURES: {optimal_count}")

    selected_mask = selector.get_support()
    selected_features = original_feature_names[selected_mask]
    
    X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features)
    X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features)

    X_train_selected_df.to_csv(TRAIN_PROCESSED_SELECTED_CSV, index=False)
    X_test_selected_df.to_csv(TEST_PROCESSED_SELECTED_CSV, index=False)
    print(f"\nSelected data saved in -> {DATA_PIPELINE_DIR}")

    pd.Series(selected_features).to_csv(SELECTED_FEATURES_FILE, index=False, header=False)
    print(f"Feature names saved in -> {SELECTED_FEATURES_FILE}")

    print("\nSaving RFECV performance plot...")
    try:
        scores = selector.cv_results_['mean_test_score']
        n_features_tested = selector.cv_results_['n_features']
        
        plt.figure(figsize=(12, 7))
        plt.plot(n_features_tested, scores, marker='o')
        best_score = np.max(scores)
        best_n_features = n_features_tested[np.argmax(scores)]
        plt.axvline(x=best_n_features, color='red', linestyle='--', 
                    label=f'Optimal: {best_n_features} features (AUC: {best_score:.4f})')
        plt.title('RFECV Performance vs. Number of Features')
        plt.xlabel('Number of Features Selected')
        plt.ylabel('CV Score (roc_auc)')
        plt.legend()
        plt.grid(True)
        plt.gca().invert_xaxis()
        plt.savefig(RFECV_PLOT_FILE)
        plt.close()
        print(f"RFECV performance plot saved to: {RFECV_PLOT_FILE}")
    except Exception as e:
        print(f"Error during plot creation: {e}")

    print("--- Completed 08_feature_selection (Run RFECV) ---")
    return True

def run_09_data_splitter():
    """PHASE 4: Splits the selected data."""
    print("\n--- Starting 09_data_splitter ---")
    
    try:
        X = pd.read_csv(TRAIN_PROCESSED_SELECTED_CSV)
        y = pd.read_csv(TARGET_TRAIN_CSV) 
    except FileNotFoundError:
        print(f"ERROR: Files not found in {DATA_PIPELINE_DIR}.")
        return False

    print(f"Data loaded: {len(X)} samples.")

    X_temp, X_holdout, y_temp, y_holdout = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=84, stratify=y_temp
    )

    print(f"Saving splits to {DATA_PIPELINE_DIR}...")
    X_train.to_csv(X_TRAIN_SPLIT_FILE, index=False)
    y_train.to_csv(Y_TRAIN_SPLIT_FILE, index=False)
    X_val.to_csv(X_VAL_SPLIT_FILE, index=False)
    y_val.to_csv(Y_VAL_SPLIT_FILE, index=False)
    X_holdout.to_csv(X_HOLDOUT_SPLIT_FILE, index=False)
    y_holdout.to_csv(Y_HOLDOUT_SPLIT_FILE, index=False)

    print(f"Save complete in folder: {DATA_PIPELINE_DIR}")
    print("--- Completed 09_data_splitter ---")
    return True


# --- 8. PIPELINE FUNCTIONS (PHASE 5 & 6 - TRAINING AND SUBMISSION) ---

def run_10_optimize_and_validate(run_grid_search=False, recalculate_iterations=False):
    """
    PHASE 5: Performs optimisation (optional) and validation on the 20% holdout.
    Saves parameters, reports, and plots.

    Args:
        run_grid_search (bool): If True, execute GridSearchCV to find new parameters.
                                If False, load parameters from PARAMS_OUTPUT_FILE.
        recalculate_iterations (bool): If True, run early stopping to find the best iteration.
                                       If False, load iteration from ITERATION_OUTPUT_FILE.
    """
    print("START PHASE 5: Optimization and Validation")

    print("Loading 60% Train and 20% Validation data...")
    try:
        X_train = pd.read_csv(X_TRAIN_SPLIT_FILE)
        y_train = pd.read_csv(Y_TRAIN_SPLIT_FILE).values.ravel()
        X_val = pd.read_csv(X_VAL_SPLIT_FILE)
        y_val = pd.read_csv(Y_VAL_SPLIT_FILE).values.ravel()
        
        best_params_clean = {}
        if not run_grid_search:
            if os.path.exists(PARAMS_OUTPUT_FILE):
                print(f"Loading existing parameters from {PARAMS_OUTPUT_FILE}...")
                with open(PARAMS_OUTPUT_FILE, 'r') as f:
                    best_params_clean = json.load(f)
            else:
                print("WARNING: run_grid_search=False but parameter file not found. Defaults will be used.")
        else:
            print("run_grid_search=True. Existing parameters will be overwritten.")

    except FileNotFoundError as e:
        print(f"ERROR: Files not found in {DATA_PIPELINE_DIR}.")
        print(e)
        return False

    if run_grid_search:
        print(f"\nPhase 5.1: Starting GridSearchCV (on 60% Train):\n")
        start_time = time.time()
        param_grid = {
            'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2],
            'depth': [4, 5, 6, 8, 10],                            
            'l2_leaf_reg': [1.0, 2.0, 2.5, 3.0, 7.0, 10.0],
        }
        
        combinations = 1
        for k in param_grid: combinations *= len(param_grid[k])
        print(f"GridSearch parameters to test: {param_grid}")

        base_model = CatBoostClassifier(
            objective='Logloss', eval_metric='AUC', verbose=0,
            random_seed=42, n_estimators=1000
        )
        kf_inner = KFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            estimator=base_model, param_grid=param_grid, cv=kf_inner,
            scoring='roc_auc', n_jobs=-1, verbose=2
        )
        grid_search.fit(X_train, y_train)
        end_time = time.time()
        print(f"GridSearch completed in {end_time - start_time:.2f} seconds.")

        best_params_clean = grid_search.best_params_
        best_score_clean = grid_search.best_score_
        print(f"\nBest AUC Score (CV on 60%): {best_score_clean:.6f}")
        print(f"Best Parameters found: {best_params_clean}")

        # Save parameters
        final_params_to_save = {
            **best_params_clean,
            'objective': 'Logloss', 'eval_metric': 'AUC',
            'verbose': 0, 'random_seed': 42
        }
        with open(PARAMS_OUTPUT_FILE, 'w') as f:
            json.dump(final_params_to_save, f, indent=2)
        print(f"Parameters saved to: {PARAMS_OUTPUT_FILE}")
        best_params_clean = final_params_to_save

    else:
        print(f"\nPhase 5.1: GridSearchCV skipped (run_grid_search=False):\n")
        if not best_params_clean:
            print("No parameters found, using CatBoost defaults.")
            best_params_clean = {'objective': 'Logloss', 'eval_metric': 'AUC', 'verbose': 0, 'random_seed': 42}

    print("\nPhase 5.2: Training on 60% and Diagnostics on 20% (Validation):\n")

    loaded_iteration = None
    if not recalculate_iterations:
        if os.path.exists(ITERATION_OUTPUT_FILE):
            print(f"Found existing iteration file: {ITERATION_OUTPUT_FILE}")
            try:
                with open(ITERATION_OUTPUT_FILE, 'r') as f:
                    loaded_iteration = json.load(f)['best_iteration']
                print(f"Pre-calculated iteration count loaded: {loaded_iteration}")
            except Exception as e:
                print(f"Error loading {ITERATION_OUTPUT_FILE}: {e}. Recalculating...")
                loaded_iteration = None 
        else:
             print(f"Iteration file '{ITERATION_OUTPUT_FILE.name}' not found. Will run early stopping.")
    else:
        print("recalculate_iterations=True. Forcing re-calculation of best iteration count.")

    final_params_fit = best_params_clean.copy()
    if loaded_iteration:
        final_params_fit.update({
            'n_estimators': loaded_iteration,
            'early_stopping_rounds': None, 
            'eval_metric': 'Logloss', 'custom_metric': ['AUC'],
            'verbose': 1000, 'random_seed': 42
        })
    else:
        final_params_fit.update({
            'n_estimators': 2000,
            'early_stopping_rounds': 50, 
            'eval_metric': 'Logloss', 'custom_metric': ['AUC'],
            'verbose': 1000, 'random_seed': 42
        })
    
    final_params_fit.pop('objective', None) 

    print("Training model: Train (60%), Eval (20%)...")
    model = CatBoostClassifier(**final_params_fit)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        plot=False,
        use_best_model=True, 
        verbose=1000
    )

    if loaded_iteration:
        best_iteration = loaded_iteration
        print(f"\nUsed pre-loaded iteration: {best_iteration}")
    else:
        best_iteration = model.get_best_iteration()
        print(f"\nOptimal number of iterations (trees) found: {best_iteration}")
        with open(ITERATION_OUTPUT_FILE, 'w') as f:
            json.dump({'best_iteration': best_iteration}, f, indent=2)
        print(f"Best iteration saved to: {ITERATION_OUTPUT_FILE}")

    print("\nPhase 5.3: Performance Metrics (on 20% Validation):\n")
    y_pred_val = model.predict(X_val)
    y_pred_proba_val = model.predict_proba(X_val)[:, 1]

    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    val_accuracy = accuracy_score(y_val, y_pred_val)
    val_auc = roc_auc_score(y_val, y_pred_proba_val)

    print(f"  Accuracy (Training 60%): {train_accuracy:.4f}")
    print(f"  Accuracy (Validation 20%): {val_accuracy:.4f}")
    print(f"  AUC (Validation 20%): {val_auc:.4f}")

    if train_accuracy > (val_accuracy + 0.05):
        print("\033[93m  WARNING: Possible Overfitting! (Delta > 5%)\033[0m")
    else:
        print("\033[92m  OK: No obvious overfitting.\033[0m")

    report_text_val = classification_report(y_val, y_pred_val, target_names=['False (0)', 'True (1)'], digits=4)

    print(f"Saving metrics (Validation) to: {REPORT_TXT_FILE}")
    with open(REPORT_TXT_FILE, 'w') as f:
        f.write("Classification Report (on 20% Validation):\n\n")
        f.write(report_text_val)

    print("\nPhase 5.4: Saving Diagnostic Plots (Validation):\n")

    cm_val = confusion_matrix(y_val, y_pred_val)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted False (0)', 'Predicted True (1)'],
                yticklabels=['Actual False (0)', 'Actual True (1)'])
    plt.title("Confusion Matrix (20% Validation Set)")
    plt.savefig(CM_OUTPUT_FILE)
    plt.close()

    importances = model.get_feature_importance()
    feature_names = X_train.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    top_20_features = importance_df.sort_values(by='Importance', ascending=False).head(20)
    print("\nTop 20 Most Important Features:")
    print(top_20_features.to_string(index=False))
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', data=top_20_features, palette='viridis')
    plt.title('Top 20 Feature Importance (CatBoost)')
    plt.tight_layout()
    plt.savefig(IMPORTANCE_OUTPUT_FILE)
    plt.close()

    results = model.get_evals_result()
    train_set_name = 'validation_0'
    valid_set_name = 'validation_1'
    if 'Logloss' in results.get(train_set_name, {}):
        plt.figure(figsize=(10, 6))
        plt.plot(results[train_set_name]['Logloss'], label='Training Loss (60%)')
        plt.plot(results[valid_set_name]['Logloss'], label='Validation Loss (20%)')
        plt.title('Learning Curve (Logloss)')
        plt.xlabel('Iterations'); plt.ylabel('Logloss'); plt.legend(); plt.grid(True)
        plt.savefig(LOSS_CURVE_FILE)
        plt.close()
    if 'AUC' in results.get(train_set_name, {}):
        plt.figure(figsize=(10, 6))
        plt.plot(results[train_set_name]['AUC'], label='Training AUC (60%)')
        plt.plot(results[valid_set_name]['AUC'], label='Validation AUC (20%)')
        plt.title('Learning Curve (AUC)')
        plt.xlabel('Iterations'); plt.ylabel('AUC'); plt.legend(); plt.grid(True)
        plt.savefig(AUC_CURVE_FILE)
        plt.close()

    fpr_val, tpr_val, _ = roc_curve(y_val, y_pred_proba_val)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_val, tpr_val, color='blue', lw=2, label=f'ROC Curve (Validation AUC = {val_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Guess (AUC = 0.50)')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC Curve (20% Validation Set)')
    plt.legend(loc="lower right"); plt.grid(True)
    plt.savefig(ROC_CURVE_FILE)
    plt.close()
    
    print("PHASE 5 COMPLETE")
    return True

def run_11_create_submission():
    """
    PHASE 6: Runs OOF, trains on 100%, generates .npy and .csv files.
    """
    print("START PHASE 6: OOF Creation and Submission")
    
    N_SPLITS = 10

    print("Loading 100% Training, Test data, and Parameters...")
    try:
        X_train_full = pd.read_csv(TRAIN_PROCESSED_SELECTED_CSV)
        y_train_full = pd.read_csv(TARGET_TRAIN_CSV).values.ravel()
        X_test_kaggle = pd.read_csv(TEST_PROCESSED_SELECTED_CSV)

        with open(PARAMS_OUTPUT_FILE, 'r') as f:
            best_params = json.load(f)
        print(f"Parameters loaded from {PARAMS_OUTPUT_FILE}.")

        with open(ITERATION_OUTPUT_FILE, 'r') as f:
            best_iteration = json.load(f)['best_iteration'] 
        print(f"Optimal iteration count loaded: {best_iteration}")

    except FileNotFoundError as e:
        print("ERROR: Files not found.")
        print(e)
        print(f"Ensure CSV files are in {DATA_PIPELINE_DIR}")
        print(f"and parameter JSON files are in {MODEL_OUTPUT_DIR} (from Phase 5).")
        return False

    # --- SECTION 6.1: GENERATE OOF PREDICTIONS (CROSS-VALIDATION) ---
    print(f"\n--- Starting Out-of-Fold (OOF) generation ({N_SPLITS} Folds) ---")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train_full))

    oof_params = best_params.copy()
    oof_params.update({
        'n_estimators': best_iteration,
        'early_stopping_rounds': None,
        'verbose': 0, 'random_seed': 42
    })
    for key in ['eval_metric', 'custom_metric', 'objective']:
        oof_params.pop(key, None)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
        print(f"--- Processing Fold {fold+1}/{N_SPLITS} ---")
        X_train_fold, X_val_fold = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_train_fold = y_train_full[train_idx]
        
        model_fold = CatBoostClassifier(**oof_params)
        model_fold.fit(X_train_fold, y_train_fold)
        oof_preds[val_idx] = model_fold.predict_proba(X_val_fold)[:, 1]

    print("OOF prediction generation complete.")
    print(f"Saving OOF predictions to {OOF_FILE_NPY}...")
    np.save(OOF_FILE_NPY, oof_preds)
    print(f"OOF file saved to: {OOF_FILE_NPY} (Shape: {oof_preds.shape})")

    # --- SECTION 6.2: TRAIN FINAL MODEL ON 100% ---
    print(f"\n--- Starting Final Model Training ---")

    final_params = best_params.copy()
    final_params.update({
        'n_estimators': best_iteration,
        'early_stopping_rounds': None,
        'verbose': 200, 'random_seed': 42
    })
    for key in ['eval_metric', 'custom_metric', 'objective']:
        final_params.pop(key, None)

    final_model = CatBoostClassifier(**final_params)
    final_model.fit(X_train_full, y_train_full)
    print("Final training complete.")

    # --- SECTION 6.3: GENERATE SUBMISSION ON TEST SET (CSV and NPY) ---

    y_pred_proba = final_model.predict_proba(X_test_kaggle)[:, 1]

    print(f"Saving test set predictions to {TEST_PREDS_NPY_FILE}...")
    np.save(TEST_PREDS_NPY_FILE, y_pred_proba)
    print(f"Test set .npy file for stacking saved to: {TEST_PREDS_NPY_FILE}")

    # Load TEST SET battle_ids for the .CSV file
    try:
        print(f"Loading 'battle_id' from: '{TEST_FEATURES_FINAL_CSV}'...")
        test_ids_df = pd.read_csv(TEST_FEATURES_FINAL_CSV)
        test_ids = test_ids_df['battle_id']
        
        if not X_test_kaggle.index.equals(test_ids_df.index):
            print("WARNING: Index mismatch. Realigning indices.")
            test_ids = test_ids_df.loc[X_test_kaggle.index, 'battle_id'].values
        else:
             test_ids = test_ids.values

    except Exception as e:
        print(f"ERROR loading battle_ids from {TEST_FEATURES_FINAL_CSV}: {e}")
        return False

    if len(test_ids) != len(y_pred_proba):
        print(f"ERROR: Mismatch between number of test IDs ({len(test_ids)}) and predictions ({len(y_pred_proba)}).")
        return False

    # Save the final submission .CSV file
    print(f"Saving submission file with predictions to {SUBMISSION_FILE_CSV}...")
    submission_df = pd.DataFrame({
        'battle_id': test_ids,
        'player_won_proba': y_pred_proba
    })
    submission_df.to_csv(SUBMISSION_FILE_CSV, index=False)

    print("\n   SUBMISSION READY  ")
    print(f"File saved to: {SUBMISSION_FILE_CSV}")
    print(submission_df.head())

    print("PHASE 6 COMPLETE")
    return True
