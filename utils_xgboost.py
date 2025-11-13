"""
Utility library for the XGBoost Pokémon pipeline (v6).

This module contains all the necessary functions, constants, and path configurations
for the entire pipeline:
- Data Loading
- All Feature Engineering logic (v6)
- Preprocessing (Imputation)
- Training (K-Fold OOF and final model)
- Submission creation

"""

# --- 1. CONSOLIDATED IMPORTS ---
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
from collections import Counter
import warnings
from pathlib import Path  # Modern path management
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# --- 2. GLOBAL CONFIGURATION AND PATHS ---

# Define the project base directory
BASE_DIR = Path(__file__).resolve().parent

# Folders
INPUT_DIR_JSONL = BASE_DIR / 'Input'
DATA_DIR = BASE_DIR / 'XGBoost_Data_Pipeline'
MODEL_OUTPUT_DIR = BASE_DIR / 'XGBoost_Model_Outputs'
SUBMISSION_DIR = BASE_DIR / 'Submissions'
OOF_DIR = BASE_DIR / 'OOF_Predictions'

# Input Files (raw)
TRAIN_JSON_IN = INPUT_DIR_JSONL / 'train.jsonl'
TEST_JSON_IN = INPUT_DIR_JSONL / 'test.jsonl'

# Intermediate Files (Phase 1 -> Phase 2)
TRAIN_CSV_OUT = DATA_DIR / 'train_features.csv'
TEST_CSV_OUT = DATA_DIR / 'test_features.csv'
TEST_IDS_OUT = DATA_DIR / 'test_ids.csv' # IDs for submission

# Output Files (Preprocessing Metadata)
FEATURES_JSON_OUT = MODEL_OUTPUT_DIR / 'features_final.json'
MEDIANS_JSON_OUT = MODEL_OUTPUT_DIR / 'preprocessing_medians.json'

# Output Files (Training)
OOF_XGB_OUT_FILE = OOF_DIR / 'oof_xgboost_proba.npy'
TEST_PREDS_XGB_OUT_FILE = OOF_DIR / 'test_preds_xgboost_proba.npy'
SUBMISSION_XGB_PROBA_FILE = SUBMISSION_DIR / 'submission_xgboost_100pct_PROBA.csv'

# --- 3. MODEL AND GAME CONSTANTS ---

SEED = 42
N_SPLITS = 10 # Number of folds

# HYPERPARAMETERS (From Optuna in XGBoost_v6 notebook, Trial 84)
BEST_PARAMS = {
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'eval_metric': 'logloss',
    'use_label_encoder': False,
    'random_state': SEED,
    'n_estimators': 1000, 
    'learning_rate': 0.019161090151695974,
    'max_depth': 3,
    'min_child_weight': 6,
    'gamma': 2.4267729113636345,
    'subsample': 0.6391418336680764,
    'colsample_bytree': 0.8034979811909722,
    'reg_alpha': 0.04656745646903133,
    'reg_lambda': 0.25114366021463247
}

# Type effectiveness constants (V6)
TYPE_CHART = {
    'normal': {'rock': 0.5, 'ghost': 0},
    'fire': {'fire': 0.5, 'water': 0.5, 'grass': 2, 'ice': 2, 'bug': 2, 'rock': 0.5, 'dragon': 0.5},
    'water': {'fire': 2, 'water': 0.5, 'grass': 0.5, 'ground': 2, 'rock': 2, 'dragon': 0.5},
    'grass': {'fire': 0.5, 'water': 2, 'grass': 0.5, 'poison': 0.5, 'ground': 2, 'flying': 0.5, 'bug': 0.5, 'rock': 2, 'dragon': 0.5},
    'electric': {'water': 2, 'grass': 0.5, 'electric': 0.5, 'ground': 0, 'flying': 2, 'dragon': 0.5},
    'ice': {'fire': 0.5, 'water': 0.5, 'grass': 2, 'ground': 2, 'flying': 2, 'dragon': 2},
    'fighting': {'normal': 2, 'ice': 2, 'poison': 0.5, 'flying': 0.5, 'psychic': 0.5, 'bug': 0.5, 'rock': 2, 'ghost': 0},
    'poison': {'grass': 2, 'poison': 0.5, 'ground': 0.5, 'bug': 2, 'rock': 0.5, 'ghost': 0.5},
    'ground': {'fire': 2, 'grass': 0.5, 'electric': 2, 'poison': 2, 'flying': 0, 'bug': 0.5, 'rock': 2},
    'flying': {'grass': 2, 'electric': 0.5, 'fighting': 2, 'bug': 2, 'rock': 0.5},
    'psychic': {'fighting': 2, 'poison': 2, 'psychic': 0.5, 'ghost': 0},
    'bug': {'fire': 0.5, 'grass': 2, 'fighting': 0.5, 'poison': 2, 'flying': 0.5, 'psychic': 2, 'ghost': 0.5},
    'rock': {'fire': 2, 'ice': 2, 'fighting': 0.5, 'ground': 0.5, 'flying': 2, 'bug': 2},
    'ghost': {'normal': 0, 'psychic': 0, 'ghost': 2},
    'dragon': {'dragon': 2}
}

ALL_ATTACK_TYPES = list(TYPE_CHART.keys())


# --- 4. HELPER FUNCTIONS (Feature Engineering V6) ---

def load_jsonl(path):
    """Loads a .jsonl file and returns it as a list of dictionaries."""
    data = []
    try:
        with open(path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    except FileNotFoundError:
        print(f"ERROR: File not found: {path}")
        print("Ensure 'train.jsonl' and 'test.jsonl' are in the 'Input/' directory.")
        return None
    return data

def get_effectiveness(attack_type: str, defense_types: list) -> float:
    """Calculates the effectiveness of one type against a list of types."""
    if not attack_type or not defense_types: return 1.0
    eff = 1.0
    for d in defense_types: eff *= TYPE_CHART.get(attack_type, {}).get(d, 1.0)
    return eff

def _entropy(counter: Counter) -> float:
    """Calculates the entropy of a Counter."""
    total = sum(counter.values())
    if total == 0: return 0.0
    ent = 0.0
    for v in counter.values():
        p = v / total
        if p > 0: ent -= p * math.log(p, 2)
    return ent

def calculate_type_advantage(team1: list, team2_lead: dict) -> dict:
    """Calculates the aggregated advantage of team1 against team2's lead."""
    out = {'p1_vs_lead_avg_effectiveness': 0.0, 'p1_vs_lead_max_effectiveness': 0.0, 'p1_super_effective_options': 0}
    if not team1 or not team2_lead: return out
    lead_types = [t.lower() for t in team2_lead.get('types', [])]
    if not lead_types: return out
    effs = []
    for p in team1:
        p_types = [t.lower() for t in p.get('types', [])]
        max_eff = 0.0
        for pt in p_types: max_eff = max(max_eff, get_effectiveness(pt, lead_types))
        effs.append(max_eff)
    if not effs: return out
    out['p1_vs_lead_avg_effectiveness'] = float(np.mean(effs))
    out['p1_vs_lead_max_effectiveness'] = float(np.max(effs))
    out['p1_super_effective_options'] = int(sum(1 for e in effs if e >= 2))
    return out

def team_aggregate_features(team: list, prefix: str = 'p1_') -> dict:
    """Extracts aggregate statistics (mean, sum, std, etc.) from a team."""
    stats = ['base_hp','base_atk','base_def','base_spa','base_spd','base_spe']
    out = {}
    vals = {s: [] for s in stats}
    levels = []; types_counter = Counter(); names = []
    for p in team:
        names.append(p.get('name',''))
        for s in stats: vals[s].append(p.get(s, 0))
        levels.append(p.get('level', 0))
        for t in p.get('types', []): types_counter[t.lower()] += 1
    for s in stats:
        arr = np.array(vals[s], dtype=float)
        out[f'{prefix}{s}_sum'] = float(arr.sum())
        out[f'{prefix}{s}_mean'] = float(arr.mean())
        out[f'{prefix}{s}_max'] = float(arr.max())
        out[f'{prefix}{s}_min'] = float(arr.min())
        out[f'{prefix}{s}_std'] = float(arr.std())
    level_arr = np.array(levels, dtype=float)
    out[f'{prefix}level_mean'] = float(level_arr.mean()) if level_arr.size else 0.0
    out[f'{prefix}level_sum'] = float(level_arr.sum()) if level_arr.size else 0.0
    out[f'{prefix}n_unique_types'] = int(len(types_counter))
    for t in ['normal','fire','water','electric','grass','psychic','ice','dragon','rock','ground','flying']:
        out[f'{prefix}type_{t}_count'] = int(types_counter.get(t, 0))
    out[f'{prefix}lead_name'] = names[0] if names else ''
    out[f'{prefix}n_unique_names'] = int(len(set(names)))
    out[f'{prefix}type_entropy'] = float(_entropy(types_counter))
    spe_arr = np.array(vals['base_spe'], dtype=float)
    out[f'{prefix}spe_p25'] = float(np.percentile(spe_arr, 25)) if spe_arr.size else 0.0
    out[f'{prefix}spe_p50'] = float(np.percentile(spe_arr, 50)) if spe_arr.size else 0.0
    out[f'{prefix}spe_p75'] = float(np.percentile(spe_arr, 75)) if spe_arr.size else 0.0
    return out

def lead_vs_lead_features(p1_lead: dict, p2_lead: dict) -> dict:
    """Directly compares the stats of the two lead Pokémon."""
    out = {}
    for s in ['base_hp','base_atk','base_def','base_spa','base_spd','base_spe']:
        out[f'lead_diff_{s}'] = float(p1_lead.get(s,0) - p2_lead.get(s,0))
    out['lead_speed_advantage'] = float(p1_lead.get('base_spe',0) - p2_lead.get('base_spe',0))
    p1_types = [t.lower() for t in p1_lead.get('types', [])]
    p2_types = [t.lower() for t in p2_lead.get('types', [])]
    max_eff = 0.0
    for pt in p1_types: max_eff = max(max_eff, get_effectiveness(pt, p2_types))
    out['lead_p1_vs_p2_effectiveness'] = float(max_eff)
    return out

def lead_aggregate_features(pokemon: dict, prefix: str = 'p2_lead_') -> dict:
    """Extracts base features for a single Pokémon (the opponent's lead)."""
    out = {}
    for s in ['base_hp','base_atk','base_def','base_spa','base_spd','base_spe']:
        out[f'{prefix}{s}'] = float(pokemon.get(s,0))
    out[f'{prefix}level'] = int(pokemon.get('level',0))
    types = [x.lower() for x in pokemon.get('types', [])]
    for t in ['normal','fire','water','electric','grass','psychic','ice','dragon','rock','ground','flying']:
        out[f'{prefix}type_{t}'] = int(t in types)
    out[f'{prefix}name'] = pokemon.get('name','')
    out[f'{prefix}n_unique_types'] = int(len(set(types)))
    return out

def quick_boost_features_v2(record: dict) -> dict:
    """Extracts "quick" features comparing team vs. lead."""
    out = {}
    p1_team = record.get('p1_team_details', [])
    p2_lead = record.get('p2_lead_details', {})
    timeline = record.get('battle_timeline', [])
    if not p1_team: return out
    
    p2_lead_spe = p2_lead.get('base_spe', 0)
    faster_count = sum(1 for p in p1_team if p.get('base_spe', 0) > p2_lead_spe)
    slower_count = sum(1 for p in p1_team if p.get('base_spe', 0) <= p2_lead_spe)
    out['p1_faster_than_lead_count'] = faster_count
    out['p1_slower_than_lead_count'] = slower_count
    out['p1_speed_control_ratio'] = faster_count / max(1, len(p1_team))
    
    p1_avg_bulk = np.mean([p.get('base_hp', 0)*(p.get('base_def', 0)+p.get('base_spd', 0)) for p in p1_team])
    p2_lead_bulk = p2_lead.get('base_hp', 1)*(p2_lead.get('base_def', 1)+p2_lead.get('base_spd', 1))
    out['p1_avg_bulk_vs_lead'] = p1_avg_bulk / max(p2_lead_bulk, 1)
    
    p1_total_atk = sum(p.get('base_atk', 0) + p.get('base_spa', 0) for p in p1_team)
    p2_lead_offense = p2_lead.get('base_atk', 0) + p2_lead.get('base_spa', 0)
    out['p1_total_offense'] = p1_total_atk
    out['p1_offense_advantage'] = p1_total_atk / max(p2_lead_offense, 1)
    
    p2_lead_types = [t.lower() for t in p2_lead.get('types', [])]
    if p2_lead_types:
        coverage_scores = []
        for p in p1_team:
            p_types = [t.lower() for t in p.get('types', [])]
            max_eff = max([get_effectiveness(pt, p2_lead_types) for pt in p_types] or [1.0])
            coverage_scores.append(max_eff)
        out['p1_avg_effectiveness_vs_lead'] = float(np.mean(coverage_scores))
        out['p1_max_effectiveness_vs_lead'] = float(np.max(coverage_scores))
        out['p1_se_count_vs_lead'] = sum(1 for s in coverage_scores if s >= 2.0)
        out['p1_weak_count_vs_lead'] = sum(1 for s in coverage_scores if s <= 0.5)
        
    if timeline:
        first_p1_ko = False; first_p2_ko = False
        for turn in timeline[:30]:
            if not first_p2_ko and turn.get('p2_pokemon_state', {}).get('fainted'):
                first_p1_ko = True; out['p1_first_blood'] = 1; out['p1_first_blood_turn'] = turn.get('turn', 0); break
            if not first_p1_ko and turn.get('p1_pokemon_state', {}).get('fainted'):
                first_p2_ko = True; out['p1_first_blood'] = 0; out['p1_first_blood_turn'] = turn.get('turn', 0); break
        if not first_p1_ko and not first_p2_ko:
            out['p1_first_blood'] = -1; out['p1_first_blood_turn'] = 0
            
    p1_avg_level = np.mean([p.get('level', 50) for p in p1_team])
    out['p1_avg_level_advantage'] = p1_avg_level - p2_lead.get('level', 50)
    
    p1_stat_products = [(p.get('base_hp',1)*p.get('base_atk',1)*p.get('base_def',1)*p.get('base_spa',1)*p.get('base_spd',1)*p.get('base_spe',1)) for p in p1_team]
    out['p1_avg_stat_product'] = float(np.mean(p1_stat_products))
    out['p1_max_stat_product'] = float(np.max(p1_stat_products))
    p2_prod = p2_lead.get('base_hp',1)*p2_lead.get('base_atk',1)*p2_lead.get('base_def',1)*p2_lead.get('base_spa',1)*p2_lead.get('base_spd',1)*p2_lead.get('base_spe',1)
    out['p1_stat_product_advantage'] = out['p1_avg_stat_product'] / max(p2_prod, 1)
    return out

def summary_from_timeline(timeline: list, p1_team: list) -> dict:
    """Extracts a detailed battle summary (v6: with crit/lost turns)."""
    out = {}
    if not timeline: return {'tl_p1_moves':0,'tl_p2_moves':0,'tl_p1_est_damage':0.0,'tl_p2_est_damage':0.0,'damage_diff':0.0}
    p1_moves = p2_moves = 0; p1_damage = p2_damage = 0.0
    p1_last_active = p2_last_active = ''; p1_last_hp = p2_last_hp = np.nan
    p1_fainted = p2_fainted = 0
    p1_fainted_names = set(); p2_fainted_names = set()
    last_p1_hp = {}; last_p2_hp = {}
    p1_comeback_kos = p2_comeback_kos = 0
    p1_inflicted_statuses = Counter(); p2_inflicted_statuses = Counter()
    p1_pokemon_statuses = {}; p2_pokemon_statuses = {}
    p1_move_type_counts = Counter(); p2_move_type_counts = Counter()
    p1_damage_first2 = 0.0; p2_damage_first2 = 0.0
    p1_dmg_by_turn = {}; p2_dmg_by_turn = {}; seen_turns = set()
    first_ko_turn_p1_taken = None; first_ko_turn_p1_inflicted = None
    early_threshold = 10; p1_kos_early = p1_kos_late = p2_kos_early = p2_kos_late = 0
    
    p1_crit_count = 0; p2_crit_count = 0
    p1_lost_turns_status = 0; p2_lost_turns_status = 0

    for i, turn in enumerate(timeline[:30]):
        prev_p1_fainted, prev_p2_fainted = p1_fainted, p2_fainted
        p1_state = turn.get('p1_pokemon_state',{}) or {}; p2_state = turn.get('p2_pokemon_state',{}) or {}
        tnum = turn.get('turn', len(seen_turns) + 1); seen_turns.add(tnum)

        if p1_state.get('name'): p1_last_active = p1_state.get('name')
        if p2_state.get('name'): p2_last_active = p2_state.get('name')

        if p1_state.get('fainted') and p1_state.get('name') not in p1_fainted_names:
            p1_fainted += 1; p1_fainted_names.add(p1_state.get('name'))
            if first_ko_turn_p1_taken is None: first_ko_turn_p1_taken = tnum
            if tnum <= early_threshold: p2_kos_early += 1
            else: p2_kos_late += 1
        if p2_state.get('fainted') and p2_state.get('name') not in p2_fainted_names:
            p2_fainted += 1; p2_fainted_names.add(p2_state.get('name'))
            if first_ko_turn_p1_inflicted is None: first_ko_turn_p1_inflicted = tnum
            if tnum <= early_threshold: p1_kos_early += 1
            else: p1_kos_late += 1

        p2_name, p2_hp = p2_state.get('name'), p2_state.get('hp_pct')
        if p2_name and p2_hp is not None:
            prev_hp = last_p2_hp.get(p2_name)
            if prev_hp is not None:
                delta = max(0.0, prev_hp - p2_hp)
                p1_damage += delta
                p1_dmg_by_turn[tnum] = p1_dmg_by_turn.get(tnum, 0.0) + delta
                if turn.get('turn',999) <= 2: p1_damage_first2 += delta
            last_p2_hp[p2_name] = p2_hp

        p1_name, p1_hp = p1_state.get('name'), p1_state.get('hp_pct')
        if p1_name and p1_hp is not None:
            prev_hp = last_p1_hp.get(p1_name)
            if prev_hp is not None:
                delta = max(0.0, prev_hp - p1_hp)
                p2_damage += delta
                p2_dmg_by_turn[tnum] = p2_dmg_by_turn.get(tnum, 0.0) + delta
                if turn.get('turn',999) <= 2: p2_damage_first2 += delta
            last_p1_hp[p1_name] = p1_hp

        damage_diff_so_far = p1_damage - p2_damage
        if p2_fainted > prev_p2_fainted and damage_diff_so_far < -1.0: p1_comeback_kos += 1
        if p1_fainted > prev_p1_fainted and damage_diff_so_far > 1.0: p2_comeback_kos += 1

        p2_status = p2_state.get('status')
        if p2_name and p2_status and p2_pokemon_statuses.get(p2_name) != p2_status:
            p1_inflicted_statuses[p2_status] += 1; p2_pokemon_statuses[p2_name] = p2_status
        p1_status = p1_state.get('status')
        if p1_name and p1_status and p1_pokemon_statuses.get(p1_name) != p1_status:
            p2_inflicted_statuses[p1_status] += 1; p1_pokemon_statuses[p1_name] = p1_status

        p1_move = turn.get('p1_move_details') or {}; p2_move = turn.get('p2_move_details') or {}
        if p1_move and p1_move.get('type'): p1_move_type_counts[(p1_move.get('type') or '').lower()] += 1
        if p2_move and p2_move.get('type'): p2_move_type_counts[(p2_move.get('type') or '').lower()] += 1
        if p1_move: p1_moves += 1
        if p2_move: p2_moves += 1
        
        if p1_move.get('critical_hit', False): p1_crit_count += 1
        if p2_move.get('critical_hit', False): p2_crit_count += 1
        
        prev_p1_name = timeline[i-1].get('p1_pokemon_state',{}).get('name') if i > 0 else None
        if p1_state.get('status') in ['par', 'slp'] and not p1_move and p1_state.get('name') == prev_p1_name:
            p1_lost_turns_status += 1
            
        prev_p2_name = timeline[i-1].get('p2_pokemon_state',{}).get('name') if i > 0 else None
        if p2_state.get('status') in ['par', 'slp'] and not p2_move and p2_state.get('name') == prev_p2_name:
            p2_lost_turns_status += 1
            
        p1_last_hp = p1_state.get('hp_pct', np.nan); p2_last_hp = p2_state.get('hp_pct', np.nan)

    out['tl_p1_moves'] = int(p1_moves); out['tl_p2_moves'] = int(p2_moves)
    out['tl_p1_est_damage'] = float(p1_damage); out['tl_p2_est_damage'] = float(p2_damage)
    out['tl_p1_fainted'] = int(p1_fainted); out['tl_p2_fainted'] = int(p2_fainted)
    turns_count = max(1, len(seen_turns))
    out['tl_p1_fainted_rate'] = float(out['tl_p1_fainted'] / turns_count)
    out['tl_p2_fainted_rate'] = float(out['tl_p2_fainted'] / turns_count)
    out['damage_diff'] = float(p1_damage - p2_damage)
    out['fainted_diff'] = int(p1_fainted - p2_fainted)
    out['tl_p1_last_hp'] = float(p1_last_hp) if not np.isnan(p1_last_hp) else 0.0
    out['tl_p2_last_hp'] = float(p2_last_hp) if not np.isnan(p2_last_hp) else 0.0
    out['tl_p1_last_active'] = p1_last_active; out['tl_p2_last_active'] = p2_last_active
    
    if p1_team:
        p1_total_hp_sum = sum(p.get('base_hp',0) for p in p1_team)
        p1_avg_def = np.mean([p.get('base_def',0) for p in p1_team] or [0])
        p1_avg_spd = np.mean([p.get('base_spd',0) for p in p1_team] or [0])
        out['tl_p2_damage_vs_p1_hp_pool'] = float(p2_damage / (p1_total_hp_sum + 1e-6))
        out['tl_p1_defensive_endurance'] = float((p1_avg_def + p1_avg_spd) / (p2_damage + 1e-6))
        
    out['tl_p1_comeback_kos'] = int(p1_comeback_kos); out['tl_p2_comeback_kos'] = int(p2_comeback_kos)
    out['tl_comeback_kos_diff'] = int(p1_comeback_kos - p2_comeback_kos)

    common_statuses = ['brn','par','slp','frz','psn','tox']
    for status in common_statuses:
        out[f'tl_p1_inflicted_{status}_count'] = int(p1_inflicted_statuses.get(status,0))
        out[f'tl_p2_inflicted_{status}_count'] = int(p2_inflicted_statuses.get(status,0))
        out[f'tl_inflicted_{status}_diff'] = int(p1_inflicted_statuses.get(status,0) - p2_inflicted_statuses.get(status,0))
        c1 = p1_inflicted_statuses.get(status,0); c2 = p2_inflicted_statuses.get(status,0)
        out[f'tl_p1_inflicted_{status}_rate'] = float(c1 / turns_count)
        out[f'tl_p2_inflicted_{status}_rate'] = float(c2 / turns_count)
        out[f'tl_inflicted_{status}_rate_diff'] = float((c1 - c2) / turns_count)

    common_move_types = ['normal','fire','water','electric','grass','psychic','ice','dragon','rock','ground','flying','ghost','bug','poison','fighting']
    for mt in common_move_types:
        out[f'tl_p1_move_type_{mt}_count'] = int(p1_move_type_counts.get(mt,0))
        out[f'tl_p2_move_type_{mt}_count'] = int(p2_move_type_counts.get(mt,0))
        out[f'tl_move_type_{mt}_count_diff'] = int(p1_move_type_counts.get(mt,0) - p2_move_type_counts.get(mt,0))

    out['tl_p1_damage_first2'] = float(p1_damage_first2)
    out['tl_p2_damage_first2'] = float(p2_damage_first2)
    out['tl_first2_damage_diff'] = float(p1_damage_first2 - p2_damage_first2)
    out['tl_turns_count'] = int(turns_count)
    out['tl_p1_moves_rate'] = float(p1_moves / turns_count); out['tl_p2_moves_rate'] = float(p2_moves / turns_count)
    out['tl_p1_damage_per_turn'] = float(p1_damage / turns_count); out['tl_p2_damage_per_turn'] = float(p2_damage / turns_count)
    out['tl_damage_rate_diff'] = float(out['tl_p1_damage_per_turn'] - out['tl_p2_damage_per_turn'])

    recent_turns = sorted(seen_turns)[-5:] if seen_turns else []
    p1_last5 = sum(p1_dmg_by_turn.get(t,0.0) for t in recent_turns)
    p2_last5 = sum(p2_dmg_by_turn.get(t,0.0) for t in recent_turns)
    out['tl_p1_damage_last5'] = float(p1_last5); out['tl_p2_damage_last5'] = float(p2_last5)
    out['tl_last5_damage_diff'] = float(p1_last5 - p2_last5)
    out['tl_p1_last5_damage_ratio'] = float(p1_last5 / (p1_damage + 1e-6))
    out['tl_p2_last5_damage_ratio'] = float(p2_last5 / (p2_damage + 1e-6))
    out['tl_last5_damage_ratio_diff'] = float(out['tl_p1_last5_damage_ratio'] - out['tl_p2_last5_damage_ratio'])

    if seen_turns:
        ts = sorted(seen_turns); w = np.linspace(1.0, 2.0, num=len(ts)); w = w / (w.sum() + 1e-9)
        adv = [(p1_dmg_by_turn.get(t,0.0) - p2_dmg_by_turn.get(t,0.0)) for t in ts]
        out['tl_weighted_damage_diff'] = float(np.dot(w, adv))
        cum = 0.0; signs = []
        for t in ts:
            cum += (p1_dmg_by_turn.get(t,0.0) - p2_dmg_by_turn.get(t,0.0))
            s = 1 if cum > 1e-9 else (-1 if cum < -1e-9 else 0)
            if s != 0 and (not signs or signs[-1] != s): signs.append(s)
        out['tl_damage_adv_sign_flips'] = int(max(0, len(signs) - 1))
        out['tl_comeback_flag'] = int(1 if (len(signs) >= 2 and signs[0] != signs[-1]) else 0)
    else:
        out['tl_weighted_damage_diff'] = 0.0; out['tl_damage_adv_sign_flips'] = 0; out['tl_comeback_flag'] = 0

    out['tl_first_ko_turn_p1_inflicted'] = int(first_ko_turn_p1_inflicted or 0)
    out['tl_first_ko_turn_p1_taken'] = int(first_ko_turn_p1_taken or 0)
    out['tl_first_ko_turn_diff'] = int((first_ko_turn_p1_inflicted or 0) - (first_ko_turn_p1_taken or 0))
    out['tl_kos_early_p1'] = int(p1_kos_early); out['tl_kos_late_p1'] = int(p1_kos_late)
    out['tl_kos_early_p2'] = int(p2_kos_early); out['tl_kos_late_p2'] = int(p2_kos_late)

    out['tl_p1_crit_count'] = int(p1_crit_count)
    out['tl_p2_crit_count'] = int(p2_crit_count)
    out['tl_crit_diff'] = int(p1_crit_count - p2_crit_count)
    out['tl_p1_lost_turns_status'] = int(p1_lost_turns_status)
    out['tl_p2_lost_turns_status'] = int(p2_lost_turns_status)
    out['tl_status_luck_diff'] = int(p2_lost_turns_status - p1_lost_turns_status)
    
    return out

def extract_move_coverage_from_timeline(timeline: list, prefix: str = 'p1_') -> dict:
    """Extracts the type coverage of moves used."""
    out = {}; move_types_used = set(); move_categories_used = Counter()
    unique_moves = set(); stab_count = 0
    for turn in timeline[:30]:
        move_details = turn.get(f'{prefix[:-1]}_move_details')
        pokemon_state = turn.get(f'{prefix[:-1]}_pokemon_state', {})
        if not move_details: continue
        move_name = move_details.get('name', ''); move_type = (move_details.get('type') or '').lower()
        move_category = move_details.get('category', '')
        if move_name: unique_moves.add(move_name)
        if move_type: move_types_used.add(move_type)
        if move_category: move_categories_used[move_category] += 1
        if move_type in [t.lower() for t in pokemon_state.get('types', [])]: stab_count += 1
    
    out[f'{prefix}tl_unique_move_types'] = len(move_types_used)
    out[f'{prefix}tl_unique_moves_used'] = len(unique_moves)
    out[f'{prefix}tl_stab_moves'] = stab_count
    out[f'{prefix}tl_physical_moves'] = move_categories_used.get('physical', 0)
    out[f'{prefix}tl_special_moves'] = move_categories_used.get('special', 0)
    out[f'{prefix}tl_status_moves'] = move_categories_used.get('status', 0)
    out[f'{prefix}tl_coverage_score'] = len(move_types_used) / max(1, len(unique_moves))
    total_moves = sum(move_categories_used.values())
    if total_moves > 0:
        out[f'{prefix}tl_offensive_ratio'] = (move_categories_used.get('physical',0)+move_categories_used.get('special',0)) / total_moves
        out[f'{prefix}tl_status_ratio'] = move_categories_used.get('status', 0) / total_moves
    else:
        out[f'{prefix}tl_offensive_ratio'] = 0.0; out[f'{prefix}tl_status_ratio'] = 0.0
    return out

def ability_features(team: list, prefix: str) -> dict:
    """Counts key abilities (immunity, intimidate, weather) in a team."""
    immunity_abilities = {'levitate':0,'volt_absorb':0,'water_absorb':0,'flash_fire':0}
    stat_drop_abilities = {'intimidate':0}; weather_abilities = {'drought':0,'drizzle':0,'sand_stream':0}
    out = {}
    for pokemon in team:
        ability = (pokemon.get('ability','') or '').lower().replace(' ','_')
        if ability in immunity_abilities: immunity_abilities[ability] += 1
        if ability in stat_drop_abilities: stat_drop_abilities[ability] += 1
        if ability in weather_abilities: weather_abilities[ability] += 1
    for ability,count in immunity_abilities.items(): out[f'{prefix}ability_{ability}_count'] = int(count)
    for ability,count in stat_drop_abilities.items(): out[f'{prefix}ability_{ability}_count'] = int(count)
    for ability,count in weather_abilities.items(): out[f'{prefix}ability_{ability}_count'] = int(count)
    out[f'{prefix}total_immunity_abilities'] = int(sum(immunity_abilities.values()))
    out[f'{prefix}total_stat_drop_abilities'] = int(sum(stat_drop_abilities.values()))
    return out

def momentum_features(timeline: list) -> dict:
    """Calculates momentum (cumulative HP advantage) and its volatility."""
    out = {}; p1_advantages = []; cumulative_advantage = 0.0
    if not timeline: return out
    for i, turn in enumerate(timeline[:30]):
        p1_hp = turn.get('p1_pokemon_state', {}).get('hp_pct', 100)
        p2_hp = turn.get('p2_pokemon_state', {}).get('hp_pct', 100)
        turn_advantage = p1_hp - p2_hp
        cumulative_advantage += turn_advantage; p1_advantages.append(cumulative_advantage)
    if p1_advantages:
        x = np.arange(len(p1_advantages)); slope, intercept = np.polyfit(x, p1_advantages, 1)
        out['p1_momentum_slope'] = float(slope); out['p1_momentum_intercept'] = float(intercept)
        out['p1_final_advantage'] = float(p1_advantages[-1])
        out['p1_advantage_volatility'] = float(np.std(p1_advantages))
        out['p1_max_advantage'] = float(np.max(p1_advantages)); out['p1_min_advantage'] = float(np.min(p1_advantages))
        out['p1_advantage_range'] = float(out['p1_max_advantage'] - out['p1_min_advantage'])
    return out

def extract_opponent_team_from_timeline(timeline: list, p1_team: list) -> dict:
    """Extracts information about opponent Pokémon seen during the battle."""
    out = {}; p2_pokemon_seen = set(); p2_pokemon_types = []
    for turn in timeline[:30]:
        p2_state = turn.get('p2_pokemon_state', {})
        if not p2_state: continue
        p2_name = p2_state.get('name')
        if p2_name and p2_name not in p2_pokemon_seen:
            p2_pokemon_seen.add(p2_name)
            p2_pokemon_types.extend([t.lower() for t in p2_state.get('types', [])])
    
    out['p2_tl_unique_pokemon_seen'] = len(p2_pokemon_seen)
    out['p2_tl_switches_count'] = len(p2_pokemon_seen) - 1
    p2_type_counter = Counter(p2_pokemon_types)
    out['p2_tl_unique_types_seen'] = len(p2_type_counter)
    out['p2_tl_type_entropy'] = _entropy(p2_type_counter)
    
    if p2_pokemon_types and p1_team:
        matchup_advantages = 0
        for p1_poke in p1_team:
            p1_types = [t.lower() for t in p1_poke.get('types', [])]
            for p1_type in p1_types:
                for p2_type in set(p2_pokemon_types):
                    eff = get_effectiveness(p1_type, [p2_type])
                    if eff >= 2.0: matchup_advantages += 1
        out['p1_vs_p2_tl_type_advantages'] = matchup_advantages
        out['p1_vs_p2_tl_type_advantages_per_poke'] = matchup_advantages / max(1, len(p1_team))
    
    total_turns = len(timeline[:30])
    out['p2_tl_switch_rate'] = len(p2_pokemon_seen) / max(1, total_turns)
    return out

def extract_information_advantage(timeline: list) -> dict:
    """Calculates the information advantage (Pokémon seen)."""
    p1_rev = set(); p2_rev = set(); reveal_turns_p2 = []
    for turn in timeline[:30]:
        t = turn.get('turn', 0)
        if n1 := turn.get('p1_pokemon_state', {}).get('name'): p1_rev.add(n1)
        if n2 := turn.get('p2_pokemon_state', {}).get('name'):
            if n2 not in p2_rev: p2_rev.add(n2); reveal_turns_p2.append(t)
    return {
        'tl_p1_revealed_count': len(p1_rev),
        'tl_p2_revealed_count': len(p2_rev),
        'tl_info_advantage': len(p2_rev) - len(p1_rev),
        'tl_p2_avg_reveal_turn': float(np.mean(reveal_turns_p2)) if reveal_turns_p2 else 30.0
    }

def extract_advanced_momentum(timeline: list) -> dict:
    """Calculates advanced momentum (forced switches, immunity switches)."""
    p1_immune = 0; p2_forced = 0
    for i, turn in enumerate(timeline[:30]):
        if i == 0: continue
        prev = timeline[i-1]
        c1 = turn.get('p1_pokemon_state', {}).get('name')
        p1 = prev.get('p1_pokemon_state', {}).get('name')
        if c1 != p1 and not prev.get('p1_pokemon_state', {}).get('fainted'):
            m2_type = (turn.get('p2_move_details') or {}).get('type', '').lower()
            p1_types = [t.lower() for t in turn.get('p1_pokemon_state', {}).get('types', [])]
            if m2_type and p1_types and get_effectiveness(m2_type, p1_types) == 0.0:
                p1_immune += 1
        c2 = turn.get('p2_pokemon_state', {}).get('name')
        p2 = prev.get('p2_pokemon_state', {}).get('name')
        if c2 != p2 and not prev.get('p2_pokemon_state', {}).get('fainted'):
            if prev.get('p2_pokemon_state', {}).get('hp_pct', 1.0) < 0.50:
                p2_forced += 1
    return {'tl_p1_immune_switches': p1_immune, 'tl_p2_forced_switches': p2_forced}

def extract_gamestate_snapshots(timeline: list) -> dict:
    """Captures the gamestate (HP difference) at specific turns (10, 20, 30)."""
    turns_lead = 0; hp_diff_10 = 0.0; hp_diff_20 = 0.0; hp_diff_end = 0.0
    for i, turn in enumerate(timeline[:30]):
        t = i + 1
        h1 = turn.get('p1_pokemon_state', {}).get('hp_pct', 0)
        h2 = turn.get('p2_pokemon_state', {}).get('hp_pct', 0)
        if h1 > h2: turns_lead += 1
        if t == 10: hp_diff_10 = h1 - h2
        if t == 20: hp_diff_20 = h1 - h2
        hp_diff_end = h1 - h2 # Last available state
    return {
        'tl_turns_with_hp_lead': turns_lead,
        'tl_hp_diff_turn_10': float(hp_diff_10),
        'tl_hp_diff_turn_20': float(hp_diff_20),
        'tl_hp_diff_end': float(hp_diff_end)
    }

def extract_observed_mechanics(timeline: list) -> dict:
    """Counts specific mechanics (heals, freeze)."""
    p1_heals = 0; p2_heals = 0; p1_frz = 0; p2_frz = 0
    for i, turn in enumerate(timeline[:30]):
        if i == 0: continue
        prev = timeline[i-1]
        p1s = turn.get('p1_pokemon_state', {}); 
        p1s_prev = prev.get('p1_pokemon_state', {})
        p2s = turn.get('p2_pokemon_state', {}); 
        p2s_prev = prev.get('p2_pokemon_state', {})
        if p1s.get('name') == p1s_prev.get('name'):
             if p1s.get('hp_pct', 0) > p1s_prev.get('hp_pct', 0): p1_heals += 1
        if p2s.get('name') == p2s_prev.get('name'):
             if p2s.get('hp_pct', 0) > p2s_prev.get('hp_pct', 0): p2_heals += 1
        if p1s.get('status') == 'frz': p1_frz = 1
        if p2s.get('status') == 'frz': p2_frz = 1
    return {'tl_heal_diff': p1_heals - p2_heals, 'tl_freeze_adv': p2_frz - p1_frz}

def team_role_features(team: list, prefix: str = 'p1_') -> dict:
    """Extracts team specialists (Wall, Sweeper, etc.)"""
    if not team: return {}\
    
    spe_list = []; bulk_list = []; offense_list = []
    for p in team:
        spe_list.append(p.get('base_spe', 0))
        bulk_list.append(p.get('base_hp', 1) * (p.get('base_def', 1) + p.get('base_spd', 1)))
        offense_list.append(p.get('base_atk', 1) + p.get('base_spa', 1))
        
    return {
        f'{prefix}fastest_spe': float(np.max(spe_list)) if spe_list else 0.0,
        f'{prefix}slowest_spe': float(np.min(spe_list)) if spe_list else 0.0,
        f'{prefix}max_bulk': float(np.max(bulk_list)) if bulk_list else 0.0,
        f'{prefix}max_offense': float(np.max(offense_list)) if offense_list else 0.0
    }

def calculate_defensive_cohesion(team: list, prefix: str = 'p1_') -> dict:
    """Calculates how weak the team is to a single type (max common weakness)"""
    if not team: return {}\
    
    weakness_counts = Counter()
    for atk_type in ALL_ATTACK_TYPES:
        count = 0
        for p in team:
            def_types = [t.lower() for t in p.get('types', [])]
            if not def_types: continue
            if get_effectiveness(atk_type, def_types) >= 2.0:
                count += 1
        weakness_counts[atk_type] = count
        
    return {
        f'{prefix}max_common_weakness': float(max(weakness_counts.values())) if weakness_counts else 0.0
    }

def role_vs_lead_comparison(p1_roles: dict, p2_lead: dict) -> dict:
    """Compares P1 specialists against the P2 lead"""
    out = {}
    p2_lead_spe = p2_lead.get('base_spe', 0)
    p2_lead_offense = p2_lead.get('base_atk', 1) + p2_lead.get('base_spa', 1)
    
    out['p1_fastest_vs_lead_spe'] = p1_roles.get('p1_fastest_spe', 0) - p2_lead_spe
    out['p1_max_bulk_vs_lead_offense'] = p1_roles.get('p1_max_bulk', 1) / max(1, p2_lead_offense)
    return out


# --- 5. MASTER FUNCTIONS (Pipeline) ---

def prepare_record_features_COMPLETE(record: dict, max_turns: int = 30) -> dict:
    """Master FE Function (V6) that orchestrates all helpers for a single record."""
    out = {'battle_id': record.get('battle_id')}
    if 'player_won' in record: out['player_won'] = int(bool(record['player_won']))
    
    p1_team = record.get('p1_team_details', [])
    p2_lead = record.get('p2_lead_details', {})
    p1_lead = p1_team[0] if p1_team else {}
    tl = record.get('battle_timeline', [])
    tl_limited = tl[:max_turns]
    
    # Static features
    out.update(team_aggregate_features(p1_team, 'p1_'))
    out.update(lead_aggregate_features(p2_lead, 'p2_lead_'))
    out.update(ability_features(p1_team, 'p1_'))
    out.update(lead_vs_lead_features(p1_lead, p2_lead))
    out.update(ability_features([p2_lead], 'p2_lead_'))
    out['p1_intimidate_vs_lead'] = int(out.get('p1_ability_intimidate_count',0) > 0)
    
    # Dynamic features (from timeline, V6)
    out.update(summary_from_timeline(tl_limited, p1_team))
    out.update(extract_move_coverage_from_timeline(tl_limited, 'p1_'))
    out.update(extract_move_coverage_from_timeline(tl_limited, 'p2_'))
    out.update(extract_opponent_team_from_timeline(tl_limited, p1_team))
    out.update(momentum_features(tl_limited))
    
    # Quick features (from record)
    out.update(quick_boost_features_v2(record))
    
    # Calculated features (deltas)
    out['team_hp_sum_minus_p2lead_hp'] = out.get('p1_base_hp_sum', 0) - out.get('p2_lead_base_hp', 0)
    out['team_spa_mean_minus_p2spa'] = out.get('p1_base_spa_mean', 0) - out.get('p2_lead_base_spa', 0)
    out['speed_advantage'] = out.get('p1_base_spe_sum', 0) - out.get('p2_lead_base_spe', 0)
    out['n_unique_types_diff'] = out.get('p1_n_unique_types', 0) - out.get('p2_lead_n_unique_types', 1)
    p1_moves = max(out.get('tl_p1_moves',1),1); p2_moves = max(out.get('tl_p2_moves',1),1)
    out['damage_per_turn_diff'] = (out.get('tl_p1_est_damage',0.0)/p1_moves) - (out.get('tl_p2_est_damage',0.0)/p2_moves)
    out['last_pair'] = f"{out.get('tl_p1_last_active','')}_VS_{out.get('tl_p2_last_active','')}"
    out.update(calculate_type_advantage(p1_team, p2_lead))
    p2_lead_bulk = out.get('p2_lead_base_def',1) + out.get('p2_lead_base_spd',1)
    out['p1_se_options_vs_lead_bulk'] = out.get('p1_super_effective_options',0) / (p2_lead_bulk + 1e-6)
    
    if p2_team := record.get('p2_team_details', []):
        out.update(team_aggregate_features(p2_team, 'p2_'))
        out['team_hp_sum_diff'] = out.get('p1_base_hp_sum',0) - out.get('p2_base_hp_sum',0)
        out['team_spa_mean_diff'] = out.get('p1_base_spa_mean',0) - out.get('p2_base_spa_mean',0)
        out['team_spe_mean_diff'] = out.get('p1_base_spe_mean',0) - out.get('p2_base_spe_mean',0)
        out['n_unique_types_team_diff'] = out.get('p1_n_unique_types',0) - out.get('p2_n_unique_types',0)
        
    # Advanced features (V4)
    if tl_limited:
        out.update(extract_information_advantage(tl_limited))
        out.update(extract_advanced_momentum(tl_limited))
        out.update(extract_gamestate_snapshots(tl_limited))
        out.update(extract_observed_mechanics(tl_limited))
    else:
        out.update({
            'tl_p1_revealed_count': 1, 'tl_p2_revealed_count': 1, 'tl_info_advantage': 0,
            'tl_p2_avg_reveal_turn': 30.0, 'tl_p1_immune_switches': 0, 'tl_p2_forced_switches': 0,
            'tl_turns_with_hp_lead': 0, 'tl_hp_diff_turn_10': 0.0, 'tl_hp_diff_turn_20': 0.0,
            'tl_hp_diff_end': 0.0, 'tl_heal_diff': 0, 'tl_freeze_adv': 0
        })
        
    # Role features (V5)
    p1_role_feats = team_role_features(p1_team, 'p1_')
    out.update(p1_role_feats)
    out.update(calculate_defensive_cohesion(p1_team, 'p1_'))
    out.update(role_vs_lead_comparison(p1_role_feats, p2_lead))
        
    return out

def create_features_from_raw(data: list, feature_func=prepare_record_features_COMPLETE) -> pd.DataFrame:
    """Applies the master FE function to an entire list of records (raw data)."""
    rows = []
    for b in tqdm(data, desc='FE (V6 Complete)'):
        try:
            feat = feature_func(b, max_turns=30)
            if 'battle_id' not in feat: feat['battle_id'] = b.get('battle_id')
            rows.append(feat)
        except Exception as e:
            print(f"ERROR during FE on battle_id {b.get('battle_id')}: {e}")
            rows.append({'battle_id': b.get('battle_id'), 'error': 1})
    df = pd.DataFrame(rows)
    if 'player_won' in df.columns:
        df['player_won'] = df['player_won'].astype(int)
    return df.fillna(0)

def ensure_directories():
    """Creates all necessary output directories."""
    print("Verifying output directories...")
    dirs_to_create = [
        INPUT_DIR_JSONL, DATA_DIR, MODEL_OUTPUT_DIR,
        SUBMISSION_DIR, OOF_DIR
    ]
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
    print("Directories verified.")


# --- 6. PIPELINE FUNCTIONS (Phase Execution) ---

def run_01_feature_engineering():
    """
    PHASE 01: Executes the JSONL to CSV conversion with feature engineering (V6).
    """
    print("\n" + "="*30)
    print("START PHASE 01: Feature Engineering (XGBoost V6)")
    print("="*30)
    
    np.random.seed(SEED)
    
    print('Loading raw data...')
    train_raw = load_jsonl(TRAIN_JSON_IN)
    test_raw = load_jsonl(TEST_JSON_IN)
    
    if train_raw is None or test_raw is None:
        print("ERROR: Raw data not loaded. Aborting.")
        return False
        
    print(f'Train records: {len(train_raw)}, Test records: {len(test_raw)}')
    
    print('Creating features for Train set...')
    train_df = create_features_from_raw(train_raw) 
    
    print('Creating features for Test set...')
    test_df = create_features_from_raw(test_raw)
    
    print(f'Feature shape train/test: {train_df.shape} {test_df.shape}')
    
    # Salva i file CSV
    try:
        print(f"Saving to {TRAIN_CSV_OUT}...")
        train_df.to_csv(TRAIN_CSV_OUT, index=False)
        
        print(f"Saving to {TEST_CSV_OUT}...")
        test_df.to_csv(TEST_CSV_OUT, index=False)
        
        test_ids_df = test_df[['battle_id']].copy()
        test_ids_df.to_csv(TEST_IDS_OUT, index=False)
        print(f"Saving test IDs to {TEST_IDS_OUT}...")
        
    except Exception as e:
        print(f"ERROR during CSV saving: {e}")
        return False

    print(f"\nPHASE 01 (Feature Engineering V6) completed successfully.")
    print(f"CSV files saved in '{DATA_DIR}'")
    return True

def run_02_train_and_submit():
    """
    PHASE 02: Executes Preprocessing, K-Fold OOF Training, and creates the final submission.
    """
    print("\n" + "="*30)
    print("START PHASE 02: Preprocessing, Training and Submission (XGBoost V6)")
    print("="*30)
    
    np.random.seed(SEED)

    # --- 2. Data Loading and Preprocessing ---
    print("\nLoading .CSV data and Preprocessing...")
    try:
        train_df = pd.read_csv(TRAIN_CSV_OUT)
        test_df = pd.read_csv(TEST_CSV_OUT)
        test_ids_df = pd.read_csv(TEST_IDS_OUT)
    except FileNotFoundError:
        print(f"ERROR: .csv files not found in '{DATA_DIR}'.")
        print("Run Phase 01 (Feature Engineering) first.")
        return False

    # Exclude non-numeric columns
    exclude_cols = ['battle_id', 'player_won']
    string_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    exclude_cols.extend(string_cols)

    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    common_cols = list(train_cols.intersection(test_cols))

    FEATURES = [c for c in train_df.columns if c not in exclude_cols and c in common_cols]
    print(f"Numeric features selected: {len(FEATURES)}")
    
    with open(FEATURES_JSON_OUT, 'w') as f:
        json.dump(FEATURES, f, indent=4)
    print(f"Feature list saved to: {FEATURES_JSON_OUT}")

    # Imputation with median
    X_train_df = train_df[FEATURES].astype(float).replace([np.inf, -np.inf], np.nan)
    medians_train = X_train_df.median()
    X_train_full = X_train_df.fillna(medians_train)
    y_train_full = train_df['player_won'].values

    X_test_df = test_df.reindex(columns=FEATURES, fill_value=np.nan).astype(float).replace([np.inf, -np.inf], np.nan)
    X_test_kaggle = X_test_df.fillna(medians_train)
    
    with open(MEDIANS_JSON_OUT, 'w') as f:
        json.dump(medians_train.to_dict(), f, indent=4)
    print(f"Imputation medians saved to: {MEDIANS_JSON_OUT}")

    # --- 3. OOF Prediction Generation with K-Fold ---
    print(f"\nStarting K-Fold Cross-Validation (K={N_SPLITS}) for OOF predictions...")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_predictions = np.zeros(len(y_train_full))
    fold_auc_scores = []

    fold_params = BEST_PARAMS.copy()
    fold_params['early_stopping_rounds'] = 50
    fold_params['verbose'] = 0 

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        X_train_fold, X_val_fold = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_train_fold, y_val_fold = y_train_full[train_idx], y_train_full[val_idx]

        print(f"Training XGBoost on {len(X_train_fold)} samples...")
        fold_model = XGBClassifier(**fold_params)
        fold_model.fit(X_train_fold, y_train_fold,
                       eval_set=[(X_val_fold, y_val_fold)],
                       verbose=False)

        print(f"Generating OOF predictions on {len(X_val_fold)} samples...")
        fold_oof_preds = fold_model.predict_proba(X_val_fold)[:, 1]
        oof_predictions[val_idx] = fold_oof_preds

        fold_auc = roc_auc_score(y_val_fold, fold_oof_preds)
        fold_auc_scores.append(fold_auc)
        print(f"Fold {fold+1} AUC: {fold_auc:.4f}")

    print("\nK-Fold complete.")
    print(f"Mean OOF AUC across folds: {np.mean(fold_auc_scores):.4f}")

    print("Saving OOF predictions...")
    np.save(OOF_XGB_OUT_FILE, oof_predictions)
    print(f"OOF predictions saved to: {OOF_XGB_OUT_FILE} (Shape: {oof_predictions.shape})")

    # --- 4. Training Final Model on 100% of data ---
    print("\nTraining final XGBoost model on 100% of data...")
    final_model_params = BEST_PARAMS.copy() 
    final_model_params['verbose'] = 200 

    final_model = XGBClassifier(**final_model_params)
    final_model.fit(X_train_full, y_train_full)
    print("Final training complete.")

    # --- 5. Generate Predictions on Test Set ---
    print(f"\nGenerating predictions (probabilities) on the Test Set ({len(X_test_kaggle)} samples)...")
    test_predictions_proba = final_model.predict_proba(X_test_kaggle)[:, 1]

    # Save .npy (for stacking)
    print("Saving .npy predictions on the Test Set...")
    np.save(TEST_PREDS_XGB_OUT_FILE, test_predictions_proba)
    print(f"Test Set predictions saved to: {TEST_PREDS_XGB_OUT_FILE}")

    # Save .csv (for blending)
    print("Saving .csv predictions on the Test Set...")
    try:
        test_ids = test_ids_df['battle_id'].astype(int)
        if len(test_ids) != len(test_predictions_proba):
            raise ValueError(f"Mismatch IDs ({len(test_ids)}) vs Preds ({len(test_predictions_proba)})")
            
        submission_df = pd.DataFrame({
            'battle_id': test_ids,
            'player_won_proba': test_predictions_proba
        })
        submission_df.to_csv(SUBMISSION_XGB_PROBA_FILE, index=False)
        print(f"Submission (probabilities) saved to: {SUBMISSION_XGB_PROBA_FILE}")
    except Exception as e:
        print(f"❌ ERROR while saving the submission CSV: {e}")
        return False

    print(f"\n--- PHASE 02 (XGBoost Train & Submit V6) Complete ---")
    return True