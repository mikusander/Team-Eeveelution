"""
Utility library for the XGBoost Pokémon pipeline.

This module contains all the necessary functions, constants, and path configurations
for the entire pipeline:
- Data Loading
- All Feature Engineering logic
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
from pathlib import Path 
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

# Input Files
TRAIN_JSON_IN = INPUT_DIR_JSONL / 'train.jsonl'
TEST_JSON_IN = INPUT_DIR_JSONL / 'test.jsonl'

# Intermediate Files
TRAIN_CSV_OUT = DATA_DIR / 'train_features.csv'
TEST_CSV_OUT = DATA_DIR / 'test_features.csv'
TEST_IDS_OUT = DATA_DIR / 'test_ids.csv' 

# Output Files (Preprocessing Metadata)
FEATURES_JSON_OUT = MODEL_OUTPUT_DIR / 'features_final.json'
MEDIANS_JSON_OUT = MODEL_OUTPUT_DIR / 'preprocessing_medians.json'

# Output Files (Training)
OOF_XGB_OUT_FILE = OOF_DIR / 'oof_xgboost_proba.npy'
TEST_PREDS_XGB_OUT_FILE = OOF_DIR / 'test_preds_xgboost_proba.npy'
SUBMISSION_XGB_PROBA_FILE = SUBMISSION_DIR / 'submission_xgboost_100pct_PROBA.csv'

# --- 3. MODEL AND GAME CONSTANTS ---

SEED = 42
N_SPLITS = 10 

# HYPERPARAMETERS (From Optuna)
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

# Type effectiveness constants
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


# --- 4. HELPER FUNCTIONS (Feature Engineering) ---

def load_jsonl(path):
    """Loads a .jsonl file and returns it as a list of dictionaries."""
    data = []
    try:
        with open(path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    except FileNotFoundError:
        print(f"ERROR: File not found: {path}")
        return None
    return data

def get_effectiveness(attack_type: str, defense_types: list) -> float:
    if not attack_type or not defense_types: return 1.0
    eff = 1.0
    for d in defense_types: eff *= TYPE_CHART.get(attack_type, {}).get(d, 1.0)
    return eff

def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0: return 0.0
    ent = 0.0
    for v in counter.values():
        p = v / total
        if p > 0: ent -= p * math.log(p, 2)
    return ent

def build_pokedex(data_list: list) -> dict:
    pokedex = {}
    for battle in data_list:
        for p in battle.get('p1_team_details', []):
            if name := p.get('name'):
                if name not in pokedex:
                    stats = {k: v for k, v in p.items() if k.startswith('base_')}
                    stats['types'] = p.get('types', ['notype', 'notype'])
                    if stats: pokedex[name] = stats
        if p2 := battle.get('p2_lead_details'):
            if name := p2.get('name'):
                if name not in pokedex:
                    stats = {k: v for k, v in p2.items() if k.startswith('base_')}
                    if stats: pokedex[name] = stats
    return pokedex

def team_aggregate_features(team: list, prefix: str = 'p1_') -> dict:
    stats = ['base_hp','base_atk','base_def','base_spa','base_spd','base_spe']
    out = {}
    vals = {s: [] for s in stats}
    types_counter = Counter(); names = []
    for p in team:
        names.append(p.get('name',''))
        for s in stats: vals[s].append(p.get(s, 0))
        for t in p.get('types', []): types_counter[t.lower()] += 1
    for s in stats:
        arr = np.array(vals[s], dtype=float)
        out[f'{prefix}{s}_sum'] = float(arr.sum())
        out[f'{prefix}{s}_mean'] = float(arr.mean())
        out[f'{prefix}{s}_max'] = float(arr.max())
        out[f'{prefix}{s}_min'] = float(arr.min())
        out[f'{prefix}{s}_std'] = float(arr.std())
    out[f'{prefix}n_unique_types'] = int(len(types_counter))
    
    common_types = ['normal','fire','water','electric','grass','psychic','ice','dragon','rock','ground','flying']
    for t in common_types:
        out[f'{prefix}type_{t}_count'] = int(types_counter.get(t, 0))
        
    out[f'{prefix}type_entropy'] = float(_entropy(types_counter))
    spe_arr = np.array(vals['base_spe'], dtype=float)
    out[f'{prefix}spe_p25'] = float(np.percentile(spe_arr, 25)) if spe_arr.size else 0.0
    out[f'{prefix}spe_p50'] = float(np.percentile(spe_arr, 50)) if spe_arr.size else 0.0
    out[f'{prefix}spe_p75'] = float(np.percentile(spe_arr, 75)) if spe_arr.size else 0.0
    return out

def lead_vs_lead_features(p1_lead: dict, p2_lead: dict) -> dict:
    out = {}
    stats = ['base_hp','base_atk','base_def','base_spa','base_spd','base_spe']
    for s in stats:
        out[f'lead_diff_{s}'] = float(p1_lead.get(s,0) - p2_lead.get(s,0))
    out['lead_speed_advantage'] = float(p1_lead.get('base_spe',0) - p2_lead.get('base_spe',0))
    p1_types = [t.lower() for t in p1_lead.get('types', [])]
    p2_types = [t.lower() for t in p2_lead.get('types', [])]
    out['lead_p1_vs_p2_effectiveness'] = float(max([get_effectiveness(pt, p2_types) for pt in p1_types] or [1.0]))
    return out

def calculate_type_advantage(team1: list, team2_lead: dict) -> dict:
    out = {'p1_vs_lead_avg_effectiveness': 0.0, 'p1_vs_lead_max_effectiveness': 0.0, 'p1_super_effective_options': 0}
    lead_types = [t.lower() for t in team2_lead.get('types', [])]
    if not team1 or not lead_types: return out
    effs = []
    for p in team1:
        p_types = [t.lower() for t in p.get('types', [])]
        effs.append(max([get_effectiveness(pt, lead_types) for pt in p_types] or [1.0]))
    out['p1_vs_lead_avg_effectiveness'] = float(np.mean(effs))
    out['p1_vs_lead_max_effectiveness'] = float(np.max(effs))
    out['p1_super_effective_options'] = int(sum(1 for e in effs if e >= 2))
    return out

def lead_aggregate_features(pokemon: dict, prefix: str = 'p2_lead_') -> dict:
    out = {}
    stats = ['base_hp','base_atk','base_def','base_spa','base_spd','base_spe']
    for s in stats:
        out[f'{prefix}{s}'] = float(pokemon.get(s,0))
    out[f'{prefix}level'] = int(pokemon.get('level',0))
    types = [x.lower() for x in pokemon.get('types', [])]
    for t in ['normal','fire','water','electric','grass','psychic','ice','dragon','rock','ground','flying']:
        out[f'{prefix}type_{t}'] = int(t in types)
    out[f'{prefix}n_unique_types'] = int(len(set(types)))
    return out

def quick_boost_features_v2(record: dict) -> dict:
    out = {}
    p1_team = record.get('p1_team_details', [])
    p2_lead = record.get('p2_lead_details', {})
    if not p1_team: return out
    
    p2_lead_spe = p2_lead.get('base_spe', 0)
    out['p1_faster_than_lead_count'] = sum(1 for p in p1_team if p.get('base_spe', 0) > p2_lead_spe)
    out['p1_slower_than_lead_count'] = sum(1 for p in p1_team if p.get('base_spe', 0) <= p2_lead_spe)
    out['p1_speed_control_ratio'] = out['p1_faster_than_lead_count'] / max(1, len(p1_team))
    
    p1_avg_bulk = np.mean([p.get('base_hp', 0)*(p.get('base_def', 0)+p.get('base_spd', 0)) for p in p1_team])
    p2_lead_bulk = p2_lead.get('base_hp', 1)*(p2_lead.get('base_def', 1)+p2_lead.get('base_spd', 1))
    out['p1_avg_bulk_vs_lead'] = p1_avg_bulk / max(p2_lead_bulk, 1)
    
    p1_total_atk = sum(p.get('base_atk', 0) + p.get('base_spa', 0) for p in p1_team)
    p2_lead_offense = p2_lead.get('base_atk', 0) + p2_lead.get('base_spa', 0)
    out['p1_total_offense'] = p1_total_atk
    out['p1_offense_advantage'] = p1_total_atk / max(p2_lead_offense, 1)
    
    p2_lead_types = [t.lower() for t in p2_lead.get('types', [])]
    if p2_lead_types:
        cov = []
        for p in p1_team:
            p_types = [t.lower() for t in p.get('types', [])]
            cov.append(max([get_effectiveness(pt, p2_lead_types) for pt in p_types] or [1.0]))
        out['p1_avg_effectiveness_vs_lead'] = float(np.mean(cov))
        out['p1_max_effectiveness_vs_lead'] = float(np.max(cov))
        out['p1_se_count_vs_lead'] = sum(1 for s in cov if s >= 2.0)
        
    timeline = record.get('battle_timeline', [])
    out['p1_first_blood'] = -1
    out['p1_first_blood_turn'] = 0
    if timeline:
        for turn in timeline[:30]:
            if turn.get('p2_pokemon_state', {}).get('fainted'):
                out['p1_first_blood'] = 1; out['p1_first_blood_turn'] = turn.get('turn', 0); break
            if turn.get('p1_pokemon_state', {}).get('fainted'):
                out['p1_first_blood'] = 0; out['p1_first_blood_turn'] = turn.get('turn', 0); break
                
    p1_avg_level = np.mean([p.get('level', 50) for p in p1_team])
    out['p1_avg_level_advantage'] = p1_avg_level - p2_lead.get('level', 50)
    
    return out

def summary_from_timeline_FULL(timeline: list, p1_team: list) -> dict:
    out = {}
    if not timeline: return {}
    
    p1_moves = 0; p2_moves = 0
    p1_dmg = 0.0; p2_dmg = 0.0
    last_p1_hp = {}; last_p2_hp = {}
    p1_fainted = 0; p2_fainted = 0
    p1_fainted_names = set(); p2_fainted_names = set()
    p1_comeback_kos = 0; p2_comeback_kos = 0
    p1_statuses = Counter(); p2_statuses = Counter()
    p1_poke_statuses = {}; p2_poke_statuses = {}
    p1_move_types = Counter(); p2_move_types = Counter()
    p1_dmg_by_turn = {}; p2_dmg_by_turn = {}
    seen_turns = set()
    first_ko_turn_p1_taken = None; first_ko_turn_p1_inflicted = None
    p1_last_active = ''; p2_last_active = ''
    
    for i, turn in enumerate(timeline[:30]):
        prev_p1_f, prev_p2_f = p1_fainted, p2_fainted
        p1s = turn.get('p1_pokemon_state',{}) or {}; p2s = turn.get('p2_pokemon_state',{}) or {}
        tnum = turn.get('turn', len(seen_turns) + 1); seen_turns.add(tnum)
        
        if p1s.get('name'): p1_last_active = p1s.get('name')
        if p2s.get('name'): p2_last_active = p2s.get('name')

        if p1s.get('fainted') and p1s.get('name') not in p1_fainted_names:
            p1_fainted += 1; p1_fainted_names.add(p1s.get('name'))
            if not first_ko_turn_p1_taken: first_ko_turn_p1_taken = tnum
        if p2s.get('fainted') and p2s.get('name') not in p2_fainted_names:
            p2_fainted += 1; p2_fainted_names.add(p2s.get('name'))
            if not first_ko_turn_p1_inflicted: first_ko_turn_p1_inflicted = tnum

        if n2 := p2s.get('name'):
            if (prev := last_p2_hp.get(n2)) is not None:
                d = max(0.0, prev - p2s.get('hp_pct',0))
                p1_dmg += d; p1_dmg_by_turn[tnum] = p1_dmg_by_turn.get(tnum,0)+d
            last_p2_hp[n2] = p2s.get('hp_pct',0)
        if n1 := p1s.get('name'):
            if (prev := last_p1_hp.get(n1)) is not None:
                d = max(0.0, prev - p1s.get('hp_pct',0))
                p2_dmg += d; p2_dmg_by_turn[tnum] = p2_dmg_by_turn.get(tnum,0)+d
            last_p1_hp[n1] = p1s.get('hp_pct',0)

        dmg_diff_so_far = p1_dmg - p2_dmg
        if p2_fainted > prev_p2_f and dmg_diff_so_far < -1.0: p1_comeback_kos += 1
        if p1_fainted > prev_p1_f and dmg_diff_so_far > 1.0: p2_comeback_kos += 1

        if p2s.get('name') and p2s.get('status') and p2_poke_statuses.get(p2s['name']) != p2s['status']:
            p1_statuses[p2s['status']] += 1; p2_poke_statuses[p2s['name']] = p2s['status']
        if p1s.get('name') and p1s.get('status') and p1_poke_statuses.get(p1s['name']) != p1s['status']:
            p2_statuses[p1s['status']] += 1; p1_poke_statuses[p1s['name']] = p1s['status']

        if m1 := turn.get('p1_move_details'):
            p1_moves += 1
            if t := m1.get('type'): p1_move_types[t.lower()] += 1
        if m2 := turn.get('p2_move_details'):
            p2_moves += 1
            if t := m2.get('type'): p2_move_types[t.lower()] += 1

    out['tl_p1_moves'] = p1_moves; out['tl_p2_moves'] = p2_moves
    out['tl_p1_est_damage'] = float(p1_dmg); out['tl_p2_est_damage'] = float(p2_dmg)
    out['tl_p1_fainted'] = p1_fainted; out['tl_p2_fainted'] = p2_fainted
    turns = max(1, len(seen_turns))
    out['tl_p1_fainted_rate'] = p1_fainted/turns; out['tl_p2_fainted_rate'] = p2_fainted/turns
    out['damage_diff'] = p1_dmg - p2_dmg; out['fainted_diff'] = p1_fainted - p2_fainted
    out['tl_p1_last_active'] = p1_last_active; out['tl_p2_last_active'] = p2_last_active
    out['tl_p1_comeback_kos'] = p1_comeback_kos; out['tl_p2_comeback_kos'] = p2_comeback_kos
    
    if p1_team:
        p1_hp_sum = sum(p.get('base_hp',0) for p in p1_team)
        out['tl_p2_damage_vs_p1_hp_pool'] = p2_dmg / (p1_hp_sum + 1e-6)

    for s in ['brn','par','slp','frz','psn','tox']:
        out[f'tl_p1_inflicted_{s}_count'] = p1_statuses[s]
        out[f'tl_p2_inflicted_{s}_count'] = p2_statuses[s]
        out[f'tl_inflicted_{s}_diff'] = p1_statuses[s] - p2_statuses[s]

    for mt in ['normal','fire','water','electric','grass','psychic','ice','dragon','rock','ground','flying','ghost','bug','poison','fighting']:
        out[f'tl_p1_move_type_{mt}_count'] = p1_move_types[mt]
        out[f'tl_p2_move_type_{mt}_count'] = p2_move_types[mt]
        out[f'tl_move_type_{mt}_count_diff'] = p1_move_types[mt] - p2_move_types[mt]

    out['tl_turns_count'] = turns
    out['tl_p1_moves_rate'] = p1_moves/turns; out['tl_p2_moves_rate'] = p2_moves/turns
    out['tl_p1_damage_per_turn'] = p1_dmg/turns; out['tl_p2_damage_per_turn'] = p2_dmg/turns
    out['tl_damage_rate_diff'] = out['tl_p1_damage_per_turn'] - out['tl_p2_damage_per_turn']

    if seen_turns:
        recent = sorted(seen_turns)[-5:]
        p1_last5 = sum(p1_dmg_by_turn.get(t,0) for t in recent)
        p2_last5 = sum(p2_dmg_by_turn.get(t,0) for t in recent)
        out['tl_p1_damage_last5'] = p1_last5; out['tl_p2_damage_last5'] = p2_last5
        out['tl_last5_damage_diff'] = p1_last5 - p2_last5
        
        ts = sorted(seen_turns); w = np.linspace(1.0, 2.0, num=len(ts)); w = w/w.sum()
        adv = [(p1_dmg_by_turn.get(t,0)-p2_dmg_by_turn.get(t,0)) for t in ts]
        out['tl_weighted_damage_diff'] = float(np.dot(w, adv))
        
    out['tl_first_ko_turn_p1_inflicted'] = int(first_ko_turn_p1_inflicted or 0)
    out['tl_first_ko_turn_p1_taken'] = int(first_ko_turn_p1_taken or 0)
    out['tl_first_ko_turn_diff'] = out['tl_first_ko_turn_p1_inflicted'] - out['tl_first_ko_turn_p1_taken']

    return out

def extract_move_coverage_from_timeline(timeline: list, prefix: str = 'p1_') -> dict:
    out = {}; m_types = set(); unique_m = set(); stab = 0
    for turn in timeline[:30]:
        md = turn.get(f'{prefix[:-1]}_move_details')
        ps = turn.get(f'{prefix[:-1]}_pokemon_state', {})
        if not md: continue
        if md.get('name'): unique_m.add(md['name'])
        if t := (md.get('type') or '').lower(): m_types.add(t)
        if t in [pt.lower() for pt in ps.get('types', [])]: stab += 1
    
    out[f'{prefix}tl_unique_move_types'] = len(m_types)
    out[f'{prefix}tl_unique_moves_used'] = len(unique_m)
    out[f'{prefix}tl_stab_moves'] = stab
    out[f'{prefix}tl_coverage_score'] = len(m_types)/max(1, len(unique_m))
    return out

def ability_features(team: list, prefix: str) -> dict:
    imm = {'levitate':0,'volt_absorb':0,'water_absorb':0,'flash_fire':0}
    drop = {'intimidate':0}
    for p in team:
        a = (p.get('ability','') or '').lower().replace(' ','_')
        if a in imm: imm[a]+=1
        if a in drop: drop[a]+=1
    out = {}
    for k,v in imm.items(): out[f'{prefix}ability_{k}_count'] = v
    out[f'{prefix}total_immunity_abilities'] = sum(imm.values())
    out[f'{prefix}total_stat_drop_abilities'] = sum(drop.values())
    return out

def momentum_features(timeline: list) -> dict:
    out = {}; p1_advantages = []; cum = 0.0
    if not timeline: return out
    for i, turn in enumerate(timeline[:30]):
        p1_hp = turn.get('p1_pokemon_state', {}).get('hp_pct', 100)
        p2_hp = turn.get('p2_pokemon_state', {}).get('hp_pct', 100)
        cum += (p1_hp - p2_hp); p1_advantages.append(cum)
    if p1_advantages:
        slope, intercept = np.polyfit(np.arange(len(p1_advantages)), p1_advantages, 1)
        out['p1_momentum_slope'] = float(slope)
        out['p1_momentum_intercept'] = float(intercept)
        out['p1_final_advantage'] = float(p1_advantages[-1])
        out['p1_advantage_volatility'] = float(np.std(p1_advantages))
        out['p1_max_advantage'] = float(np.max(p1_advantages))
        out['p1_min_advantage'] = float(np.min(p1_advantages))
        out['p1_advantage_range'] = out['p1_max_advantage'] - out['p1_min_advantage']
    return out

def extract_opponent_team_from_timeline(timeline: list, p1_team: list) -> dict:
    out = {}; p2_seen = set(); p2_types = []
    for turn in timeline[:30]:
        p2s = turn.get('p2_pokemon_state', {})
        if n := p2s.get('name'):
            if n not in p2_seen:
                p2_seen.add(n)
                p2_types.extend([t.lower() for t in p2s.get('types', [])])
    out['p2_tl_unique_pokemon_seen'] = len(p2_seen)
    out['p2_tl_switches_count'] = max(0, len(p2_seen) - 1)
    out['p2_tl_unique_types_seen'] = len(set(p2_types))
    out['p2_tl_type_entropy'] = _entropy(Counter(p2_types))
    
    advs = 0
    if p2_types and p1_team:
        for p1 in p1_team:
            for t1 in [t.lower() for t in p1.get('types', [])]:
                for t2 in set(p2_types):
                    if get_effectiveness(t1, [t2]) >= 2.0: advs += 1
    out['p1_vs_p2_tl_type_advantages'] = advs
    out['p2_tl_switch_rate'] = len(p2_seen) / max(1, len(timeline[:30]))
    return out

def extract_model2_features(battle: dict, pokedex: dict) -> dict:
    out = {}
    timeline = battle.get('battle_timeline', [])
    
    p1_cond = {p.get('name'): {'hp': 1.0, 'status': 'nostatus', 'effects': []} 
               for p in battle.get('p1_team_details', [])}
    p2_lead = battle.get('p2_lead_details', {}).get('name')
    p2_cond = {p2_lead: {'hp': 1.0, 'status': 'nostatus', 'effects': []}} if p2_lead else {}

    for turn in timeline:
        if p1s := turn.get('p1_pokemon_state'):
            if name := p1s.get('name'):
                p1_cond[name] = {'hp': p1s.get('hp_pct', 1.0), 'status': p1s.get('status', 'nostatus')}
                if turn == timeline[-1]: p1_cond[name]['effects'] = p1s.get('effects', [])
        
        if p2s := turn.get('p2_pokemon_state'):
            if name := p2s.get('name'):
                p2_cond[name] = {'hp': p2s.get('hp_pct', 1.0), 'status': p2s.get('status', 'nostatus')}
                if turn == timeline[-1]: p2_cond[name]['effects'] = p2s.get('effects', [])

    out['om_p1_mean_pc_hp'] = float(np.mean([v['hp'] for v in p1_cond.values()]) if p1_cond else 0)
    out['om_p2_mean_pc_hp'] = float(np.mean([v['hp'] for v in p2_cond.values()]) if p2_cond else 0)

    out['om_p1_surviving'] = sum(1 for v in p1_cond.values() if v['hp'] > 0)
    out['om_p2_surviving'] = sum(1 for v in p2_cond.values() if v['hp'] > 0) + (6 - len(p2_cond))
    
    p1_score = sum(1 for v in p1_cond.values() if v['hp'] > 0 and v['status'] != 'nostatus')
    p1_score += sum(len(v.get('effects', [])) for v in p1_cond.values()) * 0.4
    
    p2_score = sum(1 for v in p2_cond.values() if v['hp'] > 0 and v['status'] != 'nostatus')
    p2_score += sum(len(v.get('effects', [])) for v in p2_cond.values()) * 0.4
    
    out['om_p1_status_score'] = float(p1_score)
    out['om_p2_status_score'] = float(p2_score)

    stats = ['spe', 'atk', 'def', 'spa', 'spd', 'hp']
    p1_sum = {k: 0 for k in stats}; p2_sum = {k: 0 for k in stats}
    
    for p in battle.get('p1_team_details', []):
        for k in stats: p1_sum[k] += p.get(f'base_{k}', 0)
    for name in p2_cond:
        if name in pokedex:
            for k in stats: p2_sum[k] += pokedex[name].get(f'base_{k}', 0)
            
    for k in stats:
        out[f'om_total_{k}_diff'] = p1_sum[k] - p2_sum[k]
        
    return out


# --- 5. MASTER FUNCTIONS (Pipeline) ---

def prepare_record_features_V12(record: dict, pokedex: dict, max_turns: int = 30) -> dict:
    out = {'battle_id': record.get('battle_id')}
    if 'player_won' in record: out['player_won'] = int(bool(record['player_won']))
    
    p1_team = record.get('p1_team_details', [])
    p2_lead = record.get('p2_lead_details', {})
    p1_lead = p1_team[0] if p1_team else {}
    tl = record.get('battle_timeline', [])[:max_turns]
    
    out.update(team_aggregate_features(p1_team, 'p1_'))
    out.update(lead_aggregate_features(p2_lead, 'p2_lead_'))
    out.update(ability_features(p1_team, 'p1_'))
    out.update(ability_features([p2_lead], 'p2_lead_'))
    out.update(lead_vs_lead_features(p1_lead, p2_lead))
    out.update(calculate_type_advantage(p1_team, p2_lead))
    out.update(quick_boost_features_v2(record))
    out['p1_intimidate_vs_lead'] = int(out.get('p1_ability_intimidate_count',0) > 0)
    
    out.update(summary_from_timeline_FULL(tl, p1_team))
    out.update(extract_move_coverage_from_timeline(tl, 'p1_'))
    out.update(extract_move_coverage_from_timeline(tl, 'p2_'))
    out.update(extract_opponent_team_from_timeline(tl, p1_team))
    out.update(momentum_features(tl))
    out.update(extract_model2_features(record, pokedex))
    out['team_hp_sum_minus_p2lead_hp'] = out.get('p1_base_hp_sum', 0) - out.get('p2_lead_base_hp', 0)
    out['team_spa_mean_minus_p2spa'] = out.get('p1_base_spa_mean', 0) - out.get('p2_lead_base_spa', 0)
    out['speed_advantage'] = out.get('p1_base_spe_sum', 0) - out.get('p2_lead_base_spe', 0)
    out['n_unique_types_diff'] = out.get('p1_n_unique_types', 0) - out.get('p2_lead_n_unique_types', 1)
    p1m = max(out.get('tl_p1_moves',1),1); p2m = max(out.get('tl_p2_moves',1),1)
    out['damage_per_turn_diff'] = (out.get('tl_p1_est_damage',0.0)/p1m) - (out.get('tl_p2_est_damage',0.0)/p2m)
    out['last_pair'] = f"{out.get('tl_p1_last_active','')}_VS_{out.get('tl_p2_last_active','')}"
    
    p2_lead_bulk = out.get('p2_lead_base_def',1) + out.get('p2_lead_base_spd',1)
    out['p1_se_options_vs_lead_bulk'] = out.get('p1_super_effective_options',0) / (p2_lead_bulk + 1e-6)
    
    if p2_team := record.get('p2_team_details', []):
        out.update(team_aggregate_features(p2_team, 'p2_'))
        out['team_hp_sum_diff'] = out.get('p1_base_hp_sum',0) - out.get('p2_base_hp_sum',0)
        out['team_spa_mean_diff'] = out.get('p1_base_spa_mean',0) - out.get('p2_base_spa_mean',0)
        out['team_spe_mean_diff'] = out.get('p1_base_spe_mean',0) - out.get('p2_base_spe_mean',0)
        out['n_unique_types_team_diff'] = out.get('p1_n_unique_types',0) - out.get('p2_n_unique_types',0)
        
    return out

def create_features_from_raw(data: list) -> pd.DataFrame:
    print("Building Pokedex...")
    POKEDEX = build_pokedex(data)
    
    rows = []
    for b in tqdm(data, desc='FE (V12 Maximalist)'):
        try:
            feat = prepare_record_features_V12(b, POKEDEX, max_turns=30)
            if 'battle_id' not in feat: feat['battle_id'] = b.get('battle_id')
            rows.append(feat)
        except Exception as e:
            rows.append({'battle_id': b.get('battle_id'), 'error': 1})
    df = pd.DataFrame(rows)
    if 'player_won' in df.columns: df['player_won'] = df['player_won'].astype(int)
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
    print("START PHASE 01: Feature Engineering (XGBoost V6)")
    
    np.random.seed(SEED)
    
    print('Loading raw data...')
    train_raw = load_jsonl(TRAIN_JSON_IN)
    test_raw = load_jsonl(TEST_JSON_IN)
    
    if train_raw is None or test_raw is None:
        print("ERROR: Raw data not loaded. Aborting.")
        return False
    
    print('Creating features for Train set...')
    train_df = create_features_from_raw(train_raw) 
    
    print('Creating features for Test set...')
    test_df = create_features_from_raw(test_raw)
    
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
    print("START PHASE 02: Preprocessing, Training and Submission (XGBoost V6)")
    
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
        print(f"ERROR while saving the submission CSV: {e}")
        return False

    print(f"\n--- PHASE 02 (XGBoost Train & Submit) Complete ---")
    return True
