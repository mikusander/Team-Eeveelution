# Nome file: 16_create_xgboost_stack_files.py (o 13_xgboost.py)
"""
Script adattato dal notebook 'pokemon_xgboost_cv.ipynb' (v2) per lo stacking.
UTILIZZA IL FEATURE ENGINEERING E GLI IPERPARAMETRI AGGIORNATI dal notebook.
1. Carica i dati raw (train.jsonl, test.jsonl).
2. Esegue il feature engineering completo (aggiornato).
3. Esegue K-Fold cross-validation sul 100% dei dati di train per
   generare e salvare le previsioni OOF (oof_xgboost_proba.npy).
4. Addestra il modello finale sul 100% e genera/salva le previsioni
   (probabilità) sul test set di Kaggle in *entrambi* i formati:
   - test_preds_xgboost_proba.npy (per lo stacking)
   - submission_xgboost_100pct_PROBA.csv (per il blending)
"""

import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import warnings
import math
from collections import Counter

warnings.filterwarnings('ignore')

# --- 1. Configurazione ---
SEED = 42
np.random.seed(SEED)
N_SPLITS = 5 # Numero di fold (come per gli altri modelli)

# Percorsi Input (dati raw)
train_file_path = 'Input/train.jsonl'
test_file_path = 'Input/test.jsonl'

# Cartelle di Output
OOF_PREDS_DIR = 'OOF_Predictions'
SUBMISSION_DIR = 'Submissions'
os.makedirs(OOF_PREDS_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# File di Output (.npy)
OOF_XGB_OUT_FILE = os.path.join(OOF_PREDS_DIR, 'oof_xgboost_proba.npy')
TEST_PREDS_XGB_OUT_FILE = os.path.join(OOF_PREDS_DIR, 'test_preds_xgboost_proba.npy')

# File di Output (.csv) per il blending
SUBMISSION_XGB_PROBA_FILE = os.path.join(SUBMISSION_DIR, 'submission_xgboost_100pct_PROBA.csv')


# --- IPERPARAMETRI AGGIORNATI (Dal GridSearchCV del notebook) ---
best_params = {
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'learning_rate': 0.05,
    'max_depth': 3,
    'min_child_weight': 3,
    'n_estimators': 700,
    'reg_alpha': 0.01,
    'reg_lambda': 1.5,
    'subsample': 0.8,
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'random_state': SEED
}
# --- FINE IPERPARAMETRI ---


# --- 2. Funzioni di Caricamento Dati e FE (AGGIORNATE dal Notebook) ---

def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# === TYPE CHART (Gen 1) ===
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

def get_effectiveness(attack_type: str, defense_types: list) -> float:
    if not attack_type or not defense_types:
        return 1.0
    eff = 1.0
    for d in defense_types:
        eff *= TYPE_CHART.get(attack_type, {}).get(d, 1.0)
    return eff

def calculate_type_advantage(team1: list, team2_lead: dict) -> dict:
    out = {'p1_vs_lead_avg_effectiveness': 0.0, 'p1_vs_lead_max_effectiveness': 0.0, 'p1_super_effective_options': 0}
    if not team1 or not team2_lead:
        return out
    lead_types = [t.lower() for t in team2_lead.get('types', [])]
    if not lead_types:
        return out
    effs = []
    for p in team1:
        p_types = [t.lower() for t in p.get('types', [])]
        max_eff = 0.0
        for pt in p_types:
            max_eff = max(max_eff, get_effectiveness(pt, lead_types))
        effs.append(max_eff)
    if not effs:
        return out
    out['p1_vs_lead_avg_effectiveness'] = float(np.mean(effs))
    out['p1_vs_lead_max_effectiveness'] = float(np.max(effs))
    out['p1_super_effective_options'] = int(sum(1 for e in effs if e >= 2))
    return out

def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for v in counter.values():
        p = v / total
        if p > 0:
            ent -= p * math.log(p, 2)
    return ent

def team_aggregate_features(team: list, prefix: str = 'p1_') -> dict:
    stats = ['base_hp','base_atk','base_def','base_spa','base_spd','base_spe']
    out = {}
    vals = {s: [] for s in stats}
    levels = []
    types_counter = Counter()
    names = []
    for p in team:
        names.append(p.get('name',''))
        for s in stats:
            vals[s].append(p.get(s, 0))
        levels.append(p.get('level', 0))
        for t in p.get('types', []):
            types_counter[t.lower()] += 1
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
    common_types = ['normal','fire','water','electric','grass','psychic','ice','dragon','rock','ground','flying']
    for t in common_types:
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
    out = {}
    stats = ['base_hp','base_atk','base_def','base_spa','base_spd','base_spe']
    for s in stats:
        out[f'lead_diff_{s}'] = float(p1_lead.get(s,0) - p2_lead.get(s,0))
    out['lead_speed_advantage'] = float(p1_lead.get('base_spe',0) - p2_lead.get('base_spe',0))
    p1_types = [t.lower() for t in p1_lead.get('types', [])]
    p2_types = [t.lower() for t in p2_lead.get('types', [])]
    max_eff = 0.0
    for pt in p1_types:
        max_eff = max(max_eff, get_effectiveness(pt, p2_types))
    out['lead_p1_vs_p2_effectiveness'] = float(max_eff)
    return out

def lead_aggregate_features(pokemon: dict, prefix: str = 'p2_lead_') -> dict:
    out = {}
    stats = ['base_hp','base_atk','base_def','base_spa','base_spd','base_spe']
    for s in stats:
        out[f'{prefix}{s}'] = float(pokemon.get(s,0))
    out[f'{prefix}level'] = int(pokemon.get('level',0))
    types = [x.lower() for x in pokemon.get('types', [])]
    common_types = ['normal','fire','water','electric','grass','psychic','ice','dragon','rock','ground','flying']
    for t in common_types:
        out[f'{prefix}type_{t}'] = int(t in types)
    out[f'{prefix}name'] = pokemon.get('name','')
    out[f'{prefix}n_unique_types'] = int(len(set(types)))
    return out

def summary_from_timeline(timeline: list, p1_team: list) -> dict:
    out = {}
    if not timeline:
        return {'tl_p1_moves':0,'tl_p2_moves':0,'tl_p1_est_damage':0.0,'tl_p2_est_damage':0.0,'damage_diff':0.0}
    p1_moves = p2_moves = 0
    p1_damage = p2_damage = 0.0
    p1_last_active = p2_last_active = ''
    p1_last_hp = p2_last_hp = np.nan
    p1_fainted = p2_fainted = 0
    p1_fainted_names = set()
    p2_fainted_names = set()
    last_p1_hp = {}
    last_p2_hp = {}
    p1_comeback_kos = 0
    p2_comeback_kos = 0
    p1_inflicted_statuses = Counter()
    p2_inflicted_statuses = Counter()
    p1_pokemon_statuses = {}
    p2_pokemon_statuses = {}
    p1_move_type_counts = Counter()
    p2_move_type_counts = Counter()
    p1_damage_first2 = 0.0
    p2_damage_first2 = 0.0

    p1_dmg_by_turn = {}
    p2_dmg_by_turn = {}
    seen_turns = set()
    first_ko_turn_p1_taken = None
    first_ko_turn_p1_inflicted = None
    early_threshold = 10
    p1_kos_early = p1_kos_late = 0
    p2_kos_early = p2_kos_late = 0

    for turn in timeline[:30]:
        prev_p1_fainted, prev_p2_fainted = p1_fainted, p2_fainted
        p1_state = turn.get('p1_pokemon_state',{}) or {}
        p2_state = turn.get('p2_pokemon_state',{}) or {}
        tnum = turn.get('turn', None)
        if tnum is None:
            tnum = (len(seen_turns) + 1)
        seen_turns.add(tnum)
        if p1_state.get('name'):
            p1_last_active = p1_state.get('name')
        if p2_state.get('name'):
            p2_last_active = p2_state.get('name')
        if p1_state.get('fainted') and p1_state.get('name') not in p1_fainted_names:
            p1_fainted += 1
            p1_fainted_names.add(p1_state.get('name'))
            if first_ko_turn_p1_taken is None:
                first_ko_turn_p1_taken = tnum
            if tnum <= early_threshold: p2_kos_early += 1
            else: p2_kos_late += 1
        if p2_state.get('fainted') and p2_state.get('name') not in p2_fainted_names:
            p2_fainted += 1
            p2_fainted_names.add(p2_state.get('name'))
            if first_ko_turn_p1_inflicted is None:
                first_ko_turn_p1_inflicted = tnum
            if tnum <= early_threshold: p1_kos_early += 1
            else: p1_kos_late += 1
        p2_name, p2_hp = p2_state.get('name'), p2_state.get('hp_pct')
        if p2_name and p2_hp is not None:
            prev_hp = last_p2_hp.get(p2_name)
            if prev_hp is not None:
                delta = max(0.0, prev_hp - p2_hp)
                p1_damage += delta
                p1_dmg_by_turn[tnum] = p1_dmg_by_turn.get(tnum, 0.0) + delta
                if turn.get('turn',999) <= 2:
                    p1_damage_first2 += delta
            last_p2_hp[p2_name] = p2_hp
        p1_name, p1_hp = p1_state.get('name'), p1_state.get('hp_pct')
        if p1_name and p1_hp is not None:
            prev_hp = last_p1_hp.get(p1_name)
            if prev_hp is not None:
                delta = max(0.0, prev_hp - p1_hp)
                p2_damage += delta
                p2_dmg_by_turn[tnum] = p2_dmg_by_turn.get(tnum, 0.0) + delta
                if turn.get('turn',999) <= 2:
                    p2_damage_first2 += delta
            last_p1_hp[p1_name] = p1_hp
        damage_diff_so_far = p1_damage - p2_damage
        if p2_fainted > prev_p2_fainted and damage_diff_so_far < -1.0:
            p1_comeback_kos += 1
        if p1_fainted > prev_p1_fainted and damage_diff_so_far > 1.0:
            p2_comeback_kos += 1
        p2_status = p2_state.get('status')
        if p2_name and p2_status and p2_pokemon_statuses.get(p2_name) != p2_status:
            p1_inflicted_statuses[p2_status] += 1
            p2_pokemon_statuses[p2_name] = p2_status
        p1_status = p1_state.get('status')
        if p1_name and p1_status and p1_pokemon_statuses.get(p1_name) != p1_status:
            p2_inflicted_statuses[p1_status] += 1
            p1_pokemon_statuses[p1_name] = p1_status
        p1_move = turn.get('p1_move_details') or {}
        p2_move = turn.get('p2_move_details') or {}
        if p1_move and p1_move.get('type'):
            p1_move_type_counts[(p1_move.get('type') or '').lower()] += 1
        if p2_move and p2_move.get('type'):
            p2_move_type_counts[(p2_move.get('type') or '').lower()] += 1
        if turn.get('p1_move_details'):
            p1_moves += 1
        if turn.get('p2_move_details'):
            p2_moves += 1
        p1_last_hp = p1_state.get('hp_pct', np.nan)
        p2_last_hp = p2_state.get('hp_pct', np.nan)

    out['tl_p1_moves'] = int(p1_moves)
    out['tl_p2_moves'] = int(p2_moves)
    out['tl_p1_est_damage'] = float(p1_damage)
    out['tl_p2_est_damage'] = float(p2_damage)
    out['damage_diff'] = float(p1_damage - p2_damage)
    out['fainted_diff'] = int(p1_fainted - p2_fainted)
    out['tl_p1_last_hp'] = float(p1_last_hp) if not np.isnan(p1_last_hp) else 0.0
    out['tl_p2_last_hp'] = float(p2_last_hp) if not np.isnan(p2_last_hp) else 0.0
    out['tl_p1_last_active'] = p1_last_active
    out['tl_p2_last_active'] = p2_last_active
    if p1_team:
        p1_total_hp_sum = sum(p.get('base_hp',0) for p in p1_team)
        p1_avg_def = np.mean([p.get('base_def',0) for p in p1_team] or [0])
        p1_avg_spd = np.mean([p.get('base_spd',0) for p in p1_team] or [0])
        out['tl_p2_damage_vs_p1_hp_pool'] = float(p2_damage / (p1_total_hp_sum + 1e-6))
        out['tl_p1_defensive_endurance'] = float((p1_avg_def + p1_avg_spd) / (p2_damage + 1e-6))
    out['tl_p1_comeback_kos'] = int(p1_comeback_kos)
    out['tl_p2_comeback_kos'] = int(p2_comeback_kos)
    out['tl_comeback_kos_diff'] = int(p1_comeback_kos - p2_comeback_kos)
    common_statuses = ['brn','par','slp','frz','psn','tox']
    for status in common_statuses:
        out[f'tl_p1_inflicted_{status}_count'] = int(p1_inflicted_statuses.get(status,0))
        out[f'tl_p2_inflicted_{status}_count'] = int(p2_inflicted_statuses.get(status,0))
        out[f'tl_inflicted_{status}_diff'] = int(p1_inflicted_statuses.get(status,0) - p2_inflicted_statuses.get(status,0))
    common_move_types = ['normal','fire','water','electric','grass','psychic','ice','dragon','rock','ground','flying','ghost','bug','poison','fighting']
    for mt in common_move_types:
        out[f'tl_p1_move_type_{mt}_count'] = int(p1_move_type_counts.get(mt,0))
        out[f'tl_p2_move_type_{mt}_count'] = int(p2_move_type_counts.get(mt,0))
        out[f'tl_move_type_{mt}_count_diff'] = int(p1_move_type_counts.get(mt,0) - p2_move_type_counts.get(mt,0))
    out['tl_p1_damage_first2'] = float(p1_damage_first2)
    out['tl_p2_damage_first2'] = float(p2_damage_first2)
    out['tl_first2_damage_diff'] = float(p1_damage_first2 - p2_damage_first2)
    
    # --- FE AGGIUNTO DAL NOTEBOOK ---
    turns_count = max(1, len(seen_turns))
    out['tl_turns_count'] = int(turns_count)
    out['tl_p1_moves_rate'] = float(p1_moves / turns_count)
    out['tl_p2_moves_rate'] = float(p2_moves / turns_count)
    out['tl_p1_damage_per_turn'] = float(p1_damage / turns_count)
    out['tl_p2_damage_per_turn'] = float(p2_damage / turns_count)
    out['tl_damage_rate_diff'] = float(out['tl_p1_damage_per_turn'] - out['tl_p2_damage_per_turn'])
    if seen_turns:
        recent_turns = sorted(seen_turns)[-5:]
        p1_last5 = sum(p1_dmg_by_turn.get(t,0.0) for t in recent_turns)
        p2_last5 = sum(p2_dmg_by_turn.get(t,0.0) for t in recent_turns)
    else:
        p1_last5 = p2_last5 = 0.0
    out['tl_p1_damage_last5'] = float(p1_last5)
    out['tl_p2_damage_last5'] = float(p2_last5)
    out['tl_last5_damage_diff'] = float(p1_last5 - p2_last5)
    out['tl_p1_last5_damage_ratio'] = float(p1_last5 / (p1_damage + 1e-6))
    out['tl_p2_last5_damage_ratio'] = float(p2_last5 / (p2_damage + 1e-6))
    out['tl_last5_damage_ratio_diff'] = float(out['tl_p1_last5_damage_ratio'] - out['tl_p2_last5_damage_ratio'])
    if seen_turns:
        ts = sorted(seen_turns)
        w = np.linspace(1.0, 2.0, num=len(ts))
        w = w / (w.sum() + 1e-9)
        adv = [(p1_dmg_by_turn.get(t,0.0) - p2_dmg_by_turn.get(t,0.0)) for t in ts]
        out['tl_weighted_damage_diff'] = float(np.dot(w, adv))
        cum = 0.0
        signs = []
        for t in ts:
            cum += (p1_dmg_by_turn.get(t,0.0) - p2_dmg_by_turn.get(t,0.0))
            s = 1 if cum > 1e-9 else (-1 if cum < -1e-9 else 0)
            if s != 0:
                if not signs or signs[-1] != s:
                    signs.append(s)
        sign_flips = max(0, len(signs) - 1)
        comeback_flag = 1 if (len(signs) >= 2 and signs[0] != signs[-1]) else 0
    else:
        out['tl_weighted_damage_diff'] = 0.0
        sign_flips = 0
        comeback_flag = 0
    out['tl_damage_adv_sign_flips'] = int(sign_flips)
    out['tl_comeback_flag'] = int(comeback_flag)
    out['tl_first_ko_turn_p1_inflicted'] = int(first_ko_turn_p1_inflicted or 0)
    out['tl_first_ko_turn_p1_taken'] = int(first_ko_turn_p1_taken or 0)
    out['tl_first_ko_turn_diff'] = int((first_ko_turn_p1_inflicted or 0) - (first_ko_turn_p1_taken or 0))
    out['tl_kos_early_p1'] = int(p1_kos_early)
    out['tl_kos_late_p1'] = int(p1_kos_late)
    out['tl_kos_early_p2'] = int(p2_kos_early)
    out['tl_kos_late_p2'] = int(p2_kos_late)
    for status in common_statuses:
        c1 = p1_inflicted_statuses.get(status,0)
        c2 = p2_inflicted_statuses.get(status,0)
        out[f'tl_p1_inflicted_{status}_rate'] = float(c1 / turns_count)
        out[f'tl_p2_inflicted_{status}_rate'] = float(c2 / turns_count)
        out[f'tl_inflicted_{status}_rate_diff'] = float((c1 - c2) / turns_count)
    # --- FINE FE AGGIUNTO ---
    return out

def ability_features(team: list, prefix: str) -> dict:
    immunity_abilities = {'levitate':0,'volt_absorb':0,'water_absorb':0,'flash_fire':0}
    stat_drop_abilities = {'intimidate':0}
    weather_abilities = {'drought':0,'drizzle':0,'sand_stream':0}
    out = {}
    for pokemon in team:
        ability = (pokemon.get('ability','') or '').lower().replace(' ','_')
        if ability in immunity_abilities:
            immunity_abilities[ability] += 1
        if ability in stat_drop_abilities:
            stat_drop_abilities[ability] += 1
        if ability in weather_abilities:
            weather_abilities[ability] += 1
    for ability,count in immunity_abilities.items():
        out[f'{prefix}ability_{ability}_count'] = int(count)
    for ability,count in stat_drop_abilities.items():
        out[f'{prefix}ability_{ability}_count'] = int(count)
    for ability,count in weather_abilities.items():
        out[f'{prefix}ability_{ability}_count'] = int(count)
    out[f'{prefix}total_immunity_abilities'] = int(sum(immunity_abilities.values()))
    out[f'{prefix}total_stat_drop_abilities'] = int(sum(stat_drop_abilities.values()))
    return out

def prepare_record_features(record: dict, max_turns: int = 30) -> dict:
    out = {}
    out['battle_id'] = record.get('battle_id')
    if 'player_won' in record:
        out['player_won'] = int(bool(record.get('player_won')))
    p1_team = record.get('p1_team_details', [])
    out.update(team_aggregate_features(p1_team, prefix='p1_'))
    p2_lead = record.get('p2_lead_details', {})
    out.update(lead_aggregate_features(p2_lead, prefix='p2_lead_'))
    out.update(ability_features(p1_team, prefix='p1_'))
    p1_lead = p1_team[0] if p1_team else {}
    out.update(lead_vs_lead_features(p1_lead, p2_lead))
    out.update(ability_features([p2_lead], prefix='p2_lead_'))
    out['p1_intimidate_vs_lead'] = 1 if out.get('p1_ability_intimidate_count',0) > 0 else 0
    tl = record.get('battle_timeline', [])
    out.update(summary_from_timeline(tl[:max_turns], p1_team))
    out['team_hp_sum_minus_p2lead_hp'] = out.get('p1_base_hp_sum', 0) - out.get('p2_lead_base_hp', 0)
    out['team_spa_mean_minus_p2spa'] = out.get('p1_base_spa_mean', 0) - out.get('p2_lead_base_spa', 0)
    out['speed_advantage'] = out.get('p1_base_spe_sum', 0) - out.get('p2_lead_base_spe', 0)
    out['n_unique_types_diff'] = out.get('p1_n_unique_types', 0) - out.get('p2_lead_n_unique_types', 1)
    p1_moves = max(out.get('tl_p1_moves',1),1)
    p2_moves = max(out.get('tl_p2_moves',1),1)
    out['damage_per_turn_diff'] = (out.get('tl_p1_est_damage',0.0)/p1_moves) - (out.get('tl_p2_est_damage',0.0)/p2_moves)
    out['last_pair'] = f"{out.get('tl_p1_last_active','')}_VS_{out.get('tl_p2_last_active','')}"
    out.update(calculate_type_advantage(p1_team, p2_lead))
    p2_lead_bulk = out.get('p2_lead_base_def',1) + out.get('p2_lead_base_spd',1)
    out['p1_se_options_vs_lead_bulk'] = out.get('p1_super_effective_options',0) / (p2_lead_bulk + 1e-6)
    
    # --- FE AGGIUNTO DAL NOTEBOOK (Team P2) ---
    p2_team = record.get('p2_team_details', [])
    if p2_team:
        out.update(team_aggregate_features(p2_team, prefix='p2_'))
        out['team_hp_sum_diff'] = out.get('p1_base_hp_sum',0) - out.get('p2_base_hp_sum',0)
        out['team_spa_mean_diff'] = out.get('p1_base_spa_mean',0) - out.get('p2_base_spa_mean',0)
        out['team_spe_mean_diff'] = out.get('p1_base_spe_mean',0) - out.get('p2_base_spe_mean',0)
        out['n_unique_types_team_diff'] = out.get('p1_n_unique_types',0) - out.get('p2_n_unique_types',0)
    # --- FINE FE AGGIUNTO ---
    return out

def create_features_from_raw(data: list) -> pd.DataFrame:
    rows = []
    for b in tqdm(data, desc='FE'):
        try:
            feat = prepare_record_features(b, max_turns=30)
            if 'battle_id' not in feat:
                feat['battle_id'] = b.get('battle_id')
            rows.append(feat)
        except Exception as e:
            rows.append({'battle_id': b.get('battle_id'), 'error': 1})
    df = pd.DataFrame(rows)
    if 'player_won' in df.columns:
        df['player_won'] = df['player_won'].astype(int)
    return df.fillna(0)


# --- 3. Caricamento Dati e Creazione Feature ---
print('Caricamento dati...')
try:
    train_raw = load_jsonl(train_file_path)
    test_raw = load_jsonl(test_file_path)
    print(f'Train records: {len(train_raw)}, Test records: {len(test_raw)}')
except FileNotFoundError:
    print(f"ERRORE: File '{train_file_path}' o '{test_file_path}' non trovati.")
    exit()

print("\nCreazione Feature per Train set...")
train_df = create_features_from_raw(train_raw)
print("Creazione Feature per Test set...")
test_df = create_features_from_raw(test_raw)

# --- 4. Preparazione Dati per Stacking (AGGIORNATO) ---
print("\nPreparazione dati per Stacking...")

# Escludi colonne non numeriche (come nel notebook)
exclude_cols = ['battle_id', 'player_won']
string_cols = train_df.select_dtypes(include=['object']).columns.tolist()
exclude_cols.extend(string_cols)
FEATURES = [c for c in train_df.columns if c not in exclude_cols]

# --- NUOVA SELEZIONE FEATURE (con whitelist) ---
DROP_NEW_TIMELINE_FEATURES = True 
late_new_static = [
    'tl_turns_count','tl_p1_moves_rate','tl_p2_moves_rate',
    'tl_p1_damage_per_turn','tl_p2_damage_per_turn','tl_damage_rate_diff',
    'tl_p1_damage_last5','tl_p2_damage_last5','tl_last5_damage_diff',
    'tl_weighted_damage_diff','tl_first_ko_turn_p1_inflicted','tl_first_ko_turn_p1_taken',
    'tl_first_ko_turn_diff','tl_kos_early_p1','tl_kos_late_p1','tl_kos_early_p2','tl_kos_late_p2',
    'tl_p1_last5_damage_ratio','tl_p2_last5_damage_ratio','tl_last5_damage_ratio_diff',
    'tl_damage_adv_sign_flips','tl_comeback_flag'
]
rate_cols = [c for c in train_df.columns if c.startswith('tl_') and c.endswith('_rate')]
LATE_GAME_NEW_FEATURES = sorted(set(late_new_static + rate_cols))

WHITELIST_KEEP = {
    'tl_weighted_damage_diff', 'tl_last5_damage_diff',
    'tl_damage_adv_sign_flips', 'tl_comeback_flag',
    'tl_p1_last5_damage_ratio', 'tl_p2_last5_damage_ratio', 'tl_last5_damage_ratio_diff'
}
if DROP_NEW_TIMELINE_FEATURES:
    FEATURES = [c for c in FEATURES if (c not in LATE_GAME_NEW_FEATURES) or (c in WHITELIST_KEEP)]
# --- FINE NUOVA SELEZIONE ---

print(f"Feature numeriche selezionate: {len(FEATURES)}")

# Dati di training (100%)
X_train_full = train_df[FEATURES]
y_train_full = train_df['player_won'].values
# Dati di test
X_test_kaggle = test_df[FEATURES]

# --- 5. Generazione Predizioni OOF con K-Fold ---
print(f"\nAvvio K-Fold Cross-Validation (K={N_SPLITS}) per previsioni OOF...")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_predictions = np.zeros(len(y_train_full))
fold_auc_scores = []

# Usa i parametri aggiornati
fold_params = best_params.copy()
fold_params.pop('early_stopping_rounds', None) # Rimuovi per K-Fold
fold_params['verbose'] = 0 

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
    print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")

    X_train_fold, X_val_fold = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
    y_train_fold, y_val_fold = y_train_full[train_idx], y_train_full[val_idx]

    print(f"Training XGBoost su {len(X_train_fold)} campioni...")
    fold_model = XGBClassifier(**fold_params)
    fold_model.fit(X_train_fold, y_train_fold)

    print(f"Generazione previsioni OOF su {len(X_val_fold)} campioni...")
    fold_oof_preds = fold_model.predict_proba(X_val_fold)[:, 1]
    oof_predictions[val_idx] = fold_oof_preds

    fold_auc = roc_auc_score(y_val_fold, fold_oof_preds)
    fold_auc_scores.append(fold_auc)
    print(f"Fold {fold+1} AUC: {fold_auc:.4f}")

print("\nK-Fold completato.")
print(f"Mean OOF AUC across folds: {np.mean(fold_auc_scores):.4f}")

print("Salvataggio previsioni OOF...")
np.save(OOF_XGB_OUT_FILE, oof_predictions)
print(f"Previsioni OOF salvate in: {OOF_XGB_OUT_FILE} (Shape: {oof_predictions.shape})")

# --- 6. Addestramento Modello Finale sul 100% ---
print("\nTraining modello finale XGBoost sul 100% dei dati...")
final_model_params = best_params.copy()
final_model_params['verbose'] = 200 # Più verboso

final_model = XGBClassifier(**final_model_params)
final_model.fit(X_train_full, y_train_full)
print("Training finale completato.")

# --- 7. Genera Predizioni sul Test Set (Salva sia .npy che .csv) ---
print(f"\nGenerazione previsioni (probabilità) sul Test Set ({len(X_test_kaggle)} campioni)...")
test_predictions_proba = final_model.predict_proba(X_test_kaggle)[:, 1]

# Salvataggio .npy (per lo stacking)
print("Salvataggio previsioni .npy sul Test Set...")
np.save(TEST_PREDS_XGB_OUT_FILE, test_predictions_proba)
print(f"Previsioni sul Test Set salvate in: {TEST_PREDS_XGB_OUT_FILE}")

# Salvataggio .csv (per il blending)
print("Salvataggio previsioni .csv sul Test Set...")
try:
    # Usa gli ID dal DataFrame di test (che li ha preservati)
    test_ids = test_df['battle_id'].astype(int)
    if len(test_ids) != len(test_predictions_proba):
        raise ValueError(f"Mismatch IDs ({len(test_ids)}) vs Preds ({len(test_predictions_proba)})")
        
    submission_df = pd.DataFrame({
        'battle_id': test_ids,
        'player_won_proba': test_predictions_proba
    })
    submission_df.to_csv(SUBMISSION_XGB_PROBA_FILE, index=False)
    print(f"Submission (probabilità) salvata in: {SUBMISSION_XGB_PROBA_FILE}")
except Exception as e:
    print(f"❌ ERRORE durante il salvataggio del CSV di submission: {e}")


print(f"\n--- Script 16 (XGBoost Stacking Prep) Completato ---")
print(f"File OOF e Test Preds salvati in '{OOF_PREDS_DIR}' e '{SUBMISSION_DIR}'.")