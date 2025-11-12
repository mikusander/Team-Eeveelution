"""
Libreria di utilità per la pipeline LightGBM Pokémon.

Questo modulo contiene tutte le funzioni, le costanti e le configurazioni
di percorso necessarie per l'intera pipeline, dal caricamento dei dati
alla creazione della submission.

"""

# --- 1. IMPORT CONSOLIDATI ---
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
from collections import Counter
import warnings
from pathlib import Path 
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# --- 2. CONFIGURAZIONE GLOBALE E PERCORSI ---

# Definisci la directory base del progetto
BASE_DIR = Path(__file__).resolve().parent

# Cartelle
INPUT_DIR_JSONL = BASE_DIR / 'Input'
OUTPUT_DIR_DATA = BASE_DIR / 'LightGBM_Data_Pipeline'
OUTPUT_DIR_MODEL = BASE_DIR / 'LightGBM_Model_Outputs'
SUBMISSION_DIR = BASE_DIR / 'Submissions'
OOF_DIR = BASE_DIR / 'OOF_Predictions'

# Percorsi File (Fase 01: FE)
TRAIN_JSON_IN = INPUT_DIR_JSONL / 'train.jsonl'
TEST_JSON_IN = INPUT_DIR_JSONL / 'test.jsonl'
TRAIN_CSV_OUT = OUTPUT_DIR_DATA / 'train_features.csv'
TEST_CSV_OUT = OUTPUT_DIR_DATA / 'test_features.csv'
TEST_IDS_OUT = OUTPUT_DIR_DATA / 'test_ids.csv'

# Percorsi File (Fase 02: Preprocessing)
X_TRAIN_OUT = OUTPUT_DIR_DATA / 'X_train.npy'
Y_TRAIN_OUT = OUTPUT_DIR_DATA / 'y_train.npy'
X_TEST_OUT = OUTPUT_DIR_DATA / 'X_test.npy'
FEATURES_JSON_OUT = OUTPUT_DIR_MODEL / 'features_final.json'
MEDIANS_JSON_OUT = OUTPUT_DIR_MODEL / 'preprocessing_medians.json'

# Percorsi File (Fase 03: Training/Submission)
OOF_TRAIN_OUT = OOF_DIR / 'oof_lgbm_proba.npy'
OOF_TEST_OUT = OOF_DIR / 'test_preds_lgbm_proba.npy'
SUB_ENSEMBLE_PROBA_OUT = SUBMISSION_DIR / 'submission_lgbm_100pct_PROBA.csv'

# --- 3. COSTANTI DI MODELLO E GIOCO ---

# Costanti di efficacia dei tipi
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

# IPERPARAMETRI OTTIMALI (da Notebook Lightgbm_v2, Trial 71)
BEST_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'seed': 42,
    'n_jobs': -1,
    'learning_rate': 0.013679823261876542,
    'n_estimators': 650, # Dal trial (NB: nel notebook il CV usa 650, la sub finale 1000. Usiamo 650 per coerenza)
    'num_leaves': 97,
    'max_depth': 7,
    'min_child_samples': 70,
    'subsample': 0.6161988793364829,
    'subsample_freq': 4,
    'colsample_bytree': 0.6136973726187104,
    'reg_alpha': 2.261339389106705e-06,
    'reg_lambda': 1.8764227095178463,
    'min_split_gain': 0.3554832229238166
}

# --- 4. FUNZIONI HELPER (Feature Engineering) ---

def load_jsonl(path):
    """Carica un file .jsonl e lo restituisce come lista di dizionari."""
    data = []
    try:
        with open(path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    except FileNotFoundError:
        print(f"ERRORE: File non trovato: {path}")
        print("Assicurati che 'train.jsonl' e 'test.jsonl' siano nella directory 'Input/'.")
        return None
    return data

def get_effectiveness(attack_type: str, defense_types: list) -> float:
    """Calcola l'efficacia di un tipo contro una lista di tipi."""
    if not attack_type or not defense_types: return 1.0
    eff = 1.0
    for d in defense_types: eff *= TYPE_CHART.get(attack_type, {}).get(d, 1.0)
    return eff

def _entropy(counter: Counter) -> float:
    """Calcola l'entropia di un Counter."""
    total = sum(counter.values())
    if total == 0: return 0.0
    ent = 0.0
    for v in counter.values():
        p = v / total
        if p > 0: ent -= p * math.log(p, 2)
    return ent

def calculate_type_advantage(team1: list, team2_lead: dict) -> dict:
    """Calcola il vantaggio aggregato del team1 contro il lead del team2."""
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
    """Estrae statistiche aggregate (media, somma, std, etc.) da un team."""
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
    """Confronta direttamente le statistiche dei due Pokémon lead."""
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
    """Estrae le feature di base per un singolo Pokémon (il lead avversario)."""
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
    """Estrae feature "veloci" di confronto team vs lead."""
    out = {}
    p1_team = record.get('p1_team_details', [])
    p2_lead = record.get('p2_lead_details', {})
    timeline = record.get('battle_timeline', [])
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
        out['p1_weak_count_vs_lead'] = sum(1 for s in cov if s <= 0.5)
        
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
    
    p1_prods = [(p.get('base_hp',1)*p.get('base_atk',1)*p.get('base_def',1)*p.get('base_spa',1)*p.get('base_spd',1)*p.get('base_spe',1)) for p in p1_team]
    out['p1_avg_stat_product'] = float(np.mean(p1_prods))
    out['p1_max_stat_product'] = float(np.max(p1_prods))
    p2_prod = p2_lead.get('base_hp',1)*p2_lead.get('base_atk',1)*p2_lead.get('base_def',1)*p2_lead.get('base_spa',1)*p2_lead.get('base_spd',1)*p2_lead.get('base_spe',1)
    out['p1_stat_product_advantage'] = out['p1_avg_stat_product'] / max(p2_prod, 1)
    return out

def summary_from_timeline(timeline: list, p1_team: list) -> dict:
    """Estrae un riepilogo dettagliato della battaglia dalla timeline."""
    out = {}
    if not timeline: return {'tl_p1_moves':0,'tl_p2_moves':0,'tl_p1_est_damage':0.0,'tl_p2_est_damage':0.0,'damage_diff':0.0}
    p1_moves = p2_moves = 0; p1_damage = p2_damage = 0.0
    p1_last_active = p2_last_active = ''; p1_fainted = p2_fainted = 0
    p1_fainted_names = set(); p2_fainted_names = set()
    last_p1_hp = {}; last_p2_hp = {}
    p1_comeback_kos = p2_comeback_kos = 0
    p1_statuses = Counter(); p2_statuses = Counter()
    p1_poke_statuses = {}; p2_poke_statuses = {}
    p1_move_types = Counter(); p2_move_types = Counter()
    p1_dmg_first2 = p2_dmg_first2 = 0.0
    p1_dmg_by_turn = {}; p2_dmg_by_turn = {}; seen_turns = set()
    first_ko_p1_taken = None; first_ko_p1_inflicted = None
    p1_kos_early = p1_kos_late = p2_kos_early = p2_kos_late = 0

    for turn in timeline[:30]:
        prev_p1_f, prev_p2_f = p1_fainted, p2_fainted
        p1s = turn.get('p1_pokemon_state',{}) or {}; p2s = turn.get('p2_pokemon_state',{}) or {}
        tnum = turn.get('turn', len(seen_turns) + 1); seen_turns.add(tnum)

        if p1s.get('name'): p1_last_active = p1s.get('name')
        if p2s.get('name'): p2_last_active = p2s.get('name')

        if p1s.get('fainted') and p1s.get('name') not in p1_fainted_names:
            p1_fainted += 1; p1_fainted_names.add(p1s.get('name'))
            if first_ko_p1_taken is None: first_ko_p1_taken = tnum
            if tnum <= 10: p2_kos_early += 1
            else: p2_kos_late += 1
        if p2s.get('fainted') and p2s.get('name') not in p2_fainted_names:
            p2_fainted += 1; p2_fainted_names.add(p2s.get('name'))
            if first_ko_p1_inflicted is None: first_ko_p1_inflicted = tnum
            if tnum <= 10: p1_kos_early += 1
            else: p1_kos_late += 1

        if p2s.get('name') and p2s.get('hp_pct') is not None:
            prev = last_p2_hp.get(p2s['name'])
            if prev is not None:
                delta = max(0.0, prev - p2s['hp_pct'])
                p1_damage += delta; p1_dmg_by_turn[tnum] = p1_dmg_by_turn.get(tnum,0.0)+delta
                if tnum <= 2: p1_dmg_first2 += delta
            last_p2_hp[p2s['name']] = p2s['hp_pct']

        if p1s.get('name') and p1s.get('hp_pct') is not None:
            prev = last_p1_hp.get(p1s['name'])
            if prev is not None:
                delta = max(0.0, prev - p1s['hp_pct'])
                p2_damage += delta; p2_dmg_by_turn[tnum] = p2_dmg_by_turn.get(tnum,0.0)+delta
                if tnum <= 2: p2_dmg_first2 += delta
            last_p1_hp[p1s['name']] = p1s['hp_pct']

        dmg_diff = p1_damage - p2_damage
        if p2_fainted > prev_p2_f and dmg_diff < -1.0: p1_comeback_kos += 1
        if p1_fainted > prev_p1_f and dmg_diff > 1.0: p2_comeback_kos += 1

        if p2s.get('name') and p2s.get('status') and p2_poke_statuses.get(p2s['name']) != p2s['status']:
            p1_statuses[p2s['status']] += 1; p2_poke_statuses[p2s['name']] = p2s['status']
        if p1s.get('name') and p1s.get('status') and p1_poke_statuses.get(p1s['name']) != p1s['status']:
            p2_statuses[p1s['status']] += 1; p1_poke_statuses[p1s['name']] = p1s['status']

        if turn.get('p1_move_details'):
            p1_moves += 1; mtype = (turn['p1_move_details'].get('type') or '').lower()
            if mtype: p1_move_types[mtype] += 1
        if turn.get('p2_move_details'):
            p2_moves += 1; mtype = (turn['p2_move_details'].get('type') or '').lower()
            if mtype: p2_move_types[mtype] += 1

    out['tl_p1_moves'] = p1_moves; out['tl_p2_moves'] = p2_moves
    out['tl_p1_est_damage'] = float(p1_damage); out['tl_p2_est_damage'] = float(p2_damage)
    out['tl_p1_fainted'] = p1_fainted; out['tl_p2_fainted'] = p2_fainted
    turns = max(1, len(seen_turns))
    out['tl_p1_fainted_rate'] = p1_fainted/turns; out['tl_p2_fainted_rate'] = p2_fainted/turns
    out['damage_diff'] = float(p1_damage - p2_damage); out['fainted_diff'] = int(p1_fainted - p2_fainted)
    out['tl_p1_last_hp'] = float(last_p1_hp.get(p1_last_active, 0) if p1_last_active else 0)
    out['tl_p2_last_hp'] = float(last_p2_hp.get(p2_last_active, 0) if p2_last_active else 0)
    out['tl_p1_last_active'] = p1_last_active; out['tl_p2_last_active'] = p2_last_active
    
    if p1_team:
        p1_hp_sum = sum(p.get('base_hp',0) for p in p1_team)
        p1_def_avg = np.mean([p.get('base_def',0)+p.get('base_spd',0) for p in p1_team])
        out['tl_p2_damage_vs_p1_hp_pool'] = p2_damage / (p1_hp_sum + 1e-6)
        out['tl_p1_defensive_endurance'] = p1_def_avg / (p2_damage + 1e-6)
        
    out['tl_p1_comeback_kos'] = p1_comeback_kos; out['tl_p2_comeback_kos'] = p2_comeback_kos
    out['tl_comeback_kos_diff'] = p1_comeback_kos - p2_comeback_kos

    for s in ['brn','par','slp','frz','psn','tox']:
        out[f'tl_p1_inflicted_{s}_count'] = p1_statuses[s]
        out[f'tl_p2_inflicted_{s}_count'] = p2_statuses[s]
        out[f'tl_inflicted_{s}_diff'] = p1_statuses[s] - p2_statuses[s]
        out[f'tl_p1_inflicted_{s}_rate'] = p1_statuses[s]/turns
        out[f'tl_p2_inflicted_{s}_rate'] = p2_statuses[s]/turns
        out[f'tl_inflicted_{s}_rate_diff'] = (p1_statuses[s]-p2_statuses[s])/turns

    for mt in ['normal','fire','water','electric','grass','psychic','ice','dragon','rock','ground','flying','ghost','bug','poison','fighting']:
        out[f'tl_p1_move_type_{mt}_count'] = p1_move_types[mt]
        out[f'tl_p2_move_type_{mt}_count'] = p2_move_types[mt]
        out[f'tl_move_type_{mt}_count_diff'] = p1_move_types[mt] - p2_move_types[mt]

    out['tl_p1_damage_first2'] = p1_dmg_first2; out['tl_p2_damage_first2'] = p2_dmg_first2
    out['tl_first2_damage_diff'] = p1_dmg_first2 - p2_dmg_first2
    out['tl_turns_count'] = turns
    out['tl_p1_moves_rate'] = p1_moves/turns; out['tl_p2_moves_rate'] = p2_moves/turns
    out['tl_p1_damage_per_turn'] = p1_damage/turns; out['tl_p2_damage_per_turn'] = p2_damage/turns
    out['tl_damage_rate_diff'] = out['tl_p1_damage_per_turn'] - out['tl_p2_damage_per_turn']

    recent = sorted(seen_turns)[-5:] if seen_turns else []
    p1_last5 = sum(p1_dmg_by_turn.get(t,0) for t in recent)
    p2_last5 = sum(p2_dmg_by_turn.get(t,0) for t in recent)
    out['tl_p1_damage_last5'] = p1_last5; out['tl_p2_damage_last5'] = p2_last5
    out['tl_last5_damage_diff'] = p1_last5 - p2_last5
    out['tl_p1_last5_damage_ratio'] = p1_last5/(p1_damage+1e-6)
    out['tl_p2_last5_damage_ratio'] = p2_last5/(p2_damage+1e-6)
    out['tl_last5_damage_ratio_diff'] = out['tl_p1_last5_damage_ratio'] - out['tl_p2_last5_damage_ratio']

    if seen_turns:
        ts = sorted(seen_turns); w = np.linspace(1.0, 2.0, num=len(ts)); w = w/w.sum()
        adv = [(p1_dmg_by_turn.get(t,0)-p2_dmg_by_turn.get(t,0)) for t in ts]
        out['tl_weighted_damage_diff'] = float(np.dot(w, adv))
        cum = 0.0; signs = []
        for t in ts:
            cum += (p1_dmg_by_turn.get(t,0)-p2_dmg_by_turn.get(t,0))
            s = 1 if cum > 1e-9 else (-1 if cum < -1e-9 else 0)
            if s != 0 and (not signs or signs[-1] != s): signs.append(s)
        out['tl_damage_adv_sign_flips'] = max(0, len(signs)-1)
        out['tl_comeback_flag'] = 1 if (len(signs)>=2 and signs[0]!=signs[-1]) else 0
    else:
        out['tl_weighted_damage_diff'] = 0.0; out['tl_damage_adv_sign_flips'] = 0; out['tl_comeback_flag'] = 0

    out['tl_first_ko_turn_p1_inflicted'] = int(first_ko_p1_inflicted or 0)
    out['tl_first_ko_turn_p1_taken'] = int(first_ko_p1_taken or 0)
    out['tl_first_ko_turn_diff'] = out['tl_first_ko_turn_p1_inflicted'] - out['tl_first_ko_turn_p1_taken']
    out['tl_kos_early_p1'] = p1_kos_early; out['tl_kos_late_p1'] = p1_kos_late
    out['tl_kos_early_p2'] = p2_kos_early; out['tl_kos_late_p2'] = p2_kos_late
    return out

def extract_move_coverage_from_timeline(timeline: list, prefix: str = 'p1_') -> dict:
    """Estrae la copertura dei tipi di mosse usate."""
    out = {}
    m_types = set(); m_cats = Counter(); unique_m = set(); stab = 0
    for turn in timeline[:30]:
        md = turn.get(f'{prefix[:-1]}_move_details')
        ps = turn.get(f'{prefix[:-1]}_pokemon_state', {})
        if not md: continue
        if md.get('name'): unique_m.add(md['name'])
        if t := (md.get('type') or '').lower(): m_types.add(t)
        if c := md.get('category'): m_cats[c] += 1
        if t in [pt.lower() for pt in ps.get('types', [])]: stab += 1
    
    out[f'{prefix}tl_unique_move_types'] = len(m_types)
    out[f'{prefix}tl_unique_moves_used'] = len(unique_m)
    out[f'{prefix}tl_stab_moves'] = stab
    out[f'{prefix}tl_physical_moves'] = m_cats['physical']
    out[f'{prefix}tl_special_moves'] = m_cats['special']
    out[f'{prefix}tl_status_moves'] = m_cats['status']
    out[f'{prefix}tl_coverage_score'] = len(m_types)/max(1, len(unique_m))
    tot = sum(m_cats.values())
    out[f'{prefix}tl_offensive_ratio'] = (m_cats['physical']+m_cats['special'])/tot if tot else 0.0
    out[f'{prefix}tl_status_ratio'] = m_cats['status']/tot if tot else 0.0
    return out

def ability_features(team: list, prefix: str) -> dict:
    """Conta le abilità chiave (immunità, intimidazione, meteo) in un team."""
    imm = {'levitate':0,'volt_absorb':0,'water_absorb':0,'flash_fire':0}
    drop = {'intimidate':0}; weather = {'drought':0,'drizzle':0,'sand_stream':0}
    for p in team:
        a = (p.get('ability','') or '').lower().replace(' ','_')
        if a in imm: imm[a]+=1
        if a in drop: drop[a]+=1
        if a in weather: weather[a]+=1
    out = {}
    for k,v in imm.items(): out[f'{prefix}ability_{k}_count'] = v
    for k,v in drop.items(): out[f'{prefix}ability_{k}_count'] = v
    for k,v in weather.items(): out[f'{prefix}ability_{k}_count'] = v
    out[f'{prefix}total_immunity_abilities'] = sum(imm.values())
    out[f'{prefix}total_stat_drop_abilities'] = sum(drop.values())
    return out

def momentum_features(timeline: list) -> dict:
    """Calcola il momentum (vantaggio HP cumulativo) e la sua volatilità."""
    out = {}
    if not timeline: return out
    advs = []; cum = 0.0
    for turn in timeline[:30]:
        p1h = turn.get('p1_pokemon_state', {}).get('hp_pct', 100)
        p2h = turn.get('p2_pokemon_state', {}).get('hp_pct', 100)
        cum += (p1h - p2h); advs.append(cum)
    if advs:
        slope, intercept = np.polyfit(np.arange(len(advs)), advs, 1)
        out['p1_momentum_slope'] = float(slope)
        out['p1_momentum_intercept'] = float(intercept)
        out['p1_final_advantage'] = float(advs[-1])
        out['p1_advantage_volatility'] = float(np.std(advs))
        out['p1_max_advantage'] = float(np.max(advs))
        out['p1_min_advantage'] = float(np.min(advs))
        out['p1_advantage_range'] = out['p1_max_advantage'] - out['p1_min_advantage']
    return out

def extract_opponent_team_from_timeline(timeline: list, p1_team: list) -> dict:
    """Estrae informazioni sui Pokémon avversari visti durante la battaglia."""
    out = {}
    seen = set(); types = []
    for turn in timeline[:30]:
        p2s = turn.get('p2_pokemon_state', {})
        if p2s.get('name') and p2s['name'] not in seen:
            seen.add(p2s['name'])
            types.extend([t.lower() for t in p2s.get('types', [])])
    
    out['p2_tl_unique_pokemon_seen'] = len(seen)
    out['p2_tl_switches_count'] = max(0, len(seen) - 1)
    out['p2_tl_unique_types_seen'] = len(set(types))
    out['p2_tl_type_entropy'] = _entropy(Counter(types))
    
    advs = 0
    if types and p1_team:
        for p1 in p1_team:
            for t1 in [t.lower() for t in p1.get('types', [])]:
                for t2 in set(types):
                    if get_effectiveness(t1, [t2]) >= 2.0: advs += 1
    out['p1_vs_p2_tl_type_advantages'] = advs
    out['p1_vs_p2_tl_type_advantages_per_poke'] = advs / max(1, len(p1_team))
    out['p2_tl_switch_rate'] = len(seen) / max(1, len(timeline[:30]))
    return out

def extract_information_advantage(timeline: list) -> dict:
    """Calcola il vantaggio informativo (Pokémon visti)."""
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
    """Calcola momentum avanzato (switch forzati, switch su immunità)."""
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
    """Cattura lo stato (differenza HP) a turni specifici (10, 20, 30)."""
    turns_lead = 0; hp_diff_10 = 0.0; hp_diff_20 = 0.0; hp_diff_30 = 0.0
    for i, turn in enumerate(timeline[:30]):
        t = i + 1
        h1 = turn.get('p1_pokemon_state', {}).get('hp_pct', 0)
        h2 = turn.get('p2_pokemon_state', {}).get('hp_pct', 0)
        if h1 > h2: turns_lead += 1
        if t == 10: hp_diff_10 = h1 - h2
        if t == 20: hp_diff_20 = h1 - h2
        hp_diff_30 = h1 - h2 # Ultimo stato disponibile
    return {
        'tl_turns_with_hp_lead': turns_lead,
        'tl_hp_diff_turn_10': float(hp_diff_10),
        'tl_hp_diff_turn_20': float(hp_diff_20),
        'tl_hp_diff_end': float(hp_diff_30)
    }

def extract_observed_mechanics(timeline: list) -> dict:
    """Conta meccaniche specifiche (cure, congelamento)."""
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


# --- 5. FUNZIONI MASTER (Feature Engineering) ---

def prepare_record_features_COMPLETE(record: dict, max_turns: int = 30) -> dict:
    """Funzione Master che orchestra tutti gli helper FE per un singolo record."""
    out = {'battle_id': record.get('battle_id')}
    if 'player_won' in record: out['player_won'] = int(bool(record['player_won']))
    
    p1_team = record.get('p1_team_details', [])
    p2_lead = record.get('p2_lead_details', {})
    p1_lead = p1_team[0] if p1_team else {}
    tl = record.get('battle_timeline', [])[:max_turns]
    
    # Feature statiche
    out.update(team_aggregate_features(p1_team, 'p1_'))
    out.update(lead_aggregate_features(p2_lead, 'p2_lead_'))
    out.update(ability_features(p1_team, 'p1_'))
    out.update(lead_vs_lead_features(p1_lead, p2_lead))
    out.update(ability_features([p2_lead], 'p2_lead_'))
    out['p1_intimidate_vs_lead'] = int(out.get('p1_ability_intimidate_count',0) > 0)
    
    # Feature dinamiche (da timeline)
    out.update(summary_from_timeline(tl, p1_team))
    out.update(extract_move_coverage_from_timeline(tl, 'p1_'))
    out.update(extract_move_coverage_from_timeline(tl, 'p2_'))
    out.update(extract_opponent_team_from_timeline(tl, p1_team))
    out.update(momentum_features(tl))
    
    # Feature veloci (da record)
    out.update(quick_boost_features_v2(record))
    
    # Feature calcolate (delta)
    out['team_hp_sum_minus_p2lead_hp'] = out.get('p1_base_hp_sum',0) - out.get('p2_lead_base_hp',0)
    out['team_spa_mean_minus_p2spa'] = out.get('p1_base_spa_mean',0) - out.get('p2_lead_base_spa',0)
    out['speed_advantage'] = out.get('p1_base_spe_sum',0) - out.get('p2_lead_base_spe',0)
    out['n_unique_types_diff'] = out.get('p1_n_unique_types',0) - out.get('p2_lead_n_unique_types',1)
    p1m = max(out.get('tl_p1_moves',1),1); p2m = max(out.get('tl_p2_moves',1),1)
    out['damage_per_turn_diff'] = (out.get('tl_p1_est_damage',0)/p1m) - (out.get('tl_p2_est_damage',0)/p2m)
    out['last_pair'] = f"{out.get('tl_p1_last_active','')}_VS_{out.get('tl_p2_last_active','')}"
    
    out.update(calculate_type_advantage(p1_team, p2_lead))
    p2_bulk = out.get('p2_lead_base_def',1) + out.get('p2_lead_base_spd',1)
    out['p1_se_options_vs_lead_bulk'] = out.get('p1_super_effective_options',0) / (p2_bulk + 1e-6)
    
    # Feature P2 (se presenti, es. in train)
    if p2_team := record.get('p2_team_details', []):
        out.update(team_aggregate_features(p2_team, 'p2_'))
        out['team_hp_sum_diff'] = out.get('p1_base_hp_sum',0) - out.get('p2_base_hp_sum',0)
        out['team_spa_mean_diff'] = out.get('p1_base_spa_mean',0) - out.get('p2_base_spa_mean',0)
        out['team_spe_mean_diff'] = out.get('p1_base_spe_mean',0) - out.get('p2_base_spe_mean',0)
        out['n_unique_types_team_diff'] = out.get('p1_n_unique_types',0) - out.get('p2_n_unique_types',0)
        
    # Feature avanzate (V2/V3)
    if tl:
        out.update(extract_information_advantage(tl))
        out.update(extract_advanced_momentum(tl))
        out.update(extract_gamestate_snapshots(tl))
        out.update(extract_observed_mechanics(tl))
    else:
        # Defaults per record senza timeline
        out.update({
            'tl_p1_revealed_count': 1, 'tl_p2_revealed_count': 1, 'tl_info_advantage': 0,
            'tl_p2_avg_reveal_turn': 30.0, 'tl_p1_immune_switches': 0, 'tl_p2_forced_switches': 0,
            'tl_turns_with_hp_lead': 0, 'tl_hp_diff_turn_10': 0.0, 'tl_hp_diff_turn_20': 0.0,
            'tl_hp_diff_end': 0.0, 'tl_heal_diff': 0, 'tl_freeze_adv': 0
        })
        
    return out

def create_features_from_raw(data: list) -> pd.DataFrame:
    """Applica la funzione master FE a un'intera lista di record (dati raw)."""
    rows = []
    for b in tqdm(data, desc='Feature Engineering'): 
        try:
            feat = prepare_record_features_COMPLETE(b)
            if 'battle_id' not in feat:
                feat['battle_id'] = b.get('battle_id')
            rows.append(feat)
        except Exception as e:
            print(f"ERRORE durante FE su battle_id {b.get('battle_id')}: {e}")
            rows.append({'battle_id': b.get('battle_id'), 'error': 1})
    df = pd.DataFrame(rows)
    if 'player_won' in df.columns:
        df['player_won'] = df['player_won'].astype(int)
    return df.fillna(0)


# --- 6. FUNZIONI DI PIPELINE (Esecuzione Fasi) ---

def ensure_directories():
    """Crea tutte le directory di output necessarie."""
    print("Verifica delle directory di output...")
    dirs_to_create = [
        INPUT_DIR_JSONL, OUTPUT_DIR_DATA, OUTPUT_DIR_MODEL,
        SUBMISSION_DIR, OOF_DIR
    ]
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
    print("Directory verificate.")


def run_01_feature_engineering():
    """
    FASE 01: Esegue la conversione da JSONL a CSV con feature engineering.
    """
    print("\n" + "="*30)
    print("INIZIO FASE 01: Feature Engineering (LGBM)")
    print("="*30)
    
    print('Caricamento dati raw...')
    train_raw = load_jsonl(TRAIN_JSON_IN)
    test_raw = load_jsonl(TEST_JSON_IN)
    
    if train_raw is None or test_raw is None:
        print("ERRORE: Dati raw non caricati. Interruzione.")
        return False
        
    print(f'Train records: {len(train_raw)}, Test records: {len(test_raw)}')
    
    print('Creazione feature per Train set...')
    train_df = create_features_from_raw(train_raw)
    
    print('Creazione feature per Test set...')
    test_df = create_features_from_raw(test_raw)
    
    print(f'Feature shape train/test: {train_df.shape} {test_df.shape}')
    
    # Salva i file CSV
    try:
        print(f"Salvataggio in {TRAIN_CSV_OUT}...")
        train_df.to_csv(TRAIN_CSV_OUT, index=False)
        
        print(f"Salvataggio in {TEST_CSV_OUT}...")
        test_df.to_csv(TEST_CSV_OUT, index=False)
        
        # Salva gli ID del test per la submission finale
        test_ids_df = test_df[['battle_id']].copy()
        test_ids_df.to_csv(TEST_IDS_OUT, index=False)
        print(f"Salvataggio ID test in {TEST_IDS_OUT}...")
        
    except Exception as e:
        print(f"ERRORE during il salvataggio dei CSV: {e}")
        return False

    print(f"\nFASE 01 (Feature Engineering) completata con successo.")
    print(f"File CSV salvati in '{OUTPUT_DIR_DATA}'")
    return True

def run_02_preprocessing():
    """
    FASE 02: Esegue il preprocessing (Imputazione) e salva i file .npy.
    """
    print("\n" + "="*30)
    print("INIZIO FASE 02: Preprocessing (LGBM)")
    print("="*30)

    # --- 1. Carica Dati CSV ---
    print(f"Caricamento {TRAIN_CSV_OUT.name} e {TEST_CSV_OUT.name}...")
    try:
        train_df = pd.read_csv(TRAIN_CSV_OUT)
        test_df = pd.read_csv(TEST_CSV_OUT)
    except FileNotFoundError:
        print("ERRORE: File .csv non trovati. Esegui prima la Fase 01.")
        return False

    # --- 2. Definisci Feature Numeriche ---
    print("Definizione feature numeriche...")
    exclude_cols = ['battle_id', 'player_won', 'error']
    string_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    exclude_cols.extend(string_cols)

    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    common_cols = list(train_cols.intersection(test_cols))

    ALL_NUMERIC_FEATURES = [c for c in train_df.columns if c not in exclude_cols and c in common_cols]
    FEATURES_FINAL = ALL_NUMERIC_FEATURES

    print(f"Trovate e selezionate {len(FEATURES_FINAL)} feature numeriche.")

    # Salva la lista
    with open(FEATURES_JSON_OUT, 'w') as f:
        json.dump(FEATURES_FINAL, f, indent=4)
    print(f"Lista feature salvata in: {FEATURES_JSON_OUT}")


    # --- 3. Preprocessing (Imputazione) ---
    print("\nAvvio preprocessing finale (Imputazione con Mediana)...")
    y = train_df['player_won'].astype(int).values

    final_train_df = train_df[FEATURES_FINAL].astype(float).replace([np.inf, -np.inf], np.nan)
    final_medians = final_train_df.median()
    X_train = final_train_df.fillna(final_medians).values

    final_test_df = test_df.reindex(columns=FEATURES_FINAL, fill_value=np.nan).astype(float).replace([np.inf, -np.inf], np.nan)
    X_test = final_test_df.fillna(final_medians).values

    print("Imputazione completata.")

    # Salva i file .npy
    np.save(X_TRAIN_OUT, X_train)
    np.save(Y_TRAIN_OUT, y)
    np.save(X_TEST_OUT, X_test)
    print(f"File .npy salvati: {X_TRAIN_OUT.name}, {Y_TRAIN_OUT.name}, {X_TEST_OUT.name}")

    # Salva le mediane
    with open(MEDIANS_JSON_OUT, 'w') as f:
        json.dump(final_medians.to_dict(), f, indent=4)
    print(f"Mediane di imputazione salvate in: {MEDIANS_JSON_OUT}")

    print("\nFASE 02 (Preprocessing) completata con successo.")
    return True

def run_03_training_and_submission():
    """
    FASE 03: Esegue il training CV 10-fold, salva OOF e crea la submission.
    """
    print("\n" + "="*30)
    print("INIZIO FASE 03: Training e Submission (LGBM)")
    print("="*30)

    # --- 1. Carica Dati .npy ---
    print("Caricamento dati .npy e IDs...")
    try:
        X = np.load(X_TRAIN_OUT)
        y = np.load(Y_TRAIN_OUT)
        X_test_matrix = np.load(X_TEST_OUT)
        test_ids_df = pd.read_csv(TEST_IDS_OUT)
        with open(FEATURES_JSON_OUT, 'r') as f:
            FEATURES = json.load(f)
    except FileNotFoundError:
        print("ERRORE: File .npy non trovati. Esegui prima la Fase 01 e 02.")
        return False

    print(f"Dati caricati: X_train {X.shape}, y_train {y.shape}, X_test {X_test_matrix.shape}")
    print(f"Utilizzando {len(FEATURES)} feature e i parametri V2.")

    # --- 2. Esegui 10-Fold CV e salva OOF ---
    print("\n=== Avvio 10-Fold Cross-Validation (per Ensemble) ===")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    oof_train_proba = np.zeros(len(X)) # Predizioni sul train set
    oof_test_proba_list = [] # Lista per le 10 predizioni sul test set
    fold_accuracies = []
    train_accuracies = []
    train_val_gaps = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        clf = lgb.LGBMClassifier(**BEST_PARAMS)
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
        
        # Predizioni di validazione (per OOF train e score)
        y_proba_val = clf.predict_proba(X_val)[:, 1]
        y_pred_val = (y_proba_val > 0.5).astype(int)
        oof_train_proba[val_idx] = y_proba_val
        val_acc = accuracy_score(y_val, y_pred_val)
        fold_accuracies.append(val_acc)
        
        # Calcolo gap
        y_pred_tr = clf.predict(X_tr)
        tr_acc = accuracy_score(y_tr, y_pred_tr)
        train_accuracies.append(tr_acc)
        train_val_gaps.append(tr_acc - val_acc)
        
        # Predizioni sul TEST SET (per OOF test)
        y_proba_test_fold = clf.predict_proba(X_test_matrix)[:, 1]
        oof_test_proba_list.append(y_proba_test_fold)
        
        print(f'Fold {fold_idx+1}: val_acc={val_acc*100:.2f}%, train_acc={tr_acc*100:.2f}%, gap={(tr_acc - val_acc)*100:.2f}%')

    print("\n--- Risultati CV (10 Folds) ---")
    mean_cv_acc = np.mean(fold_accuracies)
    std_cv_acc = np.std(fold_accuracies)
    mean_gap = np.mean(train_val_gaps)
    print(f'Mean CV accuracy: {mean_cv_acc*100:.2f}% ± {std_cv_acc*100:.2f}%')
    print(f'Mean gap (train - val): {mean_gap*100:.2f}%')

    # Salva OOF
    oof_test_proba = np.stack(oof_test_proba_list) # Shape (10, 5000)
    np.save(OOF_TRAIN_OUT, oof_train_proba)
    np.save(OOF_TEST_OUT, oof_test_proba)
    print(f"Predizioni OOF (Train e Test) salvate in '{OOF_DIR}'")

    # --- 3. Crea Submission ENSEMBLE ---
    print("\n=== Creazione Submission Ensemble (Media 10 Folds) ===")
    
    mean_test_proba = np.mean(oof_test_proba, axis=0)
    
    # Salva submission PROBABILITÀ (per ensemble)
    sub_ensemble_proba = pd.DataFrame({
        'battle_id': test_ids_df['battle_id'].astype(np.int64),
        'player_won': mean_test_proba
    })
    sub_ensemble_proba.to_csv(SUB_ENSEMBLE_PROBA_OUT, index=False)
    print(f"Submission ENSEMBLE (Proba) salvata in: {SUB_ENSEMBLE_PROBA_OUT}")

    print("\nFASE 03 (Training e Submission) completata con successo.")
    return True