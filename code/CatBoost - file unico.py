import json
import argparse
import os
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import xgboost as xgb
import joblib
import optuna
import shap
import matplotlib.pyplot as plt


# ---------------------------
# Utilities to parse the JSONL
# ---------------------------

def read_jsonl(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


# ---------------------------
# Feature engineering helpers
# ---------------------------

def safe_get(d: Dict, *keys, default=None):
    cur = d
    for k in keys:
        if cur is None:
            return default
        cur = cur.get(k, None) if isinstance(cur, dict) else None
    return cur if cur is not None else default


def team_aggregate_features(team: List[Dict[str, Any]], prefix: str = 'p1_') -> Dict[str, Any]:
    # team: list of 6 pokemon dicts
    stats = ['base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe']
    out = {}
    vals = {s: [] for s in stats}
    types_counter = Counter()
    names = []
    for p in team:
        names.append(p.get('name', ''))
        for s in stats:
            v = p.get(s, 0)
            vals[s].append(v)
        for t in p.get('types', []):
            types_counter[t.lower()] += 1
    # basic aggregates
    for s in stats:
        arr = np.array(vals[s], dtype=float)
        out[f'{prefix}{s}_sum'] = float(arr.sum())
        out[f'{prefix}{s}_mean'] = float(arr.mean())
        out[f'{prefix}{s}_max'] = float(arr.max())
        out[f'{prefix}{s}_min'] = float(arr.min())
        out[f'{prefix}{s}_std'] = float(arr.std())
    # types: top 6 most common types as features
    for t, cnt in types_counter.items():
        out[f'{prefix}type_{t}_count'] = int(cnt)
    # fallback for common types
    common_types = ['normal','fire','water','electric','grass','psychic','ice','dragon','rock','ground','flying']
    for t in common_types:
        out.setdefault(f'{prefix}type_{t}_count', 0)
    # names: we keep the lead name as a categorical
    out[f'{prefix}lead_name'] = names[0] if len(names) > 0 else ''
    out[f'{prefix}n_unique_names'] = len(set(names))
    return out


def summary_from_timeline(timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Create timeline-derived features limited to first 30 turns (timeline already has up to 30)
    out = {}
    if not timeline:
        return out
    # counts
    p1_moves = 0
    p2_moves = 0
    p1_damage = 0.0  # estimated by hp_pct differences of opponent's active mon between successive turns when same name
    p2_damage = 0.0
    p1_status_inflicted = 0
    p2_status_inflicted = 0
    p1_high_power_moves = 0
    p2_high_power_moves = 0
    p1_last_active = None
    p2_last_active = None
    p1_last_hp = None
    p2_last_hp = None

    # We'll track last seen hp_pct by active mon name to estimate damage done per side
    last_p2_hp_by_name = {}
    last_p1_hp_by_name = {}

    for t in timeline:
        p1_state = t.get('p1_pokemon_state', {}) or {}
        p2_state = t.get('p2_pokemon_state', {}) or {}
        # track last active
        p1_last_active = p1_state.get('name')
        p2_last_active = p2_state.get('name')
        p1_last_hp = p1_state.get('hp_pct')
        p2_last_hp = p2_state.get('hp_pct')

        # moves
        p1_move = t.get('p1_move_details')
        p2_move = t.get('p2_move_details')
        if p1_move:
            p1_moves += 1
            bp = p1_move.get('base_power', 0) or 0
            if bp >= 80:
                p1_high_power_moves += 1
            # status inducing move detection via category == STATUS and maybe move name
            if p1_move.get('category') == 'STATUS':
                p1_status_inflicted += 1
        if p2_move:
            p2_moves += 1
            bp = p2_move.get('base_power', 0) or 0
            if bp >= 80:
                p2_high_power_moves += 1
            if p2_move.get('category') == 'STATUS':
                p2_status_inflicted += 1

        # estimate damage by comparing hp_pct for same-name pokemon across turns
        # p1 damage to p2: if same p2 mon name seen previously, delta of last hp - current hp (if positive)
        name = p2_state.get('name')
        hp = p2_state.get('hp_pct')
        if name is not None and hp is not None:
            prev = last_p2_hp_by_name.get(name)
            if prev is not None:
                delta = max(0.0, prev - hp)
                p1_damage += delta
            last_p2_hp_by_name[name] = hp

        name1 = p1_state.get('name')
        hp1 = p1_state.get('hp_pct')
        if name1 is not None and hp1 is not None:
            prev1 = last_p1_hp_by_name.get(name1)
            if prev1 is not None:
                delta1 = max(0.0, prev1 - hp1)
                p2_damage += delta1
            last_p1_hp_by_name[name1] = hp1

    # populate
    out['tl_p1_moves'] = p1_moves
    out['tl_p2_moves'] = p2_moves
    out['tl_p1_high_power_moves'] = p1_high_power_moves
    out['tl_p2_high_power_moves'] = p2_high_power_moves
    out['tl_p1_status_moves'] = p1_status_inflicted
    out['tl_p2_status_moves'] = p2_status_inflicted
    out['tl_p1_est_damage'] = float(p1_damage)
    out['tl_p2_est_damage'] = float(p2_damage)
    out['tl_p1_last_active'] = p1_last_active or ''
    out['tl_p2_last_active'] = p2_last_active or ''
    out['tl_p1_last_hp'] = float(p1_last_hp) if p1_last_hp is not None else np.nan
    out['tl_p2_last_hp'] = float(p2_last_hp) if p2_last_hp is not None else np.nan
    # simple ratios
    out['tl_damage_ratio'] = float((p1_damage + 1e-6) / (p2_damage + 1e-6))
    out['tl_moves_diff'] = p1_moves - p2_moves

        # --- Nuove feature ingegnerizzate ---
    # Differenza netta di danno
    out['damage_diff'] = out['tl_p1_est_damage'] - out['tl_p2_est_damage']

    # Danno medio per mossa (robusto al numero di turni)
    out['damage_per_move_diff'] = (
        (out['tl_p1_est_damage'] / (out['tl_p1_moves'] + 1e-6))
        - (out['tl_p2_est_damage'] / (out['tl_p2_moves'] + 1e-6))
    )

    # Rapporto di HP residui (normalizzato, +1 per stabilità)
    out['hp_diff_ratio'] = (
        (out['tl_p1_last_hp'] + 1e-6) / (out['tl_p2_last_hp'] + 1e-6)
    )

    return out


# ---------------------------
# Full record -> feature row
# ---------------------------

def make_features(record: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    # ids and target
    out['battle_id'] = record.get('battle_id')
    if 'player_won' in record:
        out['player_won'] = int(bool(record.get('player_won')))

    # team features
    p1_team = record.get('p1_team_details', [])
    p2_lead = record.get('p2_lead_details', {})
    out.update(team_aggregate_features(p1_team, prefix='p1_'))

    # p2 lead aggregate (single mon)
    for s in ['base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe']:
        out[f'p2_lead_{s}'] = p2_lead.get(s, 0)
    out['p2_lead_name'] = p2_lead.get('name', '')
    types = p2_lead.get('types', [])
    for t in types:
        out[f'p2_lead_type_{t.lower()}'] = 1
    # ensure zero for common types
    common_types = ['normal','fire','water','electric','grass','psychic','ice','dragon','rock','ground','flying']
    for t in common_types:
        out.setdefault(f'p2_lead_type_{t}', 0)

    # timeline summaries
    timeline = record.get('battle_timeline', [])
    out.update(summary_from_timeline(timeline))

    # engineered interactions
    out['team_hp_sum_minus_p2lead_hp'] = out.get('p1_base_hp_sum', 0) - out.get('p2_lead_base_hp', 0)
    out['team_spa_mean_minus_p2spa'] = out.get('p1_base_spa_mean', 0) - out.get('p2_lead_base_spa', 0)
    # last active name pair hash (simple)
    out['last_pair'] = f"{out.get('tl_p1_last_active','')}_VS_{out.get('tl_p2_last_active','')}"

    return out


# ---------------------------
# Utilities to build dataframe
# ---------------------------

def build_dataframe(path: str, is_train: bool = True) -> pd.DataFrame:
    rows = []
    for rec in read_jsonl(path):
        rows.append(make_features(rec))
    df = pd.DataFrame(rows)
    # reorganize columns: battlefield id first
    cols = list(df.columns)
    if 'battle_id' in cols:
        cols = ['battle_id'] + [c for c in cols if c != 'battle_id']
    df = df[cols]
    # simple cleaning: fillna
    df.fillna(-999, inplace=True)
    return df


# ---------------------------
# Encoding helpers
# ---------------------------

def target_encode(train_ser: pd.Series, target: pd.Series, min_samples_leaf=100, smoothing=10):
    # simple target encoding with smoothing
    averages = target.groupby(train_ser).agg(['mean','count'])
    prior = target.mean()
    counts = averages['count']
    means = averages['mean']
    smooth = (counts * means + smoothing * prior) / (counts + smoothing)
    return smooth.to_dict(), prior


def apply_target_encoding(df: pd.DataFrame, col: str, enc_map: Dict, prior: float, default=None):
    df[f'{col}_te'] = df[col].map(enc_map).fillna(prior if default is None else default)
    return df

# ---------------------------
# Common feature extraction (tabellare + timeline)
# ---------------------------

def prepare_record_features(record: Dict[str, Any], max_turns: int = 30) -> Dict[str, Any]:
    """
    Estrae tutte le feature tabellari da un record singolo
    """
    out = {}

    # ID e target
    out['battle_id'] = record.get('battle_id')
    if 'player_won' in record:
        out['player_won'] = int(bool(record.get('player_won')))

    # Team features
    p1_team = record.get('p1_team_details', [])
    p2_lead = record.get('p2_lead_details', {})
    out.update(team_aggregate_features(p1_team, prefix='p1_'))

    # P2 lead aggregate
    stats = ['base_hp','base_atk','base_def','base_spa','base_spd','base_spe']
    for s in stats:
        out[f'p2_lead_{s}'] = p2_lead.get(s, 0)
    out['p2_lead_name'] = p2_lead.get('name', '')
    types = p2_lead.get('types', [])
    for t in types:
        out[f'p2_lead_type_{t.lower()}'] = 1
    # fallback zero per tipi comuni
    common_types = ['normal','fire','water','electric','grass','psychic','ice','dragon','rock','ground','flying']
    for t in common_types:
        out.setdefault(f'p2_lead_type_{t}', 0)

    # Timeline summary per ML
    timeline = record.get('battle_timeline', [])
    out.update(summary_from_timeline(timeline))

    # Feature ingegnerizzate
    out['team_hp_sum_minus_p2lead_hp'] = out.get('p1_base_hp_sum', 0) - out.get('p2_lead_base_hp', 0)
    out['team_spa_mean_minus_p2spa'] = out.get('p1_base_spa_mean', 0) - out.get('p2_lead_base_spa', 0)
    out['last_pair'] = f"{out.get('tl_p1_last_active','')}_VS_{out.get('tl_p2_last_active','')}"

    return out


# ---------------------------
# Dataframe ML
# ---------------------------

def build_ml_dataframe(jsonl_path: str, is_train: bool = True) -> pd.DataFrame:
    rows = [prepare_record_features(rec) for rec in read_jsonl(jsonl_path)]
    df = pd.DataFrame(rows)
    # riorganizza ID
    if 'battle_id' in df.columns:
        cols = ['battle_id'] + [c for c in df.columns if c != 'battle_id']
        df = df[cols]
    df.fillna(-999, inplace=True)
    return df


# ---------------------------
# Data per DL (timeline sequenziale)
# ---------------------------

def prepare_timeline_tensor(record: Dict[str, Any], max_turns: int = 30) -> np.ndarray:
    """
    Converte la timeline in un array (max_turns x feature_dim)
    Feature per turno:
      - p1_hp_pct, p2_hp_pct
      - p1_move_base_power, p2_move_base_power
      - p1_move_category (one-hot: PHYSICAL, SPECIAL, STATUS)
      - p2_move_category (one-hot)
      - optional: p1_status, p2_status come embedding
    """
    timeline = record.get('battle_timeline', [])
    tensor = []
    category_map = {'PHYSICAL':0, 'SPECIAL':1, 'STATUS':2}

    for turn in range(max_turns):
        if turn < len(timeline):
            t = timeline[turn]
            # HP%
            p1_hp = t.get('p1_pokemon_state', {}).get('hp_pct', 0.0)
            p2_hp = t.get('p2_pokemon_state', {}).get('hp_pct', 0.0)

            # base_power
            p1_bp = t.get('p1_move_details', {}).get('base_power', 0.0) if t.get('p1_move_details') else 0.0
            p2_bp = t.get('p2_move_details', {}).get('base_power', 0.0) if t.get('p2_move_details') else 0.0

            # category one-hot
            p1_cat_vec = [0,0,0]
            p2_cat_vec = [0,0,0]
            p1_cat = t.get('p1_move_details', {}).get('category') if t.get('p1_move_details') else None
            p2_cat = t.get('p2_move_details', {}).get('category') if t.get('p2_move_details') else None
            if p1_cat in category_map:
                p1_cat_vec[category_map[p1_cat]] = 1
            if p2_cat in category_map:
                p2_cat_vec[category_map[p2_cat]] = 1

            # feature turno concatenata
            turn_features = [p1_hp, p2_hp, p1_bp, p2_bp] + p1_cat_vec + p2_cat_vec
        else:
            # padding se la battaglia ha meno di max_turns
            turn_features = [0.0]*8
        tensor.append(turn_features)

    return np.array(tensor, dtype=np.float32)


def build_dl_dataset(jsonl_path: str, max_turns: int = 30) -> Dict[str, Any]:
    """
    Restituisce un dict:
      'X' : np.array (n_battles, max_turns, feature_dim)
      'y' : np.array (n_battles,) (solo se target disponibile)
      'battle_ids' : lista di id
    """
    X, y, ids = [], [], []
    for rec in read_jsonl(jsonl_path):
        X.append(prepare_timeline_tensor(rec, max_turns))
        ids.append(rec.get('battle_id'))
        if 'player_won' in rec:
            y.append(int(rec.get('player_won')))
    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.float32) if y else None
    return {'X': X, 'y': y, 'battle_ids': ids}

# ---------------------------
# CatBoost
# ---------------------------

df_train = build_ml_dataframe("/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/Pokémon/Input/fds-pokemon-battles-prediction-2025/train.jsonl")
df_test  = build_ml_dataframe("/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/Pokémon/Input/fds-pokemon-battles-prediction-2025/test.jsonl")

TARGET = "player_won"
ID_COL = "battle_id"

X_train = df_train.drop(columns=[TARGET, ID_COL])
y_train = df_train[TARGET]
X_test  = df_test.drop(columns=[ID_COL])

# ---------------------------
# Target encoding per variabili categoriali
categorical_cols = ['tl_p1_last_active','tl_p2_last_active','p1_lead_name','p2_lead_name','last_pair']

for col in categorical_cols:
    if col in X_train.columns:
        enc_map, prior = target_encode(X_train[col], y_train)
        X_train = apply_target_encoding(X_train, col, enc_map, prior)
        X_test  = apply_target_encoding(X_test, col, enc_map, prior)

# Rimuovo le colonne originali categoriali
X_train = X_train.drop(columns=[c for c in categorical_cols if c in X_train.columns])
X_test  = X_test.drop(columns=[c for c in categorical_cols if c in X_test.columns])

# ---------------------------
# Calcolo SHAP e selezione top feature
# ---------------------------
print("\nCalcolo SHAP values e selezione top feature...")
model_cb_final = CatBoostClassifier(
    iterations=3000,
    depth=5,
    learning_rate=0.02,
    l2_leaf_reg=22,
    rsm=0.4,
    bagging_temperature=1.5,
    eval_metric='Accuracy',
    random_seed=42,
    verbose=0,
    early_stopping_rounds=200,
)
model_cb_final.fit(X_train, y_train) 
explainer = shap.TreeExplainer(model_cb_final)
shap_values = explainer.shap_values(X_train)

# Se il modello è binario, prendiamo solo la classe positiva
if isinstance(shap_values, list) and len(shap_values) == 2:
    shap_values = shap_values[1]

# Importanza media assoluta
shap_mean_abs = np.abs(shap_values).mean(axis=0)
feat_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': shap_mean_abs
}).sort_values(by='importance', ascending=False)

# Top N features (puoi cambiare N)
top_n = 20
top_features = feat_importance['feature'].head(top_n).tolist()
print(f"\nTop {top_n} features:\n", top_features)

# Nuovo dataset con solo top feature
X_train_top = X_train[top_features]
X_test_top  = X_test[top_features]

""" # ---------------------------
# Optuna Hyperparameter Tuning (solo top 20 features)
# ---------------------------
import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

def objective(trial):
    params = {
        "iterations": 3000,  # allineato al fit finale
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 10, 40),
        "rsm": trial.suggest_float("rsm", 0.3, 0.8),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.5, 2.0),
        "eval_metric": "Accuracy",
        "random_seed": 42,
        "verbose": 0,
    }
    
    model = CatBoostClassifier(**params)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_top, y_train, cv=skf, scoring='accuracy')
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)  # puoi aumentare trials se vuoi più esplorazione

print("\nBest Optuna parameters (top 20 features):")
print(study.best_params)
print("Best CV Accuracy:", study.best_value) """

#{'depth': 7, 'learning_rate': 0.010282576218851393, 'l2_leaf_reg': 36, 'rsm': 0.34866409002293647, 'bagging_temperature': 0.7004868652076222}

# ---------------------------
# Stratified 5-Fold CV con top features
# ---------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cat_fold_acc_top = []
cat_test_preds_top = np.zeros(X_test_top.shape[0])

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_top, y_train)):
    print(f"\n--- CatBoost Top Features Fold {fold+1} ---")
    X_tr, X_val = X_train_top.iloc[train_idx], X_train_top.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    """ model_cb = CatBoostClassifier(
        iterations=3000,
        depth=7,
        learning_rate=0.010282576218851393,
        l2_leaf_reg=36,
        rsm=0.34866409002293647,
        bagging_temperature=0.7004868652076222,
        eval_metric='Accuracy',
        random_seed=42,
        verbose=0,
        early_stopping_rounds=200,
    ) """
    model_cb = CatBoostClassifier(
        iterations=3000,
        depth=7,
        learning_rate=0.01,
        l2_leaf_reg=36,
        rsm=0.34,
        bagging_temperature=0.7,
        eval_metric='Accuracy',
        random_seed=42,
        verbose=0,
        early_stopping_rounds=200,
    )
    model_cb.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    # Train accuracy
    y_train_pred = model_cb.predict(X_tr)
    train_acc = accuracy_score(y_tr, y_train_pred)

    # Validation accuracy
    y_val_pred = model_cb.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)

    print(f"Fold {fold+1} → Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")
    # Accuracy
    y_val_pred = model_cb.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"Fold {fold+1} Accuracy: {val_acc:.4f}")
    cat_fold_acc_top.append(val_acc)

    # Accumulo predizioni test
    y_test_pred = model_cb.predict_proba(X_test_top)[:,1]
    cat_test_preds_top += y_test_pred

mean_acc_top = np.mean(cat_fold_acc_top)
print(f"\nCatBoost Mean CV Accuracy (Top {top_n} features): {mean_acc_top:.4f}")

# ---------------------------
# Fit finale su tutto il training set con top features
# ---------------------------
model_cb_final = CatBoostClassifier(
    iterations=3000,
    depth=5,
    learning_rate=0.02,
    l2_leaf_reg=22,
    rsm=0.4,
    bagging_temperature=1.5,
    eval_metric='Accuracy',
    random_seed=42,
    verbose=0,
    early_stopping_rounds=200,
)
model_cb_final.fit(X_train_top, y_train)

# Predizioni finali su test
cat_test_preds_top /= skf.n_splits
final_labels = (cat_test_preds_top > 0.5).astype(int)

# ---------------------------
# File submission
submission = pd.DataFrame({
    "battle_id": df_test[ID_COL],
    "player_won": final_labels
})
submission.to_csv("submission_catboost_top20.csv", index=False)
print("\nSubmission file created: submission_catboost_top20.csv")