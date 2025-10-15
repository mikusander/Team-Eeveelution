"""
pokemon_battle_model.py

A self-contained script to:
- parse train.jsonl / test.jsonl provided by the challenge
- engineer a set of baseline features from team details and the first 30-turn battle_timeline
- train a LightGBM classification model with cross-validation
- produce a submission .csv with columns (battle_id, player_won)

Notes / assumptions:
- This is a robust baseline focused on feature engineering from data you described.
- It uses simple, explainable engineered features (team-level aggregates, lead matchup features,
  timeline-derived summaries like damage done, status inflicted, move-power summaries, etc.)
- The model is LightGBM with basic hyperparameters. You can swap to XGBoost / RandomForest.
"""

import json
import argparse
import os
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

import json

# Optional imports for XGBoost, CatBoost, DNN
try:
    import xgboost as xgb
    has_xgb = True
except ImportError:
    has_xgb = False

try:
    from catboost import CatBoostClassifier
    has_catboost = True
except ImportError:
    has_catboost = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers # type: ignore
    has_tf = True
except ImportError:
    has_tf = False


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
# Train / Predict pipeline
# ---------------------------

def train_and_predict(train_path: str, test_path: str, out_path: str, seed: int = 42):
    print('Loading train...')
    train_df = build_dataframe(train_path, is_train=True)
    print('Loading test...')
    test_df = build_dataframe(test_path, is_train=False)

    # Keep battle_id
    train_ids = train_df['battle_id'].values
    test_ids = test_df['battle_id'].values

    y = train_df['player_won'].values
    train_df.drop(columns=['player_won'], inplace=True)

    # Candidate categorical columns
    cat_cols = ['p1_lead_name', 'p2_lead_name', 'tl_p1_last_active', 'tl_p2_last_active', 'last_pair']
    for c in cat_cols:
        if c not in train_df.columns:
            train_df[c] = ''
        if c not in test_df.columns:
            test_df[c] = ''

    # target encode p1_lead_name and p2_lead_name on train
    for c in ['p1_lead_name', 'p2_lead_name']:
        enc_map, prior = target_encode(train_df[c], pd.Series(y), min_samples_leaf=50, smoothing=20)
        train_df[f'{c}_te'] = train_df[c].map(enc_map).fillna(prior)
        test_df[f'{c}_te'] = test_df[c].map(enc_map).fillna(prior)

    # simple label encoding for last_pair (many values) -> frequency encoding
    for c in ['last_pair']:
        freq = train_df[c].value_counts().to_dict()
        train_df[f'{c}_freq'] = train_df[c].map(freq).fillna(0)
        test_df[f'{c}_freq'] = test_df[c].map(freq).fillna(0)

    # drop raw categorical columns from modeling (we keep encoded versions)
    to_drop = ['p1_lead_name', 'p2_lead_name', 'tl_p1_last_active', 'tl_p2_last_active', 'last_pair']
    for d in to_drop:
        if d in train_df.columns:
            train_df.drop(columns=[d], inplace=True)
        if d in test_df.columns:
            test_df.drop(columns=[d], inplace=True)

    # align columns
    train_cols = [c for c in train_df.columns if c != 'battle_id']
    test_cols = [c for c in test_df.columns if c != 'battle_id']
    common = [c for c in train_cols if c in test_cols]
    X_train = train_df[common].values
    X_test = test_df[common].values

    print(f'Features used: {len(common)}')

    # --------------------------
    # Unified preprocessing for all models
    # --------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    smote = SMOTE(random_state=seed)
    scaler = StandardScaler()
    feature_names = common

    # Path for best params file
    best_params_path = '/Users/lorenzo/Desktop/Università/Sapienza/Computer Science - Magistrale/Foundations of Data Science/Kaggle/Pokemon/models/best_params.json'
    # Try to load best params if exists
    if os.path.exists(best_params_path):
        with open(best_params_path, 'r') as f:
            best_params = json.load(f)
    else:
        best_params = {}

    # --------------------------
    # XGBoost - Optuna hyperparameter search, then CV, then final fit and save
    # --------------------------
    import optuna
    if not has_xgb:
        raise ImportError("xgboost is not installed. Please install xgboost to use this script.")
    if 'XGBoost' in best_params:
        xgb_best = best_params['XGBoost']
        print(f"Using saved best params for XGBoost: {xgb_best}")
    else:
        print("Running Optuna hyperparameter search for XGBoost (maximizing accuracy)...")
        X_sample, y_sample = X_train[:10000], y[:10000]
        scaler_grid = StandardScaler()
        X_sample_scaled = scaler_grid.fit_transform(X_sample)
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 2000, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_weight': trial.suggest_float('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.8, 1.2),
                'random_state': seed,
                'eval_metric': 'logloss',
            }

            clf_trial = xgb.XGBClassifier(**params)
            scores = []
            skf_opt = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

            for tr_idx, va_idx in skf_opt.split(X_sample_scaled, y_sample):
                X_tr, y_tr = X_sample_scaled[tr_idx], y_sample[tr_idx]
                X_va, y_va = X_sample_scaled[va_idx], y_sample[va_idx]

                sm = SMOTE(random_state=seed)
                X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)

                clf_trial.fit(
                    X_tr_sm, y_tr_sm,
                    eval_set=[(X_va, y_va)],
                    verbose=False
                )

                y_pred = clf_trial.predict(X_va)
                acc = accuracy_score(y_va, y_pred)
                scores.append(acc)

            return np.mean(scores)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=200)
        xgb_best = study.best_params
        print(f"Best params for XGBoost (Optuna): {xgb_best}")
        best_params['XGBoost'] = xgb_best
        with open(best_params_path, 'w') as f:
            json.dump(best_params, f, indent=4)

    # Cross-validation for XGBoost with SMOTE and StandardScaler
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y)):
        print(f'XGBoost Fold {fold+1}')
        X_tr, y_tr = X_train[tr_idx], y[tr_idx]
        X_va, y_va = X_train[va_idx], y[va_idx]
        # Apply SMOTE
        X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)
        # Standardize
        scaler_fold = StandardScaler()
        X_tr_sm_scaled = scaler_fold.fit_transform(X_tr_sm)
        X_va_scaled = scaler_fold.transform(X_va)
        X_test_scaled = scaler_fold.transform(X_test)
        # Fit model
        clf_fold = xgb.XGBClassifier(**xgb_best)
        clf_fold.fit(X_tr_sm_scaled, y_tr_sm)
        oof_preds[va_idx] = clf_fold.predict(X_va_scaled)
        # Predict test set (average over folds)
        if hasattr(clf_fold, "predict_proba"):
            test_preds += clf_fold.predict_proba(X_test_scaled)[:,1] / skf.n_splits
        else:
            test_preds += clf_fold.predict(X_test_scaled) / skf.n_splits
    oof_preds_bin = (oof_preds >= 0.5).astype(int)
    acc = accuracy_score(y, oof_preds_bin)
    print(f"XGBoost OOF accuracy: {acc:.5f}")
    # Prepare submission for XGBoost
    sub = pd.DataFrame({'battle_id': test_ids, 'player_won': (test_preds >= 0.5).astype(int)})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f'Wrote submission to {out_path}')
    # Final fit on all training data and save
    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train)
    clf_final = xgb.XGBClassifier(**xgb_best)
    clf_final.fit(X_train_scaled, y)
    final_model_path = f"/Users/lorenzo/Desktop/Università/Sapienza/Computer Science - Magistrale/Foundations of Data Science/Kaggle/Pokemon/models/XGBoost_final_model.joblib"
    joblib.dump(clf_final, final_model_path)
    print(f"Saved final XGBoost model to {final_model_path}")


# ---------------------------
# CLI and default paths
# ---------------------------

DATA_PATH = '/Users/lorenzo/Desktop/Università/Sapienza/Computer Science - Magistrale/Foundations of Data Science/Kaggle/Pokemon/input/fds-pokemon-battles-prediction-2025'
train_file_path = os.path.join(DATA_PATH, 'train.jsonl')
test_file_path = os.path.join(DATA_PATH, 'test.jsonl')
output_dir = '/Users/lorenzo/Desktop/Università/Sapienza/Computer Science - Magistrale/Foundations of Data Science/Kaggle/Pokemon/output'
output_file_path = os.path.join(output_dir, 'submission.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default=None)
    parser.add_argument('--test', type=str, default=None)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    if args.train is not None and args.test is not None and args.out is not None:
        train_and_predict(args.train, args.test, args.out)
    else:
        print('Running train_and_predict with default file paths...')
        train_and_predict(train_file_path, test_file_path, output_file_path)
