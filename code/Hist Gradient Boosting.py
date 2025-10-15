import json
import argparse
import os
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, GridSearchCV

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
    # Unified preprocessing for Gradient Boosting only
    # --------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    smote = SMOTE(random_state=seed)
    scaler = StandardScaler()
    feature_names = common

    # Path for best params file
    best_params_path = '/Users/lorenzo/Desktop/Università/Sapienza/Computer Science - Magistrale/Foundations of Data Science/Kaggle/Pokemon/Github/Team-Eeveelution/models/params/best_params_hist_gradient_boosting.json'
    # Try to load best params if exists
    if os.path.exists(best_params_path):
        with open(best_params_path, 'r') as f:
            best_params = json.load(f)
    else:
        best_params = {}

    # Prepare result records for model
    model_results = []

    # HistGradientBoosting - Grid Search outside CV, then CV, then final fit and save
    gb_param_grid = {
        'max_iter': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'l2_regularization': [0.0, 0.1, 1.0]
    }
    if 'HistGradientBoosting' in best_params:
        gb_best = best_params['HistGradientBoosting']
        print(f"Using saved best params for HistGradientBoosting: {gb_best}")
    else:
        print("Running GridSearchCV for HistGradientBoosting (maximizing accuracy)...")
        # Use a sample of up to 10,000 rows for search
        X_sample, y_sample = X_train[:10000], y[:10000]
        scaler_grid = StandardScaler()
        X_sample_scaled = scaler_grid.fit_transform(X_sample)
        grid = GridSearchCV(
            HistGradientBoostingClassifier(random_state=seed),
            param_grid=gb_param_grid,
            scoring='accuracy',
            cv=5,
            n_jobs=-1,
        )
        grid.fit(X_sample_scaled, y_sample)
        best_params['HistGradientBoosting'] = grid.best_params_
        print(f"Best params for HistGradientBoosting: {grid.best_params_}")
        with open(best_params_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        gb_best = grid.best_params_

    # Cross-validation for HistGradientBoosting
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y)):
        print(f'HistGradientBoosting Fold {fold+1}')
        X_tr, y_tr = X_train[tr_idx], y[tr_idx]
        X_va, y_va = X_train[va_idx], y[va_idx]
        # SMOTE
        X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)
        # Scaling
        scaler_fold = StandardScaler()
        X_tr_sm_scaled = scaler_fold.fit_transform(X_tr_sm)
        X_va_scaled = scaler_fold.transform(X_va)
        X_test_scaled = scaler_fold.transform(X_test)
        # Fit
        clf_fold = HistGradientBoostingClassifier(random_state=seed, **gb_best)
        clf_fold.fit(X_tr_sm_scaled, y_tr_sm)
        train_pred = clf_fold.predict(X_tr_sm_scaled)
        val_pred = clf_fold.predict(X_va_scaled)
        train_acc = accuracy_score(y_tr_sm, train_pred)
        val_acc = accuracy_score(y_va, val_pred)
        print(f"Fold {fold+1} - Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")
        oof_preds[va_idx] = val_pred
        # For test set, accumulate average predictions
        if hasattr(clf_fold, "predict_proba"):
            test_preds += clf_fold.predict_proba(X_test_scaled)[:,1] / skf.n_splits
        else:
            test_preds += clf_fold.predict(X_test_scaled) / skf.n_splits
    oof_preds_bin = (oof_preds >= 0.5).astype(int)
    acc = accuracy_score(y, oof_preds_bin)
    report = classification_report(y, oof_preds_bin, output_dict=True)
    cm = confusion_matrix(y, oof_preds_bin)
    print(f"HistGradientBoosting OOF accuracy: {acc:.5f}")
    model_results.append({
        'model': 'HistGradientBoosting',
        'accuracy': acc,
        'report': report,
        'confusion_matrix': cm.tolist()
    })
    # After CV, fit on all train and save final model
    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train)
    clf_final = HistGradientBoostingClassifier(random_state=seed, **gb_best)
    clf_final.fit(X_train_scaled, y)
    final_model_path = f"/Users/lorenzo/Desktop/Università/Sapienza/Computer Science - Magistrale/Foundations of Data Science/Kaggle/Pokemon/Github/Team-Eeveelution/models/HistGradientBoosting_model.joblib"
    print(f"Saved final HistGradientBoosting model to {final_model_path}")
    # Prepare submission for HistGradientBoosting
    X_test_scaled_final = scaler_final.transform(X_test)
    preds_final = clf_final.predict_proba(X_test_scaled_final)[:,1] if hasattr(clf_final, "predict_proba") else clf_final.predict(X_test_scaled_final)
    sub = pd.DataFrame({'battle_id': test_ids, 'player_won': (preds_final >= 0.5).astype(int)})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f'Wrote submission to {out_path}')

    # Save model results to CSV
    output_rows = []
    for res in model_results:
        row = {
            'model': res['model'],
            'accuracy': res['accuracy'],
            'confusion_matrix': str(res['confusion_matrix'])
        }
        # Flatten main metrics from classification report
        for label in ['0', '1', 'macro avg', 'weighted avg']:
            if label in res['report']:
                for metric in ['precision', 'recall', 'f1-score', 'support']:
                    key = f"{label}_{metric}"
                    row[key] = res['report'][label].get(metric, None)
        output_rows.append(row)
    comparison_path = '/Users/lorenzo/Desktop/Università/Sapienza/Computer Science - Magistrale/Foundations of Data Science/Kaggle/Pokemon/Github/Team-Eeveelution/models/comparisons/model_hist_gradient_boosting.csv'
    pd.DataFrame(output_rows).to_csv(comparison_path, index=False)
    print(f"Saved model metrics to {comparison_path}")


# ---------------------------
# CLI and default paths
# ---------------------------

DATA_PATH = '/Users/lorenzo/Desktop/Università/Sapienza/Computer Science - Magistrale/Foundations of Data Science/Kaggle/Pokemon/input/fds-pokemon-battles-prediction-2025'
train_file_path = os.path.join(DATA_PATH, 'train.jsonl')
test_file_path = os.path.join(DATA_PATH, 'test.jsonl')
output_dir = '/Users/lorenzo/Desktop/Università/Sapienza/Computer Science - Magistrale/Foundations of Data Science/Kaggle/Pokemon/Github/Team-Eeveelution/output'
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
