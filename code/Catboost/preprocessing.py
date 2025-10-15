import pandas as pd
import numpy as np
from data_utils import read_jsonl
from features import prepare_record_features
from typing import Dict

def build_ml_dataframe(jsonl_path: str, is_train: bool = True) -> pd.DataFrame:
    rows = [prepare_record_features(rec) for rec in read_jsonl(jsonl_path)]
    df = pd.DataFrame(rows)
    # riorganizza ID
    if 'battle_id' in df.columns:
        cols = ['battle_id'] + [c for c in df.columns if c != 'battle_id']
        df = df[cols]
    df.fillna(-999, inplace=True)
    return df

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