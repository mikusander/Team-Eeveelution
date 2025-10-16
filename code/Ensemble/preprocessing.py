import pandas as pd
import numpy as np
from data_utils import read_jsonl
from features import prepare_record_features
from typing import Dict

def build_ml_dataframe(jsonl_path: str, is_train: bool = True, save_path: str = None) -> pd.DataFrame:
    rows = [prepare_record_features(rec) for rec in read_jsonl(jsonl_path)]
    df = pd.DataFrame(rows)

    if 'battle_id' in df.columns:
        cols = ['battle_id'] + [c for c in df.columns if c != 'battle_id']
        df = df[cols]

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna('missing', inplace=True)
        else:
            df[col].fillna(-999, inplace=True)
    
    if save_path is not None:
        df.to_csv(save_path, index=False)
        print(f"DataFrame salvato in {save_path}")

    return df