import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import re
from preprocessing import build_ml_dataframe 
from encoders import TargetEncoder
import joblib
import os

def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in df.columns}
    df = df.rename(columns=new_cols)
    return df

def run_lightgbm_pipeline(train_path: str, test_path: str, output_csv: str):
    
    df_train_full = build_ml_dataframe(train_path, save_path='train_features.csv')
    df_test = build_ml_dataframe(test_path, is_train=False, save_path='test_features.csv')

    TARGET = "player_won"
    ID_COL = "battle_id"
    X_train_full = df_train_full.drop(columns=[TARGET, ID_COL])
    y_train_full = df_train_full[TARGET]
    X_test_final = df_test.drop(columns=[ID_COL])

    X_train_full = sanitize_feature_names(X_train_full)
    X_test_final = sanitize_feature_names(X_test_final)
    print(f"\nTraining finale su {len(X_train_full)} campioni del dataset completo.")

    CATEGORICAL_COLS = ['tl_p1_last_active','tl_p2_last_active','p1_lead_name','p2_lead_name','last_pair']

    OPTIMAL_PARAMS = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1,
        'boosting_type': 'gbdt', 'random_state': 42, 'n_estimators': 10000,
        'learning_rate': 0.01, 'num_leaves': 11, 'max_depth': 3,
        'lambda_l1': 0.05, 'lambda_l2': 0.1, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 1,
    }

    print("\n--- AVVIO CROSS-VALIDATION SUL TRAINING SET ---")

    models_dir = '/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/LightGBM - 0.8129/Models'
    encoders_dir = '/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/LightGBM - 0.8129/Encoders'

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_accuracies = []
    test_preds_ensemble = np.zeros(X_test_final.shape[0])

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
        print(f"--- Fold {fold+1}/{skf.n_splits} ---")
        
        X_tr_fold, X_val_fold = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_tr_fold, y_val_fold = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
        
        encoder_fold = TargetEncoder(cols=CATEGORICAL_COLS)
        X_tr_encoded = encoder_fold.fit_transform(X_tr_fold, y_tr_fold) 
        
        encoder_filename = os.path.join(encoders_dir, f"encoder_fold_{fold+1}.joblib")
        joblib.dump(encoder_fold, encoder_filename)
        print(f"Encoder del Fold {fold+1} salvato.")

        X_val_encoded = encoder_fold.transform(X_val_fold)
        
        X_tr_final = X_tr_encoded.drop(columns=CATEGORICAL_COLS)
        X_val_final = X_val_encoded.drop(columns=CATEGORICAL_COLS)

        model_lgbm = lgb.LGBMClassifier(**OPTIMAL_PARAMS)
        model_lgbm.fit(X_tr_final, y_tr_fold, 
                     eval_set=[(X_val_final, y_val_fold)], 
                     eval_metric='accuracy',
                     callbacks=[lgb.early_stopping(300, verbose=False)])
        
        model_filename = os.path.join(models_dir, f"lgbm_fold_{fold+1}.joblib")
        joblib.dump(model_lgbm, model_filename)
        print(f"Modello del Fold {fold+1} salvato.")
        
        y_val_pred = model_lgbm.predict(X_val_final)
        val_acc = accuracy_score(y_val_fold, y_val_pred)
        cv_accuracies.append(val_acc)
        
        X_test_encoded_fold = encoder_fold.transform(X_test_final)
        X_test_final_fold = X_test_encoded_fold.drop(columns=CATEGORICAL_COLS)
        test_preds_ensemble += model_lgbm.predict_proba(X_test_final_fold)[:, 1]


    test_preds_ensemble /= skf.n_splits
    mean_cv_acc = np.mean(cv_accuracies)

    print(f"\n--- RISULTATI FINALE ---")
    print(f"Mean CV Accuracy: {mean_cv_acc:.4f}")

    final_labels = (test_preds_ensemble > 0.5).astype(int)
    submission = pd.DataFrame({"battle_id": df_test[ID_COL], "player_won": final_labels})
    submission.to_csv(output_csv, index=False)
