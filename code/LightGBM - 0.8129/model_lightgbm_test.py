import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import optuna
from preprocessing import build_ml_dataframe 
import re
from encoders import TargetEncoder 

def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in df.columns}
    df = df.rename(columns=new_cols)
    return df

def run_lightgbm_pipeline_test(train_path: str, test_path: str, output_csv: str):
    
    TUNE_HYPERPARAMETERS = False

    df_train_full = build_ml_dataframe(train_path, save_path='train_features.csv')
    df_test = build_ml_dataframe(test_path, is_train=False, save_path='test_features.csv')

    TARGET = "player_won"
    ID_COL = "battle_id"
    X_train_full = df_train_full.drop(columns=[TARGET, ID_COL])
    y_train_full = df_train_full[TARGET]
    X_test_final = df_test.drop(columns=[ID_COL])

    X_train_full = sanitize_feature_names(X_train_full)
    X_test_final = sanitize_feature_names(X_test_final)
    print("\nNomi delle feature sanificati.")

    X_train_main, X_holdout, y_train_main, y_holdout = train_test_split(
        X_train_full, y_train_full, 
        test_size=0.20, 
        random_state=42, 
        stratify=y_train_full 
    )
    print(f"\nDati di training divisi in: {len(X_train_main)} righe per allenamento/CV e {len(X_holdout)} per la validazione hold-out.")

    CATEGORICAL_COLS_original = ['tl_p1_last_active','tl_p2_last_active','p1_lead_name','p2_lead_name','last_pair']
    CATEGORICAL_COLS = [re.sub(r'[^A-Za-z0-9_]+', '', col) for col in CATEGORICAL_COLS_original]

    def objective(trial: optuna.Trial) -> float:
        encoder = TargetEncoder(cols=CATEGORICAL_COLS)
        X_train_encoded = encoder.fit_transform(X_train_main, y_train_main)
        X_train_obj = X_train_encoded.drop(columns=CATEGORICAL_COLS)

        params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1,
            'boosting_type': 'gbdt', 'random_state': 42,
            'n_estimators': 10000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.04, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 30, 150),
            'max_depth': trial.suggest_int('max_depth', 7, 11),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        }
        X_tr, X_val, y_tr, y_val = train_test_split(X_train_obj, y_train_main, test_size=0.25, random_state=42, stratify=y_train_main)
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)], eval_metric='accuracy',
                  callbacks=[lgb.early_stopping(150, verbose=False)])
        
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)

    if TUNE_HYPERPARAMETERS:
        print("\n--- AVVIO OTTIMIZZAZIONE IPERPARAMETRI CON OPTUNA ---")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        
        print("\n--- OTTIMIZZAZIONE COMPLETATA ---")
        print(f"Migliore accuracy di validazione: {study.best_value:.4f}")
        print("Parametri ottimali trovati:")
        
        best_params = study.best_trial.params
        print("OPTIMAL_PARAMS = {")
        for key, value in best_params.items():
            if isinstance(value, str):
                print(f"    '{key}': '{value}',")
            else:
                print(f"    '{key}': {value},")
        print("}")
        
        print("\n--> Copia il dizionario qui sopra e sostituiscilo a 'OPTIMAL_PARAMS'.")
        print("--> Poi, imposta TUNE_HYPERPARAMETERS = False per eseguire l'allenamento finale.")
        return

    """ OPTIMAL_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': 10000,          # Costruisci fino a 10000 alberi (ma si fermerà prima)
        'learning_rate': 0.01,          # Impara MOLTO lentamente
        'num_leaves': 31,               # Alberi di media complessità
        'max_depth': 7,                 # Limita la profondità massima
        'lambda_l1': 0.1,               # Aumenta la regolarizzazione L1
        'lambda_l2': 0.1,               # Aumenta la regolarizzazione L2
        'feature_fraction': 0.8,        # Usa l'80% delle feature per ogni albero
        'bagging_fraction': 0.8,        # Usa l'80% dei dati per ogni albero
        'bagging_freq': 1,
    } """ #0.8160
    OPTIMAL_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': 10000,
        'learning_rate': 0.01,  
        'num_leaves': 11,       # -- 21 = 31, 11 = 0.8175
        'max_depth': 3,         # 7 = 6 = 5, 4 = 0.8180, 3 = 0.8205
        'lambda_l1': 0.05,      # 0.05 = 0.8210
        'lambda_l2': 0.1,       
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
    }

    print("\n--- AVVIO CROSS-VALIDATION SUL TRAINING SET (80%) ---")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_accuracies = []
    holdout_preds_ensemble = np.zeros(X_holdout.shape[0])
    test_preds_ensemble = np.zeros(X_test_final.shape[0])

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_main, y_train_main)):
        print(f"\n--- Fold {fold+1} ---")
        
        X_tr_fold, X_val_fold = X_train_main.iloc[train_idx], X_train_main.iloc[val_idx]
        y_tr_fold, y_val_fold = y_train_main.iloc[train_idx], y_train_main.iloc[val_idx]
        
        encoder_fold = TargetEncoder(cols=CATEGORICAL_COLS)
        X_tr_encoded = encoder_fold.fit_transform(X_tr_fold, y_tr_fold)
        
        X_val_encoded = encoder_fold.transform(X_val_fold)
        X_holdout_encoded_fold = encoder_fold.transform(X_holdout)
        X_test_final_encoded_fold = encoder_fold.transform(X_test_final)

        X_tr_final = X_tr_encoded.drop(columns=CATEGORICAL_COLS)
        X_val_final = X_val_encoded.drop(columns=CATEGORICAL_COLS)
        X_holdout_final_fold = X_holdout_encoded_fold.drop(columns=CATEGORICAL_COLS)
        X_test_final_final_fold = X_test_final_encoded_fold.drop(columns=CATEGORICAL_COLS)

        model_lgbm = lgb.LGBMClassifier(**OPTIMAL_PARAMS, random_state=42)
        model_lgbm.fit(X_tr_final, y_tr_fold, 
                     eval_set=[(X_val_final, y_val_fold)], eval_metric='accuracy',
                     callbacks=[lgb.early_stopping(200, verbose=False)])
        
        y_val_pred = model_lgbm.predict(X_val_final)
        val_acc = accuracy_score(y_val_fold, y_val_pred)
        print(f"Fold {fold+1} → CV Accuracy: {val_acc:.4f}")
        cv_accuracies.append(val_acc)

        holdout_preds_ensemble += model_lgbm.predict_proba(X_holdout_final_fold)[:, 1]
        test_preds_ensemble += model_lgbm.predict_proba(X_test_final_final_fold)[:, 1]

    mean_cv_acc = np.mean(cv_accuracies)
    holdout_preds_ensemble /= skf.n_splits
    test_preds_ensemble /= skf.n_splits
    holdout_labels = (holdout_preds_ensemble > 0.5).astype(int)
    holdout_accuracy = accuracy_score(y_holdout, holdout_labels)
    
    print("\n" + "="*50)
    print("---               REPORT FINALE                ---")
    print("="*50)
    print(f"Accuracy media di Cross-Validation (su 80% dati): {mean_cv_acc:.4f}")
    print(f"Accuracy VERA sul Hold-out Set (su 20% dati):   {holdout_accuracy:.4f}  <-- IL TUO INDICATORE CHIAVE")
    print("-"*50)
    
    overfitting_gap = mean_cv_acc - holdout_accuracy
    if overfitting_gap > 0.01: 
        print(f"ATTENZIONE: Overfitting Rilevato! (Gap: {overfitting_gap:.4f})")
        print("Il modello performa peggio su dati mai visti. Prova a usare più regolarizzazione.")
    elif overfitting_gap < -0.005: 
        print("INFO: Performance sul hold-out è migliore della CV. Potrebbe essere un caso fortunato.")
    else:
        print("OTTIMO: Il modello sembra generalizzare bene! (Gap:")
        print(f" {overfitting_gap:.4f})")
    print("="*50 + "\n")

    final_labels = (test_preds_ensemble > 0.5).astype(int)
    submission = pd.DataFrame({"battle_id": df_test[ID_COL], "player_won": final_labels})
    submission.to_csv(output_csv, index=False)
    print(f"Submission file created: {output_csv}")