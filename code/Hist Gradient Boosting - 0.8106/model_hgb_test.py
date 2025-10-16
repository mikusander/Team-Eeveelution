import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
import optuna
from preprocessing import build_ml_dataframe 
import re
from encoders import TargetEncoder 

def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in df.columns}
    df = df.rename(columns=new_cols)
    return df

def run_hgb_pipeline_test(train_path: str, test_path: str, output_csv: str):
    
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
            'max_iter': 1000, 
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 20, 100), 
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'l2_regularization': trial.suggest_float('l2_regularization', 1e-3, 10.0, log=True), 
            'validation_fraction': 0.1, 
            'n_iter_no_change': 15,     
            'random_state': 42,
        }
        X_tr, X_val, y_tr, y_val = train_test_split(X_train_obj, y_train_main, test_size=0.25, random_state=42, stratify=y_train_main)
        
        model = HistGradientBoostingClassifier(**params)
        
        model.fit(X_tr, y_tr)
        
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)

    if TUNE_HYPERPARAMETERS:
        print("\n--- AVVIO OTTIMIZZAZIONE IPERPARAMETRI CON OPTUNA ---")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
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
        'learning_rate': 0.014866670192170072, #0.014866670192170072 = 0.8043-0.8170
        'max_leaf_nodes': 35,#35 = 0.8043-0.8170
        'max_depth': 8, #10 = 0.8031-0.8155/ 9 = 0.8027-0.8160/ 8 = 0.8043-0.8170
        'l2_regularization': 0.04575698125967396, #0.04575698125967396 = 0.8043-0.8170
    }

    DEFAULT_PARAMS = {
        'max_iter': 1000,
        'validation_fraction': 0.1,
        'n_iter_no_change': 20,
        'random_state': 42,
    }

    OPTIMAL_PARAMS = {**DEFAULT_PARAMS, **OPTIMAL_PARAMS}  """

    OPTIMAL_PARAMS = {
        'learning_rate': 0.05, #0.014866670192170072 = 0.8043-0.8170
        'max_leaf_nodes': 35,#35 = 0.8043-0.8170
        'max_depth': 3, #10 = 0.8031-0.8155/ 9 = 0.8027-0.8160/ 8 = 0.8043-0.8170
        'l2_regularization': 0.1, #0.04575698125967396 = 0.8043-0.8170
    }

    DEFAULT_PARAMS = {
        'max_iter': 500,
        'validation_fraction': 0.1,
        'n_iter_no_change': 20,
        'random_state': 42,
    }

    OPTIMAL_PARAMS = {**DEFAULT_PARAMS, **OPTIMAL_PARAMS} # 0.8065-0.8210

    """ OPTIMAL_PARAMS = {
        'learning_rate': 0.20393765741141795, #0.014866670192170072 = 0.8043-0.8170
        'max_leaf_nodes': 140,#35 = 0.8043-0.8170
        'max_depth': 3, #10 = 0.8031-0.8155/ 9 = 0.8027-0.8160/ 8 = 0.8043-0.8170
        'l2_regularization': 0.1, #0.04575698125967396 = 0.8043-0.8170
        'min_samples_leaf': 41,
        'early_stopping': True,
    }

    DEFAULT_PARAMS = {
        'max_iter': 751,
        'validation_fraction': 0.1,
        'n_iter_no_change': 10,
        'random_state': 42,
    }  """#0.8039-0.8190

    OPTIMAL_PARAMS = {**DEFAULT_PARAMS, **OPTIMAL_PARAMS} 

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

        model_hgb = HistGradientBoostingClassifier(**OPTIMAL_PARAMS)
        model_hgb.fit(X_tr_final, y_tr_fold)
        
        y_val_pred = model_hgb.predict(X_val_final)
        val_acc = accuracy_score(y_val_fold, y_val_pred)
        print(f"Fold {fold+1} → CV Accuracy: {val_acc:.4f}")
        cv_accuracies.append(val_acc)

        holdout_preds_ensemble += model_hgb.predict_proba(X_holdout_final_fold)[:, 1]
        test_preds_ensemble += model_hgb.predict_proba(X_test_final_final_fold)[:, 1]

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