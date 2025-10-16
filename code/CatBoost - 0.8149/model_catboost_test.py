import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import shap
import optuna
from preprocessing import build_ml_dataframe 

def run_catboost_pipeline_test(train_path: str, test_path: str, output_csv: str):
    
    TUNE_HYPERPARAMETERS = True

    df_train_full = build_ml_dataframe(train_path, save_path='train_features.csv')
    df_test = build_ml_dataframe(test_path, is_train=False, save_path='test_features.csv')

    TARGET = "player_won"
    ID_COL = "battle_id"
    X_train_full = df_train_full.drop(columns=[TARGET, ID_COL])
    y_train_full = df_train_full[TARGET]
    X_test_final = df_test.drop(columns=[ID_COL])

    X_train_main, X_holdout, y_train_main, y_holdout = train_test_split(
        X_train_full, y_train_full, 
        test_size=0.20, 
        random_state=42, 
        stratify=y_train_full 
    )
    print(f"\nDati di training divisi in: {len(X_train_main)} righe per allenamento/CV e {len(X_holdout)} per la validazione hold-out.")

    CATEGORICAL_COLS = ['tl_p1_last_active','tl_p2_last_active','p1_lead_name','p2_lead_name','last_pair']
    for col in CATEGORICAL_COLS:
        if col in X_train_main.columns:
            X_train_main[col] = X_train_main[col].astype(str)
            X_holdout[col] = X_holdout[col].astype(str)
            X_test_final[col] = X_test_final[col].astype(str)
            
    CAT_FEATURES = [c for c in CATEGORICAL_COLS if c in X_train_main.columns]

    print("\nCalcolo SHAP values e selezione top feature...")
    model_cb_shap = CatBoostClassifier(iterations=1000, depth=5, learning_rate=0.05, eval_metric='Accuracy', random_seed=42, verbose=0, cat_features=CAT_FEATURES)
    model_cb_shap.fit(X_train_main, y_train_main) 
    
    explainer = shap.TreeExplainer(model_cb_shap)
    shap_values = explainer.shap_values(X_train_main)

    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]

    shap_mean_abs = np.abs(shap_values).mean(axis=0)
    feat_importance = pd.DataFrame({'feature': X_train_main.columns, 'importance': shap_mean_abs}).sort_values(by='importance', ascending=False)

    top_n = 35
    top_features = feat_importance['feature'].head(top_n).tolist()
    print(f"\nTop {top_n} features basate su SHAP:\n", top_features) 
    
    X_train_top = X_train_main[top_features]
    X_holdout_top = X_holdout[top_features]
    X_test_top  = X_test_final[top_features]
    
    cv_cat_features = [f for f in CAT_FEATURES if f in top_features]

    def objective(trial: optuna.Trial) -> float:
        params = {
            'iterations': trial.suggest_int('iterations', 1000, 4000),
            'depth': trial.suggest_int('depth', 5, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 50.0, log=True),
            'rsm': trial.suggest_float('rsm', 0.3, 1.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'eval_metric': 'Accuracy', 'random_seed': 42, 'verbose': 0, 'cat_features': cv_cat_features,
        }
        X_tr, X_val, y_tr, y_val = train_test_split(X_train_top, y_train_main, test_size=0.25, random_state=42, stratify=y_train_main)
        model = CatBoostClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=150, verbose=0)
        preds = model.predict(X_val)
        accuracy = accuracy_score(y_val, preds)
        return accuracy

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

    OPTIMAL_PARAMS = {
        'iterations': 1924,
        'depth': 5,
        'learning_rate': 0.038756425850563964,
        'l2_leaf_reg': 2.63350046084044,
        'rsm': 0.8600441898727658,
        'bagging_temperature': 0.1916032066181844,
    }

    print("\n--- AVVIO CROSS-VALIDATION SUL TRAINING SET (80%) ---")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_accuracies = []
    holdout_preds_ensemble = np.zeros(X_holdout_top.shape[0])
    test_preds_ensemble = np.zeros(X_test_top.shape[0])

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_top, y_train_main)):
        print(f"\n--- Fold {fold+1} ---")
        X_tr, X_val = X_train_top.iloc[train_idx], X_train_top.iloc[val_idx]
        y_tr, y_val = y_train_main.iloc[train_idx], y_train_main.iloc[val_idx]

        final_params = OPTIMAL_PARAMS.copy()
        final_params.update({'eval_metric': 'Accuracy', 'random_seed': 42, 'verbose': 0, 'early_stopping_rounds': 200, 'cat_features': cv_cat_features})

        model_cb = CatBoostClassifier(**final_params)
        model_cb.fit(X_tr, y_tr, eval_set=(X_val, y_val)) 
        
        y_val_pred = model_cb.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        print(f"Fold {fold+1} → CV Accuracy: {val_acc:.4f}")
        cv_accuracies.append(val_acc)

        holdout_preds_ensemble += model_cb.predict_proba(X_holdout_top)[:, 1]
        test_preds_ensemble += model_cb.predict_proba(X_test_top)[:, 1]

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