import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import shap
import optuna
from preprocessing import build_ml_dataframe 

def run_xgboost_pipeline_test(train_path: str, test_path: str, output_csv: str):
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
    print(f"\nTraining data split into: {len(X_train_main)} rows for training/CV and {len(X_holdout)} for hold-out validation.")

    print("\nCalculating SHAP values and selecting top features...")
    model_xgb_shap = xgb.XGBClassifier(iterations=1000, max_depth=5, learning_rate=0.05, eval_metric='logloss', random_state=42, use_label_encoder=False, verbosity=0, enable_categorical=True)
    model_xgb_shap.fit(X_train_main, y_train_main)
    explainer = shap.TreeExplainer(model_xgb_shap)
    shap_values = explainer.shap_values(X_train_main)
    shap_mean_abs = np.abs(shap_values).mean(axis=0)
    feat_importance = pd.DataFrame({'feature': X_train_main.columns, 'importance': shap_mean_abs}).sort_values(by='importance', ascending=False)

    top_n = 35
    top_features = feat_importance['feature'].head(top_n).tolist()
    print(f"\nTop {top_n} SHAP-based features:\n", top_features)

    X_train_top = X_train_main[top_features]
    X_holdout_top = X_holdout[top_features]
    X_test_top  = X_test_final[top_features]

    def objective(trial: optuna.Trial) -> float:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 1000, 4000),
            'max_depth': trial.suggest_int('max_depth', 5, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.3, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'eval_metric': 'logloss', 'random_state': 42, 'use_label_encoder': False, 'verbosity': 0, 'enable_categorical': True,
        }
        X_tr, X_val, y_tr, y_val = train_test_split(X_train_top, y_train_main, test_size=0.25, random_state=42, stratify=y_train_main)
        model = xgb.XGBClassifier(**params, early_stopping_rounds=150)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
        preds = model.predict(X_val)
        accuracy = accuracy_score(y_val, preds)
        return accuracy

    if TUNE_HYPERPARAMETERS:
        print("\n--- STARTING HYPERPARAMETER OPTIMIZATION WITH OPTUNA ---")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        print("\n--- OPTIMIZATION COMPLETED ---")
        print(f"Best validation accuracy: {study.best_value:.4f}")
        print("Optimal parameters found:")
        best_params = study.best_trial.params
        print("OPTIMAL_PARAMS = {")
        for key, value in best_params.items():
            if isinstance(value, str):
                print(f"    '{key}': '{value}',")
            else:
                print(f"    '{key}': {value},")
        print("}")
        print("\n--> Copy the dictionary above and replace 'OPTIMAL_PARAMS'.")
        print("--> Then set TUNE_HYPERPARAMETERS = False to run final training.")
        return

    OPTIMAL_PARAMS = {
        'n_estimators': 1924,
        'max_depth': 5,
        'learning_rate': 0.038756425850563964,
        'subsample': 0.8600441898727658,
        'colsample_bytree': 0.8600441898727658,
        'eval_metric': 'logloss',
        'random_state': 42,
        'use_label_encoder': False,
        'verbosity': 0,
        'enable_categorical': True,
    }

    print("\n--- STARTING CROSS-VALIDATION ON TRAINING SET (80%) ---")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_accuracies = []
    holdout_preds_ensemble = np.zeros(X_holdout_top.shape[0])
    test_preds_ensemble = np.zeros(X_test_top.shape[0])

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_top, y_train_main)):
        print(f"\n--- Fold {fold+1} ---")
        X_tr, X_val = X_train_top.iloc[train_idx], X_train_top.iloc[val_idx]
        y_tr, y_val = y_train_main.iloc[train_idx], y_train_main.iloc[val_idx]
        final_params = OPTIMAL_PARAMS.copy()
        final_params.update({'early_stopping_rounds': 200})
        model_xgb = xgb.XGBClassifier(**final_params)
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
        y_val_pred = model_xgb.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        print(f"Fold {fold+1} → CV Accuracy: {val_acc:.4f}")
        cv_accuracies.append(val_acc)
        holdout_preds_ensemble += model_xgb.predict_proba(X_holdout_top)[:, 1]
        test_preds_ensemble += model_xgb.predict_proba(X_test_top)[:, 1]

    mean_cv_acc = np.mean(cv_accuracies)
    holdout_preds_ensemble /= skf.n_splits
    test_preds_ensemble /= skf.n_splits

    holdout_labels = (holdout_preds_ensemble > 0.5).astype(int)
    holdout_accuracy = accuracy_score(y_holdout, holdout_labels)
    print("\n" + "="*50)
    print("---               FINAL REPORT                ---")
    print("="*50)
    print(f"Mean Cross-Validation Accuracy (on 80% data): {mean_cv_acc:.4f}")
    print(f"TRUE Accuracy on Hold-out Set (on 20% data):   {holdout_accuracy:.4f}  <-- YOUR KEY INDICATOR")
    print("-"*50)
    overfitting_gap = mean_cv_acc - holdout_accuracy
    if overfitting_gap > 0.01: 
        print(f"WARNING: Overfitting Detected! (Gap: {overfitting_gap:.4f})")
        print("Model performs worse on unseen data. Try more regularization.")
    elif overfitting_gap < -0.005: 
        print("INFO: Hold-out performance is better than CV. Might be lucky.")
    else:
        print("GREAT: Model seems to generalize well! (Gap:")
        print(f" {overfitting_gap:.4f})")
    print("="*50 + "\n")
    final_labels = (test_preds_ensemble > 0.5).astype(int)
    submission = pd.DataFrame({"battle_id": df_test[ID_COL], "player_won": final_labels})
    submission.to_csv(output_csv, index=False)
    print(f"Submission file created: {output_csv}")
