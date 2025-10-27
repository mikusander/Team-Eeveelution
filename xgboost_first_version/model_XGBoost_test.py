import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import shap
import optuna
from preprocessing import build_ml_dataframe 

# --- NUOVA FUNZIONE PER L'ANALISI DEGLI ERRORI ---
def perform_error_analysis(X_val, y_val, model, original_df, fold_num):
    """
    Esegue un'analisi dettagliata degli errori del modello su un set di validazione.
    """
    # print(f"\n--- Inizio Analisi Errori per Fold {fold_num} ---")

    # 1. Calcola la loss per ogni singola previsione
    y_val_probs = model.predict_proba(X_val)[:, 1]
    epsilon = 1e-15
    y_val_probs = np.clip(y_val_probs, epsilon, 1 - epsilon)
    
    # Assicurati che y_val sia un array numpy per i calcoli
    y_val_numpy = y_val.to_numpy()
    
    individual_loss = - (y_val_numpy * np.log(y_val_probs) + (1 - y_val_numpy) * np.log(1 - y_val_probs))

    # 2. Crea un DataFrame per l'analisi
    # Usiamo gli indici di X_val per recuperare le righe originali complete da original_df
    df_val_analysis = original_df.loc[X_val.index].copy()
    df_val_analysis['true_winner'] = y_val
    df_val_analysis['predicted_prob'] = y_val_probs
    df_val_analysis['loss'] = individual_loss

    # 3. Ispeziona i peggiori errori
    worst_errors = df_val_analysis.sort_values(by='loss', ascending=False)
    # print("\nLe 5 previsioni con la loss più alta (peggiori errori):")
    # Mostriamo solo le colonne più rilevanti per la leggibilità
    display_cols = ['true_winner', 'predicted_prob', 'loss', 'lead_speed_advantage', 'p1_super_effective_options', 'damage_diff']
    # Aggiungi altre colonne di interesse se necessario
    # print(worst_errors[display_cols].head(5))

    # 4. Raggruppa gli errori per trovare pattern
    # print("\nAnalisi Aggregata degli Errori:")
    
    # Esempio: Analisi per vantaggio di velocità
    if 'lead_speed_advantage' in df_val_analysis.columns:
        speed_bins = pd.cut(df_val_analysis['lead_speed_advantage'], 
                            bins=[-np.inf, -50, -10, 10, 50, np.inf],
                            labels=["Molto Lento", "Lento", "Pari", "Veloce", "Molto Veloce"])
        loss_by_speed = df_val_analysis.groupby(speed_bins, observed=False)['loss'].mean()
        # print("\nLoss media in base al vantaggio di velocità del leader:")
        # print(loss_by_speed)

    # Esempio: Analisi per opzioni super-efficaci
    if 'p1_super_effective_options' in df_val_analysis.columns:
        loss_by_se_options = df_val_analysis.groupby('p1_super_effective_options')['loss'].mean()
        # print("\nLoss media in base al numero di opzioni super-efficaci:")
        # print(loss_by_se_options)

    # print(f"--- Fine Analisi Errori per Fold {fold_num} ---\n")

def run_xgboost_pipeline_test(train_path: str, test_path: str, output_csv: str):
    TUNE_HYPERPARAMETERS = False # Imposta a True per rieseguire l'ottimizzazione
    df_train_full = build_ml_dataframe(train_path, save_path='train_features.csv')
    df_test = build_ml_dataframe(test_path, is_train=False, save_path='test_features.csv')

    TARGET = "player_won"
    ID_COL = "battle_id"
    
    # Conserviamo le colonne categoriche e ID per l'analisi successiva
    cols_to_drop_for_training = [TARGET, ID_COL]
    
    X_train_full = df_train_full.drop(columns=cols_to_drop_for_training)
    y_train_full = df_train_full[TARGET]
    X_test_final = df_test.drop(columns=[col for col in [ID_COL] if col in df_test.columns])

    X_train_main, X_holdout, y_train_main, y_holdout = train_test_split(
        X_train_full, y_train_full, 
        test_size=0.20, 
        random_state=42, 
        stratify=y_train_full 
    )
    # print(f"\nTraining data split into: {len(X_train_main)} rows for training/CV and {len(X_holdout)} for hold-out validation.")

    # print("\nCalculating SHAP values and selecting top features...")
    # NOTA: Per un'analisi più veloce, si potrebbe ridurre il numero di alberi (n_estimators) qui.
    model_xgb_shap = xgb.XGBClassifier(n_estimators=1000, max_depth=5, learning_rate=0.05, eval_metric='logloss', random_state=42, use_label_encoder=False, verbosity=0)
    model_xgb_shap.fit(X_train_main, y_train_main)
    explainer = shap.TreeExplainer(model_xgb_shap)
    shap_values = explainer.shap_values(X_train_main)
    shap_mean_abs = np.abs(shap_values).mean(axis=0)
    feat_importance = pd.DataFrame({'feature': X_train_main.columns, 'importance': shap_mean_abs}).sort_values(by='importance', ascending=False)

    top_n = 35
    top_features = feat_importance['feature'].head(top_n).tolist()
    # print(f"\nTop {top_n} SHAP-based features:\n", top_features)

    X_train_top = X_train_main[top_features]
    X_holdout_top = X_holdout[top_features]
    X_test_top  = X_test_final[top_features]

    if TUNE_HYPERPARAMETERS:

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 2, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 5.0),
                'eval_metric': 'logloss',
                'random_state': 42,
                'use_label_encoder': False,
                'verbosity': 0,
            }
            model = xgb.XGBClassifier(**params)
            model.fit(X_train_top, y_train_main)
            preds = model.predict(X_holdout_top)
            acc = accuracy_score(y_holdout, preds)
            return acc

        # print("\n--- STARTING HYPERPARAMETER OPTIMIZATION WITH OPTUNA ---")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30)
        # print("Best params:", study.best_params)
        # print("Best accuracy:", study.best_value)
        # Forza la stampa su stdout e flush immediato
        import sys
        sys.stdout.flush()
        # print("\n=== PARAMETRI OTTIMALI TROVATI ===")
        #for k, v in study.best_params.items():
            # print(f"{k}: {v}")
        # print(f"Best accuracy: {study.best_value}")
        sys.stdout.flush()
        return

    OPTIMAL_PARAMS = {
        'n_estimators': 217,
        'max_depth': 3,
        'learning_rate': 0.14575402365967152,
        'subsample': 0.9285942951393896,
        'colsample_bytree': 0.6450099647987623,
        'reg_alpha': 0.8880236166542238,
        'reg_lambda': 2.0963775264052527,
        'eval_metric': 'logloss',
        'random_state': 42,
        'use_label_encoder': False,
        'verbosity': 0,
    }

    # print("\n--- STARTING CROSS-VALIDATION ON TRAINING SET (80%) ---")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_accuracies = []
    holdout_preds_ensemble = np.zeros(X_holdout_top.shape[0])
    test_preds_ensemble = np.zeros(X_test_top.shape[0])

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_top, y_train_main)):
        # print(f"\n--- Fold {fold+1} ---")
        X_tr, X_val = X_train_top.iloc[train_idx], X_train_top.iloc[val_idx]
        y_tr, y_val = y_train_main.iloc[train_idx], y_train_main.iloc[val_idx]
        
        final_params = OPTIMAL_PARAMS.copy()
        final_params.update({'early_stopping_rounds': 300})  # aumentato early stopping
        
        model_xgb = xgb.XGBClassifier(**final_params)
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
        
        y_val_pred = model_xgb.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        # print(f"Fold {fold+1} → CV Accuracy: {val_acc:.4f}")
        cv_accuracies.append(val_acc)

        # --- INIZIO SEZIONE ANALISI ERRORI ---
        perform_error_analysis(X_val, y_val, model_xgb, df_train_full, fold + 1)
        # --- FINE SEZIONE ANALISI ERRORI ---

        holdout_preds_ensemble += model_xgb.predict_proba(X_holdout_top)[:, 1]
        test_preds_ensemble += model_xgb.predict_proba(X_test_top)[:, 1]

    mean_cv_acc = np.mean(cv_accuracies)
    holdout_preds_ensemble /= skf.n_splits
    test_preds_ensemble /= skf.n_splits

    holdout_labels = (holdout_preds_ensemble > 0.5).astype(int)
    holdout_accuracy = accuracy_score(y_holdout, holdout_labels)
    
    print("\n" + "="*50)
    print("---               FINAL REPORT                ---")
    print(f"Mean CV Accuracy: {mean_cv_acc:.4f}")
    print(f"Holdout Accuracy: {holdout_accuracy:.4f}")
    print("="*50 + "\n")
    
    final_labels = (test_preds_ensemble > 0.5).astype(int)
    submission = pd.DataFrame({"battle_id": df_test[ID_COL], "player_won": final_labels})
    submission.to_csv(output_csv, index=False)
    # print(f"Submission file created: {output_csv}")

    # Addestramento su tutto il training set
    model_final = xgb.XGBClassifier(
        n_estimators=217,
        max_depth=3,
        learning_rate=0.14575402365967152,
        subsample=0.9285942951393896,
        colsample_bytree=0.6450099647987623,
        reg_alpha=0.8880236166542238,
        reg_lambda=2.0963775264052527,
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
        use_label_encoder=False
    )
    model_final.fit(X_train_full, y_train_full)

    # Calcola l'accuracy totale sul training set
    y_train_full_pred = model_final.predict(X_train_full)
    total_acc = accuracy_score(y_train_full, y_train_full_pred)
    print(f"Total Training Accuracy: {total_acc:.4f}")

    # Salva l'accuracy su file
    with open("total_accuracy.txt", "w") as f:
        f.write(f"Total Training Accuracy: {total_acc:.4f}\n")