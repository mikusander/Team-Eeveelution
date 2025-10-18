import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import shap
from preprocessing import build_ml_dataframe
from data_utils import read_jsonl
import joblib
import os

def run_xgboost_pipeline_final(train_path: str, test_path: str, output_csv: str):
    """
    Esegue la pipeline completa senza PCA, utilizzando solo le top feature selezionate con SHAP.
    """
    
    # --- 1. Preprocessing (Invariato) ---
    df_train_full = build_ml_dataframe(train_path, save_path='train_features.csv')
    df_test = build_ml_dataframe(test_path, is_train=False, save_path='test_features.csv')

    TARGET = "player_won"
    ID_COL = "battle_id"
    cols_to_drop = [TARGET, ID_COL] + [col for col in df_train_full.columns if df_train_full[col].dtype == 'category']
    X_train_full = df_train_full.drop(columns=cols_to_drop)
    y_train_full = df_train_full[TARGET]
    X_test_final = df_test.drop(columns=[col for col in cols_to_drop if col in df_test.columns])

    # --- 2. Selezione Features con SHAP (Invariato) ---
    print("\nCalculating SHAP values on original features...")
    xgb_model_shap = XGBClassifier(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.05,
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
        enable_categorical=True
    )
    xgb_model_shap.fit(X_train_full, y_train_full)
    explainer = shap.TreeExplainer(xgb_model_shap)
    shap_values = explainer.shap_values(X_train_full)
    shap_mean_abs = np.abs(shap_values).mean(axis=0)
    feat_importance = pd.DataFrame({'feature': X_train_full.columns, 'importance': shap_mean_abs}).sort_values(by='importance', ascending=False)
    
    # Stampa le migliori feature
    print("\nTop 20 features by SHAP importance:")
    print(feat_importance.head(20))

    # Salva le migliori feature su file
    feat_importance.to_csv("feature_importance.csv", index=False)
    print("Feature importance saved to feature_importance.csv")

    top_n = 40
    top_features = feat_importance['feature'].head(top_n).tolist()
    print(f"\nUsing top {top_n} original features for training.")
    X_train_top = X_train_full[top_features]
    X_test_top  = X_test_final[top_features]

    # --- 3. Cross-Validation e Addestramento (senza PCA) ---
    OPTIMAL_PARAMS = {
        'n_estimators': 2946, 'max_depth': 10, 'learning_rate': 0.015010541507164693,
        'subsample': 0.5140216420139342, 'colsample_bytree': 0.3948685851902122,
    }

    print("\n--- Starting 10-fold cross-validation on selected features ---")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_accuracies = []
    test_preds_ensemble = np.zeros(X_test_top.shape[0])

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_top, y_train_full)):
        print(f"\n--- Fold {fold+1}/10 ---")
        X_tr, X_val = X_train_top.iloc[train_idx], X_train_top.iloc[val_idx]
        y_tr, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        model_xgb = XGBClassifier(**OPTIMAL_PARAMS, early_stopping_rounds=200)
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        y_val_pred = model_xgb.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        cv_accuracies.append(val_acc)
        print(f"Fold {fold+1} → Validation Accuracy: {val_acc:.4f}")

        test_preds_ensemble += model_xgb.predict_proba(X_test_top)[:, 1]

        # Analisi errori per fold 1 e 6
        if fold+1 in [1, 6]:
            battle_ids = df_train_full.iloc[val_idx][ID_COL]
            errors = pd.DataFrame({
                "battle_id": battle_ids,
                "true_label": y_val,
                "pred_label": y_val_pred
            })
            errors_wrong = errors[errors["true_label"] != errors["pred_label"]]
            errors_wrong.to_csv(f"errors_fold_{fold+1}.csv", index=False)
            print(f"Saved errors for fold {fold+1} to errors_fold_{fold+1}.csv")

    # --- 4. Risultati Finali e Sottomissione ---
    mean_cv_acc = np.mean(cv_accuracies)
    print(f"\n--- FINAL RESULTS ---")
    print(f"Mean CV Accuracy: {mean_cv_acc:.4f}")

    # Salva l'accuracy su file
    with open("accuracy.txt", "w") as f:
        f.write(f"Mean CV Accuracy: {mean_cv_acc:.4f}\n")

    test_preds_ensemble /= skf.n_splits
    final_labels = (test_preds_ensemble > 0.5).astype(int)

    submission = pd.DataFrame({"battle_id": df_test[ID_COL], "player_won": final_labels})
    submission.to_csv(output_csv, index=False)
    print(f"Submission file created: {output_csv}")

    # --- Hyperparameter Tuning con GridSearchCV ---
    print("\n--- Hyperparameter Tuning with GridSearchCV ---")
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0]
    }
    xgb = XGBClassifier(eval_metric='logloss')
    grid = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_top, y_train_full)
    print("Best parameters:", grid.best_params_)
    print("Best cross-val accuracy:", grid.best_score_)

    # Salva anche la best cross-val accuracy della GridSearchCV
    with open("accuracy.txt", "a") as f:
        f.write(f"Best GridSearchCV Accuracy: {grid.best_score_:.4f}\n")

    best_model = grid.best_estimator_
    test_preds_best_model = best_model.predict_proba(X_test_top)[:, 1]
    final_labels_best_model = (test_preds_best_model > 0.5).astype(int)

    submission_best_model = pd.DataFrame({"battle_id": df_test[ID_COL], "player_won": final_labels_best_model})
    submission_best_model.to_csv(output_csv.replace('.csv', '_best_model.csv'), index=False)
    print(f"Submission file created for best model: {output_csv.replace('.csv', '_best_model.csv')}")

if __name__ == '__main__':
    TRAIN_FILE_PATH = 'train.jsonl'
    TEST_FILE_PATH = 'test.jsonl'
    OUTPUT_CSV_PATH = 'submission_pca.csv'
    
    if os.path.exists(TRAIN_FILE_PATH) and os.path.exists(TEST_FILE_PATH):
        run_xgboost_pipeline_final(TRAIN_FILE_PATH, TEST_FILE_PATH, OUTPUT_CSV_PATH)
    else:
        print(f"Errore: Assicurati che i file necessari si trovino nella stessa cartella.")
