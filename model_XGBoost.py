import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
import shap
from preprocessing import build_ml_dataframe
from data_utils import read_jsonl
import joblib
import os

def run_xgboost_pipeline_final(train_path: str, test_path: str, output_csv: str):
    """
    Esegue la pipeline completa, inclusa la trasformazione PCA.
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
    xgb_model_shap = xgb.XGBClassifier(n_estimators=1000, max_depth=5, learning_rate=0.05, eval_metric='logloss', random_state=42, use_label_encoder=False, verbosity=0, enable_categorical=True)
    xgb_model_shap.fit(X_train_full, y_train_full)
    explainer = shap.TreeExplainer(xgb_model_shap)
    shap_values = explainer.shap_values(X_train_full)
    shap_mean_abs = np.abs(shap_values).mean(axis=0)
    feat_importance = pd.DataFrame({'feature': X_train_full.columns, 'importance': shap_mean_abs}).sort_values(by='importance', ascending=False)
    
    top_n = 40
    top_features = feat_importance['feature'].head(top_n).tolist()
    print(f"\nUsing top {top_n} original features for PCA transformation.")
    X_train_top = X_train_full[top_features]
    X_test_top  = X_test_final[top_features]

    # --- 3. NUOVO: Scaling e Applicazione della PCA ---
    print("\n--- Applying StandardScaler and PCA ---")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_top)
    X_test_scaled = scaler.transform(X_test_top)

    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f"PCA applied. Original number of features: {X_train_scaled.shape[1]}")
    print(f"Number of Principal Components selected: {pca.n_components_}")

    # --- 4. Cross-Validation e Addestramento (sui dati PCA) ---
    OPTIMAL_PARAMS = {
        'n_estimators': 2946, 'max_depth': 10, 'learning_rate': 0.015010541507164693,
        'subsample': 0.5140216420139342, 'colsample_bytree': 0.3948685851902122,
    }

    print("\n--- Starting 10-fold cross-validation on PCA-transformed data ---")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_accuracies = []
    test_preds_ensemble = np.zeros(X_test_pca.shape[0])
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_pca, y_train_full)):
        print(f"\n--- Fold {fold+1}/10 ---")
        X_tr, X_val = X_train_pca[train_idx], X_train_pca[val_idx]
        y_tr, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        model_xgb = xgb.XGBClassifier(**OPTIMAL_PARAMS, early_stopping_rounds=200)
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        y_val_pred = model_xgb.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        cv_accuracies.append(val_acc)
        print(f"Fold {fold+1} → Validation Accuracy: {val_acc:.4f}")
        
        # ### CORREZIONE CHIAVE ###
        # Prediciamo sui dati di test TRASFORMATI DALLA PCA (X_test_pca)
        test_preds_ensemble += model_xgb.predict_proba(X_test_pca)[:, 1]

    # --- 5. Risultati Finali e Sottomissione ---
    mean_cv_acc = np.mean(cv_accuracies)
    print(f"\n--- FINAL RESULTS ---")
    print(f"Mean CV Accuracy on PCA data: {mean_cv_acc:.4f}")

    # Media delle probabilità per l'ensemble finale
    test_preds_ensemble /= skf.n_splits
    final_labels = (test_preds_ensemble > 0.5).astype(int)
    
    submission = pd.DataFrame({"battle_id": df_test[ID_COL], "player_won": final_labels})
    submission.to_csv(output_csv, index=False)
    print(f"Submission file created: {output_csv}")

if __name__ == '__main__':
    TRAIN_FILE_PATH = 'train.jsonl'
    TEST_FILE_PATH = 'test.jsonl'
    OUTPUT_CSV_PATH = 'submission_pca.csv'
    
    if os.path.exists(TRAIN_FILE_PATH) and os.path.exists(TEST_FILE_PATH):
        run_xgboost_pipeline_final(TRAIN_FILE_PATH, TEST_FILE_PATH, OUTPUT_CSV_PATH)
    else:
        print(f"Errore: Assicurati che i file necessari si trovino nella stessa cartella.")
