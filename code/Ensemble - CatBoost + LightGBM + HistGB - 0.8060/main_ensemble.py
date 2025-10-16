import pandas as pd
import numpy as np
import joblib
import os
import re
from preprocessing import build_ml_dataframe
from encoders import TargetEncoder 
from catboost import Pool 

def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in df.columns}
    df = df.rename(columns=new_cols)
    return df

def predict_ensemble(base_path: str):
    print("--- Avvio del processo di Predizione con Grande Ensemble ---")
    
    CATBOOST_MODELS_DIR = os.path.join(base_path, "Models - CatBoost")
    LGBM_MODELS_DIR = os.path.join(base_path, "Models - LightGBM")
    HGB_MODELS_DIR = os.path.join(base_path, "Models - Hist Gradient Boosting")

    LGBM_ENCODERS_DIR = os.path.join(base_path, "Encoders - LightGBM")
    HGB_ENCODERS_DIR = os.path.join(base_path, "Encoders - Hist Gradient Boosting")
    TEST_PATH = os.path.join(base_path, "test.jsonl")
    OUTPUT_CSV = os.path.join(base_path, "submission.csv")

    print(f"Caricamento dati di test da: {TEST_PATH}")
    if not os.path.exists(TEST_PATH):
        print(f"ERRORE: File 'test.jsonl' non trovato. Assicurati che sia nella cartella: {base_path}")
        return
        
    df_test = build_ml_dataframe(TEST_PATH, is_train=False)
    X_test_original = df_test.drop(columns=['battle_id'])

    print("\n--- Inizio previsioni con i modelli CatBoost ---")
    
    X_test_catboost = X_test_original.copy()
    CATEGORICAL_COLS = ['tl_p1_last_active','tl_p2_last_active','p1_lead_name','p2_lead_name','last_pair']
    
    for col in CATEGORICAL_COLS:
        if col in X_test_catboost.columns:
            X_test_catboost[col] = X_test_catboost[col].astype(str)
    
    cat_features_indices = [X_test_catboost.columns.get_loc(col) for col in CATEGORICAL_COLS if col in X_test_catboost.columns]
    catboost_pool = Pool(data=X_test_catboost, cat_features=cat_features_indices)
    
    catboost_preds_ensemble = np.zeros(X_test_catboost.shape[0])
    num_catboost_models = 0
    for fold in range(1, 11):
        model_filename = os.path.join(CATBOOST_MODELS_DIR, f"catboost_fold_{fold}.joblib")
        try:
            model = joblib.load(model_filename)
            print(f"Modello CatBoost Fold {fold} caricato. Eseguo previsione...")
            catboost_preds_ensemble += model.predict_proba(catboost_pool)[:, 1]
            num_catboost_models += 1
        except FileNotFoundError:
            print(f"ATTENZIONE: Modello CatBoost non trovato: {model_filename}. Salto.")
    
    catboost_final_preds = catboost_preds_ensemble / num_catboost_models if num_catboost_models > 0 else 0
    print("--- Previsioni CatBoost completate ---")

    print("\n--- Inizio previsioni con i modelli LightGBM ---")
    
    X_test_lgbm = sanitize_feature_names(X_test_original.copy())
    CATEGORICAL_COLS_sanitized = [re.sub(r'[^A-Za-z0-9_]+', '', col) for col in CATEGORICAL_COLS]
    lgbm_preds_ensemble = np.zeros(X_test_lgbm.shape[0])
    num_lgbm_models = 0
    
    for fold in range(1, 11):
        model_filename = os.path.join(LGBM_MODELS_DIR, f"lgbm_fold_{fold}.joblib")
        encoder_filename = os.path.join(LGBM_ENCODERS_DIR, f"encoder_fold_{fold}.joblib")
        try:
            model = joblib.load(model_filename)
            encoder = joblib.load(encoder_filename)
            print(f"Modello e Encoder LGBM Fold {fold} caricati. Eseguo previsione...")
            
            X_test_encoded = encoder.transform(X_test_lgbm)
            X_test_final_fold = X_test_encoded.drop(columns=CATEGORICAL_COLS_sanitized)
            
            lgbm_preds_ensemble += model.predict_proba(X_test_final_fold)[:, 1]
            num_lgbm_models += 1
        except FileNotFoundError:
            print(f"ATTENZIONE: Modello o Encoder LGBM non trovato per il Fold {fold}. Salto.")

    lgbm_final_preds = lgbm_preds_ensemble / num_lgbm_models if num_lgbm_models > 0 else 0
    print("--- Previsioni LightGBM completate ---")

    print("\n--- Inizio previsioni con i modelli Hist Gradient Boosting ---")
    X_test_hgb = sanitize_feature_names(X_test_original.copy())
    hgb_preds_ensemble = np.zeros(X_test_hgb.shape[0])
    num_hgb_models = 0

    for fold in range(1, 11):
        model_filename = os.path.join(HGB_MODELS_DIR, f"hgb_fold_{fold}.joblib")
        encoder_filename = os.path.join(HGB_ENCODERS_DIR, f"hgb_encoder_fold_{fold+1}.joblib") # Usa il nome corretto del file
        try:
            model = joblib.load(model_filename)
            encoder = joblib.load(encoder_filename)
            print(f"Modello e Encoder HGB Fold {fold} caricati. Eseguo previsione...")

            X_test_encoded = encoder.transform(X_test_hgb)
            X_test_final_fold = X_test_encoded.drop(columns=CATEGORICAL_COLS_sanitized)
            
            hgb_preds_ensemble += model.predict_proba(X_test_final_fold)[:, 1]
            num_hgb_models += 1
        except FileNotFoundError:
            print(f"ATTENZIONE: Modello o Encoder HGB non trovato per il Fold {fold}. Salto.")
    
    hgb_final_preds = hgb_preds_ensemble / num_hgb_models if num_hgb_models > 0 else 0
    print("--- Previsioni Hist Gradient Boosting completate ---")

    print("\n--- Combinazione delle previsioni dei modelli (Ensemble) ---")
    WEIGHT_CATBOOST = 0.7
    WEIGHT_LGBM = 0.15
    WEIGHT_HGB = 0.15
    
    final_preds = (WEIGHT_CATBOOST * catboost_final_preds) + \
                  (WEIGHT_LGBM * lgbm_final_preds) + \
                  (WEIGHT_HGB * hgb_final_preds)
                  
    print(f"Previsioni combinate con pesi: CatBoost={WEIGHT_CATBOOST*100}%, LightGBM={WEIGHT_LGBM*100}%, HGB={WEIGHT_HGB*100}%")

    final_labels = (final_preds > 0.5).astype(int)
    submission = pd.DataFrame({ "battle_id": df_test["battle_id"], "player_won": final_labels })
    submission.to_csv(OUTPUT_CSV, index=False)
    
    print("\n" + "="*50)
    print("---      SUBMISSION FINALE ENSEMBLE CREATA      ---")
    print("="*50)
    print(f"File salvato in: {OUTPUT_CSV}")
    print("Questa è la tua mossa migliore. In bocca al lupo per la leaderboard! 🚀")

if __name__ == "__main__":
    script_base_path = os.path.dirname(os.path.abspath(__file__))
    predict_ensemble(script_base_path)
