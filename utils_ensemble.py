"""
Libreria di utilità per lo Stacking Ensemble.

Questo modulo contiene tutte le funzioni, le costanti e le configurazioni
di percorso necessarie per addestrare i meta-modelli (L1 e L2) e
generare le submission finali.

MODIFICATO: La funzione 'train_and_evaluate_models' è stata divisa in:
- train_evaluate_logreg: Gestisce e salva solo il modello LogReg.
- train_and_select_best_model: Confronta tutti i modelli e salva solo il 'BEST'.
"""

# --- 1. IMPORT CONSOLIDATI ---
import numpy as np
import pandas as pd
import os
import joblib
from pathlib import Path
import warnings

# Modelli
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

# Metriche
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings('ignore')

# --- 2. CONFIGURAZIONE GLOBALE E PERCORSI ---

SEED = 42

# Definisci la directory base del progetto
BASE_DIR = Path(__file__).resolve().parent

# Cartelle di Input/Output
OOF_PREDS_DIR = BASE_DIR / 'OOF_Predictions'
META_MODEL_DIR = BASE_DIR / 'Meta_Model'
SUBMISSION_DIR = BASE_DIR / 'Submissions'
DATA_PIPELINE_DIR_CB = BASE_DIR / 'CatBoost_Data_Pipeline'
DATA_PIPELINE_DIR_LGBM = BASE_DIR / 'LightGBM_Data_Pipeline'
OOF_LGBM_FILE = OOF_PREDS_DIR / 'oof_lgbm_proba.npy'
OOF_CATBOOST_FILE = OOF_PREDS_DIR / 'oof_catboost_proba.npy'
OOF_XGBOOST_FILE = OOF_PREDS_DIR / 'oof_xgboost_proba.npy'
TARGET_FILE_IN = DATA_PIPELINE_DIR_CB / 'target_train.csv'
TEST_PREDS_LGBM_FILE = OOF_PREDS_DIR / 'test_preds_lgbm_proba.npy'
TEST_PREDS_CATBOOST_FILE = OOF_PREDS_DIR / 'test_preds_catboost_proba.npy'
TEST_PREDS_XGBOOST_FILE = OOF_PREDS_DIR / 'test_preds_xgboost_proba.npy'
TEST_IDS_FILE_IN = DATA_PIPELINE_DIR_LGBM / 'test_ids.csv'
META_MODEL_FILE_LOGREG = META_MODEL_DIR / 'stacking_meta_model_logreg.joblib'
META_MODEL_FILE_BEST = META_MODEL_DIR / 'stacking_meta_model_BEST.joblib'
SUBMISSION_FILE_LOGREG = SUBMISSION_DIR / 'submission_stacking_logreg_3models.csv'
SUBMISSION_FILE_BEST = SUBMISSION_DIR / 'submission_stacking_BEST_L1_model.csv'

# --- 3. FUNZIONI DI CARICAMENTO DATI ---

def ensure_directories():
    """Crea tutte le directory di output necessarie."""
    print("Verifica delle directory di output...")
    for dir_path in [OOF_PREDS_DIR, META_MODEL_DIR, SUBMISSION_DIR, DATA_PIPELINE_DIR_CB, DATA_PIPELINE_DIR_LGBM]:
        os.makedirs(dir_path, exist_ok=True)
    print("Directory verificate.")

def load_oof_data():
    """Carica i dati OOF (X) e il target (y) per addestrare i meta-modelli."""
    print("Caricamento dati OOF e Target per l'addestramento...")
    try:
        oof_lgbm = np.load(OOF_LGBM_FILE)
        oof_catboost = np.load(OOF_CATBOOST_FILE)
        oof_xgboost = np.load(OOF_XGBOOST_FILE)

        y_meta = pd.read_csv(TARGET_FILE_IN).values.ravel()

        if not (len(oof_lgbm) == len(y_meta) and 
                len(oof_catboost) == len(y_meta) and 
                len(oof_xgboost) == len(y_meta)):
            raise ValueError(f"Mismatch lunghezze OOF! LGBM({len(oof_lgbm)}), CB({len(oof_catboost)}), XGB({len(oof_xgboost)}), Y({len(y_meta)})")

        X_meta = np.column_stack((oof_lgbm, oof_catboost, oof_xgboost))
        print(f"Input Meta-Modello (X_meta) shape: {X_meta.shape}") 
        print(f"Target Meta-Modello (y_meta) shape: {y_meta.shape}")
        
        return X_meta, y_meta

    except FileNotFoundError as e:
        print(f"ERRORE: File non trovato: {e}")
        print("Assicurati di aver eseguito le pipeline dei modelli base (LGBM, CB, XGB).")
        return None, None
    except Exception as e:
        print(f"ERRORE durante il caricamento dei dati OOF: {e}")
        return None, None

def load_test_data():
    """Carica le previsioni sul test set e gli ID per la submission finale."""
    print("Caricamento previsioni Test Set e ID...")
    try:
        test_preds_lgbm_all_folds = np.load(TEST_PREDS_LGBM_FILE)
        test_preds_catboost = np.load(TEST_PREDS_CATBOOST_FILE)
        test_preds_xgboost = np.load(TEST_PREDS_XGBOOST_FILE)
        if test_preds_lgbm_all_folds.ndim == 2:
            print(f"Trovate {test_preds_lgbm_all_folds.shape[0]} fold per LGBM. Calcolo la media...")
            test_preds_lgbm = np.mean(test_preds_lgbm_all_folds, axis=0)
        else:
            test_preds_lgbm = test_preds_lgbm_all_folds
        
        test_ids_df = pd.read_csv(TEST_IDS_FILE_IN)
        test_ids = test_ids_df['battle_id']

        if not (len(test_preds_lgbm) == len(test_preds_catboost) and len(test_preds_lgbm) == len(test_preds_xgboost)):
            raise ValueError("Mismatch lunghezze previsioni Test Set!")
        
        if len(test_ids) != len(test_preds_lgbm):
            raise ValueError(f"Mismatch tra ID ({len(test_ids)}) e Predizioni ({len(test_preds_lgbm)})")

        X_test_meta = np.column_stack((test_preds_lgbm, test_preds_catboost, test_preds_xgboost))
        print(f"Input Test per Meta-Modello (X_test_meta) shape: {X_test_meta.shape}")
        
        return X_test_meta, test_ids

    except FileNotFoundError as e:
        print(f"ERRORE: File .npy o .csv non trovato: {e}")
        print("Assicurati di aver generato le previsioni sul test set da TUTTI E 3 i modelli base.")
        return None, None
    except Exception as e:
        print(f"ERRORE durante il caricamento dei dati Test: {e}")
        return None, None


# --- 4. FUNZIONI DI TRAINING E SUBMISSION ---

def _evaluate_model(model, X_meta, y_meta):
    """Helper interno per addestrare e valutare un modello."""
    model.fit(X_meta, y_meta)
    
    # Valutazione (Predizioni Binarie per Accuracy)
    meta_preds_binary = model.predict(X_meta)
    oof_accuracy = accuracy_score(y_meta, meta_preds_binary)

    # Valutazione (Score per AUC)
    if hasattr(model, "predict_proba"):
        meta_preds_scores = model.predict_proba(X_meta)[:, 1]
        oof_auc = roc_auc_score(y_meta, meta_preds_scores)
    elif hasattr(model, "decision_function"):
        meta_preds_scores = model.decision_function(X_meta)
        oof_auc = roc_auc_score(y_meta, meta_preds_scores)
    else: # VotingHard
        oof_auc = roc_auc_score(y_meta, meta_preds_binary) 
            
    return oof_accuracy, oof_auc

def train_evaluate_logreg(X_meta, y_meta):
    """
    [FUNZIONE SPECIFICA LOGREG]
    Addestra, valuta e salva il meta-modello LogReg.
    """
    print("\n" + "="*30)
    print("INIZIO FASE 2a: Addestramento Modello LogReg")
    print("="*30)
    
    model_logreg = LogisticRegression(random_state=SEED, C=1.0, solver='liblinear')
    
    print("Addestramento Regressione Logistica...")
    model_logreg.fit(X_meta, y_meta)
    print("Addestramento completato.")

    # Valutazione
    meta_preds_proba = model_logreg.predict_proba(X_meta)[:, 1]
    meta_preds_binary = model_logreg.predict(X_meta)
    oof_accuracy = accuracy_score(y_meta, meta_preds_binary)
    oof_auc = roc_auc_score(y_meta, meta_preds_proba)

    print("PESI IMPARATI (Coefficienti) per [LGBM, CatBoost, XGBoost]:")
    print(f"  Coefficienti: {model_logreg.coef_}")
    print(f"  Intercetta: {model_logreg.intercept_}")
    print(f"\nValutazione (sulle OOF):")
    print(f"  Accuracy: {oof_accuracy:.6f}")
    print(f"  AUC:      {oof_auc:.6f}")

    # Salva il modello LogReg
    print(f"\nSalvataggio Modello LogReg in: {META_MODEL_FILE_LOGREG}")
    joblib.dump(model_logreg, META_MODEL_FILE_LOGREG)
    
    print("="*30)
    print("FASE 2a Completata: Modello 'LogReg' salvato.")
    print("="*30)
    
    return model_logreg

def train_and_select_best_model(X_meta, y_meta):
    """
    [FUNZIONE MODELLI MULTIPLI]
    Addestra tutti i meta-modelli (L1 e L2), li confronta,
    e salva solo il modello MIGLIORE.
    """
    print("\n" + "="*30)
    print("INIZIO FASE 2b: Selezione Meta-Modello MIGLIORE")
    print("="*30)

    # --- Definizione Modelli ---
    model_logreg = LogisticRegression(random_state=SEED, C=1.0, solver='liblinear')
    model_ridge = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=5)
    model_lgbm = LGBMClassifier(n_estimators=100, max_depth=2, random_state=SEED, force_col_wise=True, verbose=-1)
    
    estimators_soft = [
        ('logreg', LogisticRegression(random_state=SEED, C=1.0, solver='liblinear')),
        ('lgbm', LGBMClassifier(n_estimators=100, max_depth=2, random_state=SEED, force_col_wise=True, verbose=-1))
    ]
    estimators_hard = [
        ('logreg', LogisticRegression(random_state=SEED, C=1.0, solver='liblinear')),
        ('ridge', RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=5)),
        ('lgbm', LGBMClassifier(n_estimators=100, max_depth=2, random_state=SEED, force_col_wise=True, verbose=-1))
    ]

    models_to_train = {
        'LogReg': model_logreg,
        'RidgeCV': model_ridge,
        'LGBM (L1)': model_lgbm,
        'VotingHard (L2)': VotingClassifier(estimators=estimators_hard, voting='hard'),
        'VotingSoft (L2)': VotingClassifier(estimators=estimators_soft, voting='soft')
    }

    results = {}
    best_accuracy = -1.0
    best_model_name = ""
    best_model_obj = None

    # --- Addestramento, Valutazione e Selezione ---
    for name, model in models_to_train.items():
        print(f"\n--- Confronto: {name} ---")
        
        # Usa l'helper per addestrare e valutare
        oof_accuracy, oof_auc = _evaluate_model(model, X_meta, y_meta)

        print(f"  OOF Accuracy: {oof_accuracy:.6f}")
        print(f"  OOF AUC:      {oof_auc:.6f}")
        
        results[name] = {'accuracy': oof_accuracy, 'auc': oof_auc, 'model': model}

        # Selezione basata su ACCURACY
        if oof_accuracy > best_accuracy:
            best_accuracy = oof_accuracy
            best_model_name = name
            best_model_obj = model

    # --- Selezione e Salvataggio ---
    print("\n--- Selezione Modello Migliore (basata su OOF Accuracy) ---")
    print(f"Modello migliore: {best_model_name}")
    print(f"  OOF Accuracy: {results[best_model_name]['accuracy']:.6f}")
    print(f"  OOF AUC: {results[best_model_name]['auc']:.6f}")

    # Salva SOLO il modello migliore
    print(f"Salvataggio Modello MIGLIORE in: {META_MODEL_FILE_BEST}")
    joblib.dump(best_model_obj, META_MODEL_FILE_BEST)
    
    print("="*30)
    print("FASE 2b Completata: Modello 'BEST' salvato.")
    print("="*30)
    
    return best_model_obj

def generate_submission(meta_model, X_test_meta, test_ids, output_filename):
    """
    Genera un file di submission .csv usando un meta-modello addestrato.
    """
    print(f"\nGenerazione submission per: {output_filename.name}...")
    
    try:
        final_predictions = meta_model.predict(X_test_meta)
        
        submission_df = pd.DataFrame({
            'battle_id': test_ids, 
            'player_won': final_predictions
        })
        submission_df.to_csv(output_filename, index=False)
        
        print(f"Submission salvata in: {output_filename}")
        print(submission_df.head())
        return True
        
    except Exception as e:
        print(f"ERRORE durante la generazione della submission {output_filename.name}: {e}")
        return False