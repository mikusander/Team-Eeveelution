"""
Utility library for Ensemble methods (Stacking and Blending).

This module contains all the necessary functions, constants, and path configurations
for:
1. Stacking Ensemble (L1/L2 meta-models, Logistic Regression)
2. Blending Ensemble (Simple weighted average)

"""

# --- 1. CONSOLIDATED IMPORTS ---
import numpy as np
import pandas as pd
import os
import joblib
from pathlib import Path
import warnings

# Models
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

# Metrics
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings('ignore')

# --- 2. GLOBAL CONFIGURATION AND PATHS ---

SEED = 42

# Define the project base directory
BASE_DIR = Path(__file__).resolve().parent

# Input/Output Folders
OOF_PREDS_DIR = BASE_DIR / 'OOF_Predictions'
META_MODEL_DIR = BASE_DIR / 'Meta_Model'
SUBMISSION_DIR = BASE_DIR / 'Submissions'
DATA_PIPELINE_DIR_CB = BASE_DIR / 'CatBoost_Data_Pipeline'
DATA_PIPELINE_DIR_LGBM = BASE_DIR / 'LightGBM_Data_Pipeline'

# --- Paths for Stacking Ensemble (Meta-Modeling) ---
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

# --- Paths for Blending Ensemble (Averaging) ---
LGBM_PROBA_FILE = SUBMISSION_DIR / 'submission_lgbm_100pct_PROBA.csv'
CATBOOST_PROBA_FILE = SUBMISSION_DIR / 'submission_catboost_100pct_PROBA.csv'
XGBOOST_PROBA_FILE = SUBMISSION_DIR / 'submission_xgboost_100pct_PROBA.csv'
BLENDED_SUBMISSION_CLASS_OUT = SUBMISSION_DIR / 'submission_blended_LGBM_CAT_XGB_CLASS.csv'


# --- 3. DATA LOADING FUNCTIONS (For STACKING) ---

def ensure_directories():
    """Creates all necessary output directories."""
    print("Checking output directories...")
    for dir_path in [OOF_PREDS_DIR, META_MODEL_DIR, SUBMISSION_DIR, DATA_PIPELINE_DIR_CB, DATA_PIPELINE_DIR_LGBM]:
        os.makedirs(dir_path, exist_ok=True)
    print("Directories verified.")

def load_oof_data():
    """Loads the OOF data (X) and target (y) to train the meta-models."""
    print("Loading OOF data and Target for training...")
    try:
        oof_lgbm = np.load(OOF_LGBM_FILE)
        oof_catboost = np.load(OOF_CATBOOST_FILE)
        oof_xgboost = np.load(OOF_XGBOOST_FILE)

        y_meta = pd.read_csv(TARGET_FILE_IN).values.ravel()

        if not (len(oof_lgbm) == len(y_meta) and 
                len(oof_catboost) == len(y_meta) and 
                len(oof_xgboost) == len(y_meta)):
            raise ValueError(f"OOF length mismatch! LGBM({len(oof_lgbm)}), CB({len(oof_catboost)}), XGB({len(oof_xgboost)}), Y({len(y_meta)})")

        X_meta = np.column_stack((oof_lgbm, oof_catboost, oof_xgboost))
        print(f"Meta-Model Input (X_meta) shape: {X_meta.shape}") 
        print(f"Meta-Model Target (y_meta) shape: {y_meta.shape}")
        
        return X_meta, y_meta

    except FileNotFoundError as e:
        print(f"ERROR: File not found: {e}")
        print("Make sure you have run the base model pipelines (LGBM, CB, XGB).")
        return None, None
    except Exception as e:
        print(f"ERROR while loading OOF data: {e}")
        return None, None

def load_test_data():
    """Loads the test set predictions and IDs for the final submission."""
    print("Loading Test Set predictions and IDs...")
    try:
        test_preds_lgbm_all_folds = np.load(TEST_PREDS_LGBM_FILE)
        test_preds_catboost = np.load(TEST_PREDS_CATBOOST_FILE)
        test_preds_xgboost = np.load(TEST_PREDS_XGBOOST_FILE)
        if test_preds_lgbm_all_folds.ndim == 2:
            print(f"Found {test_preds_lgbm_all_folds.shape[0]} folds for LGBM. Calculating mean...")
            test_preds_lgbm = np.mean(test_preds_lgbm_all_folds, axis=0)
        else:
            test_preds_lgbm = test_preds_lgbm_all_folds
        
        test_ids_df = pd.read_csv(TEST_IDS_FILE_IN)
        test_ids = test_ids_df['battle_id']

        if not (len(test_preds_lgbm) == len(test_preds_catboost) and len(test_preds_lgbm) == len(test_preds_xgboost)):
            raise ValueError("Test Set prediction length mismatch!")
        
        if len(test_ids) != len(test_preds_lgbm):
            raise ValueError(f"Mismatch between IDs ({len(test_ids)}) and Predictions ({len(test_preds_lgbm)})")

        X_test_meta = np.column_stack((test_preds_lgbm, test_preds_catboost, test_preds_xgboost))
        print(f"Test Input for Meta-Model (X_test_meta) shape: {X_test_meta.shape}")
        
        return X_test_meta, test_ids

    except FileNotFoundError as e:
        print(f"ERROR: .npy or .csv file not found: {e}")
        print("Make sure you have generated test set predictions from ALL 3 base models.")
        return None, None
    except Exception as e:
        print(f"ERROR while loading Test data: {e}")
        return None, None


# --- 4. STACKING ENSEMBLE FUNCTIONS (Training & Submission) ---

def _evaluate_model(model, X_meta, y_meta):
    """Internal helper to train and evaluate a model."""
    model.fit(X_meta, y_meta)
    
    # Evaluation (Binary Predictions for Accuracy)
    meta_preds_binary = model.predict(X_meta)
    oof_accuracy = accuracy_score(y_meta, meta_preds_binary)

    # Evaluation (Scores for AUC)
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
    [STACKING - LOGREG]
    Trains, evaluates, and saves the LogReg meta-model.
    """
    print("\n" + "="*30)
    print("START PHASE 2a: LogReg Model Training (Stacking)")
    print("="*30)
    
    model_logreg = LogisticRegression(random_state=SEED, C=1.0, solver='liblinear')
    
    print("Training Logistic Regression...")
    model_logreg.fit(X_meta, y_meta)
    print("Training complete.")

    # Evaluation
    meta_preds_proba = model_logreg.predict_proba(X_meta)[:, 1]
    meta_preds_binary = model_logreg.predict(X_meta)
    oof_accuracy = accuracy_score(y_meta, meta_preds_binary)
    oof_auc = roc_auc_score(y_meta, meta_preds_proba)

    print("LEARNED WEIGHTS (Coefficients) for [LGBM, CatBoost, XGBoost]:")
    print(f"  Coefficients: {model_logreg.coef_}")
    print(f"  Intercept: {model_logreg.intercept_}")
    print(f"\nEvaluation (on OOF):")
    print(f"  Accuracy: {oof_accuracy:.6f}")
    print(f"  AUC:      {oof_auc:.6f}")

    # Save the LogReg model
    print(f"\nSaving LogReg Model to: {META_MODEL_FILE_LOGREG}")
    joblib.dump(model_logreg, META_MODEL_FILE_LOGREG)
    
    print("="*30)
    print("PHASE 2a Complete: 'LogReg' Model saved.")
    print("="*30)
    
    return model_logreg

def train_and_select_best_model(X_meta, y_meta):
    """
    [STACKING - BEST MODEL]
    Trains all meta-models (L1 and L2), compares them,
    and saves only the BEST model.
    """
    print("\n" + "="*30)
    print("START PHASE 2b: Selecting BEST Meta-Model (Stacking)")
    print("="*30)

    # --- Model Definitions ---
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

    # --- Training, Evaluation, and Selection ---
    for name, model in models_to_train.items():
        print(f"\n--- Comparison: {name} ---")
        
        # Use the helper to train and evaluate
        oof_accuracy, oof_auc = _evaluate_model(model, X_meta, y_meta)

        print(f"  OOF Accuracy: {oof_accuracy:.6f}")
        print(f"  OOF AUC:      {oof_auc:.6f}")
        
        results[name] = {'accuracy': oof_accuracy, 'auc': oof_auc, 'model': model}

        # Selection based on ACCURACY
        if oof_accuracy > best_accuracy:
            best_accuracy = oof_accuracy
            best_model_name = name
            best_model_obj = model

    # --- Selection and Saving ---
    print("\n--- Best Model Selection (based on OOF Accuracy) ---")
    print(f"Best model: {best_model_name}")
    print(f"  OOF Accuracy: {results[best_model_name]['accuracy']:.6f}")
    print(f"  OOF AUC: {results[best_model_name]['auc']:.6f}")

    # Save ONLY the best model
    print(f"Saving BEST Model to: {META_MODEL_FILE_BEST}")
    joblib.dump(best_model_obj, META_MODEL_FILE_BEST)
    
    print("="*30)
    print("PHASE 2b Complete: 'BEST' Model saved.")
    print("="*30)
    
    return best_model_obj

def generate_submission(meta_model, X_test_meta, test_ids, output_filename):
    """
    [STACKING - SUBMISSION]
    Generates a .csv submission file using a trained meta-model.
    """
    print(f"\nGenerating stacking submission for: {output_filename.name}...")
    
    try:
        final_predictions = meta_model.predict(X_test_meta)
        
        submission_df = pd.DataFrame({
            'battle_id': test_ids, 
            'player_won': final_predictions
        })
        submission_df.to_csv(output_filename, index=False)
        
        print(f"Submission saved to: {output_filename}")
        print(submission_df.head())
        return True
        
    except Exception as e:
        print(f"ERROR during submission generation {output_filename.name}: {e}")
        return False


# --- 5. BLENDING ENSEMBLE FUNCTION (Averaging) ---

def run_blending_ensemble():
    """
    [BLENDING - Averaging]
    Runs the simple 3-model probability blending.
    Loads CSVs, merges, calculates weighted average, and saves.
    """
    print("\n" + "="*30)
    print("START PHASE: Simple Blending Ensemble (LGBM+CAT+XGB)")
    print("="*30)
    
    # Weights for blending
    LGBM_WEIGHT = 1/3
    CATBOOST_WEIGHT = 1/3
    XGBOOST_WEIGHT = 1/3

    # --- 1. Loading ---
    try:
        print(f"Loading LGBM proba from: {LGBM_PROBA_FILE}")
        lgbm_df = pd.read_csv(LGBM_PROBA_FILE)
        
        print(f"Loading CatBoost proba from: {CATBOOST_PROBA_FILE}")
        catboost_df = pd.read_csv(CATBOOST_PROBA_FILE)
        
        print(f"Loading XGBoost proba from: {XGBOOST_PROBA_FILE}")
        xgboost_df = pd.read_csv(XGBOOST_PROBA_FILE)
        
    except FileNotFoundError as e:
        print(f"ERROR: File not found: {e}")
        print("Make sure you have generated ALL THREE probability .csv files.")
        print(f"(Check: '{LGBM_PROBA_FILE}', '{CATBOOST_PROBA_FILE}', '{XGBOOST_PROBA_FILE}')")
        return False
    except Exception as e:
        print(f"ERROR during loading: {e}")
        return False

    # --- 2. Merging and Blending ---
    print("Merging DataFrames by battle_id...")
    try:
        # Rename probability columns for clarity
        lgbm_df = lgbm_df.rename(columns={'player_won': 'proba_lgbm'})
        catboost_df = catboost_df.rename(columns={'player_won_proba': 'proba_catboost'})
        xgboost_df = xgboost_df.rename(columns={'player_won_proba': 'proba_xgboost'})

        # Merge using battle_id
        blended_df = pd.merge(lgbm_df[['battle_id', 'proba_lgbm']],
                              catboost_df[['battle_id', 'proba_catboost']],
                              on='battle_id')
        blended_df = pd.merge(blended_df,
                              xgboost_df[['battle_id', 'proba_xgboost']],
                              on='battle_id')
        
        if len(blended_df) != len(lgbm_df):
            print(f"WARNING: Length mismatch after merge! ({len(blended_df)} vs {len(lgbm_df)})")
            print("Check that all submission files have the same number of rows (e.g., 5000).")

    except KeyError as e:
        print(f"ERROR: Missing column: {e}")
        print("Ensure the CSVs contain 'battle_id' and the correct probability column.")
        print(f"LGBM ('{LGBM_PROBA_FILE}') expects 'player_won' (containing probabilities).")
        print(f"CatBoost/XGBoost expect 'player_won_proba'.")
        return False
    except Exception as e:
        print(f"ERROR during merge: {e}")
        return False

    print("Calculating weighted average (LGBM + CAT + XGB)...")
    # Calculate the final probability
    blended_df['final_proba'] = (LGBM_WEIGHT * blended_df['proba_lgbm']) + \
                                (CATBOOST_WEIGHT * blended_df['proba_catboost']) + \
                                (XGBOOST_WEIGHT * blended_df['proba_xgboost'])

    # Convert final probability to 0 or 1
    blended_df['player_won'] = (blended_df['final_proba'] > 0.5).astype(int)

    # --- 3. Saving Final Submission ---
    final_submission_class_df = blended_df[['battle_id', 'player_won']]
    try:
        final_submission_class_df.to_csv(BLENDED_SUBMISSION_CLASS_OUT, index=False)
        print("\n--- BLENDING (3 MODELS) COMPLETE ---")
        print(f"Final submission (CLASSES 0/1) saved to: {BLENDED_SUBMISSION_CLASS_OUT}")
        print(final_submission_class_df.head())
        print(f"Class Distribution: \n{final_submission_class_df['player_won'].value_counts(normalize=True)}")
        return True
    except Exception as e:
        print(f"ERROR during save (Classes): {e}")
        return False