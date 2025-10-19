import pandas as pd
import numpy as np
import os
import json
import optuna
import time

from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Configurazione ---
PREPROCESSED_DIR = 'preprocessed_data'
MODEL_PARAMS_DIR = 'model_params'
os.makedirs(MODEL_PARAMS_DIR, exist_ok=True)
ANALYSIS_DIR = 'analysis_output'
os.makedirs(ANALYSIS_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(PREPROCESSED_DIR, 'train_processed.csv')
TARGET_FILE = os.path.join(PREPROCESSED_DIR, 'target_train.csv')
PARAMS_FILE = os.path.join(MODEL_PARAMS_DIR, 'best_catboost_params_final.json')
PARAMS_OUTPUT_FILE = os.path.join(MODEL_PARAMS_DIR, 'best_catboost_params_final.json')
CM_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'final_honest_confusion_matrix.png')
IMPORTANCE_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'final_feature_importance.png') # <-- NUOVO
N_TRIALS_OPTIMIZATION = 100 # Numero di tentativi Optuna

print("Avvio Script di Ottimizzazione e Validazione 'pulito'...")

OPTUNA = False

# --- 2. Caricamento Dati Completi ---
print("Caricamento dati...")
try:
    X = pd.read_csv(TRAIN_FILE)
    y = pd.read_csv(TARGET_FILE).values.ravel()
    if(OPTUNA == False):
        with open(PARAMS_FILE, 'r') as f:
            best_params_clean = json.load(f)
except FileNotFoundError:
    print("ERRORE: File non trovati. Assicurati di aver eseguito lo script 12.")
    exit()

# --- 3. Split Esterno 80/20 (Train / Validation Hold-out) ---
print("Divisione 80/20 (Train/Validation)...")
# Il 20% (X_val, y_val) sarà il nostro set di validazione 'pulito'
X_train_outer, X_val_outer, y_train_outer, y_val_outer = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=42, 
    stratify=y
)
print(f"  Training set per Optuna: {X_train_outer.shape[0]} campioni")
print(f"  Validation set 'pulito': {X_val_outer.shape[0]} campioni")


# --- 4. Fase 1: Ottimizzazione (Solo sull'80%) ---
if(OPTUNA == True):
    print(f"\nAvvio studio Optuna per {N_TRIALS_OPTIMIZATION} tentativi (solo sull'80% dei dati)...")
    start_time = time.time()

    # Definiamo la funzione objective che Optuna ottimizzerà
    def objective(trial, X_data, y_data):
        # La CV interna per Optuna (5-fold)
        kf_inner = KFold(n_splits=5, shuffle=True, random_state=42)
        
        params = {
            'objective': 'Logloss', 'eval_metric': 'AUC', 'verbose': 0,
            'random_seed': 42, 'n_estimators': 1000, 'early_stopping_rounds': 50,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
        }
        model = CatBoostClassifier(**params)
        
        # Eseguiamo la CV interna *solo* sui dati di training (80%)
        auc_scores = cross_val_score(model, X_data, y_data, cv=kf_inner, scoring='roc_auc', n_jobs=-1)
        return auc_scores.mean()

    # Creiamo lo studio e ottimizziamo
    study = optuna.create_study(direction='maximize')
    # Usiamo 'lambda' per passare i dati di training (80%) alla funzione objective
    study.optimize(lambda trial: objective(trial, X_train_outer, y_train_outer), 
                n_trials=N_TRIALS_OPTIMIZATION)

    end_time = time.time()
    print(f"Ottimizzazione completata in {end_time - start_time:.2f} secondi.")

    # --- 5. Salvataggio Parametri "Puliti" ---
    best_params_clean = study.best_params
    best_score_clean = study.best_value

    print("\n--- Risultati Ottimizzazione (sull'80%) ---")
    print(f"Miglior Punteggio AUC (medio, CV interna): {best_score_clean:.6f}")
    print("Parametri 'Puliti' Trovati:")
    print(json.dumps(best_params_clean, indent=2))

    final_params_to_save = {
        **best_params_clean, 
        'objective': 'Logloss', 'eval_metric': 'AUC',
        'verbose': 0, 'random_seed': 42
    }
    with open(PARAMS_OUTPUT_FILE, 'w') as f:
        json.dump(final_params_to_save, f, indent=2)
    print(f"Parametri 'puliti' salvati in: {PARAMS_OUTPUT_FILE}")


# --- 6. Fase 2: Diagnostica Finale (sul 20% 'pulito') ---
print("\n--- Fase 2: Validazione Finale sul 20% 'pulito' ---")
print("Addestramento modello con parametri 'puliti' sull'80%...")

if(OPTUNA == True):
    final_model = CatBoostClassifier(
        **final_params_to_save, # Parametri da Optuna
        n_estimators=2000,      # Numero alto
        early_stopping_rounds=50 # Ma con early stopping
    )
else:
    final_model = CatBoostClassifier(
    **best_params_clean, # Parametri da Optuna
    n_estimators=2000,
    early_stopping_rounds=50
)

# Addestriamo sull'80% (train) e validiamo sul 20% (validation)
final_model.fit(
    X_train_outer, y_train_outer,
    eval_set=(X_val_outer, y_val_outer),
    verbose=1000
)

# Predizioni sul 20% 'pulito'
y_pred_final = final_model.predict(X_val_outer)
y_pred_proba_final = final_model.predict_proba(X_val_outer)[:, 1]

# 6.1. Controllo Overfitting
train_accuracy = final_model.score(X_train_outer, y_train_outer)
val_accuracy = accuracy_score(y_val_outer, y_pred_final)
val_auc = roc_auc_score(y_val_outer, y_pred_proba_final)

print(f"\nAccuracy (sul Training 80%): {train_accuracy:.4f}")
print(f"Accuracy (sul Validation 20% 'pulito'): {val_accuracy:.4f}  <-- Questo è il nostro punteggio 'onesto'")
print(f"AUC (sul Validation 20% 'pulito'): {val_auc:.4f}      <-- Questo è il nostro punteggio 'onesto'")

if train_accuracy > (val_accuracy + 0.05):
    print("\033[93mATTENZIONE: Possibile Overfitting!\033[0m")
else:
    print("\033[92mOK: Nessun segno evidente di overfitting.\033[0m")

# 6.2. Confusion Matrix
print("\n--- Confusion Matrix 'Onesta' (sul 20%) ---")
cm = confusion_matrix(y_val_outer, y_pred_final)
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predetto Falso (0)', 'Predetto Vero (1)'], 
            yticklabels=['Reale Falso (0)', 'Reale Vero (1)'])
plt.title("Confusion Matrix 'Onesta' (su 20% di dati mai visti)")
plt.savefig(CM_OUTPUT_FILE)
print(f"Grafico della Confusion Matrix salvato in: {CM_OUTPUT_FILE}")

# 6.3. Classification Report
print("\n--- Classification Report 'Onesto' (sul 20%) ---")
print(classification_report(y_val_outer, y_pred_final, target_names=['Falso (0)', 'Vero (1)']))

print("\n--- Fase 3: Feature Importance ---")

# Estraiamo l'importanza dal modello
importances = final_model.get_feature_importance()
feature_names = X_train_outer.columns

# Creiamo un DataFrame per gestirli
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Prendiamo le 20 più importanti
top_20_features = importance_df.head(20)

print("Top 20 Features più importanti:")
print(top_20_features.to_string(index=False))

# Creiamo il grafico
plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=top_20_features, palette='viridis')
plt.title('Top 20 Feature Importance (CatBoost)')
plt.tight_layout()
plt.savefig(IMPORTANCE_OUTPUT_FILE)
plt.close()
print(f"Grafico Feature Importance salvato in: {IMPORTANCE_OUTPUT_FILE}")

print("\nDiagnostica 'onesta' completata.")