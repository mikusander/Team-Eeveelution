import pandas as pd
import numpy as np
import os
import json
import time

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Configurazione ---
# (Invariata)
PREPROCESSED_DIR = 'preprocessed_data'
MODEL_PARAMS_DIR = 'model_params'
ANALYSIS_DIR = 'analysis_output'
os.makedirs(ANALYSIS_DIR, exist_ok=True) 

TRAIN_FILE = os.path.join(PREPROCESSED_DIR, 'train_processed.csv')
TARGET_FILE = os.path.join(PREPROCESSED_DIR, 'target_train.csv')
PARAMS_FILE = os.path.join(MODEL_PARAMS_DIR, 'best_catboost_params_final.json')

METRICS_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'final_metrics_report.csv') 
CM_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'final_honest_confusion_matrix.png')
IMPORTANCE_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'final_feature_importance.png')
LOSS_CURVE_FILE = os.path.join(ANALYSIS_DIR, 'final_loss_curve.png') 
AUC_CURVE_FILE = os.path.join(ANALYSIS_DIR, 'final_auc_learning_curve.png') 
ROC_CURVE_FILE = os.path.join(ANALYSIS_DIR, 'final_roc_auc_curve.png') 

print("Avvio Script di Validazione Finale (Diagnostica COMPLETA)...")

# --- 2. Caricamento Dati ---
# (Invariata)
print("Caricamento dati...")
try:
    X = pd.read_csv(TRAIN_FILE)
    y = pd.read_csv(TARGET_FILE).values.ravel()
    with open(PARAMS_FILE, 'r') as f:
        best_params_clean = json.load(f)
except FileNotFoundError:
    print("ERRORE: File non trovati.")
    exit()

# --- 3. Split ---
# (Invariata)
print("Divisione 80/20 (Train/Validation)...")
X_train_outer, X_val_outer, y_train_outer, y_val_outer = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Training set: {X_train_outer.shape[0]} campioni")
print(f"  Validation set 'pulito': {X_val_outer.shape[0]} campioni")


# --- 4. Addestramento Modello Ottimizzato ---
print("\nAddestramento modello con parametri ottimizzati sull'80%...")

# --- MODIFICA QUI ---
# Diciamo a CatBoost di calcolare *entrambe* le metriche
# e usiamo AUC come metrica per l'early stopping (la prima della lista)
final_params_fit = {
    **best_params_clean,
    'n_estimators': 2000,
    'early_stopping_rounds': 50,
    'eval_metric': 'Logloss',           # <-- usa Logloss come metrica principale
    'custom_metric': ['AUC'], # registra entrambe per i grafici
}
# Pulizia per sicurezza
if 'eval_metric' in best_params_clean: # Rimuovi se era nei parametri di Optuna
     final_params_fit.pop('eval_metric', None) 
# --------------------

model = CatBoostClassifier(**final_params_fit)

model.fit(
    X_train_outer, y_train_outer,
    eval_set=[(X_train_outer, y_train_outer), (X_val_outer, y_val_outer)],
    plot=False,
    use_best_model=True, # Carica il modello dal bestIteration basato su AUC (la prima eval_metric)
    verbose=1000
)

# --- 5. Diagnostica ---
# (Invariata)
print("\n--- Fase 2: Validazione Finale sul 20% 'pulito' ---")
y_pred_final = model.predict(X_val_outer)
y_pred_proba_final = model.predict_proba(X_val_outer)[:, 1]
train_accuracy = accuracy_score(y_train_outer, model.predict(X_train_outer))
val_accuracy = accuracy_score(y_val_outer, y_pred_final)
val_auc = roc_auc_score(y_val_outer, y_pred_proba_final)

print(f"\nAccuracy (sul Training 80%): {train_accuracy:.4f}")
print(f"Accuracy (sul Validation 20% 'pulito'): {val_accuracy:.4f}")
print(f"AUC (sul Validation 20% 'pulito'): {val_auc:.4f}")

report_dict = classification_report(y_val_outer, y_pred_final, target_names=['Falso (0)', 'Vero (1)'], output_dict=True)
print("\n--- Classification Report 'Onesto' (sul 20%) ---")
print(classification_report(y_val_outer, y_pred_final, target_names=['Falso (0)', 'Vero (1)']))

print(f"\nSalvataggio metriche (tabella pulita) in: {METRICS_OUTPUT_FILE}")

# --- Tabella delle metriche più carina ---
metrics_data = [
    ['🏋️ Prestazioni Globali', '', ''],
    ['Accuracy (Training)', '', f"{train_accuracy:.3f}"],
    ['Accuracy (Validation)', '', f"{val_accuracy:.3f}"],
    ['AUC (Validation)', '', f"{val_auc:.3f}"],
    ['', '', ''],
    ['🎯 Metriche per Classe', '', ''],
    ['Classe: Falso (0)', 'Precision', f"{report_dict['Falso (0)']['precision']:.3f}"],
    ['Classe: Falso (0)', 'Recall', f"{report_dict['Falso (0)']['recall']:.3f}"],
    ['Classe: Falso (0)', 'F1-Score', f"{report_dict['Falso (0)']['f1-score']:.3f}"],
    ['Classe: Vero (1)', 'Precision', f"{report_dict['Vero (1)']['precision']:.3f}"],
    ['Classe: Vero (1)', 'Recall', f"{report_dict['Vero (1)']['recall']:.3f}"],
    ['Classe: Vero (1)', 'F1-Score', f"{report_dict['Vero (1)']['f1-score']:.3f}"],
]

metrics_df = pd.DataFrame(metrics_data, columns=['Categoria', 'Metrica', 'Valore'])
metrics_df.to_csv(METRICS_OUTPUT_FILE, index=False)
print("\n✨ Tabella delle metriche salvata con stile in:", METRICS_OUTPUT_FILE)
print(metrics_df.to_string(index=False))

# --- Grafico barplot delle metriche per classe ---
plt.figure(figsize=(6, 4))
sns.barplot(
    x=['Precision (0)', 'Recall (0)', 'F1 (0)', 'Precision (1)', 'Recall (1)', 'F1 (1)'],
    y=[
        report_dict['Falso (0)']['precision'],
        report_dict['Falso (0)']['recall'],
        report_dict['Falso (0)']['f1-score'],
        report_dict['Vero (1)']['precision'],
        report_dict['Vero (1)']['recall'],
        report_dict['Vero (1)']['f1-score']
    ]
)
plt.title("Metriche per Classe (Validation Set)")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
barplot_path = os.path.join(ANALYSIS_DIR, 'final_metrics_barplot.png')
plt.savefig(barplot_path)
plt.close()
print("📊 Grafico delle metriche per classe salvato in:", barplot_path)

cm = confusion_matrix(y_val_outer, y_pred_final)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predetto Falso (0)', 'Predetto Vero (1)'],
            yticklabels=['Reale Falso (0)', 'Reale Vero (1)'])
plt.title("Confusion Matrix 'Onesta' (su 20% di dati mai visti)")
plt.savefig(CM_OUTPUT_FILE)
plt.close()
print(f"Grafico della Confusion Matrix salvato in: {CM_OUTPUT_FILE}")

# --- 6. Feature Importance ---
# (Invariata)
print("\n--- Fase 3: Feature Importance ---")
importances = model.get_feature_importance()
feature_names = X_train_outer.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
top_20_features = importance_df.sort_values(by='Importance', ascending=False).head(20)

print("Top 20 Features più importanti:")
print(top_20_features.to_string(index=False))
plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=top_20_features, palette='viridis')
plt.title('Top 20 Feature Importance (CatBoost)')
plt.tight_layout()
plt.savefig(IMPORTANCE_OUTPUT_FILE)
plt.close()
print(f"Grafico Feature Importance salvato in: {IMPORTANCE_OUTPUT_FILE}")

# --- 7. Grafici di Apprendimento ---
print("\n--- Fase 4: Grafico di Apprendimento (solo Logloss) ---")
results = model.get_evals_result()
print("Chiavi disponibili nei risultati di addestramento:", results.keys())

train_set_name = 'learn' if 'learn' in results else 'validation_0'
valid_set_name = 'validation_1' if 'validation_1' in results else 'validation'

if 'Logloss' in results[train_set_name]:
    plt.figure(figsize=(10, 6))
    plt.plot(results[train_set_name]['Logloss'], label='Training Loss')
    plt.plot(results[valid_set_name]['Logloss'], label='Validation Loss')
    plt.title('Curva di Apprendimento (Logloss)')
    plt.xlabel('Iterazioni (Alberi)')
    plt.ylabel('Logloss')
    plt.legend()
    plt.grid(True)
    plt.savefig(LOSS_CURVE_FILE)
    plt.close()
    print(f"Grafico curva Loss salvato in: {LOSS_CURVE_FILE}")
else:
    print("⚠️ Nessuna metrica Logloss trovata nei risultati di addestramento.")

# --- 8. Grafico ROC-AUC ---
# (Invariato)
print("\n--- Fase 5: Grafico Curva ROC-AUC ---")
fpr, tpr, thresholds = roc_curve(y_val_outer, y_pred_proba_final)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Curva ROC (AUC = {val_auc:.4f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Caso (AUC = 0.50)')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Curva ROC-AUC (sul 20% di validazione)')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(ROC_CURVE_FILE)
plt.close()
print(f"Grafico curva ROC-AUC salvato in: {ROC_CURVE_FILE}")


print("\nDiagnostica DEFINITIVA completata.")
print("Ora siamo pronti per lo script di submission finale.")