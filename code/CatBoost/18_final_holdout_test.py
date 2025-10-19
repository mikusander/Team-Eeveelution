import pandas as pd
import numpy as np
import os
import json

from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Configurazione ---
SPLIT_DIR = 'preprocessed_splits' # Cartella con i dati divisi
MODEL_PARAMS_DIR = 'model_params'
ANALYSIS_DIR = 'Model_Analysis_Holdout' # Cartella SEPARATA per i risultati finali
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Input (Train e HOLDOUT)
X_TRAIN_FILE = os.path.join(SPLIT_DIR, 'train_split_60_X.csv')
Y_TRAIN_FILE = os.path.join(SPLIT_DIR, 'train_split_60_y.csv')
X_HOLDOUT_FILE = os.path.join(SPLIT_DIR, 'holdout_split_20_X.csv')
Y_HOLDOUT_FILE = os.path.join(SPLIT_DIR, 'holdout_split_20_y.csv')
PARAMS_FILE = os.path.join(MODEL_PARAMS_DIR, 'best_catboost_params.json') # I parametri finali!

# Output (Grafici di HOLDOUT)
METRICS_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'holdout_metrics_report.csv')
CM_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'holdout_confusion_matrix.png')
ROC_CURVE_FILE = os.path.join(ANALYSIS_DIR, 'holdout_roc_auc_curve.png')

print("Avvio Script di Test Finale (Fase 3) su HOLDOUT Set...")

# --- 2. Caricamento Dati (Train 60% e Holdout 20%) ---
print("Caricamento dati 60% (Train) e 20% (Holdout)...")
try:
    X_train = pd.read_csv(X_TRAIN_FILE)
    y_train = pd.read_csv(Y_TRAIN_FILE).values.ravel()
    X_holdout = pd.read_csv(X_HOLDOUT_FILE)
    y_holdout = pd.read_csv(Y_HOLDOUT_FILE).values.ravel()
    
    with open(PARAMS_FILE, 'r') as f:
        best_params_clean = json.load(f)
    print(f"Caricati parametri finali da {PARAMS_FILE}.")

except FileNotFoundError:
    print(f"ERRORE: File non trovati in {SPLIT_DIR} o {PARAMS_FILE}.")
    print("Assicurati di aver eseguito '16_data_splitter.py' e '17_optimize_and_validate.py'.")
    exit()

# --- 3. Addestramento Finale e Test su Holdout ---
print("\n--- Addestramento su 60% e Test su 20% (Holdout) ---")

# Nota: questa volta NON usiamo early stopping con il set di holdout
# per non "inquinare" il test. Addestriamo sul 60% e basta.
# Potremmo usare il 60% come eval_set per trovare il best_iteration,
# ma per un test pulito, addestriamo e testiamo.

# NOTA: Per un test ancora più rigoroso, dovremmo caricare il numero
# di iterazioni (alberi) trovati nel file precedente.
# Per semplicità, ri-addestriamo usando il 20% di validation come
# eval_set per trovare il numero ottimale di alberi, MA PREDIREMO SULL'HOLDOUT.
# *** CORREZIONE: Manteniamo la stessa logica di training del file 2 per coerenza ***
# Carichiamo anche i dati di validation per usarli come eval_set
print("Caricamento 20% (Validation) per usarlo come eval_set (per early stopping)...")
X_val = pd.read_csv(os.path.join(SPLIT_DIR, 'validation_split_20_X.csv'))
y_val = pd.read_csv(os.path.join(SPLIT_DIR, 'validation_split_20_y.csv')).values.ravel()

final_params_fit = best_params_clean.copy() 
final_params_fit.update({
    'n_estimators': 2000, 'early_stopping_rounds': 50,
    'eval_metric': 'Logloss', 'custom_metric': ['AUC'],
    'verbose': 0, # Mettiamo 0 per non mostrare l'output
    'random_seed': 42
})

print("Addestramento modello finale (Train 60%, Eval 20%)...")
final_model = CatBoostClassifier(**final_params_fit)
final_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val), # Usiamo ancora il validation set per l'early stopping
    plot=False,
    use_best_model=True, 
    verbose=0
)
print("Modello addestrato sull'iterazione migliore (trovata sul validation set).")

# --- 4. Predizione e Metriche su HOLDOUT ---
print("\n--- Test 'Onesto' sul 20% (Holdout) ---")
y_pred_holdout = final_model.predict(X_holdout)
y_pred_proba_holdout = final_model.predict_proba(X_holdout)[:, 1]

# Calcoliamo le metriche finali
holdout_accuracy = accuracy_score(y_holdout, y_pred_holdout)
holdout_auc = roc_auc_score(y_holdout, y_pred_proba_holdout)

print(f"  Accuracy (sul Holdout 20%): {holdout_accuracy:.4f}  <-- PUNTEGGIO FINALE")
print(f"  AUC (sul Holdout 20%): {holdout_auc:.4f}      <-- PUNTEGGIO FINALE")

print("\n--- Classification Report 'Finale' (su 20% Holdout) ---")
print(classification_report(y_holdout, y_pred_holdout, target_names=['Falso (0)', 'Vero (1)']))
report_dict_holdout = classification_report(y_holdout, y_pred_holdout, target_names=['Falso (0)', 'Vero (1)'], output_dict=True)

# Salvataggio report metriche finali
print(f"Salvataggio metriche (Holdout) in: {METRICS_OUTPUT_FILE}")
metrics_data_holdout = {
    'Metric': ['Accuracy (Holdout 20%)', 'AUC (Holdout 20%)',
               'Precision (Falso 0)', 'Recall (Falso 0)', 'F1-Score (Falso 0)',
               'Precision (Vero 1)', 'Recall (Vero 1)', 'F1-Score (Vero 1)'],
    'Score': [holdout_accuracy, holdout_auc,
              report_dict_holdout['Falso (0)']['precision'], report_dict_holdout['Falso (0)']['recall'], report_dict_holdout['Falso (0)']['f1-score'],
              report_dict_holdout['Vero (1)']['precision'], report_dict_holdout['Vero (1)']['recall'], report_dict_holdout['Vero (1)']['f1-score']]
}
pd.DataFrame(metrics_data_holdout).to_csv(METRICS_OUTPUT_FILE, index=False, float_format='%.4f')

# --- 5. Grafici Finali (Holdout) ---
print("Salvataggio grafici finali (Holdout)...")

# --- Confusion Matrix (Holdout) ---
cm_holdout = confusion_matrix(y_holdout, y_pred_holdout)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_holdout, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Predetto Falso (0)', 'Predetto Vero (1)'], 
            yticklabels=['Reale Falso (0)', 'Reale Vero (1)'])
plt.title("Confusion Matrix 'Finale' (su 20% Holdout Set)")
plt.savefig(CM_OUTPUT_FILE)
plt.close()
print(f"Grafico CM (Holdout) salvato in: {CM_OUTPUT_FILE}")

# --- Curva ROC-AUC (Holdout) ---
fpr_hold, tpr_hold, _ = roc_curve(y_holdout, y_pred_proba_holdout)
plt.figure(figsize=(10, 8))
plt.plot(fpr_hold, tpr_hold, color='green', lw=2, label=f'Curva ROC (Holdout AUC = {holdout_auc:.4f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Caso (AUC = 0.50)')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Curva ROC-AUC \'Finale\' (su 20% Holdout Set)')
plt.legend(loc="lower right"); plt.grid(True)
plt.savefig(ROC_CURVE_FILE)
plt.close()
print(f"Grafico curva ROC-AUC (Holdout) salvato in: {ROC_CURVE_FILE}")

print(f"\n--- TEST FINALE COMPLETATO ---")
print(f"I risultati 'veri' del tuo modello sono in: {ANALYSIS_DIR}")