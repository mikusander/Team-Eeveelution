import pandas as pd
import numpy as np
import os
import json

from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Configurazione ---
SPLIT_DIR = 'preprocessed_splits' 
MODEL_PARAMS_DIR = 'model_params'
# Cartella di output dedicata per questo test
ANALYSIS_DIR = 'Model_Analysis_Holdout_80' 
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Input
X_TRAIN_FILE = os.path.join(SPLIT_DIR, 'train_split_60_X.csv')
Y_TRAIN_FILE = os.path.join(SPLIT_DIR, 'train_split_60_y.csv')
X_VAL_FILE = os.path.join(SPLIT_DIR, 'validation_split_20_X.csv')
Y_VAL_FILE = os.path.join(SPLIT_DIR, 'validation_split_20_y.csv')
X_HOLDOUT_FILE = os.path.join(SPLIT_DIR, 'holdout_split_20_X.csv')
Y_HOLDOUT_FILE = os.path.join(SPLIT_DIR, 'holdout_split_20_y.csv')

# Parametri
PARAMS_FILE = os.path.join(MODEL_PARAMS_DIR, 'best_catboost_params.json') 
ITERATION_FILE = os.path.join(MODEL_PARAMS_DIR, 'best_iteration.json') 

# Output
METRICS_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'holdout_80_metrics_report.csv')
CM_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'holdout_80_confusion_matrix.png')
ROC_CURVE_FILE = os.path.join(ANALYSIS_DIR, 'holdout_80_roc_auc_curve.png')

print("Avvio Script di Test Finale (Fase 3b) - Addestramento su 80%...")

# --- 2. Caricamento Dati e Parametri ---
print("Caricamento dati 60% (Train), 20% (Validation) e 20% (Holdout)...")
try:
    X_train = pd.read_csv(X_TRAIN_FILE)
    y_train = pd.read_csv(Y_TRAIN_FILE).values.ravel()
    X_val = pd.read_csv(X_VAL_FILE)
    y_val = pd.read_csv(Y_VAL_FILE).values.ravel()
    X_holdout = pd.read_csv(X_HOLDOUT_FILE)
    y_holdout = pd.read_csv(Y_HOLDOUT_FILE).values.ravel()
    
    with open(PARAMS_FILE, 'r') as f:
        best_params = json.load(f)
    print(f"Caricati parametri da {PARAMS_FILE}.")
    
    with open(ITERATION_FILE, 'r') as f:
        best_iteration = json.load(f)['best_iteration']
    print(f"Caricato numero iterazioni ottimale: {best_iteration}")

except FileNotFoundError:
    print(f"ERRORE: File non trovati.")
    print("Assicurati di aver eseguito '16_data_splitter.py' e '17_optimize_and_validate.py' (quello modificato).")
    exit()

# --- 3. Unione Dati di Addestramento (60% + 20%) ---
print("\nUnione di Train (60%) e Validation (20%) in un unico set (80%)...")
X_train_80 = pd.concat([X_train, X_val], ignore_index=True)
y_train_80 = np.concatenate([y_train, y_val])
print(f"Dimensione totale del set di addestramento: {len(X_train_80)} campioni.")

# --- 4. Addestramento Finale (su 80%) e Test su Holdout (20%) ---
print(f"\nAddestramento modello su 80% dei dati per {best_iteration} iterazioni...")

final_params_fit = best_params.copy()
final_params_fit.update({
    # Impostiamo il numero esatto di alberi
    'n_estimators': best_iteration, 
    # Disattiviamo l'early stopping, non serve
    'early_stopping_rounds': None, 
    'verbose': 200, 
    'random_seed': 42
})
# Rimuoviamo le metriche di valutazione
final_params_fit.pop('eval_metric', None)
final_params_fit.pop('custom_metric', None)


final_model = CatBoostClassifier(**final_params_fit)

# Addestriamo sul set unito da 80%. Non c'è eval_set.
final_model.fit(X_train_80, y_train_80)
print("Modello addestrato.")


# --- 5. Predizione e Metriche su HOLDOUT ---
print("\n--- Test 'Onesto' sul 20% (Holdout) ---")
y_pred_holdout = final_model.predict(X_holdout)
y_pred_proba_holdout = final_model.predict_proba(X_holdout)[:, 1]

# Calcoliamo le metriche finali
holdout_accuracy = accuracy_score(y_holdout, y_pred_holdout)
holdout_auc = roc_auc_score(y_holdout, y_pred_proba_holdout)

print(f"  Accuracy (sul Holdout 20%): {holdout_accuracy:.4f}  <-- PUNTEGGIO FINALE (da 80%)")
print(f"  AUC (sul Holdout 20%): {holdout_auc:.4f}      <-- PUNTEGGIO FINALE (da 80%)")

print("\n--- Classification Report 'Finale' (su 20% Holdout) ---")
print(classification_report(y_holdout, y_pred_holdout, target_names=['Falso (0)', 'Vero (1)']))
report_dict_holdout = classification_report(y_holdout, y_pred_holdout, target_names=['Falso (0)', 'Vero (1)'], output_dict=True)

# Salvataggio report metriche finali
print(f"Salvataggio metriche (Holdout da 80%) in: {METRICS_OUTPUT_FILE}")
metrics_data_holdout = {
    'Metric': ['Accuracy (Holdout 20%)', 'AUC (Holdout 20%)',
               'Precision (Falso 0)', 'Recall (Falso 0)', 'F1-Score (Falso 0)',
               'Precision (Vero 1)', 'Recall (Vero 1)', 'F1-Score (Vero 1)'],
    'Score': [holdout_accuracy, holdout_auc,
              report_dict_holdout['Falso (0)']['precision'], report_dict_holdout['Falso (0)']['recall'], report_dict_holdout['Falso (0)']['f1-score'],
              report_dict_holdout['Vero (1)']['precision'], report_dict_holdout['Vero (1)']['recall'], report_dict_holdout['Vero (1)']['f1-score']]
}
pd.DataFrame(metrics_data_holdout).to_csv(METRICS_OUTPUT_FILE, index=False, float_format='%.4f')

# --- 6. Grafici Finali (Holdout) ---
print("Salvataggio grafici finali (Holdout da 80%)...")

# --- Confusion Matrix (Holdout) ---
cm_holdout = confusion_matrix(y_holdout, y_pred_holdout)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_holdout, annot=True, fmt='d', cmap='Oranges', # Colore diverso
            xticklabels=['Predetto Falso (0)', 'Predetto Vero (1)'], 
            yticklabels=['Reale Falso (0)', 'Reale Vero (1)'])
plt.title("Confusion Matrix 'Finale' (Addestrato su 80%)")
plt.savefig(CM_OUTPUT_FILE)
plt.close()
print(f"Grafico CM (Holdout) salvato in: {CM_OUTPUT_FILE}")

# --- Curva ROC-AUC (Holdout) ---
fpr_hold, tpr_hold, _ = roc_curve(y_holdout, y_pred_proba_holdout)
plt.figure(figsize=(10, 8))
plt.plot(fpr_hold, tpr_hold, color='orange', lw=2, label=f'Curva ROC (Holdout AUC = {holdout_auc:.4f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Caso (AUC = 0.50)')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Curva ROC-AUC \'Finale\ (Addestrato su 80%)')
plt.legend(loc="lower right"); plt.grid(True)
plt.savefig(ROC_CURVE_FILE)
plt.close()
print(f"Grafico curva ROC-AUC (Holdout) salvato in: {ROC_CURVE_FILE}")

print(f"\n--- TEST FINALE (DA 80%) COMPLETATO ---")
print(f"I risultati di questo approccio sono in: {ANALYSIS_DIR}")
print("Ora puoi confrontare questi risultati con quelli dello script 18.")