import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 

# --- 1. Configurazione ---
FEATURES_DIR = 'features_expert'
PREPROCESSED_DIR = 'preprocessed_data' 
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

TRAIN_IN = os.path.join(FEATURES_DIR, 'features_expert_train.csv')
TEST_IN = os.path.join(FEATURES_DIR, 'features_expert_test.csv')

TRAIN_OUT = os.path.join(PREPROCESSED_DIR, 'train_processed.csv')
TEST_OUT = os.path.join(PREPROCESSED_DIR, 'test_processed.csv')
TARGET_OUT = os.path.join(PREPROCESSED_DIR, 'target_train.csv')

print(f"I dati processati verranno salvati in: {PREPROCESSED_DIR}")

# --- 2. Caricamento Dati ---
print(f"Caricamento {TRAIN_IN}...")
train_df = pd.read_csv(TRAIN_IN)
test_df = pd.read_csv(TEST_IN)
print("Dati caricati.")

# --- 3. Preparazione e Unione ---
y_train = train_df['player_won']
test_ids = test_df['battle_id']

# Rimuoviamo colonne dal TRAIN set
train_df = train_df.drop(columns=['player_won', 'battle_id']) 

# --- MODIFICA QUI ---
# Rimuoviamo colonne dal TEST set (aggiungendo 'player_won')
test_df = test_df.drop(columns=['battle_id', 'player_won'])
# ---------------------

# Aggiungiamo flag per unione
train_df['is_train'] = 1
test_df['is_train'] = 0

# Uniamo i due DataFrame
combined_df = pd.concat([train_df, test_df], ignore_index=True)
print(f"DataFrame combinato creato con {combined_df.shape[0]} righe.")

# --- 4. Identificazione Tipi di Colonne ---
categorical_features = [
    'p2_lead.name', 'p2_lead.type1', 'p2_lead.type2',
]
for i in range(6):
    categorical_features.append(f'p1_team.{i}.name')
    categorical_features.append(f'p1_team.{i}.type1')
    categorical_features.append(f'p1_team.{i}.type2')

numeric_features = [
    col for col in combined_df.columns 
    if col not in categorical_features and col != 'is_train'
]
print(f"Trovate {len(numeric_features)} features numeriche.")
print(f"Trovate {len(categorical_features)} features categoriche.")

# --- 5. Esecuzione del Preprocessing ---

# 5.1. One-Hot Encoding
print("Esecuzione One-Hot Encoding (pd.get_dummies)...")
# Il 'combined_df' ora non ha 'player_won', quindi get_dummies funziona
processed_df = pd.get_dummies(
    combined_df, 
    columns=categorical_features, 
    dummy_na=False, 
    drop_first=False
)
# Le colonne totali ora dovrebbero essere 350 (non 351)
print(f"DataFrame trasformato in {processed_df.shape[1]} colonne totali.")

# 5.2. Scaling delle Features Numeriche (CORRETTO)
print("Applicazione Imputer e StandardScaler alle features numeriche...")

# 1. PRIMA riempiamo i NaN
numeric_imputer = SimpleImputer(strategy='median')
# (ora 'numeric_features' non contiene 'player_won', quindi l'imputer non darà warning)
processed_df[numeric_features] = numeric_imputer.fit_transform(processed_df[numeric_features])

# 2. ORA scaliamo
scaler = StandardScaler()
processed_df[numeric_features] = scaler.fit_transform(processed_df[numeric_features])

print("Imputazione e Scaling completati.")


# --- 6. Separazione e Salvataggio ---
print("Separazione in Train e Test processati...")
X_train_processed = processed_df[processed_df['is_train'] == 1].drop(columns=['is_train'])
X_test_processed = processed_df[processed_df['is_train'] == 0].drop(columns=['is_train'])

print(f"Salvataggio di {TRAIN_OUT}...")
X_train_processed.to_csv(TRAIN_OUT, index=False)
print(f"Salvataggio di {TEST_OUT}...")
X_test_processed.to_csv(TEST_OUT, index=False)
print(f"Salvataggio di {TARGET_OUT}...")
y_train.to_csv(TARGET_OUT, index=False, header=True)

print("\nPreprocessing completato. I dati sono 100% numerici e pronti per i modelli.")