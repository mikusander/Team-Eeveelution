"""
This script preprocesses the expert Pokémon battle dataset (v5 features) for modeling.
It performs One-Hot Encoding for categorical features, imputes missing numeric values with median,
scales numeric features, and separates the dataset into processed training and test sets along
with the target variable.
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 

#FEATURES_DIR = 'Features_v5' #Features_v6
FEATURES_DIR = 'Features_v6'
PREPROCESSED_DIR = 'Preprocessed_Data' 
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

#TRAIN_IN = os.path.join(FEATURES_DIR, 'features_expert_train.csv') #features_team_stats_train
#TEST_IN = os.path.join(FEATURES_DIR, 'features_expert_test.csv') #features_team_stats_test

TRAIN_IN = os.path.join(FEATURES_DIR, 'features_team_stats_train.csv') #features_team_stats_train
TEST_IN = os.path.join(FEATURES_DIR, 'features_team_stats_test.csv') #features_team_stats_test

TRAIN_OUT = os.path.join(PREPROCESSED_DIR, 'train_processed.csv')
TEST_OUT = os.path.join(PREPROCESSED_DIR, 'test_processed.csv')
TARGET_OUT = os.path.join(PREPROCESSED_DIR, 'target_train.csv')

print(f"Processed data will be saved in: {PREPROCESSED_DIR}")

# Load training and test datasets
print(f"Loading {TRAIN_IN}...")
train_df = pd.read_csv(TRAIN_IN)
test_df = pd.read_csv(TEST_IN)
print("Data loaded.")

y_train = train_df['player_won']
test_ids = test_df['battle_id']

train_df = train_df.drop(columns=['player_won', 'battle_id']) 

test_df = test_df.drop(columns=['battle_id', 'player_won'])

train_df['is_train'] = 1
test_df['is_train'] = 0

# Combine training and test sets for consistent preprocessing
combined_df = pd.concat([train_df, test_df], ignore_index=True)
print(f"Combined DataFrame created with {combined_df.shape[0]} rows.")

# Identify categorical and numeric features
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
print(f"Found {len(numeric_features)} numeric features.")
print(f"Found {len(categorical_features)} categorical features.")

# Perform One-Hot Encoding for categorical features
print("Performing One-Hot Encoding (pd.get_dummies)...")
processed_df = pd.get_dummies(
    combined_df, 
    columns=categorical_features, 
    dummy_na=False, 
    drop_first=False
)
print(f"DataFrame transformed to {processed_df.shape[1]} total columns.")

# Impute missing numeric values and scale numeric features
print("Applying SimpleImputer and StandardScaler to numeric features...")

numeric_imputer = SimpleImputer(strategy='median')
processed_df[numeric_features] = numeric_imputer.fit_transform(processed_df[numeric_features])

scaler = StandardScaler()
processed_df[numeric_features] = scaler.fit_transform(processed_df[numeric_features])

print("Imputation and scaling completed.")


# Split back into processed Train and Test sets and save
print("Separating processed Train and Test sets...")
X_train_processed = processed_df[processed_df['is_train'] == 1].drop(columns=['is_train'])
X_test_processed = processed_df[processed_df['is_train'] == 0].drop(columns=['is_train'])

print(f"Saving {TRAIN_OUT}...")
X_train_processed.to_csv(TRAIN_OUT, index=False)
print(f"Saving {TEST_OUT}...")
X_test_processed.to_csv(TEST_OUT, index=False)
print(f"Saving {TARGET_OUT}...")
y_train.to_csv(TARGET_OUT, index=False, header=True)

print("\nPreprocessing completed.")
print("\n12_preprocessing.py executed successfully.")