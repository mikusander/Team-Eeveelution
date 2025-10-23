"""
This script performs exploratory data analysis on the Pokémon battle dataset with v3 features,
including dynamic type-effectiveness features. It analyzes the target distribution, computes mean
feature values by battle outcome, calculates feature correlations with the target, and saves a
correlation heatmap.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

FEATURES_DIR = 'Features_v3' 
ANALYSIS_DIR = 'Analysis_Output'
os.makedirs(ANALYSIS_DIR, exist_ok=True)

FEATURES_TRAIN_FILE = os.path.join(FEATURES_DIR, 'features_train_v3.csv') 
HEATMAP_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'correlation_heatmap_v3.png') 

sns.set(style="whitegrid")

# Load training dataset
print(f"Loading training data from {FEATURES_TRAIN_FILE}...")
try:
    df = pd.read_csv(FEATURES_TRAIN_FILE)
except FileNotFoundError:
    print(f"Error: File not found. Ensure '06_feature_engineering_03.py' has been run first.")
    exit()

print(f"Loaded {len(df)} rows.")

# Analyze target distribution
print("\nTarget Analysis: 'player_won'")
target_distribution = df['player_won'].value_counts(normalize=True) * 100
print(target_distribution)
print("Confirming target balance.\n")

# Compute mean feature values grouped by battle outcome
print("\nMean Features by Battle Outcome")

df['player_won_int'] = df['player_won'].astype(int)

features_to_analyze = [
    'faint_delta',
    'hp_avg_delta',
    'p1_pokemon_used_count',
    'p2_pokemon_revealed_count',
    'p1_avg_effectiveness',
    'p1_super_effective_hits',
    'p1_resisted_hits',
    'p2_avg_effectiveness',
    'p2_super_effective_hits',
    'p2_resisted_hits'
]

grouped_analysis = df.groupby('player_won')[features_to_analyze].mean().T
grouped_analysis.columns = ['Mean (Lost)', 'Mean (Won)']
grouped_analysis['Difference (Won - Lost)'] = grouped_analysis['Mean (Won)'] - grouped_analysis['Mean (Lost)']

print(grouped_analysis)

# Compute feature correlations with target and generate heatmap
print("\nCorrelation Analysis")

correlation_features = ['player_won_int'] + features_to_analyze
corr_matrix = df[correlation_features].corr()

corr_with_target = corr_matrix['player_won_int'].sort_values(ascending=False)
print("Feature correlations with 'player_won_int':")
print(corr_with_target)

print("Saving correlation heatmap...")
plt.figure(figsize=(14, 12)) 
sns.heatmap(
    corr_matrix, 
    annot=True,     
    cmap='coolwarm',
    fmt=".2f"       
)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(HEATMAP_OUTPUT_FILE)

print(f"Heatmap saved to: {HEATMAP_OUTPUT_FILE}")
print("\nAnalysis v3 completed. Check console output and saved PNG.")
print("\n07_analysis_v3.py executed successfully.")