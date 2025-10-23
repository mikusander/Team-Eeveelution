"""
This script performs exploratory data analysis on the Pokémon battle dataset with static 
and dynamic features (v2). It analyzes the target distribution, computes mean feature values 
by battle outcome, calculates feature correlations with the target, and saves a correlation heatmap.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

FEATURES_DIR = 'Features_v2'  
ANALYSIS_DIR = 'Analysis_Output'
os.makedirs(ANALYSIS_DIR, exist_ok=True)

FEATURES_TRAIN_FILE = os.path.join(FEATURES_DIR, 'features_train_v2.csv') 
HEATMAP_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'correlation_heatmap_v2.png') 

sns.set(style="whitegrid")

# Load training dataset
print(f"Loading training data from {FEATURES_TRAIN_FILE}...")
try:
    df = pd.read_csv(FEATURES_TRAIN_FILE)
except FileNotFoundError:
    print(f"Error: File not found. Ensure '04_feature_engineering_02.py' has been run first.")
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
    'p1_fainted_count',
    'p2_fainted_count',
    'final_hp_delta',
    'hp_avg_delta',
    'total_boosts_delta',
    'p1_pokemon_used_count',
    'p2_pokemon_revealed_count',
    'lead_offense_delta',
    'team_counters_vs_lead'
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
plt.figure(figsize=(12, 10)) 
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
print("\nAnalysis v2 completed. Check console output and saved PNG.")
print("\n05_analysis_v2.py executed successfully.")