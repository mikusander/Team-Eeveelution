"""
This script performs exploratory data analysis on the final Pokémon battle dataset (v4 features),
analyzing the target distribution, computing mean feature values by battle outcome, calculating
feature correlations with the target, and saving a correlation heatmap.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

FEATURES_DIR = 'Features_v4'  
ANALYSIS_DIR = 'Analysis_Output'
os.makedirs(ANALYSIS_DIR, exist_ok=True)

FEATURES_TRAIN_FILE = os.path.join(FEATURES_DIR, 'features_final_train.csv') 
HEATMAP_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'correlation_heatmap_final.png') 

sns.set(style="whitegrid")

# Load training dataset
print(f"Loading training data from {FEATURES_TRAIN_FILE}...")
try:
    df = pd.read_csv(FEATURES_TRAIN_FILE)
except FileNotFoundError:
    print(f"Error: File not found. Ensure '08_feature_engineering_04.py' has been run first.")
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
    'p2_avg_effectiveness',
    'p2_super_effective_hits',
    'p1_total_status_turns',
    'p2_total_status_turns',
    'status_turns_delta'
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
plt.figure(figsize=(16, 14)) 
sns.heatmap(
    corr_matrix, 
    annot=True,     
    cmap='coolwarm',
    fmt=".2f"       
)
plt.title("Correlation Matrix: Final Features")
plt.tight_layout()
plt.savefig(HEATMAP_OUTPUT_FILE)

print(f"Heatmap saved to: {HEATMAP_OUTPUT_FILE}")
print("\nFinal analysis completed. Check console output and saved PNG.")
print("\n09_analysis_v4.py executed successfully.")