"""
This script performs exploratory data analysis on the Pokémon battle feature dataset.
It analyzes the target distribution, calculates mean dynamic feature values per battle outcome,
computes feature correlations with the target, and saves a correlation heatmap for visualization.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

FEATURES_DIR = 'Features_v1'
ANALYSIS_DIR = 'Analysis_Output'
os.makedirs(ANALYSIS_DIR, exist_ok=True)

FEATURES_TRAIN_FILE = os.path.join(FEATURES_DIR, 'features_train.csv')
HEATMAP_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'correlation_heatmap.png')

sns.set(style="whitegrid")

# Load training dataset
print(f"Loading training data from {FEATURES_TRAIN_FILE}...")
try:
    df = pd.read_csv(FEATURES_TRAIN_FILE)
except FileNotFoundError:
    print(f"Error: File not found. Ensure '02_feature_engineering_01.py' has been run first.")
    exit()

print(f"Loaded {len(df)} rows.")

# Analyze target distribution
print("\nTarget Analysis: 'player_won'")
target_distribution = df['player_won'].value_counts(normalize=True) * 100
print(target_distribution)

# Compute mean values of dynamic features grouped by battle outcome
print("\nMean Dynamic Features by Battle Outcome")

df['player_won_int'] = df['player_won'].astype(int)

dynamic_features = [
    'faint_delta',
    'p1_fainted_count',
    'p2_fainted_count',
    'final_hp_delta',
    'hp_avg_delta',
    'total_boosts_delta',
    'p1_pokemon_used_count',
    'p2_pokemon_revealed_count'
]

grouped_analysis = df.groupby('player_won')[dynamic_features].mean().T
grouped_analysis.columns = ['Mean (Lost)', 'Mean (Won)']
grouped_analysis['Difference (Won - Lost)'] = grouped_analysis['Mean (Won)'] - grouped_analysis['Mean (Lost)']

print(grouped_analysis)

# Compute feature correlations with target and generate heatmap
print("\nCorrelation Analysis")

correlation_features = ['player_won_int'] + dynamic_features
corr_matrix = df[correlation_features].corr()

corr_with_target = corr_matrix['player_won_int'].sort_values(ascending=False)
print("Feature correlations with 'player_won_int':")
print(corr_with_target)

# Save the correlation heatmap
print("Saving correlation heatmap")
plt.figure(figsize=(10, 8))
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
print("\nAnalysis completed. Check console output and saved PNG.")
print("\n03_analysis_v1.py executed successfully.")