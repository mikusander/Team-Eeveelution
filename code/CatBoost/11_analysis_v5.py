"""
This script performs exploratory data analysis on the expert Pokémon battle dataset (v5 features).
It analyzes feature distributions, computes mean values by battle outcome, calculates correlations,
and generates visualizations including KDE and scatter plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings


FEATURES_DIR = 'Features_v5'  
ANALYSIS_DIR = 'Analysis_Output'
os.makedirs(ANALYSIS_DIR, exist_ok=True)

FEATURES_TRAIN_FILE = os.path.join(FEATURES_DIR, 'features_expert_train.csv') 
HEATMAP_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'correlation_heatmap_expert.png') 

#sns.set(style="whitegrid")
sns.set_theme(style='whitegrid')

# Load expert feature dataset
print(f"Loading training data from {FEATURES_TRAIN_FILE}...")
try:
    df = pd.read_csv(FEATURES_TRAIN_FILE)
except FileNotFoundError:
    print(f"Error: File not found. Ensure '10_feature_engineering_05.py' has been run first.")
    exit()

print(f"Loaded {len(df)} rows.")
df['player_won_int'] = df['player_won'].astype(int)

# Standard analysis: means and correlation
print("\nStandard Analysis (Correlation and Means)")

features_to_analyze = [
    'faint_delta',
    'hp_avg_delta',
    'status_turns_delta',
    'p1_pokemon_used_count',
    'p2_pokemon_revealed_count',
    'stab_delta',
    'status_move_delta',
    'healing_move_delta'
]

print("Mean Feature Values by Battle Outcome")
grouped_analysis = df.groupby('player_won')[features_to_analyze].mean().T
grouped_analysis.columns = ['Mean (Lost)', 'Mean (Won)']
grouped_analysis['Difference (Won - Lost)'] = grouped_analysis['Mean (Won)'] - grouped_analysis['Mean (Lost)']
print(grouped_analysis)

print("\nFeature correlations with 'player_won_int'")
correlation_features = ['player_won_int'] + features_to_analyze
corr_matrix = df[correlation_features].corr()
corr_with_target = corr_matrix['player_won_int'].sort_values(ascending=False)
print(corr_with_target)

print(f"\nSaving correlation heatmap to: {HEATMAP_OUTPUT_FILE}")
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix, 
    annot=True,     
    cmap='coolwarm',
    fmt=".2f"       
)
plt.title("Correlation Matrix: Expert Features")
plt.tight_layout()
plt.savefig(HEATMAP_OUTPUT_FILE)
plt.close() 

# Advanced analysis: distribution plots (KDE)
print("\nAnalysis (Distributions)")
print("Saving distribution (KDE) plots in 'analysis_output'...")

expert_features = ['stab_delta', 'status_move_delta', 'healing_move_delta']

warnings.filterwarnings("ignore", category=RuntimeWarning)

for feature in expert_features:
    plt.figure(figsize=(10, 6))
    
    sns.kdeplot(df[df['player_won'] == True][feature], label='Won (True)', color='blue', fill=True)
    sns.kdeplot(df[df['player_won'] == False][feature], label='Lost (False)', color='red', fill=True)
    
    plt.title(f'Distribution of "{feature}" by Battle Outcome', fontsize=16)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    output_filename = os.path.join(ANALYSIS_DIR, f'distribuzione_{feature}.png')
    plt.savefig(output_filename)
    plt.close() 
    print(f"Plot saved: {output_filename}")

# Advanced analysis: scatter plot for top features
print("\nAnalysis (Interaction)")
print("Saving scatter plot in 'analysis_output'...")

top_feature_1 = 'faint_delta'
top_feature_2 = 'hp_avg_delta'

plt.figure(figsize=(12, 8))

sns.scatterplot(
    data=df,
    x=top_feature_1,
    y=top_feature_2,
    hue='player_won', 
    palette={True: 'blue', False: 'red'}, 
    alpha=0.5 
)

plt.title('Top Feature Interaction: HP vs KO', fontsize=16)
plt.xlabel('KO Difference (faint_delta)', fontsize=12)
plt.ylabel('Average HP Difference (hp_avg_delta)', fontsize=12)
plt.axvline(0, color='grey', linestyle='--', linewidth=0.5) 
plt.axhline(0, color='grey', linestyle='--', linewidth=0.5) 
plt.legend(title='Outcome')
plt.tight_layout()

output_filename_scatter = os.path.join(ANALYSIS_DIR, 'scatter_hp_vs_faint.png')
plt.savefig(output_filename_scatter)
plt.close() 
print(f"Plot saved: {output_filename_scatter}")

print("\nAnalysis completed. Check all new .png files in 'analysis_output'.")
print("\n11_analysis_v5.py executed successfully.")