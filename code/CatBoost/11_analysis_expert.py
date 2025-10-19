import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. Configurazione ---
FEATURES_DIR = 'Features_Expert'  # <-- MODIFICATO: leggiamo da expert
ANALYSIS_DIR = 'Analysis_Output'
os.makedirs(ANALYSIS_DIR, exist_ok=True)

FEATURES_TRAIN_FILE = os.path.join(FEATURES_DIR, 'features_expert_train.csv') # <-- MODIFICATO
HEATMAP_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'correlation_heatmap_expert.png') # <-- MODIFICATO

sns.set(style="whitegrid")

# --- 2. Caricamento Dati ---
print(f"Caricamento dati da {FEATURES_TRAIN_FILE}...")
try:
    df = pd.read_csv(FEATURES_TRAIN_FILE)
except FileNotFoundError:
    print(f"Errore: File non trovato. Assicurati di aver eseguito prima '10_feature_engineering_expert.py'")
    exit()

print(f"Caricate {len(df)} righe.")
df['player_won_int'] = df['player_won'].astype(int)

# --- 3. Analisi Standard (Correlazione e Medie) ---
print("\n--- 1. Analisi Standard (Correlazione e Medie) ---")

features_to_analyze = [
    # Le nostre "Top 4"
    'faint_delta',
    'hp_avg_delta',
    'status_turns_delta',
    'p1_pokemon_used_count',
    'p2_pokemon_revealed_count',
    
    # Le nuove "Expert"
    'stab_delta',
    'status_move_delta',
    'healing_move_delta'
]

# 3.1. Analisi delle Medie
print("--- 1.1 Medie Features per Esito Battaglia ---")
grouped_analysis = df.groupby('player_won')[features_to_analyze].mean().T
grouped_analysis.columns = ['Media (Perso)', 'Media (Vinto)']
grouped_analysis['Differenza (Vinto - Perso)'] = grouped_analysis['Media (Vinto)'] - grouped_analysis['Media (Perso)']
print(grouped_analysis)

# 3.2. Analisi di Correlazione
print("\n--- 1.2 Correlazione delle features con 'player_won_int' ---")
correlation_features = ['player_won_int'] + features_to_analyze
corr_matrix = df[correlation_features].corr()
corr_with_target = corr_matrix['player_won_int'].sort_values(ascending=False)
print(corr_with_target)

# 3.3. Salvataggio Heatmap
print(f"\nSalvataggio Heatmap in: {HEATMAP_OUTPUT_FILE}")
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix, 
    annot=True,     
    cmap='coolwarm',
    fmt=".2f"       
)
plt.title("Matrice di Correlazione (Features EXPERT)")
plt.tight_layout()
plt.savefig(HEATMAP_OUTPUT_FILE)
plt.close() # Chiudiamo il plot

# --- 4. Analisi Avanzata (Distribuzioni - KDE Plots) ---
print("\n--- 2. Analisi Avanzata (Distribuzioni) ---")
print("Salvataggio grafici di distribuzione (KDE) in 'analysis_output'...")

expert_features = ['stab_delta', 'status_move_delta', 'healing_move_delta']

for feature in expert_features:
    plt.figure(figsize=(10, 6))
    
    # Plot per chi ha vinto (True)
    sns.kdeplot(df[df['player_won'] == True][feature], label='Vinto (True)', color='blue', shade=True)
    # Plot per chi ha perso (False)
    sns.kdeplot(df[df['player_won'] == False][feature], label='Perso (False)', color='red', shade=True)
    
    plt.title(f'Distribuzione di "{feature}" per Esito Battaglia', fontsize=16)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Densità', fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    output_filename = os.path.join(ANALYSIS_DIR, f'distribuzione_{feature}.png')
    plt.savefig(output_filename)
    plt.close() # Chiudiamo il plot
    print(f"Grafico salvato: {output_filename}")


# --- 5. Analisi Avanzata (Interazione - Scatter Plot) ---
print("\n--- 3. Analisi Avanzata (Interazione) ---")
print("Salvataggio grafico a dispersione (Scatter) in 'analysis_output'...")

# Prendiamo le nostre 2 feature più forti
top_feature_1 = 'faint_delta'
top_feature_2 = 'hp_avg_delta'

plt.figure(figsize=(12, 8))

# Creiamo lo scatter plot, colorando i punti in base a 'player_won'
sns.scatterplot(
    data=df,
    x=top_feature_1,
    y=top_feature_2,
    hue='player_won', # Colora in base alla vittoria
    palette={True: 'blue', False: 'red'}, # Colori personalizzati
    alpha=0.5 # Trasparenza per vedere i punti sovrapposti
)

plt.title('Interazione Top Features: HP vs KO', fontsize=16)
plt.xlabel('Differenza KO (faint_delta) - Negativo è buono', fontsize=12)
plt.ylabel('Differenza HP Medio (hp_avg_delta) - Positivo è buono', fontsize=12)
plt.axvline(0, color='grey', linestyle='--', linewidth=0.5) # Linea a x=0
plt.axhline(0, color='grey', linestyle='--', linewidth=0.5) # Linea a y=0
plt.legend(title='Esito')
plt.tight_layout()

output_filename_scatter = os.path.join(ANALYSIS_DIR, 'scatter_hp_vs_faint.png')
plt.savefig(output_filename_scatter)
plt.close() # Chiudiamo il plot
print(f"Grafico salvato: {output_filename_scatter}")

print("\nAnalisi EXPERT completata. Controlla tutti i nuovi file .png in 'analysis_output'.")