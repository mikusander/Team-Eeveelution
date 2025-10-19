import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. Configurazione ---
FEATURES_DIR = 'Features_v2'  # <-- MODIFICATO: leggiamo dalla nuova cartella
ANALYSIS_DIR = 'Analysis_Output'
os.makedirs(ANALYSIS_DIR, exist_ok=True)

FEATURES_TRAIN_FILE = os.path.join(FEATURES_DIR, 'features_train_v2.csv') # <-- MODIFICATO
HEATMAP_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'correlation_heatmap_v2.png') # <-- MODIFICATO

sns.set(style="whitegrid")

# --- 2. Caricamento Dati ---
print(f"Caricamento dati da {FEATURES_TRAIN_FILE}...")
try:
    df = pd.read_csv(FEATURES_TRAIN_FILE)
except FileNotFoundError:
    print(f"Errore: File non trovato. Assicurati di aver eseguito prima '04_feature_engineering_static.py'")
    exit()

print(f"Caricate {len(df)} righe.")

# --- 3. Analisi del Target (Invariata) ---
print("\n--- 1. Analisi Target (player_won) ---")
target_distribution = df['player_won'].value_counts(normalize=True) * 100
print(target_distribution)
print("Confermiamo che il target è bilanciato.\n")


# --- 4. Analisi Informatività Features (raggruppate per Vittoria) ---
print("--- 2. Medie Features per Esito Battaglia ---")

df['player_won_int'] = df['player_won'].astype(int)

# Selezioniamo le features
features_to_analyze = [
    # Le vecchie features dinamiche
    'faint_delta',
    'p1_fainted_count',
    'p2_fainted_count',
    'final_hp_delta',
    'hp_avg_delta',
    'total_boosts_delta',
    'p1_pokemon_used_count',
    'p2_pokemon_revealed_count',
    
    # <-- NOVITÀ: Le nostre features statiche calcolate -->
    'lead_offense_delta',
    'team_counters_vs_lead'
]

# Calcoliamo la media
grouped_analysis = df.groupby('player_won')[features_to_analyze].mean().T
grouped_analysis.columns = ['Media (Perso)', 'Media (Vinto)']
grouped_analysis['Differenza (Vinto - Perso)'] = grouped_analysis['Media (Vinto)'] - grouped_analysis['Media (Perso)']

print(grouped_analysis)
print("\n* Spiegazione: *")
print("  - Ci aspettiamo 'lead_offense_delta' > 0 per i vincitori.")
print("  - Ci aspettiamo 'team_counters_vs_lead' > 0 per i vincitori.\n")


# --- 5. La "Big Picture": Matrice di Correlazione ---
print("--- 3. Analisi di Correlazione ---")

correlation_features = ['player_won_int'] + features_to_analyze
corr_matrix = df[correlation_features].corr()

# Estraiamo solo la correlazione con 'player_won_int' e ordiniamola
corr_with_target = corr_matrix['player_won_int'].sort_values(ascending=False)
print("Correlazione delle features con 'player_won_int':")
print(corr_with_target)

print("\n* Spiegazione: *")
print("  - Valori vicini a 1 o -1 sono OTTIMI (segnale forte).")


# --- 6. Visualizzazione: Heatmap ---
print(f"--- 4. Salvataggio Heatmap di Correlazione ---")
plt.figure(figsize=(12, 10)) # <-- Leggermente più grande per le nuove features
sns.heatmap(
    corr_matrix, 
    annot=True,     
    cmap='coolwarm',
    fmt=".2f"       
)
plt.title("Matrice di Correlazione (Features Statiche e Dinamiche vs Vittoria)")
plt.tight_layout()
plt.savefig(HEATMAP_OUTPUT_FILE)

print(f"Heatmap salvata in: {HEATMAP_OUTPUT_FILE}")
print("\nAnalisi v2 completata. Controlla l'output e il nuovo file PNG.")