import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. Configurazione ---
FEATURES_DIR = 'Features_v3'  # <-- MODIFICATO: leggiamo da v3
ANALYSIS_DIR = 'Analysis_Output'
os.makedirs(ANALYSIS_DIR, exist_ok=True)

FEATURES_TRAIN_FILE = os.path.join(FEATURES_DIR, 'features_train_v3.csv') # <-- MODIFICATO
HEATMAP_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'correlation_heatmap_v3.png') # <-- MODIFICATO

sns.set(style="whitegrid")

# --- 2. Caricamento Dati ---
print(f"Caricamento dati da {FEATURES_TRAIN_FILE}...")
try:
    df = pd.read_csv(FEATURES_TRAIN_FILE)
except FileNotFoundError:
    print(f"Errore: File non trovato. Assicurati di aver eseguito prima '06_feature_engineering_dynamic_types.py'")
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
    # Le vecchie features dinamiche "forti"
    'faint_delta',
    'hp_avg_delta',
    'p1_pokemon_used_count',
    'p2_pokemon_revealed_count',
    
    # <-- NOVITÀ: Le nostre features di EFFICACIA DINAMICA -->
    'p1_avg_effectiveness',
    'p1_super_effective_hits',
    'p1_resisted_hits',
    'p2_avg_effectiveness',
    'p2_super_effective_hits',
    'p2_resisted_hits'
]

# Calcoliamo la media
grouped_analysis = df.groupby('player_won')[features_to_analyze].mean().T
grouped_analysis.columns = ['Media (Perso)', 'Media (Vinto)']
grouped_analysis['Differenza (Vinto - Perso)'] = grouped_analysis['Media (Vinto)'] - grouped_analysis['Media (Perso)']

print(grouped_analysis)
print("\n* Spiegazione: *")
print("  - Ci aspettiamo 'p1_avg_effectiveness' > 1 per i vincitori.")
print("  - Ci aspettiamo 'p2_avg_effectiveness' < 1 per i vincitori.\n")


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
plt.figure(figsize=(14, 12)) # <-- Ancora più grande per tutte le features
sns.heatmap(
    corr_matrix, 
    annot=True,     
    cmap='coolwarm',
    fmt=".2f"       
)
plt.title("Matrice di Correlazione (Features V3 - Efficacia Dinamica)")
plt.tight_layout()
plt.savefig(HEATMAP_OUTPUT_FILE)

print(f"Heatmap salvata in: {HEATMAP_OUTPUT_FILE}")
print("\nAnalisi v3 completata. Controlla l'output e il nuovo file PNG.")