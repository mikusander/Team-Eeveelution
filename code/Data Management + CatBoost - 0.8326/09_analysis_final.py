import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. Configurazione ---
FEATURES_DIR = 'Features_final'  # <-- MODIFICATO: leggiamo da final
ANALYSIS_DIR = 'Analysis_Output'
os.makedirs(ANALYSIS_DIR, exist_ok=True)

FEATURES_TRAIN_FILE = os.path.join(FEATURES_DIR, 'features_final_train.csv') # <-- MODIFICATO
HEATMAP_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'correlation_heatmap_final.png') # <-- MODIFICATO

sns.set(style="whitegrid")

# --- 2. Caricamento Dati ---
print(f"Caricamento dati da {FEATURES_TRAIN_FILE}...")
try:
    df = pd.read_csv(FEATURES_TRAIN_FILE)
except FileNotFoundError:
    print(f"Errore: File non trovato. Assicurati di aver eseguito prima '08_feature_engineering_final.py'")
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

# Selezioniamo TUTTE le features che pensiamo siano utili
features_to_analyze = [
    # Da V2 (KO, HP, Pressione)
    'faint_delta',
    'hp_avg_delta',
    'p1_pokemon_used_count',
    'p2_pokemon_revealed_count',
    
    # Da V3 (Efficacia Dinamica)
    'p1_avg_effectiveness',
    'p1_super_effective_hits',
    'p2_avg_effectiveness',
    'p2_super_effective_hits',
    
    # <-- NOVITÀ: Le nostre features di STATO -->
    'p1_total_status_turns',
    'p2_total_status_turns',
    'status_turns_delta'
]

# Calcoliamo la media
grouped_analysis = df.groupby('player_won')[features_to_analyze].mean().T
grouped_analysis.columns = ['Media (Perso)', 'Media (Vinto)']
grouped_analysis['Differenza (Vinto - Perso)'] = grouped_analysis['Media (Vinto)'] - grouped_analysis['Media (Perso)']

print(grouped_analysis)
print("\n* Spiegazione: *")
print("  - Ci aspettiamo 'status_turns_delta' < 0 per i vincitori (hanno subito meno stati).\n")


# --- 5. La "Big Picture": Matrice di Correlazione ---
print("--- 3. Analisi di Correlazione ---")

# Aggiungiamo il target
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
plt.figure(figsize=(16, 14)) # <-- Ancora più grande per tutte le features
sns.heatmap(
    corr_matrix, 
    annot=True,     
    cmap='coolwarm',
    fmt=".2f"       
)
plt.title("Matrice di Correlazione (Features FINALI - Stati Alterati)")
plt.tight_layout()
plt.savefig(HEATMAP_OUTPUT_FILE)

print(f"Heatmap salvata in: {HEATMAP_OUTPUT_FILE}")
print("\nAnalisi FINALE completata. Controlla l'output e il nuovo file PNG.")