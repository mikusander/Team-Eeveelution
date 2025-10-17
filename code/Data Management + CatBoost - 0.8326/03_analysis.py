import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. Configurazione ---
FEATURES_DIR = 'Features'
ANALYSIS_DIR = 'Analysis_Output'
os.makedirs(ANALYSIS_DIR, exist_ok=True)

FEATURES_TRAIN_FILE = os.path.join(FEATURES_DIR, 'features_train.csv')
HEATMAP_OUTPUT_FILE = os.path.join(ANALYSIS_DIR, 'correlation_heatmap.png')

# Impostiamo lo stile dei grafici
sns.set(style="whitegrid")

# --- 2. Caricamento Dati ---
print(f"Caricamento dati da {FEATURES_TRAIN_FILE}...")
try:
    df = pd.read_csv(FEATURES_TRAIN_FILE)
except FileNotFoundError:
    print(f"Errore: File non trovato. Assicurati di aver eseguito prima '02_feature_engineering.py'")
    exit()

print(f"Caricate {len(df)} righe.")

# --- 3. Analisi del Target ---
print("\n--- 1. Analisi Target (player_won) ---")
target_distribution = df['player_won'].value_counts(normalize=True) * 100
print(target_distribution)
print("Confermiamo che il target è (o non è) bilanciato.\n")


# --- 4. Analisi Informatività Features (raggruppate per Vittoria) ---
# Analizziamo come le medie delle nostre nuove features cambiano tra chi vince e chi perde.

print("--- 2. Medie Features per Esito Battaglia ---")

# Convertiamo bool in int (True=1, False=0) per il calcolo delle medie
df['player_won_int'] = df['player_won'].astype(int)

# Selezioniamo solo le features numeriche che abbiamo creato
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

# Calcoliamo la media di queste features, raggruppando per player_won
grouped_analysis = df.groupby('player_won')[dynamic_features].mean().T
grouped_analysis.columns = ['Media (Perso)', 'Media (Vinto)']
grouped_analysis['Differenza (Vinto - Perso)'] = grouped_analysis['Media (Vinto)'] - grouped_analysis['Media (Perso)']

print(grouped_analysis)
print("\n* Spiegazione: *")
print("  - 'faint_delta': Ci aspettiamo un valore negativo per i vincitori (hanno più KO di P2).")
print("  - 'final_hp_delta': Ci aspettiamo un valore positivo per i vincitori (più HP alla fine).")
print("  - 'p2_pokemon_revealed_count': Ci aspettiamo un valore più alto per i vincitori (hanno forzato P2 a cambiare).\n")


# --- 5. La "Big Picture": Matrice di Correlazione ---
# Questo è il modo più rapido per vedere quali features sono legate alla vittoria.

print("--- 3. Analisi di Correlazione ---")

# Selezioniamo le features per la matrice
correlation_features = ['player_won_int'] + dynamic_features
corr_matrix = df[correlation_features].corr()

# Estraiamo solo la correlazione con 'player_won_int' e ordiniamola
corr_with_target = corr_matrix['player_won_int'].sort_values(ascending=False)
print("Correlazione delle features con 'player_won_int':")
print(corr_with_target)

print("\n* Spiegazione: *")
print("  - Valori > 0: La feature aumenta, la probabilità di vincere aumenta.")
print("  - Valori < 0: La feature aumenta, la probabilità di vincere diminuisce.")
print("  - Valori vicini a 1 o -1 sono OTTIMI (segnale forte).")
print("  - Valori vicini a 0 sono INUTILI (nessun segnale).\n")

# --- 6. Visualizzazione: Heatmap ---
# Un grafico vale più di mille numeri.

print(f"--- 4. Salvataggio Heatmap di Correlazione ---")
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix, 
    annot=True,     # Mostra i numeri dentro le celle
    cmap='coolwarm',# Scala di colori (Rosso=positivo, Blu=negativo)
    fmt=".2f"       # Formato a due decimali
)
plt.title("Matrice di Correlazione (Features Dinamiche vs Vittoria)")
plt.tight_layout()
plt.savefig(HEATMAP_OUTPUT_FILE)

print(f"Heatmap salvata in: {HEATMAP_OUTPUT_FILE}")
print("\nAnalisi completata. Controlla l'output della console e il file PNG.")