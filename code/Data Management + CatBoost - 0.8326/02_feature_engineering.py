import pandas as pd
import numpy as np
import os

# --- 1. Configurazione dei Percorsi ---
INPUT_DIR = 'Output_CSVs'  # Leggiamo i CSV creati al Passo 1
OUTPUT_DIR = 'Features'    # Salviamo i nuovi file di features qui
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"I CSV processati verranno salvati in: {OUTPUT_DIR}")

# Percorsi file di input
BATTLES_TRAIN_STATIC_IN = os.path.join(INPUT_DIR, 'battles_train_static.csv')
TIMELINES_TRAIN_DYNAMIC_IN = os.path.join(INPUT_DIR, 'timelines_train_dynamic.csv')
BATTLES_TEST_STATIC_IN = os.path.join(INPUT_DIR, 'battles_test_static.csv')
TIMELINES_TEST_DYNAMIC_IN = os.path.join(INPUT_DIR, 'timelines_test_dynamic.csv')

# Percorsi file di output
FEATURES_TRAIN_OUT = os.path.join(OUTPUT_DIR, 'features_train.csv')
FEATURES_TEST_OUT = os.path.join(OUTPUT_DIR, 'features_test.csv')


# --- 2. Funzione per creare Features Dinamiche ---

def create_dynamic_features(timelines_df):
    """
    Prende il DataFrame 'lungo' (timelines) e crea 
    features aggregate per ogni battle_id.
    """
    
    # Assicuriamoci che i dati siano ordinati per turno
    # Questo è cruciale per features come 'last_hp'
    timelines_df = timelines_df.sort_values(by=['battle_id', 'turn'])
    
    # Calcoliamo le somme dei boost
    boost_cols_p1 = [col for col in timelines_df.columns if 'p1_pokemon_state.boosts' in col]
    boost_cols_p2 = [col for col in timelines_df.columns if 'p2_pokemon_state.boosts' in col]
    
    timelines_df['p1_boosts_sum_turn'] = timelines_df[boost_cols_p1].sum(axis=1)
    timelines_df['p2_boosts_sum_turn'] = timelines_df[boost_cols_p2].sum(axis=1)

    # Calcoliamo i Pokémon KO (fainted)
    timelines_df['p1_fainted_turn'] = (timelines_df['p1_pokemon_state.status'] == 'fnt').astype(int)
    timelines_df['p2_fainted_turn'] = (timelines_df['p2_pokemon_state.status'] == 'fnt').astype(int)

    # Aggreghiamo per battle_id
    # usiamo .agg() per fare tutti i calcoli in una volta
    
    print("Inizio aggregazione features dinamiche...")
    
    # Definiamo le aggregazioni
    aggregations = {
        # Features di Progresso
        'p1_pokemon_state.name': 'nunique',
        'p2_pokemon_state.name': 'nunique',
        'p1_fainted_turn': 'sum',
        'p2_fainted_turn': 'sum',
        
        # Features di Stato (HP)
        'p1_pokemon_state.hp_pct': ['mean', 'last'],
        'p2_pokemon_state.hp_pct': ['mean', 'last'],
        
        # Features di Stato (Boosts)
        'p1_boosts_sum_turn': 'sum',
        'p2_boosts_sum_turn': 'sum',
    }
    
    # Eseguiamo il groupby
    dynamic_features_df = timelines_df.groupby('battle_id').agg(aggregations)
    
    # Puliamo i nomi delle colonne
    # (Pandas crea nomi multi-livello, es. ('p1_pokemon_state.hp_pct', 'mean'))
    new_cols = []
    for col in dynamic_features_df.columns:
        if col[1] == '':
             new_cols.append(col[0])
        elif col[1] == 'nunique':
            if col[0] == 'p1_pokemon_state.name':
                new_cols.append('p1_pokemon_used_count')
            else:
                new_cols.append('p2_pokemon_revealed_count')
        else:
             new_cols.append(f'{col[0]}_{col[1]}')
            
    dynamic_features_df.columns = new_cols
    
    # Rinominiamo le colonne calcolate
    dynamic_features_df = dynamic_features_df.rename(columns={
        'p1_fainted_turn_sum': 'p1_fainted_count',
        'p2_fainted_turn_sum': 'p2_fainted_count',
        'p1_pokemon_state.hp_pct_mean': 'p1_avg_hp_pct',
        'p2_pokemon_state.hp_pct_mean': 'p2_avg_hp_pct',
        'p1_pokemon_state.hp_pct_last': 'p1_hp_at_turn_30',
        'p2_pokemon_state.hp_pct_last': 'p2_hp_at_turn_30',
        'p1_boosts_sum_turn_sum': 'p1_total_boosts',
        'p2_boosts_sum_turn_sum': 'p2_total_boosts',
    })

    # Creiamo le features "delta" (differenza)
    dynamic_features_df['faint_delta'] = dynamic_features_df['p1_fainted_count'] - dynamic_features_df['p2_fainted_count']
    dynamic_features_df['hp_avg_delta'] = dynamic_features_df['p1_avg_hp_pct'] - dynamic_features_df['p2_avg_hp_pct']
    dynamic_features_df['final_hp_delta'] = dynamic_features_df['p1_hp_at_turn_30'] - dynamic_features_df['p2_hp_at_turn_30']
    dynamic_features_df['total_boosts_delta'] = dynamic_features_df['p1_total_boosts'] - dynamic_features_df['p2_total_boosts']

    print("Aggregazione completata.")
    return dynamic_features_df


# --- 3. Caricamento e Processamento Dati ---

# --- TRAIN ---
print("\n--- Processo di TRAIN ---")
print(f"Caricamento {BATTLES_TRAIN_STATIC_IN}...")
battles_train_df = pd.read_csv(BATTLES_TRAIN_STATIC_IN)
print(f"Caricamento {TIMELINES_TRAIN_DYNAMIC_IN}...")
timelines_train_df = pd.read_csv(TIMELINES_TRAIN_DYNAMIC_IN)

# Creiamo le features dinamiche per il train
dynamic_features_train = create_dynamic_features(timelines_train_df)

# Uniamo (merge) le features dinamiche al DataFrame statico
print("Unione features statiche e dinamiche per il TRAIN...")
features_train_df = pd.merge(
    battles_train_df, 
    dynamic_features_train, 
    on='battle_id', 
    how='left'
)
print(f"DataFrame finale di TRAIN creato con {features_train_df.shape[1]} colonne.")


# --- TEST ---
print("\n--- Processo di TEST ---")
print(f"Caricamento {BATTLES_TEST_STATIC_IN}...")
battles_test_df = pd.read_csv(BATTLES_TEST_STATIC_IN)
print(f"Caricamento {TIMELINES_TEST_DYNAMIC_IN}...")
timelines_test_df = pd.read_csv(TIMELINES_TEST_DYNAMIC_IN)

# Creiamo le features dinamiche per il test
dynamic_features_test = create_dynamic_features(timelines_test_df)

# Uniamo (merge) le features dinamiche al DataFrame statico
print("Unione features statiche e dinamiche per il TEST...")
features_test_df = pd.merge(
    battles_test_df, 
    dynamic_features_test, 
    on='battle_id', 
    how='left'
)
print(f"DataFrame finale di TEST creato con {features_test_df.shape[1]} colonne.")


# --- 4. Salvataggio DataFrame Finali ---
print("\nSalvataggio dei DataFrame finali...")
features_train_df.to_csv(FEATURES_TRAIN_OUT, index=False)
features_test_df.to_csv(FEATURES_TEST_OUT, index=False)

print(f"Salvataggio completato! File creati:")
print(f"  {FEATURES_TRAIN_OUT}")
print(f"  {FEATURES_TEST_OUT}")

print("\nPasso 2 completato.")