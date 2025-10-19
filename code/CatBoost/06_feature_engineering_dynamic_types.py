import pandas as pd
import numpy as np
import os

# --- 1. Configurazione dei Percorsi ---
CSV_DIR = 'Output_CSVs'       # Dati grezzi puliti (per pokedex e timeline)
FEATURES_V2_DIR = 'Features_v2' # Dati con features dinamiche (v2)
FEATURES_V3_DIR = 'Features_v3' # Output finale
os.makedirs(FEATURES_V3_DIR, exist_ok=True)

print(f"I file di features finali (v3) verranno salvati in: {FEATURES_V3_DIR}")

# --- 2. Conoscenza di Dominio: Matrice Tipi (Invariata) ---
# Usiamo 'notype' e 'none' come i dati, e tutto minuscolo.
TYPE_EFFECTIVENESS = {
    'normal': {'rock': 0.5, 'ghost': 0.0},
    'fire': {'fire': 0.5, 'water': 0.5, 'grass': 2.0, 'ice': 2.0, 'bug': 2.0, 'rock': 0.5, 'dragon': 0.5},
    'water': {'fire': 2.0, 'water': 0.5, 'grass': 0.5, 'ground': 2.0, 'rock': 2.0, 'dragon': 0.5},
    'electric': {'water': 2.0, 'electric': 0.5, 'grass': 0.5, 'ground': 0.0, 'flying': 2.0, 'dragon': 0.5},
    'grass': {'fire': 0.5, 'water': 2.0, 'electric': 0.5, 'grass': 0.5, 'poison': 0.5, 'ground': 2.0, 'flying': 0.5, 'bug': 0.5, 'rock': 2.0, 'dragon': 0.5},
    'ice': {'fire': 0.5, 'water': 0.5, 'grass': 2.0, 'ice': 0.5, 'ground': 2.0, 'flying': 2.0, 'dragon': 2.0},
    'fighting': {'normal': 2.0, 'ice': 2.0, 'poison': 0.5, 'flying': 0.5, 'psychic': 0.5, 'bug': 0.5, 'rock': 2.0, 'ghost': 0.0},
    'poison': {'grass': 2.0, 'poison': 0.5, 'ground': 0.5, 'bug': 2.0, 'rock': 0.5, 'ghost': 0.5},
    'ground': {'fire': 2.0, 'electric': 2.0, 'grass': 0.5, 'poison': 2.0, 'flying': 0.0, 'bug': 0.5, 'rock': 2.0},
    'flying': {'electric': 0.5, 'grass': 2.0, 'fighting': 2.0, 'bug': 2.0, 'rock': 0.5},
    'psychic': {'fighting': 2.0, 'poison': 2.0, 'psychic': 0.5, 'ghost': 0.0},
    'bug': {'fire': 0.5, 'grass': 2.0, 'fighting': 0.5, 'poison': 2.0, 'flying': 0.5, 'psychic': 2.0},
    'rock': {'fire': 2.0, 'ice': 2.0, 'fighting': 0.5, 'ground': 0.5, 'flying': 2.0, 'bug': 2.0},
    'ghost': {'normal': 0.0, 'psychic': 0.0, 'ghost': 2.0},
    'dragon': {'dragon': 2.0},
    'notype': {}, 'none': {}
}

# --- 3. Funzione Helper (Invariata) ---
def get_effectiveness(move_type, target_types):
    if move_type is None or pd.isna(move_type) or move_type in ['notype', 'none']:
        return np.nan # np.nan così non viene contato nelle medie
    
    effectiveness_map = TYPE_EFFECTIVENESS.get(str(move_type).lower(), {})
    
    multiplier = 1.0
    for target_type in target_types:
        if target_type is not None and not pd.isna(target_type) and target_type not in ['notype', 'none']:
            multiplier *= effectiveness_map.get(str(target_type).lower(), 1.0)
            
    return multiplier

# --- 4. Creazione del Pokédex ---
def build_pokedex():
    """
    Costruisce un dizionario {nome: [tipo1, tipo2]} 
    usando i dati statici di P1 come fonte di verità.
    """
    print("Costruzione del Pokédex...")
    static_train_df = pd.read_csv(os.path.join(CSV_DIR, 'battles_train_static.csv'))
    pokedex = {}
    
    for i in range(6): # Per ogni Pokémon nel team di P1
        name_col = f'p1_team.{i}.name'
        type1_col = f'p1_team.{i}.type1'
        type2_col = f'p1_team.{i}.type2'
        
        # Estraiamo le coppie nome-tipi
        pkmn_data = static_train_df[[name_col, type1_col, type2_col]].drop_duplicates()
        
        for _, row in pkmn_data.iterrows():
            name = row[name_col]
            if name not in pokedex:
                pokedex[name] = [row[type1_col], row[type2_col]]
                
    # Aggiungiamo anche il P2 Lead, non si sa mai
    p2_lead_data = static_train_df[['p2_lead.name', 'p2_lead.type1', 'p2_lead.type2']].drop_duplicates()
    for _, row in p2_lead_data.iterrows():
        name = row['p2_lead.name']
        if name not in pokedex:
            pokedex[name] = [row['p2_lead.type1'], row['p2_lead.type2']]

    print(f"Pokédex costruito. Contiene {len(pokedex)} Pokémon unici.")
    return pokedex

# --- 5. Funzione per Processare le Timeline ---
def process_timeline_effectiveness(timeline_df, pokedex):
    """
    Applica la logica di efficacia dinamica a un intero DataFrame timeline.
    """
    print("Inizio calcolo efficacia dinamica (turno per turno)...")
    
    # 1. Mappiamo i tipi ai Pokémon attivi
    #    pokedex.get(x, [None, None]) gestisce Pokémon non trovati
    timeline_df['p1_active_types'] = timeline_df['p1_pokemon_state.name'].apply(lambda x: pokedex.get(x, [None, None]))
    timeline_df['p2_active_types'] = timeline_df['p2_pokemon_state.name'].apply(lambda x: pokedex.get(x, [None, None]))
    
    # 2. Calcoliamo l'efficacia per ogni mossa
    #    usiamo 'apply' con axis=1 per passare intere righe alla funzione
    
    print("... calcolo p1_move_effectiveness")
    timeline_df['p1_move_effectiveness'] = timeline_df.apply(
        lambda row: get_effectiveness(row['p1_move_details.type'], row['p2_active_types']),
        axis=1
    )
    
    print("... calcolo p2_move_effectiveness")
    timeline_df['p2_move_effectiveness'] = timeline_df.apply(
        lambda row: get_effectiveness(row['p2_move_details.type'], row['p1_active_types']),
        axis=1
    )
    
    print("Calcolo efficacia completato. Inizio aggregazione...")
    
    # 3. Aggreghiamo i risultati
    aggregations = {
        'p1_move_effectiveness': ['mean', lambda x: (x > 1.0).sum(), lambda x: (x < 1.0).sum()],
        'p2_move_effectiveness': ['mean', lambda x: (x > 1.0).sum(), lambda x: (x < 1.0).sum()],
    }
    
    dynamic_type_features = timeline_df.groupby('battle_id').agg(aggregations)
    
    # Puliamo i nomi delle colonne
    dynamic_type_features.columns = [
        'p1_avg_effectiveness', 'p1_super_effective_hits', 'p1_resisted_hits',
        'p2_avg_effectiveness', 'p2_super_effective_hits', 'p2_resisted_hits'
    ]
    
    # Riempiamo i NaN con 0 (per hits) e 1 (per avg effectiveness)
    # Se un giocatore non ha mai attaccato, la sua efficacia media è 'neutra' (1.0)
    # e i suoi "colpi" sono 0.
    dynamic_type_features['p1_avg_effectiveness'] = dynamic_type_features['p1_avg_effectiveness'].fillna(1.0)
    dynamic_type_features['p2_avg_effectiveness'] = dynamic_type_features['p2_avg_effectiveness'].fillna(1.0)
    dynamic_type_features = dynamic_type_features.fillna(0) # Per i conteggi 'hits'

    print("Aggregazione features dinamiche di tipo completata.")
    return dynamic_type_features

# --- 6. Esecuzione del Processo ---

# 6.1. Costruiamo il Pokédex UNA SOLA VOLTA
pokedex = build_pokedex()

# 6.2. Processo di TRAIN
print("\n--- Processo di TRAIN ---")
print("Caricamento timeline di TRAIN...")
timeline_train = pd.read_csv(os.path.join(CSV_DIR, 'timelines_train_dynamic.csv'))
dynamic_type_features_train = process_timeline_effectiveness(timeline_train, pokedex)

print("Caricamento features_v2 di TRAIN...")
features_train_v2 = pd.read_csv(os.path.join(FEATURES_V2_DIR, 'features_train_v2.csv'))
features_train_v3 = pd.merge(features_train_v2, dynamic_type_features_train, on='battle_id', how='left')
features_train_v3.to_csv(os.path.join(FEATURES_V3_DIR, 'features_train_v3.csv'), index=False)
print(f"File finale di TRAIN v3 salvato con {features_train_v3.shape[1]} colonne.")

# 6.3. Processo di TEST
print("\n--- Processo di TEST ---")
print("Caricamento timeline di TEST...")
timeline_test = pd.read_csv(os.path.join(CSV_DIR, 'timelines_test_dynamic.csv'))
dynamic_type_features_test = process_timeline_effectiveness(timeline_test, pokedex)

print("Caricamento features_v2 di TEST...")
features_test_v2 = pd.read_csv(os.path.join(FEATURES_V2_DIR, 'features_test_v2.csv'))
features_test_v3 = pd.merge(features_test_v2, dynamic_type_features_test, on='battle_id', how='left')
features_test_v3.to_csv(os.path.join(FEATURES_V3_DIR, 'features_test_v3.csv'), index=False)
print(f"File finale di TEST v3 salvato con {features_test_v3.shape[1]} colonne.")

print("\nPasso 6 completato. Abbiamo aggiunto le features di efficacia dinamica.")