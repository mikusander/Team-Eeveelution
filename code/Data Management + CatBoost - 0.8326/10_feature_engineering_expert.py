import pandas as pd
import numpy as np
import os

# --- 1. Configurazione dei Percorsi ---
CSV_DIR = 'Output_CSVs'       # Dati grezzi puliti
FEATURES_FINAL_DIR = 'Features_Final' # Il nostro ultimo set di features
FEATURES_EXPERT_DIR = 'Features_Expert' # L'output definitivo!
os.makedirs(FEATURES_EXPERT_DIR, exist_ok=True)

print(f"I file di features EXPERT verranno salvati in: {FEATURES_EXPERT_DIR}")

# --- 2. Conoscenza di Dominio: Categorie Mosse ---
# Queste sono le mosse più comuni di Gen 1 per queste categorie
STATUS_MOVES = ['thunderwave', 'sleeppowder', 'poisonpowder', 'stunspore', 'glare', 'sing', 'lovelykiss', 'spore']
HEALING_MOVES = ['recover', 'softboiled', 'rest']

# --- 3. Creazione del Pokédex (Funzione Invariata) ---
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
        
        pkmn_data = static_train_df[[name_col, type1_col, type2_col]].drop_duplicates()
        for _, row in pkmn_data.iterrows():
            name = row[name_col]
            if name not in pokedex:
                pokedex[name] = [str(row[type1_col]).lower(), str(row[type2_col]).lower()]
                
    # Aggiungiamo anche il P2 Lead
    p2_lead_data = static_train_df[['p2_lead.name', 'p2_lead.type1', 'p2_lead.type2']].drop_duplicates()
    for _, row in p2_lead_data.iterrows():
        name = row['p2_lead.name']
        if name not in pokedex:
            pokedex[name] = [str(row['p2_lead.type1']).lower(), str(row['p2_lead.type2']).lower()]

    print(f"Pokédex costruito. Contiene {len(pokedex)} Pokémon unici.")
    return pokedex

# --- 4. Funzione per Processare la Timeline (Versione Expert) ---
def process_timeline_expert_features(timeline_df, pokedex):
    """
    Calcola le features STAB e di Categoria.
    """
    print("Inizio calcolo features esperte (STAB, Categoria)...")
    
    # 1. Mappiamo i tipi ai Pokémon attivi (come prima)
    timeline_df['p1_active_types'] = timeline_df['p1_pokemon_state.name'].apply(lambda x: pokedex.get(x, [None, None]))
    timeline_df['p2_active_types'] = timeline_df['p2_pokemon_state.name'].apply(lambda x: pokedex.get(x, [None, None]))
    
    # 2. Calcoliamo le features per ogni turno (riga)
    
    def calculate_turn_features(row):
        move_name_p1 = str(row['p1_move_details.name']).lower()
        move_type_p1 = str(row['p1_move_details.type']).lower()
        active_types_p1 = row['p1_active_types']
        
        move_name_p2 = str(row['p2_move_details.name']).lower()
        move_type_p2 = str(row['p2_move_details.type']).lower()
        active_types_p2 = row['p2_active_types']

        # Calcolo STAB
        p1_is_stab = 1 if (move_type_p1 in active_types_p1) and (move_type_p1 not in ['nan', 'none']) else 0
        p2_is_stab = 1 if (move_type_p2 in active_types_p2) and (move_type_p2 not in ['nan', 'none']) else 0
        
        # Calcolo Categoria
        p1_is_status_move = 1 if move_name_p1 in STATUS_MOVES else 0
        p1_is_healing_move = 1 if move_name_p1 in HEALING_MOVES else 0
        
        p2_is_status_move = 1 if move_name_p2 in STATUS_MOVES else 0
        p2_is_healing_move = 1 if move_name_p2 in HEALING_MOVES else 0
        
        return pd.Series([
            p1_is_stab, p2_is_stab,
            p1_is_status_move, p1_is_healing_move,
            p2_is_status_move, p2_is_healing_move
        ])

    print("... applicazione calcoli turno per turno (richiede tempo)...")
    new_features_per_turn = timeline_df.apply(calculate_turn_features, axis=1)
    new_features_per_turn.columns = [
        'p1_is_stab', 'p2_is_stab',
        'p1_is_status_move', 'p1_is_healing_move',
        'p2_is_status_move', 'p2_is_healing_move'
    ]
    
    # Uniamo queste nuove colonne alla timeline
    timeline_df = pd.concat([timeline_df, new_features_per_turn], axis=1)
    
    # 3. Aggreghiamo i risultati
    print("Calcolo per turno completato. Inizio aggregazione...")
    
    aggregations = {
        'p1_is_stab': 'sum',
        'p2_is_stab': 'sum',
        'p1_is_status_move': 'sum',
        'p1_is_healing_move': 'sum',
        'p2_is_status_move': 'sum',
        'p2_is_healing_move': 'sum'
    }
    
    expert_features_df = timeline_df.groupby('battle_id').agg(aggregations)
    
    # Rinominiamo le colonne per chiarezza
    expert_features_df = expert_features_df.rename(columns={
        'p1_is_stab': 'p1_stab_move_count',
        'p2_is_stab': 'p2_stab_move_count',
        'p1_is_status_move': 'p1_status_move_count',
        'p1_is_healing_move': 'p1_healing_move_count',
        'p2_is_status_move': 'p2_status_move_count',
        'p2_is_healing_move': 'p2_healing_move_count'
    })

    # Creiamo le features "delta"
    expert_features_df['stab_delta'] = expert_features_df['p1_stab_move_count'] - expert_features_df['p2_stab_move_count']
    expert_features_df['status_move_delta'] = expert_features_df['p1_status_move_count'] - expert_features_df['p2_status_move_count']
    expert_features_df['healing_move_delta'] = expert_features_df['p1_healing_move_count'] - expert_features_df['p2_healing_move_count']

    print("Aggregazione features esperte completata.")
    return expert_features_df

# --- 5. Esecuzione del Processo ---

# 5.1. Costruiamo il Pokédex
pokedex = build_pokedex()

# 5.2. Processo di TRAIN
print("\n--- Processo di TRAIN ---")
print("Caricamento timeline di TRAIN...")
timeline_train = pd.read_csv(os.path.join(CSV_DIR, 'timelines_train_dynamic.csv'))
expert_features_train = process_timeline_expert_features(timeline_train, pokedex)

print("Caricamento features_final di TRAIN...")
features_train_final = pd.read_csv(os.path.join(FEATURES_FINAL_DIR, 'features_final_train.csv'))
features_train_expert = pd.merge(features_train_final, expert_features_train, on='battle_id', how='left')
features_train_expert.to_csv(os.path.join(FEATURES_EXPERT_DIR, 'features_expert_train.csv'), index=False)
print(f"File finale EXPERT di TRAIN salvato con {features_train_expert.shape[1]} colonne.")

# 5.3. Processo di TEST
print("\n--- Processo di TEST ---")
print("Caricamento timeline di TEST...")
timeline_test = pd.read_csv(os.path.join(CSV_DIR, 'timelines_test_dynamic.csv'))
expert_features_test = process_timeline_expert_features(timeline_test, pokedex)

print("Caricamento features_final di TEST...")
features_test_final = pd.read_csv(os.path.join(FEATURES_FINAL_DIR, 'features_final_test.csv'))
features_test_expert = pd.merge(features_test_final, expert_features_test, on='battle_id', how='left')
features_test_expert.to_csv(os.path.join(FEATURES_EXPERT_DIR, 'features_expert_test.csv'), index=False)
print(f"File finale EXPERT di TEST salvato con {features_test_expert.shape[1]} colonne.")

print("\nProcesso 'Massimo Sforzo' completato.")
print("Abbiamo i nostri dataset definitivi: features_expert_train.csv e features_expert_test.csv")