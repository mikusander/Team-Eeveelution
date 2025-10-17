import pandas as pd
import numpy as np
import os

# --- 1. Configurazione dei Percorsi ---
INPUT_DIR = 'Features'
OUTPUT_DIR = 'Features_v2' # Creiamo una nuova cartella
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES_TRAIN_IN = os.path.join(INPUT_DIR, 'features_train.csv')
FEATURES_TEST_IN = os.path.join(INPUT_DIR, 'features_test.csv')

FEATURES_TRAIN_OUT = os.path.join(OUTPUT_DIR, 'features_train_v2.csv')
FEATURES_TEST_OUT = os.path.join(OUTPUT_DIR, 'features_test_v2.csv')

# --- 2. Conoscenza di Dominio: Matrice Tipi (Gen 1) ---
# Dobbiamo usare 'notype' e 'none' come i dati, e tutto minuscolo.

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
    'psychic': {'fighting': 2.0, 'poison': 2.0, 'psychic': 0.5, 'ghost': 0.0}, # In Gen 1, Psychic era immune a Ghost
    'bug': {'fire': 0.5, 'grass': 2.0, 'fighting': 0.5, 'poison': 2.0, 'flying': 0.5, 'psychic': 2.0},
    'rock': {'fire': 2.0, 'ice': 2.0, 'fighting': 0.5, 'ground': 0.5, 'flying': 2.0, 'bug': 2.0},
    'ghost': {'normal': 0.0, 'psychic': 0.0, 'ghost': 2.0}, # In Gen 1, colpiva solo Ghost
    'dragon': {'dragon': 2.0},
    # Tipi "speciali" trovati nei dati
    'notype': {},
    'none': {}
}

# --- 3. Funzione Helper per calcolare l'efficacia ---
def get_effectiveness(move_type, target_types):
    """
    Calcola il moltiplicatore di una mossa contro i tipi di un target.
    """
    if move_type is None or pd.isna(move_type) or move_type in ['notype', 'none']:
        return 1.0
    
    # Mappa delle efficacie per il tipo di mossa
    effectiveness_map = TYPE_EFFECTIVENESS.get(move_type, {})
    
    multiplier = 1.0
    for target_type in target_types:
        if target_type is not None and not pd.isna(target_type) and target_type not in ['notype', 'none']:
            multiplier *= effectiveness_map.get(target_type, 1.0) # default 1.0 se non c'è interazione
            
    return multiplier

# --- 4. Funzione per creare le nuove Features di Matchup ---
def create_static_matchup_features(row):
    """
    Calcola le features di vantaggio di tipo statico
    basandosi su una riga del DataFrame.
    """
    # 1. Lead vs Lead
    p1_lead_types = [row.get('p1_team.0.type1'), row.get('p1_team.0.type2')]
    p2_lead_types = [row.get('p2_lead.type1'), row.get('p2_lead.type2')]
    
    # Calcoliamo il miglior moltiplicatore offensivo di P1 contro P2
    p1_offense_score = max(
        get_effectiveness(p1_lead_types[0], p2_lead_types),
        get_effectiveness(p1_lead_types[1], p2_lead_types)
    )
    
    # Calcoliamo il miglior moltiplicatore offensivo di P2 contro P1
    p2_offense_score = max(
        get_effectiveness(p2_lead_types[0], p1_lead_types),
        get_effectiveness(p2_lead_types[1], p1_lead_types)
    )
    
    # Feature 1: Vantaggio offensivo
    lead_offense_delta = p1_offense_score - p2_offense_score
    
    # 2. Team P1 vs Lead P2
    team_counters = 0
    for i in range(6): # Per ogni Pokémon nel team di P1
        p1_pkmn_types = [row.get(f'p1_team.{i}.type1'), row.get(f'p1_team.{i}.type2')]
        
        # Calcoliamo il miglior moltiplicatore offensivo di questo Pokémon contro il lead di P2
        pkmn_offense_score = max(
            get_effectiveness(p1_pkmn_types[0], p2_lead_types),
            get_effectiveness(p1_pkmn_types[1], p2_lead_types)
        )
        
        if pkmn_offense_score > 1.0: # È super-efficace?
            team_counters += 1
            
    # Feature 2: Numero di counter nel team
    return pd.Series([lead_offense_delta, team_counters], index=['lead_offense_delta', 'team_counters_vs_lead'])


# --- 5. Esecuzione del Processo ---
def process_dataframe(input_path, output_path):
    """
    Carica il DataFrame, applica il feature engineering e salva.
    """
    print(f"\nCaricamento dati da {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Errore: File non trovato. Assicurati di aver eseguito prima '02_feature_engineering.py'")
        return
    
    print("Inizio calcolo features di vantaggio di tipo (statico)...")
    
    # Applichiamo la funzione riga per riga
    # 'axis=1' significa "per riga"
    # 'result_type='expand'' trasforma l'output di pd.Series in colonne
    new_features = df.apply(create_static_matchup_features, axis=1)
    
    # Uniamo le nuove features al DataFrame esistente
    df_final = pd.concat([df, new_features], axis=1)
    
    print(f"Features aggiunte. Numero totale di colonne: {df_final.shape[1]}")
    
    # Salviamo il nuovo DataFrame
    df_final.to_csv(output_path, index=False)
    print(f"Nuovo file di features salvato in: {output_path}")

# Eseguiamo per Train e Test
process_dataframe(FEATURES_TRAIN_IN, FEATURES_TRAIN_OUT)
process_dataframe(FEATURES_TEST_IN, FEATURES_TEST_OUT)

print("\nPasso 4 completato.")