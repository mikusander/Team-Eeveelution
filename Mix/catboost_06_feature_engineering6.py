# Nome file: 14_feature_engineering_06.py
"""
Questo script aggiunge l'ultimo set di feature statiche: aggregati di
squadra. Legge le statistiche base (HP, Atk, Spe, ecc.) già presenti
nei file CSV e calcola le medie, la diversità dei tipi e le debolezze
condivise per la squadra di P1.
"""

import pandas as pd
import numpy as np
import os
import json
from collections import defaultdict

# --- 1. CONFIGURAZIONE ---
FEATURES_V5_DIR = 'Features_v5'
FEATURES_V6_DIR = 'Features_v6' 
os.makedirs(FEATURES_V6_DIR, exist_ok=True)

# File di Input (da 13_...)
TRAIN_IN = os.path.join(FEATURES_V5_DIR, 'features_expert_train.csv')
TEST_IN = os.path.join(FEATURES_V5_DIR, 'features_expert_test.csv')

# File di Output (per 15_...)
TRAIN_OUT = os.path.join(FEATURES_V6_DIR, 'features_team_stats_train.csv')
TEST_OUT = os.path.join(FEATURES_V6_DIR, 'features_team_stats_test.csv')

# Grafico dei tipi (copiato da 07_feature_engineering_02.py)
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

REMOVE_OLD_STATS_COLUMNS = False

# --- 2. FUNZIONI HELPER ---

def get_effectiveness(move_type, target_types):
    """Calcola l'efficacia di un tipo contro uno o due tipi difensivi."""
    move_type = str(move_type).lower()
    if move_type in ['notype', 'none', 'nan']:
        return 1.0
    
    effectiveness_map = TYPE_EFFECTIVENESS.get(move_type, {})
    multiplier = 1.0
    for target_type in target_types:
        target_type = str(target_type).lower()
        if target_type not in ['notype', 'none', 'nan']:
            multiplier *= effectiveness_map.get(target_type, 1.0)
    return multiplier

def create_team_aggregate_features(row):
    """
    Funzione principale per calcolare le nuove feature statiche
    basate sulla riga del DataFrame.
    """
    team_stats = []
    team_types = set()
    
    # 1. Raccogli statistiche e tipi per tutti i 6 Pokémon dalla riga
    for i in range(6):
        # Leggiamo i dati direttamente dalle colonne del DataFrame
        stats = {
            'hp': row.get(f'p1_team.{i}.base_hp'),
            'atk': row.get(f'p1_team.{i}.base_atk'),
            'def': row.get(f'p1_team.{i}.base_def'),
            'spa': row.get(f'p1_team.{i}.base_spa'),
            'spd': row.get(f'p1_team.{i}.base_spd'),
            'spe': row.get(f'p1_team.{i}.base_spe'),
            'type1': row.get(f'p1_team.{i}.type1'),
            'type2': row.get(f'p1_team.{i}.type2')
        }
        
        # Controlla se le statistiche base sono presenti
        if pd.notna(stats['hp']) and pd.notna(stats['spe']):
            team_stats.append(stats)
            team_types.add(stats['type1'])
            if pd.notna(stats['type2']) and stats['type2'] != 'none':
                team_types.add(stats['type2'])

    if not team_stats: # Se non abbiamo dati (improbabile)
        return pd.Series({
            'p1_avg_team_speed': np.nan, 'p1_total_team_hp': np.nan,
            'p1_avg_team_atk': np.nan, 'p1_avg_team_def': np.nan,
            'p1_type_diversity': 0, 'p1_team_weaknesses_count': 0
        })

    # 2. Calcola le feature aggregate
    p1_avg_team_speed = np.mean([s['spe'] for s in team_stats])
    p1_total_team_hp = np.sum([s['hp'] for s in team_stats])
    
    # Calcola attacco misto (media di Atk fisico e speciale)
    p1_avg_team_atk = np.mean([max(s['atk'], s['spa']) for s in team_stats])
    # Calcola difesa mista (media di Def fisica e speciale)
    p1_avg_team_def = np.mean([(s['def'] + s['spd']) / 2 for s in team_stats])
    
    # Rimuovi 'none' e 'nan' se presenti
    team_types.discard('none')
    team_types.discard(np.nan)
    p1_type_diversity = len(team_types)

    # 3. Calcola debolezze condivise
    p1_team_weaknesses_count = 0
    attacking_types = [t for t in TYPE_EFFECTIVENESS.keys() if t not in ['notype', 'none']]
    
    for att_type in attacking_types:
        weak_pokemon_count = 0
        for pkmn in team_stats:
            def_types = [pkmn['type1'], pkmn.get('type2', 'none')]
            effectiveness = get_effectiveness(att_type, def_types)
            if effectiveness > 1.0: # È debole
                weak_pokemon_count += 1
        
        if weak_pokemon_count >= 3: # 3 o più Pokémon sono deboli a questo tipo
            p1_team_weaknesses_count += 1

    return pd.Series({
        'p1_avg_team_speed': p1_avg_team_speed,
        'p1_total_team_hp': p1_total_team_hp,
        'p1_avg_team_atk': p1_avg_team_atk,
        'p1_avg_team_def': p1_avg_team_def,
        'p1_type_diversity': p1_type_diversity,
        'p1_team_weaknesses_count': p1_team_weaknesses_count
    })

# --- 3. FUNZIONE DI PROCESSO PRINCIPALE ---

def process_dataframe(input_path, output_path):
    """Carica i dati v5, applica le nuove feature e salva in v6."""
    print(f"\nCaricamento dati da {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"ERRORE: File non trovato: {input_path}")
        print("Esegui prima 13_feature_engineering_05.py")
        return

    print("Avvio calcolo feature statiche di squadra (v6)...")
    
    # Applica la funzione a ogni riga
    new_features = df.apply(create_team_aggregate_features, axis=1)
    
    df_final = pd.concat([df, new_features], axis=1)
    
    print(f"Feature aggiunte. Totale colonne: {df_final.shape[1]}")
    
    if REMOVE_OLD_STATS_COLUMNS:
        print("Rimozione delle colonne di statistiche base originali (REMOVE_OLD_STATS_COLUMNS=True)...")
        
        cols_to_drop = []
        stats_keys = ['base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe', 'level', 'item', 'ability']
        for i in range(6):
            for key in stats_keys:
                cols_to_drop.append(f'p1_team.{i}.{key}')
        
        # Rimuovi anche quelle del P2 Lead (che sono nel set di feature)
        stats_keys_p2 = ['level', 'item', 'ability', 'base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe']
        for key in stats_keys_p2:
            cols_to_drop.append(f'p2_lead.{key}')

        # Rimuovi solo le colonne che esistono effettivamente
        cols_that_exist = [col for col in cols_to_drop if col in df_final.columns]
        df_final = df_final.drop(columns=cols_that_exist)
        
        print(f"Rimosse {len(cols_that_exist)} colonne di statistiche base originali per pulizia.")
    
    else:
        print("Mantenute le colonne di statistiche base originali (REMOVE_OLD_STATS_COLUMNS=False).")

    # --- FINE BLOCCO MODIFICATO ---

    print(f"Totale colonne finali: {df_final.shape[1]}")

    df_final.to_csv(output_path, index=False)
    print(f"File feature V6 salvato in: {output_path}")

# --- 4. ESECUZIONE SCRIPT ---


process_dataframe(TRAIN_IN, TRAIN_OUT)
process_dataframe(TEST_IN, TEST_OUT)

print("\n14_feature_engineering_06.py eseguito con successo.")