import pandas as pd
import json
import os

# --- 1. Configurazione dei Percorsi ---
INPUT_DIR = 'Input'
OUTPUT_DIR = 'Output_CSVs' 
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"I CSV verranno salvati in: {OUTPUT_DIR}")

TRAIN_FILE = os.path.join(INPUT_DIR, 'train.jsonl')
TEST_FILE = os.path.join(INPUT_DIR, 'test.jsonl')

# --- 2. Funzione di Caricamento JSONL (invariata) ---
def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Errore nel decodificare una linea nel file: {filepath}")
    return data

# --- 3. Caricamento Dati Grezzi (invariato) ---
print(f"Caricamento dati da {TRAIN_FILE}...")
train_data_raw = load_jsonl(TRAIN_FILE)
print(f"Caricamento dati da {TEST_FILE}...")
test_data_raw = load_jsonl(TEST_FILE)
print(f"Caricate {len(train_data_raw)} battaglie di training.")
print(f"Caricate {len(test_data_raw)} battaglie di test.")

# --- 4. FUNZIONE AGGIORNATA per Appiattire i Dati Statici (V3) ---
def process_battles_data(raw_data):
    """
    Appiattisce manualmente i dati statici, gestendo
    correttamente p1_team_details (lista), p2_lead_details (dizionario)
    E ORA ANCHE I TIPI (types).
    """
    processed_list = []
    for battle in raw_data:
        flat_battle = {
            'battle_id': battle.get('battle_id'),
            'player_won': battle.get('player_won') 
        }
        
        # 2. Appiattisci p2_lead_details
        if 'p2_lead_details' in battle and battle['p2_lead_details']:
            for key, value in battle['p2_lead_details'].items():
                
                # --- GESTIONE TIPI (p2_lead) ---
                if key == 'types':
                    flat_battle['p2_lead.type1'] = value[0] if len(value) > 0 else 'none'
                    flat_battle['p2_lead.type2'] = value[1] if len(value) > 1 else 'none'
                # ------------------------------
                
                else:
                    flat_battle[f'p2_lead.{key}'] = value
        
        # 3. Appiattisci p1_team_details
        if 'p1_team_details' in battle and battle['p1_team_details']:
            for i, pokemon in enumerate(battle['p1_team_details']):
                if pokemon: 
                    for key, value in pokemon.items():
                        
                        # --- GESTIONE TIPI (p1_team) ---
                        if key == 'types':
                            flat_battle[f'p1_team.{i}.type1'] = value[0] if len(value) > 0 else 'none'
                            flat_battle[f'p1_team.{i}.type2'] = value[1] if len(value) > 1 else 'none'
                        # ---------------------------------
                        
                        else:
                            flat_battle[f'p1_team.{i}.{key}'] = value
        
        processed_list.append(flat_battle)
    
    return pd.DataFrame(processed_list)

# --- 5. Creazione DataFrame (Strategia Aggiornata) ---

# --- A. battles_df (Dati Statici) ---
print("\nCreazione 'battles_df' (dati statici) con gestione tipi...")

battles_train_df = process_battles_data(train_data_raw)
battles_test_df = process_battles_data(test_data_raw)

print("--- Esempio battles_train_df (statici) ---")
battles_train_df.info() # Ora info() mostrerà p1_team.0.type1, p1_team.0.type2, ecc.


# --- B. timelines_df (Dati Dinamici) ---
print("\nCreazione 'timelines_df' (dati dinamici)...")

timelines_train_df = pd.json_normalize(
    train_data_raw,
    record_path='battle_timeline',
    meta=['battle_id'],
    errors='ignore'
)
timelines_test_df = pd.json_normalize(
    test_data_raw,
    record_path='battle_timeline',
    meta=['battle_id'],
    errors='ignore'
)

cols_to_drop = ['p1_move_details', 'p2_move_details']
timelines_train_df = timelines_train_df.drop(columns=cols_to_drop, errors='ignore')
timelines_test_df = timelines_test_df.drop(columns=cols_to_drop, errors='ignore')

print("--- Esempio timelines_train_df (dinamici) ---")
timelines_train_df.info()


# --- 6. SALVATAGGIO IN CSV (invariato) ---
print("\nSalvataggio dei DataFrame in file CSV...")

battles_train_path = os.path.join(OUTPUT_DIR, 'battles_train_static.csv')
timelines_train_path = os.path.join(OUTPUT_DIR, 'timelines_train_dynamic.csv')
battles_test_path = os.path.join(OUTPUT_DIR, 'battles_test_static.csv')
timelines_test_path = os.path.join(OUTPUT_DIR, 'timelines_test_dynamic.csv')

battles_train_df.to_csv(battles_train_path, index=False)
timelines_train_df.to_csv(timelines_train_path, index=False)
battles_test_df.to_csv(battles_test_path, index=False)
timelines_test_df.to_csv(timelines_test_path, index=False)

print(f"Salvataggio completato! File creati:")
print(f"  {battles_train_path}")
print(f"  {timelines_train_path}")
print(f"  {battles_test_path}")
print(f"  {timelines_test_path}")

print("\nPasso 1 (Versione 3) completato.")