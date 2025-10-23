import pprint
from collections import defaultdict

# --- 1. IMPORTA I DATI GREZZI ---
try:
    from move_stats_raw import MOVE_STATS_RAW
    from move_categories_raw import MOVE_CATEGORIES
except ImportError:
    print("ERRORE: File 'move_stats_raw.py' o 'move_categories_raw.py' non trovati.")
    print("Assicurati di eseguire prima lo script 'counter.py'.")
    exit()

# --- 2. DEFINISCI LE TUE SOGLIE (Thresholds) ---
# Queste soglie sono per la logica PRINCIPALE
PRIMARY_THRESHOLD = 0.30 # 30%
SECONDARY_THRESHOLD = 0.01 # 1%
MIN_USES = 1


# --- 3. ESEGUI IL FILTRO (FASE 2) ---
final_move_effects = {}

for move_name, data in sorted(MOVE_STATS_RAW.items()):
    
    total_uses = data['total_uses']
    if total_uses < MIN_USES:
        continue 

    category = MOVE_CATEGORIES.get(move_name, "UNKNOWN")
    effects = data['effects']
    
    final_effects = set()

    # --- Logica Principale (Prova prima questa) ---
    
    if category in ["PHYSICAL", "SPECIAL"]:
        if 'damage' in effects and effects['damage'] > 0:
            final_effects.add('damage')
        if 'healing' in effects and (effects['healing'] / total_uses) >= PRIMARY_THRESHOLD:
            final_effects.add('healing')
        if 'opponent_status' in effects and (effects['opponent_status'] / total_uses) >= SECONDARY_THRESHOLD:
            final_effects.add('opponent_status')
        if 'opponent_debuff' in effects and (effects['opponent_debuff'] / total_uses) >= SECONDARY_THRESHOLD:
            final_effects.add('opponent_debuff')
            
    elif category == "STATUS":
        if 'healing' in effects and (effects['healing'] / total_uses) >= PRIMARY_THRESHOLD:
            final_effects.add('healing')
        if 'user_boost' in effects and (effects['user_boost'] / total_uses) >= PRIMARY_THRESHOLD:
            final_effects.add('user_boost')
        if 'user_status' in effects and (effects['user_status'] / total_uses) >= PRIMARY_THRESHOLD:
            final_effects.add('user_status')
        if 'opponent_status' in effects and (effects['opponent_status'] / total_uses) >= PRIMARY_THRESHOLD:
            final_effects.add('opponent_status')
        if 'opponent_debuff' in effects and (effects['opponent_debuff'] / total_uses) >= PRIMARY_THRESHOLD:
            final_effects.add('opponent_debuff')

    # --- NUOVA Logica di Fallback (Come da tua richiesta) ---
    # Se la logica principale ha fallito e non ha trovato NESSUN effetto...
    if not final_effects and effects: # 'effects' non è vuoto
        
        # ...allora trova l'effetto (non-rumore) con il conteggio più alto.
        possible_effects = {}
        if category == "STATUS":
            for effect, count in effects.items():
                if effect != 'damage': # Ignora il rumore del danno
                    possible_effects[effect] = count
        elif category in ["PHYSICAL", "SPECIAL"]:
            for effect, count in effects.items():
                if effect not in ['user_boost', 'user_status']: # Ignora il rumore dei boost
                    possible_effects[effect] = count
        
        if possible_effects:
            # Prendi solo l'effetto migliore
            best_effect = max(possible_effects, key=possible_effects.get)
            final_effects.add(best_effect)
            
    # --- Fine Logica di Fallback ---

    if final_effects:
        final_move_effects[move_name] = sorted(list(final_effects))

# --- 4. SALVA L'OUTPUT FINALE PULITO ---
try:
    with open("move_effects_final.py", "w", encoding="utf-8") as f:
        f.write("# Questo file è stato generato automaticamente\n")
        f.write("# Contiene la lista di effetti PULITA, basata sui filtri\n")
        f.write(f"# (Soglia Primaria: {PRIMARY_THRESHOLD*100}%, Fallback: Max Conteggio)\n\n")
        f.write("MOVE_EFFECTS_DETAILED = ")
        pprint.pprint(final_move_effects, stream=f, indent=2, width=120)
        
    print(f"Output finale pulito salvato in 'move_effects_final.py'")
    
except IOError as e:
    print(f"Errore durante il salvataggio del file finale: {e}")