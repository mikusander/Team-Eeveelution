import pprint
from collections import defaultdict

try:
    from move_stats_raw import MOVE_STATS_RAW
    from move_categories_raw import MOVE_CATEGORIES
except ImportError:
    print("ERROR: File 'move_stats_raw.py' or 'move_categories_raw.py' not found.")
    print("Make sure to run the 'counter.py' script first.")
    exit()

PRIMARY_THRESHOLD = 0.30 
SECONDARY_THRESHOLD = 0.01 
MIN_USES = 1

final_move_effects = {}

for move_name, data in sorted(MOVE_STATS_RAW.items()):
    
    total_uses = data['total_uses']
    if total_uses < MIN_USES:
        continue 

    category = MOVE_CATEGORIES.get(move_name, "UNKNOWN")
    effects = data['effects']
    
    final_effects = set()
    
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

    if not final_effects and effects: 
        possible_effects = {}
        if category == "STATUS":
            for effect, count in effects.items():
                if effect != 'damage': 
                    possible_effects[effect] = count
        elif category in ["PHYSICAL", "SPECIAL"]:
            for effect, count in effects.items():
                if effect not in ['user_boost', 'user_status']: 
                    possible_effects[effect] = count
        
        if possible_effects:
            best_effect = max(possible_effects, key=possible_effects.get)
            final_effects.add(best_effect)
            
    if final_effects:
        final_move_effects[move_name] = sorted(list(final_effects))

try:
    with open("move_effects_final.py", "w", encoding="utf-8") as f:
        f.write("# This file was generated automatically\n")
        f.write("# Contains the CLEANED list of effects, based on filters\n")
        f.write(f"# (Primary Threshold: {PRIMARY_THRESHOLD*100}%, Fallback: Max Count)\n\n")
        f.write("MOVE_EFFECTS_DETAILED = ")
        pprint.pprint(final_move_effects, stream=f, indent=2, width=120)

    print(f"Cleaned final output saved to 'move_effects_final.py'")

except IOError as e:
    print(f"Error saving final file: {e}")