import json
import pprint
import traceback
from collections import defaultdict

def save_raw_counts(counts, filename="move_stats_raw.py"):
    try:
        cleaned_counts = {}
        for move, data in counts.items():
            cleaned_counts[move] = {
                'total_uses': data['total_uses'],
                'effects': dict(data['effects'])  
            }

        with open(filename, "w", encoding="utf-8") as f:
            f.write("# This file was generated automatically\n")
            f.write("# Contains the RAW counts of uses/effects for each move.\n\n")
            f.write("MOVE_STATS_RAW = ")
            pprint.pprint(cleaned_counts, stream=f, indent=2, width=120) 

        print(f"Raw counts successfully saved to '{filename}'")
        
    except IOError as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    
    move_stats = defaultdict(lambda: {'total_uses': 0, 'effects': defaultdict(int)})
    move_categories = {}
    input_file = "Input/train.jsonl"
    
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            
            for line_number, line in enumerate(f, 1):
                try:
                    battle_data = json.loads(line)
                    timeline = battle_data.get("battle_timeline", [])
                    
                    if len(timeline) < 2:
                        continue 

                    for i in range(1, len(timeline)):
                        curr_turn = timeline[i]
                        prev_turn = timeline[i-1]

                        p1_curr = curr_turn["p1_pokemon_state"]
                        p2_curr = curr_turn["p2_pokemon_state"]
                        p1_prev = prev_turn["p1_pokemon_state"]
                        p2_prev = prev_turn["p2_pokemon_state"]
                        
                        p1_move = curr_turn.get("p1_move_details")
                        p2_move = curr_turn.get("p2_move_details")
                        
                        if p1_move and p1_move["name"] not in move_categories:
                            move_categories[p1_move["name"]] = p1_move.get("category", "UNKNOWN")
                        if p2_move and p2_move["name"] not in move_categories:
                            move_categories[p2_move["name"]] = p2_move.get("category", "UNKNOWN")

                        p1_did_switch = p1_curr["name"] != p1_prev["name"]
                        p2_did_switch = p2_curr["name"] != p2_prev["name"]

                        if p1_move and not p1_did_switch:
                            move_stats[p1_move["name"]]['total_uses'] += 1
                        if p2_move and not p2_did_switch:
                            move_stats[p2_move["name"]]['total_uses'] += 1

                        if not p1_did_switch:
                            if p1_curr["hp_pct"] < p1_prev["hp_pct"] and p2_move:
                                move_stats[p2_move["name"]]['effects']['damage'] += 1
                            if p1_curr["hp_pct"] > p1_prev["hp_pct"] and p1_move:
                                move_stats[p1_move["name"]]['effects']['healing'] += 1
                            if (p1_curr["boosts"] != p1_prev["boosts"] and any(p1_curr["boosts"][s] > p1_prev["boosts"][s] for s in p1_curr["boosts"])) and p1_move:
                                move_stats[p1_move["name"]]['effects']['user_boost'] += 1
                            if p1_curr["effects"] != p1_prev["effects"] and "noeffect" not in p1_curr["effects"] and p1_move:
                                move_stats[p1_move["name"]]['effects']['user_boost'] += 1
                            if p1_curr["status"] != p1_prev["status"] and p1_curr["status"] not in ["nostatus", "fnt"]:
                                if p1_move and p1_move["name"] == "rest" and p1_curr["status"] == "slp":
                                    move_stats[p1_move["name"]]['effects']['user_status'] += 1
                                elif p2_move: 
                                    move_stats[p2_move["name"]]['effects']['opponent_status'] += 1
                            if (p1_curr["boosts"] != p1_prev["boosts"] and any(p1_curr["boosts"][s] < p1_prev["boosts"][s] for s in p1_curr["boosts"])) and p2_move:
                                move_stats[p2_move["name"]]['effects']['opponent_debuff'] += 1

                        if not p2_did_switch:
                            if p2_curr["hp_pct"] < p2_prev["hp_pct"] and p1_move:
                                move_stats[p1_move["name"]]['effects']['damage'] += 1
                            if p2_curr["hp_pct"] > p2_prev["hp_pct"] and p2_move:
                                move_stats[p2_move["name"]]['effects']['healing'] += 1
                            if (p2_curr["boosts"] != p2_prev["boosts"] and any(p2_curr["boosts"][s] > p2_prev["boosts"][s] for s in p2_curr["boosts"])) and p2_move:
                                move_stats[p2_move["name"]]['effects']['user_boost'] += 1
                            if p2_curr["effects"] != p2_prev["effects"] and "noeffect" not in p2_curr["effects"] and p2_move:
                                move_stats[p2_move["name"]]['effects']['user_boost'] += 1
                            if p2_curr["status"] != p2_prev["status"] and p2_curr["status"] not in ["nostatus", "fnt"]:
                                if p2_move and p2_move["name"] == "rest" and p2_curr["status"] == "slp":
                                    move_stats[p2_move["name"]]['effects']['user_status'] += 1
                                elif p1_move: 
                                    move_stats[p1_move["name"]]['effects']['opponent_status'] += 1
                            if (p2_curr["boosts"] != p2_prev["boosts"] and any(p2_curr["boosts"][s] < p2_prev["boosts"][s] for s in p2_curr["boosts"])) and p1_move:
                                move_stats[p1_move["name"]]['effects']['opponent_debuff'] += 1
                        
                except json.JSONDecodeError:
                    print(f"Attention: skipped non-JSON line (line {line_number}).")
                except Exception as e:
                    print(f"Unexpected error parsing line {line_number}: {e}")

        save_raw_counts(move_stats)

        with open("move_categories_raw.py", "w", encoding="utf-8") as f:
            f.write("# Move categories\n")
            f.write("MOVE_CATEGORIES = ")
            pprint.pprint(move_categories, stream=f, indent=2)

        print("Move categories saved to 'move_categories_raw.py'")

    except FileNotFoundError:
        print(f"ERROR: File not found. Make sure '{input_file}' exists.")
    except Exception as e:
        print(f"Unexpected error reading file: {e}")
        traceback.print_exc()