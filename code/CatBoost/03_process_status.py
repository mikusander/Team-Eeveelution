import json
from pathlib import Path

INPUT_DIR = Path("Input")
INPUT_FILE = INPUT_DIR / "train.jsonl"

OUTPUT_FILE = Path("unique_statuses.py")

NEGATIVE_STATUSES = set()

print(f"Starting analisys of: {INPUT_FILE}")

try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f):
            try:
                data = json.loads(line)
                
                timeline = data.get("battle_timeline")
                if not timeline:
                    continue

                for turn in timeline:
                    p1_state = turn.get("p1_pokemon_state")
                    if p1_state and p1_state.get("status"):
                        if p1_state.get("status") != 'fnt' and p1_state.get("status") != 'nostatus':
                            NEGATIVE_STATUSES.add(p1_state["status"])
                        
                    p2_state = turn.get("p2_pokemon_state")
                    if p2_state and p2_state.get("status"):
                        if p2_state.get("status") != 'fnt' and p2_state.get("status") != 'nostatus':
                            NEGATIVE_STATUSES.add(p2_state["status"])

            except json.JSONDecodeError:
                print(f"Attention: Error in reading the JSON at line {line_number + 1}. Line skipped.")
            except Exception as e:
                print(f"Attention: Unexpected error processing the line {line_number + 1}: {e}")

    
    print("-" * 30)
    print(f"Analisys completed. Found {len(NEGATIVE_STATUSES)} negative status.")
    print(f"Status found: {NEGATIVE_STATUSES}")
    print(f"I'm writing the file: {OUTPUT_FILE}...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(f"STATUSES = {NEGATIVE_STATUSES}\n")

    print(f"File '{OUTPUT_FILE}' successfully created!")

except FileNotFoundError:
    print(f"ERROR: File not found: '{INPUT_FILE}'")
    print("Make sure that the file 'train.jsonl' is in the folder 'Input',")
except Exception as e:
    print(f"ERROR: A general error occurred: {e}")