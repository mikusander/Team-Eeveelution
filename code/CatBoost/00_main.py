import subprocess
import sys
import os
import time


RERUN_DATA_PREPROCESSING = False

data_scripts = [
    '01_load_data.py',
    '02_feature_engineering_01.py',
    '03_analysis_v1.py',
    '04_feature_engineering_02.py',
    '05_analysis_v2.py',
    '06_feature_engineering_03.py',
    '07_analysis_v3.py',
    '08_feature_engineering_04.py',
    '09_analysis_v4.py',
    '10_feature_engineering_05.py',
    '11_analysis_v5.py',
    '12_preprocessing.py',
    '13_verify_preprocessing.py'
]

model_scripts = [
    '14_data_splitter.py',
    '15_optimize_and_validate.py',
    '16_final_holdout_test_60.py',
    '17_final_holdout_test_80.py',
    '18_create_submission.py'
]

PREPROCESSED_DIR = 'Preprocessed_Data'
CHECK_FILE = os.path.join(PREPROCESSED_DIR, 'train_processed.csv')

def execute_script(script_name):
    """
    Esegue un singolo script Python e gestisce gli errori.
    Utilizza lo stesso interprete Python che sta eseguendo main.py.
    """
    print("\n" + "="*70)
    print(f"--- ESECUZIONE: {script_name} ---")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    try:
        subprocess.run(
            [sys.executable, script_name], 
            check=True,
            encoding='utf-8' #
        )
        
        end_time = time.time()
        print("\n" + "-"*70)
        print(f"--- COMPLETATO: {script_name} (Durata: {end_time - start_time:.2f} sec) ---")
        print("-"*70 + "\n")
        
    except subprocess.CalledProcessError as e:
        print(f"\n\n\033[91m" + "!"*70)
        print(f"ERRORE: L'esecuzione di '{script_name}' è fallita!")
        print(f"Codice di uscita: {e.returncode}")
        print("PIPELINE INTERROTTA.")
        print("!"*70 + "\033[0m\n\n")
        sys.exit(1) 
        
    except FileNotFoundError:
        print(f"\n\n\033[91m" + "!"*70)
        print(f"ERRORE: Script non trovato: '{script_name}'")
        print("Assicurati che tutti gli script siano nella stessa cartella di main.py.")
        print("PIPELINE INTERROTTA.")
        print("!"*70 + "\033[0m\n\n")
        sys.exit(1)

def main():
    """
    Funzione principale che orchestra l'intera pipeline.
    """
    pipeline_start_time = time.time()
    
    print("*"*70)
    print("AVVIO PIPELINE DI MACHINE LEARNING - POKÉMON")
    print("*"*70)

    # --- FASE 1: DATA PROCESSING ---
    if RERUN_DATA_PREPROCESSING:
        print("\n[FASE 1] Avvio: Data Processing e Feature Engineering...")
        for script in data_scripts:
            execute_script(script)
        print("\n\033[92m[FASE 1] Completata: Dati processati e verificati.\033[0m")
    else:
        print("\n[FASE 1] Saltata: 'RERUN_DATA_PREPROCESSING' è False.")
        # Verifichiamo se i file necessari per la Fase 2 esistono
        print(f"Controllo esistenza file necessari in '{PREPROCESSED_DIR}'...")
        if not os.path.exists(CHECK_FILE):
            print(f"\n\033[91mERRORE: File '{CHECK_FILE}' non trovato!\033[0m")
            print("Impossibile saltare la Fase 1.")
            print("Imposta 'RERUN_DATA_PREPROCESSING = True' per generare i file.")
            sys.exit(1)
        print(f"\033[92mOK: File trovati. Si procede con la Fase 2.\033[0m")

    # --- FASE 2: MODELING ---
    print("\n[FASE 2] Avvio: Modeling, Validazione e Creazione Submission...")
    for script in model_scripts:
        execute_script(script)
    print("\n\033[92m[FASE 2] Completata: Modeling terminato.\033[0m")

    pipeline_end_time = time.time()
    total_duration = pipeline_end_time - pipeline_start_time
    
    print("\n" + "*"*70)
    print(f"\033[92mPIPELINE COMPLETATA CON SUCCESSO!\033[0m")
    print(f"Durata totale: {total_duration:.2f} secondi.")
    print("Controlla la cartella 'Submissions' per il file finale.")
    print("*"*70)

if __name__ == "__main__":
    main()