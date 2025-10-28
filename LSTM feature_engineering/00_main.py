import subprocess
import sys
import os
import time


RERUN_DATA_PREPROCESSING = False

data_scripts = [
    '01_counter.py',
    '02_filter.py',
    '03_process_status.py',
    '04_load_data.py',
    '05_lstm_preprocessing.py',
    '05b_static_features.py'
]

model_scripts = [
    '06_data_splitter.py',
    '07_optimize_and_validate_ADVANCED.py',
    '08_final_holdout_test_60.py',
    '09_final_holdout_test_80.py',
    '10_create_submission_80.py',
    '11_create_submission_100.py',
    '12_create_submission_ensemble.py',
    '13_blender.py',
    '14_compare_submissions.py'
]

PREPROCESSED_DIR = 'Preprocessed_LSTM'
CHECK_FILE = os.path.join(PREPROCESSED_DIR, 'target.npy')

def execute_script(script_name):
    print("\n" + "="*70)
    print(f"--- Execution: {script_name} ---")
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
        print(f"--- Completed: {script_name} (Duration: {end_time - start_time:.2f} sec) ---")
        print("-"*70 + "\n")
        
    except subprocess.CalledProcessError as e:
        print(f"\n\n\033[91m" + "!"*70)
        print(f"ERROR: Execution of '{script_name}' failed!")
        print(f"Exit code: {e.returncode}")
        print("PIPELINE INTERRUPTED.")
        print("!"*70 + "\033[0m\n\n")
        sys.exit(1) 
        
    except FileNotFoundError:
        print(f"\n\n\033[91m" + "!"*70)
        print(f"ERROR: Script not found: '{script_name}'")
        print("Make sure all scripts are in the same folder as main.py.")
        print("PIPELINE INTERRUPTED.")
        print("!"*70 + "\033[0m\n\n")
        sys.exit(1)

def main():
    pipeline_start_time = time.time()
    
    print("*"*70)
    print("STARTING PIPELINE OF MACHINE LEARNING - POKÉMON")
    print("*"*70)

    if RERUN_DATA_PREPROCESSING:
        print("\n[PHASE 1] Starting: Data Processing and Feature Engineering...")
        for script in data_scripts:
            execute_script(script)
        print("\n\033[92m[PHASE 1] Completed: Data processed and verified.\033[0m")
    else:
        print("\n[PHASE 1] Skipped: 'RERUN_DATA_PREPROCESSING' is False.")
        # Check if the necessary files for Phase 2 exist
        print(f"Checking existence of necessary files in '{PREPROCESSED_DIR}'...")
        if not os.path.exists(CHECK_FILE):
            print(f"\n\033[91mERROR: File '{CHECK_FILE}' not found!\033[0m")
            print("Cannot skip Phase 1.")
            print("Set 'RERUN_DATA_PREPROCESSING = True' to generate the files.")
            sys.exit(1)
        print(f"\033[92mOK: Files found. Proceeding to Phase 2.\033[0m")

    # --- PHASE 2: MODELING ---
    print("\n[PHASE 2] Starting: Modeling, Validation and Submission Creation...")
    for script in model_scripts:
        execute_script(script)
    print("\n\033[92m[PHASE 2] Completed: Modeling finished.\033[0m")

    pipeline_end_time = time.time()
    total_duration = pipeline_end_time - pipeline_start_time
    
    print("\n" + "*"*70)
    print(f"\033[92mPIPELINE COMPLETED SUCCESSFULLY!\033[0m")
    print(f"Total duration: {total_duration:.2f} seconds.")
    print("Check the 'Submissions' folder for the final file.")
    print("*"*70)

if __name__ == "__main__":
    main()