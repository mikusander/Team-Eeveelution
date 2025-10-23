import subprocess
import sys
import os
import time


RERUN_DATA_PREPROCESSING = True

data_scripts = [
    '01_counter.py',
    '02_filter.py',
    '03_load_data.py',
    '04_feature_engineering_01.py',
    '05_analysis_v1.py',
    '06_feature_engineering_02.py',
    '07_analysis_v2.py',
    '08_feature_engineering_03.py',
    '09_analysis_v3.py',
    '10_feature_engineering_04.py',
    '11_analysis_v4.py',
    '12_feature_engineering_05.py',
    '13_analysis_v5.py',
    '14_preprocessing.py',
    '15_verify_preprocessing.py',
    '16_feature_selection.py'
]

model_scripts = [
    '17_data_splitter.py',
    '18_optimize_and_validate.py',
    '19_final_holdout_test_60.py',
    '20_final_holdout_test_80.py',
    '21_create_submission.py'
]

PREPROCESSED_DIR = 'Preprocessed_Data'
CHECK_FILE = os.path.join(PREPROCESSED_DIR, 'train_processed.csv')

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