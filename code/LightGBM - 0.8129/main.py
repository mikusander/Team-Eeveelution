from model_lightgbm_final import run_lightgbm_pipeline
from model_lightgbm_test import run_lightgbm_pipeline_test


if __name__ == "__main__":
    TESTING = True  
    if TESTING:
        run_lightgbm_pipeline_test(
            train_path="/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/LightGBM - 0.8129/Input/train.jsonl",
            test_path="/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/LightGBM - 0.8129/Input/test.jsonl",
            output_csv="/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/LightGBM - 0.8129/Output/submission.csv"
        )
    else:
        run_lightgbm_pipeline(
            train_path="/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/LightGBM - 0.8129/Input/train.jsonl",
            test_path="/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/LightGBM - 0.8129/Input/test.jsonl",
            output_csv="/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/LightGBM - 0.8129/Output/submission.csv"
        )

