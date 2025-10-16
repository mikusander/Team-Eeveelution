from model_catboost_final import run_catboost_pipeline_final
from model_catboost_test import run_catboost_pipeline_test

if __name__ == "__main__":
    TESTING = False  
    if TESTING:
        run_catboost_pipeline_test(
        train_path="/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/CatBoost - 0.8149/Input/train.jsonl",
        test_path="/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/CatBoost - 0.8149/Input/test.jsonl",
        output_csv="/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/CatBoost - 0.8149/Output/submission.csv"
    )
    else:
        run_catboost_pipeline_final(
        train_path="/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/CatBoost - 0.8149/Input/train.jsonl",
        test_path="/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/CatBoost - 0.8149/Input/test.jsonl",
        output_csv="/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/CatBoost - 0.8149/Output/submission.csv"
    )

