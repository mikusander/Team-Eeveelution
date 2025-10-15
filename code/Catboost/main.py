from model_catboost import run_catboost_pipeline

if __name__ == "__main__":
    run_catboost_pipeline(
    train_path="/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/CatBoost - 0.8149/Input/train.jsonl",
    test_path="/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/CatBoost - 0.8149/Input/test.jsonl",
    output_csv="/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/CatBoost - 0.8149/Output/submission.csv"
)

