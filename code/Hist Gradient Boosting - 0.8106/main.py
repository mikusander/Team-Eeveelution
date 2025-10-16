from model_hgb_final import run_hgb_pipeline
from model_hgb_test import run_hgb_pipeline_test


if __name__ == "__main__":
    TESTING = False  
    if TESTING:
        run_hgb_pipeline_test(
            train_path="/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/Hist Gradient Boosting - 0.8106/Input/train.jsonl",
            test_path="/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/Hist Gradient Boosting - 0.8106/Input/test.jsonl",
            output_csv="/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/Hist Gradient Boosting - 0.8106/Output/submission.csv"
        )
    else:
        run_hgb_pipeline(
            train_path="/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/Hist Gradient Boosting - 0.8106/Input/train.jsonl",
            test_path="/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/Hist Gradient Boosting - 0.8106/Input/test.jsonl",
            output_csv="/Users/giulia/Documents/Università/Magistrale/FDS/Kaggle/Hist Gradient Boosting - 0.8106/Output/submission.csv"
        )

