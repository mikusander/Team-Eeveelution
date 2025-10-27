import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_XGBoost import run_xgboost_pipeline_final
from model_XGBoost_test import run_xgboost_pipeline_test

if __name__ == "__main__":
    TESTING = True
    if TESTING:
        run_xgboost_pipeline_test(
        train_path="train.jsonl",
        test_path="test.jsonl",
        output_csv="submission.csv"
    )
    else:
        run_xgboost_pipeline_final(
        train_path="train.jsonl",
        test_path="test.jsonl",
        output_csv="submission.csv"
    )