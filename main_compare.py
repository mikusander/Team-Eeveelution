from model_XGBoost_test import run_xgboost_pipeline_test
from model_RF_test import run_rf_pipeline
from model_LR_test import run_lr_pipeline

def read_accuracy_file(filepath):
    try:
        with open(filepath, "r") as f:
            return f.read().strip()
    except Exception:
        return "File not found or error."

def main():
    train_path = "train.jsonl"
    test_path = "test.jsonl"

    print("== XGBoost ==")
    run_xgboost_pipeline_test(train_path, test_path, "xgb_submission.csv")
    print("== RandomForest ==")
    run_rf_pipeline(train_path, test_path, "rf_submission.csv")
    print("== LogisticRegression ==")
    run_lr_pipeline(train_path, test_path, "lr_submission.csv")

    print("\n=== MODEL ACCURACY COMPARISON ===")
    print("\n[XGBoost]\n" + read_accuracy_file("total_accuracy.txt"))
    print("\n[RandomForest]\n" + read_accuracy_file("rf_accuracy.txt"))
    print("\n[LogisticRegression]\n" + read_accuracy_file("lr_accuracy.txt"))

if __name__ == "__main__":
    main()
