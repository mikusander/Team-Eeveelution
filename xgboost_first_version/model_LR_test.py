import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from preprocessing import build_ml_dataframe

def run_lr_pipeline(train_path: str, test_path: str, output_csv: str):
    df_train_full = build_ml_dataframe(train_path, save_path='train_features.csv')
    df_test = build_ml_dataframe(test_path, is_train=False, save_path='test_features.csv')

    TARGET = "player_won"
    ID_COL = "battle_id"
    cols_to_drop_for_training = [TARGET, ID_COL]

    X_train_full = df_train_full.drop(columns=cols_to_drop_for_training)
    y_train_full = df_train_full[TARGET]
    X_test_final = df_test.drop(columns=[col for col in [ID_COL] if col in df_test.columns])

    X_train_main, X_holdout, y_train_main, y_holdout = train_test_split(
        X_train_full, y_train_full, 
        test_size=0.20, 
        random_state=42, 
        stratify=y_train_full 
    )

    X_train_top = X_train_main
    X_holdout_top = X_holdout
    X_test_top = X_test_final

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_accuracies = []
    holdout_preds_ensemble = np.zeros(X_holdout_top.shape[0])
    test_preds_ensemble = np.zeros(X_test_top.shape[0])

    for train_idx, val_idx in skf.split(X_train_top, y_train_main):
        X_tr, X_val = X_train_top.iloc[train_idx], X_train_top.iloc[val_idx]
        y_tr, y_val = y_train_main.iloc[train_idx], y_train_main.iloc[val_idx]

        model_lr = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42, n_jobs=-1)
        model_lr.fit(X_tr, y_tr)

        y_val_pred = model_lr.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        cv_accuracies.append(val_acc)

        holdout_preds_ensemble += model_lr.predict_proba(X_holdout_top)[:, 1]
        test_preds_ensemble += model_lr.predict_proba(X_test_top)[:, 1]

    mean_cv_acc = np.mean(cv_accuracies)
    holdout_preds_ensemble /= skf.n_splits
    test_preds_ensemble /= skf.n_splits

    holdout_labels = (holdout_preds_ensemble > 0.5).astype(int)
    holdout_accuracy = accuracy_score(y_holdout, holdout_labels)

    final_labels = (test_preds_ensemble > 0.5).astype(int)
    submission = pd.DataFrame({"battle_id": df_test[ID_COL], "player_won": final_labels})
    submission.to_csv(output_csv, index=False)

    # Addestramento su tutto il training set
    model_final = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42, n_jobs=-1)
    model_final.fit(X_train_full, y_train_full)
    y_train_full_pred = model_final.predict(X_train_full)
    total_acc = accuracy_score(y_train_full, y_train_full_pred)

    with open("lr_accuracy.txt", "w") as f:
        f.write(f"Mean CV Accuracy: {mean_cv_acc:.4f}\n")
        f.write(f"Holdout Accuracy: {holdout_accuracy:.4f}\n")
        f.write(f"Total Training Accuracy: {total_acc:.4f}\n")
