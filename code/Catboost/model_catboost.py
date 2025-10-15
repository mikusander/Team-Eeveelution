import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import shap
from preprocessing import build_ml_dataframe, target_encode, apply_target_encoding

def run_catboost_pipeline(train_path: str, test_path: str, output_csv: str):
    # 1. Build ML DataFrames
    df_train = build_ml_dataframe(train_path)
    df_test  = build_ml_dataframe(test_path)

    # 2. Preparazione target e ID
    TARGET = "player_won"
    ID_COL = "battle_id"
    X_train = df_train.drop(columns=[TARGET, ID_COL])
    y_train = df_train[TARGET]
    X_test  = df_test.drop(columns=[ID_COL])

    # 3. Target encoding
    categorical_cols = ['tl_p1_last_active','tl_p2_last_active','p1_lead_name','p2_lead_name','last_pair']
    for col in categorical_cols:
        if col in X_train.columns:
            enc_map, prior = target_encode(X_train[col], y_train)
            X_train = apply_target_encoding(X_train, col, enc_map, prior)
            X_test  = apply_target_encoding(X_test, col, enc_map, prior)
    X_train = X_train.drop(columns=[c for c in categorical_cols if c in X_train.columns])
    X_test  = X_test.drop(columns=[c for c in categorical_cols if c in X_test.columns])

    # 4. Calcolo SHAP, top features e CV
    # Qui va esattamente il tuo codice di CatBoost + SHAP + StratifiedKFold
    # senza cambiare nulla, usando X_train, y_train, X_test
    # ---------------------------
    # Calcolo SHAP e selezione top feature
    # ---------------------------
    print("\nCalcolo SHAP values e selezione top feature...")
    model_cb_final = CatBoostClassifier(
        iterations=3000,
        depth=5,
        learning_rate=0.02,
        l2_leaf_reg=22,
        rsm=0.4,
        bagging_temperature=1.5,
        eval_metric='Accuracy',
        random_seed=42,
        verbose=0,
        early_stopping_rounds=200,
    )
    model_cb_final.fit(X_train, y_train) 
    explainer = shap.TreeExplainer(model_cb_final)
    shap_values = explainer.shap_values(X_train)

    # Se il modello è binario, prendiamo solo la classe positiva
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]

    # Importanza media assoluta
    shap_mean_abs = np.abs(shap_values).mean(axis=0)
    feat_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': shap_mean_abs
    }).sort_values(by='importance', ascending=False)

    # Top N features (puoi cambiare N)
    top_n = 20
    top_features = feat_importance['feature'].head(top_n).tolist()
    print(f"\nTop {top_n} features:\n", top_features)

    # Nuovo dataset con solo top feature
    X_train_top = X_train[top_features]
    X_test_top  = X_test[top_features]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cat_fold_acc_top = []
    cat_test_preds_top = np.zeros(X_test_top.shape[0])

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_top, y_train)):
        print(f"\n--- CatBoost Top Features Fold {fold+1} ---")
        X_tr, X_val = X_train_top.iloc[train_idx], X_train_top.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        """ model_cb = CatBoostClassifier(
            iterations=3000,
            depth=7,
            learning_rate=0.010282576218851393,
            l2_leaf_reg=36,
            rsm=0.34866409002293647,
            bagging_temperature=0.7004868652076222,
            eval_metric='Accuracy',
            random_seed=42,
            verbose=0,
            early_stopping_rounds=200,
        ) """
        model_cb = CatBoostClassifier(
            iterations=3000,
            depth=7,
            learning_rate=0.01,
            l2_leaf_reg=36,
            rsm=0.34,
            bagging_temperature=0.7,
            eval_metric='Accuracy',
            random_seed=42,
            verbose=0,
            early_stopping_rounds=200,
        )
        model_cb.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        # Train accuracy
        y_train_pred = model_cb.predict(X_tr)
        train_acc = accuracy_score(y_tr, y_train_pred)

        # Validation accuracy
        y_val_pred = model_cb.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)

        print(f"Fold {fold+1} → Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")
        # Accuracy
        y_val_pred = model_cb.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        print(f"Fold {fold+1} Accuracy: {val_acc:.4f}")
        cat_fold_acc_top.append(val_acc)

        # Accumulo predizioni test
        y_test_pred = model_cb.predict_proba(X_test_top)[:,1]
        cat_test_preds_top += y_test_pred

    mean_acc_top = np.mean(cat_fold_acc_top)
    print(f"\nCatBoost Mean CV Accuracy (Top {top_n} features): {mean_acc_top:.4f}")

    # ---------------------------
    # Fit finale su tutto il training set con top features
    # ---------------------------
    model_cb_final = CatBoostClassifier(
        iterations=3000,
        depth=5,
        learning_rate=0.02,
        l2_leaf_reg=22,
        rsm=0.4,
        bagging_temperature=1.5,
        eval_metric='Accuracy',
        random_seed=42,
        verbose=0,
        early_stopping_rounds=200,
    )
    model_cb_final.fit(X_train_top, y_train)

    # Predizioni finali su test
    cat_test_preds_top /= skf.n_splits
    final_labels = (cat_test_preds_top > 0.5).astype(int)



    # 5. Submission
    submission = pd.DataFrame({
        "battle_id": df_test[ID_COL],
        "player_won": final_labels
    })
    submission.to_csv(output_csv, index=False)
    print(f"\nSubmission file created: {output_csv}")