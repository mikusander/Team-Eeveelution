import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import shap
from preprocessing import build_ml_dataframe 

def run_catboost_pipeline(train_path: str, test_path: str, output_csv: str):
    
    df_train_full = build_ml_dataframe(train_path, save_path='train_features.csv')
    df_test = build_ml_dataframe(test_path, is_train=False, save_path='test_features.csv')

    TARGET = "player_won"
    ID_COL = "battle_id"
    X_train_full = df_train_full.drop(columns=[TARGET, ID_COL])
    y_train_full = df_train_full[TARGET]
    X_test_final = df_test.drop(columns=[ID_COL])

    CATEGORICAL_COLS = ['tl_p1_last_active','tl_p2_last_active','p1_lead_name','p2_lead_name','last_pair']
    for col in CATEGORICAL_COLS:
        if col in X_train_full.columns:
            X_train_full[col] = X_train_full[col].astype(str)
            X_test_final[col] = X_test_final[col].astype(str)

    CAT_FEATURES = [c for c in CATEGORICAL_COLS if c in X_train_full.columns]

    print("\nCalcolo SHAP values e selezione top feature...")
    model_cb_shap = CatBoostClassifier(iterations=1000, depth=5, learning_rate=0.05, eval_metric='Accuracy', random_seed=42, verbose=0, cat_features=CAT_FEATURES)
    model_cb_shap.fit(X_train_full, y_train_full)
    
    explainer = shap.TreeExplainer(model_cb_shap)
    shap_values = explainer.shap_values(X_train_full)

    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]

    shap_mean_abs = np.abs(shap_values).mean(axis=0)
    feat_importance = pd.DataFrame({'feature': X_train_full.columns, 'importance': shap_mean_abs}).sort_values(by='importance', ascending=False)

    top_n = 35
    top_features = feat_importance['feature'].head(top_n).tolist()
    print(f"\nTop {top_n} features basate su SHAP:\n", top_features) 
    
    X_train_top = X_train_full[top_features]
    X_test_top  = X_test_final[top_features]
    
    cv_cat_features = [f for f in CAT_FEATURES if f in top_features]

    """ OPTIMAL_PARAMS = {
        'iterations': 1924,
        'depth': 5,
        'learning_rate': 0.038756425850563964,
        'l2_leaf_reg': 2.63350046084044,
        'rsm': 0.8600441898727658,
        'bagging_temperature': 0.1916032066181844,
    } """ #0.8073

    OPTIMAL_PARAMS = {
        'iterations': 2848,
        'depth': 5,
        'learning_rate': 0.07237695803152867,
        'l2_leaf_reg': 21.174775148438027,
        'rsm': 0.8404794027781933,
        'bagging_temperature': 0.7580211003293981,
    } #0.8113

    """ OPTIMAL_PARAMS = {
        'iterations': 2717,
        'depth': 5,
        'learning_rate': 0.030143919645181167,
        'l2_leaf_reg': 2.964573861262902,
        'rsm': 0.8549333864720773,
        'bagging_temperature': 0.5328732564135111,
    } """ # 0.8046

    print("\n--- AVVIO CROSS-VALIDATION SUL TRAINING SET ---")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_accuracies = []
    test_preds_ensemble = np.zeros(X_test_top.shape[0])

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_top, y_train_full)):
        print(f"\n--- Fold {fold+1} ---")
        X_tr, X_val = X_train_top.iloc[train_idx], X_train_top.iloc[val_idx]
        y_tr, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        final_params = OPTIMAL_PARAMS.copy()
        final_params.update({'eval_metric': 'Accuracy', 'random_seed': 42, 'verbose': 0, 'early_stopping_rounds': 200, 'cat_features': cv_cat_features})

        model_cb = CatBoostClassifier(**final_params)
        model_cb.fit(X_tr, y_tr, eval_set=(X_val, y_val)) 
        
        y_val_pred = model_cb.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        cv_accuracies.append(val_acc)
        
        print(f"Fold {fold+1} → Validation Accuracy: {val_acc:.4f}")
        test_preds_ensemble += model_cb.predict_proba(X_test_top)[:, 1]

    test_preds_ensemble /= skf.n_splits
    mean_cv_acc = np.mean(cv_accuracies)

    print(f"\n--- RISULTATI FINALE ---")
    print(f"Mean CV Accuracy: {mean_cv_acc:.4f}")

    final_labels = (test_preds_ensemble > 0.5).astype(int)
    submission = pd.DataFrame({"battle_id": df_test[ID_COL], "player_won": final_labels})
    submission.to_csv(output_csv, index=False)
