# Team Eeveelution: Pokémon Battle Prediction

This repository contains the code for a machine learning pipeline designed to predict the outcome of Pokémon battles.

The framework utilizes **CatBoost**, **LightGBM**, and **XGBoost** as its core models, and is explicitly designed to produce submissions and files suitable for **ensembling** (stacking/blending).

## Repository Guide

Below is a brief description of the main files and folders and their purpose.

### Main Folders

* **/CatBoost_Data_Pipeline**:
    This is the output folder for all intermediate `.csv` files. It contains the processed data, engineered features, and split datasets (train/validation/holdout) ready to be used by the CatBoost model.

* **/*_Model_Outputs**:
    Saves all analysis results for each model. For CatBoost, it includes diagnostic plots (e.g., Feature Importance, ROC/AUC, Confusion Matrix), classification reports, and the `.json` files with optimal parameters and iterations. For all others, it includes the preprocessing_medians and final_features.json.

* **/Meta_Model**:
    Saves the models that generated the stacking, including both LogReg and the BEST_model (chosen from LGBM, LogReg, Ridge, VotingSoft, VotingHard).

* **/OOF_Predictions**:
    A crucial folder for ensembling. It contains the final predictions for both the training set (Out-of-Fold) and the test set, saved as `.npy` files.

* **/Submissions**:
    Contains the `submission.csv` files, both those useful for blending (based on probabilities) and the final blended and stacked ones.

### Main Files

* `utils_*.py`:
    The "brain" of the entire pipeline. These files contain **all the logic**, utility functions, paths, and definitions for each phase of data loading, feature engineering, validation, training, and blending/stacking.

* `Report_Team_Eeveelution.pdf`:
    The PDF report file related to the challenge.
