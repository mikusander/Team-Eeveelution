import pandas as pd
from sklearn.model_selection import StratifiedKFold

class TargetEncoder:
    def __init__(self, cols, n_splits=5, random_state=42):
        self.cols = cols
        self.n_splits = n_splits
        self.random_state = random_state
        self.encoders = {}

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        X_encoded = X.copy()
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for col in self.cols:
            global_mean = y.mean()
            self.encoders[col] = {'global_mean': global_mean}
            
            X_encoded[f'{col}_te'] = 0.0
            
            for train_idx, val_idx in skf.split(X, y):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_val = X.iloc[val_idx]
                
                means = y_train.groupby(X_train[col]).mean()
                X_encoded.iloc[val_idx, X_encoded.columns.get_loc(f'{col}_te')] = X_val[col].map(means).fillna(global_mean)

            self.encoders[col]['means'] = y.groupby(X[col]).mean()
        
        return X_encoded

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_encoded = X.copy()
        for col in self.cols:
            global_mean = self.encoders[col]['global_mean']
            means = self.encoders[col]['means']
            X_encoded[f'{col}_te'] = X[col].map(means).fillna(global_mean)
        return X_encoded
