import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os


def load_and_clean(filepath):
    """Load SECOM CSV, drop junk columns, impute missing values."""
    df = pd.read_csv(filepath)
    df = df.drop(columns=['Time'])

    # drop features where more than half the values are missing
    missing_pct = df.isnull().sum() / len(df)
    bad_cols = missing_pct[missing_pct > 0.5].index.tolist()
    # don't accidentally drop the label column
    bad_cols = [c for c in bad_cols if c != 'Pass/Fail']
    df = df.drop(columns=bad_cols)

    print(f"Dropped {len(bad_cols)} columns with >50% missing")
    print(f"Remaining features: {len(df.columns) - 1}")

    # fill remaining NaNs with column median
    feature_cols = [c for c in df.columns if c != 'Pass/Fail']
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    return df


def prepare_features(df, scaler_path=None):
    """Split into X/y, convert labels to 0/1, scale features.

    Returns X (scaled numpy array), y (numpy array), and the fitted scaler.
    Saves scaler to disk if scaler_path is provided.
    """
    X = df.drop(columns=['Pass/Fail']).values
    y = df['Pass/Fail'].values

    # convert from -1/1 to 0/1 (0 = pass, 1 = fail)
    y = (y == 1).astype(int)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")

    return X, y, scaler


if __name__ == '__main__':
    df = load_and_clean('../data/uci-secom.csv')
    X, y, scaler = prepare_features(df, scaler_path='../models/scaler.pkl')
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Class distribution: pass={sum(y==0)}, fail={sum(y==1)}")
