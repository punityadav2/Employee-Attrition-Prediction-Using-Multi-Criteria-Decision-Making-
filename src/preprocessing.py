"""
Preprocessing pipeline for IBM HR Employee Attrition dataset.

Steps covered:
  1. Encode target  (Yes/No -> 1/0)
  2. Drop constant columns
  3. Handle missing values
  4. Encode categorical features (Label Encoding)
  5. Stratified 80:20 train-test split
  6. StandardScaler on numerical features
  7. Random Oversampling (ROS) on training data ONLY
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ---------------------------------------------------------------------------
# Step 1  Encode target
# ---------------------------------------------------------------------------
def encode_target(df):
    """Convert Attrition Yes/No to 1/0."""
    df = df.copy()
    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].map({'Yes': 1, 'No': 0})
    print(f"[OK] Target encoded  -> Attrition Yes=1 ({df[config.TARGET_COLUMN].sum()}), "
          f"No=0 ({(df[config.TARGET_COLUMN]==0).sum()})")
    return df


# ---------------------------------------------------------------------------
# Step 2  Drop constant / useless columns
# ---------------------------------------------------------------------------
def drop_constant_columns(df):
    """Drop EmployeeCount, Over18, StandardHours, EmployeeNumber."""
    df = df.copy()
    cols_to_drop = [c for c in config.DROP_COLUMNS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"[OK] Dropped constant columns: {cols_to_drop}")
    return df


# ---------------------------------------------------------------------------
# Step 3  Missing values
# ---------------------------------------------------------------------------
def handle_missing_values(df):
    """Impute: mode for categoricals, median for numerics."""
    df = df.copy()
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) == 0:
        print("[OK] No missing values found.")
        return df

    for col in missing.index:
        if df[col].dtype == 'object':
            fill = df[col].mode()[0]
            df[col].fillna(fill, inplace=True)
        else:
            fill = df[col].median()
            df[col].fillna(fill, inplace=True)
        print(f"  Filled '{col}' with {fill}")

    return df


# ---------------------------------------------------------------------------
# Step 4  Encode categorical features
# ---------------------------------------------------------------------------
def encode_categorical(df, fit=True, encoders=None):
    """
    Label-encode all object columns.

    Parameters
    ----------
    df       : input dataframe
    fit      : True = fit new encoders; False = use provided encoders
    encoders : dict of pre-fitted LabelEncoders (for inference)

    Returns
    -------
    df_encoded, encoders_dict
    """
    df = df.copy()
    if encoders is None:
        encoders = {}

    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Never encode the target here (already numeric)
    cat_cols = [c for c in cat_cols if c != config.TARGET_COLUMN]

    print(f"[OK] Encoding {len(cat_cols)} categorical columns...")
    for col in cat_cols:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            if col in encoders:
                le = encoders[col]
                df[col] = df[col].apply(
                    lambda x: le.transform([str(x)])[0]
                    if str(x) in le.classes_ else -1
                )
            else:
                print(f"  [!] No encoder for '{col}', filling -1")
                df[col] = -1

    return df, encoders


# ---------------------------------------------------------------------------
# Step 5  Train / test split (stratified 80:20)
# ---------------------------------------------------------------------------
def split_data(df):
    """Stratified 80:20 train-test split. Returns X_train, X_test, y_train, y_test."""
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    print(f"[OK] Stratified split: train={len(X_train)} | test={len(X_test)}")
    print(f"  Train attrition rate: {y_train.mean()*100:.1f}%")
    print(f"  Test  attrition rate: {y_test.mean()*100:.1f}%")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Step 6  Scale numerical features
# ---------------------------------------------------------------------------
def scale_features(X_train, X_test=None, scaler=None):
    """
    Fit StandardScaler on X_train, transform both splits.

    Returns X_train_scaled, X_test_scaled (or None), scaler
    """
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    if scaler is None:
        scaler = StandardScaler()
        X_train = X_train.copy()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        print(f"[OK] Scaled {len(num_cols)} numerical features (StandardScaler)")
    else:
        X_train = X_train.copy()
        X_train[num_cols] = scaler.transform(X_train[num_cols])

    if X_test is not None:
        X_test = X_test.copy()
        X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test, scaler


# ---------------------------------------------------------------------------
# Step 7  Random Oversampling (training data ONLY)
# ---------------------------------------------------------------------------
def apply_random_oversampling(X_train, y_train):
    """
    Balance class distribution using RandomOverSampler.
    Applied ONLY to training data  test data is never touched.
    """
    ros = RandomOverSampler(random_state=config.RANDOM_STATE)
    X_res, y_res = ros.fit_resample(X_train, y_train)

    print(f"[OK] Random Oversampling applied:")
    print(f"  Before -> Attrition=1: {y_train.sum():>4} | Attrition=0: {(y_train==0).sum():>4}")
    print(f"  After  -> Attrition=1: {y_res.sum():>4} | Attrition=0: {(y_res==0).sum():>4}")

    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res, name=config.TARGET_COLUMN)


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------
def full_preprocessing_pipeline(df, fit=True, encoders=None, scaler=None):
    """
    Run all preprocessing steps in sequence.

    Returns
    -------
    X_train_ros, X_test, y_train_ros, y_test, encoders, scaler
    """
    print("\n" + "=" * 70)
    print("  STEP 1-3: PREPROCESSING")
    print("=" * 70)

    df = encode_target(df)
    df = drop_constant_columns(df)
    df = handle_missing_values(df)
    df, encoders = encode_categorical(df, fit=fit, encoders=encoders)

    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test, scaler = scale_features(X_train, X_test, scaler=scaler)
    X_train_ros, y_train_ros = apply_random_oversampling(X_train, y_train)

    return X_train_ros, X_test, y_train_ros, y_test, encoders, scaler
