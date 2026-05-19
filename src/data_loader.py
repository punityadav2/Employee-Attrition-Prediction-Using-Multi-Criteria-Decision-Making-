"""
Data loader for IBM HR Employee Attrition dataset.
"""

import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_attrition_data(filepath=None):
    """
    Load the IBM HR Employee Attrition dataset.

    Returns
    -------
    pd.DataFrame   raw dataframe with original columns intact
    """
    if filepath is None:
        filepath = config.ATTRITION_FILE

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at: {filepath}\n"
            "Please place 'WA_Fn-UseC_-HR-Employee-Attrition.csv' in data/raw/"
        )

    df = pd.read_csv(filepath)
    print("=" * 70)
    print("  IBM HR EMPLOYEE ATTRITION  DATASET LOADED")
    print("=" * 70)
    print(f"  Shape          : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Target column  : {config.TARGET_COLUMN}")

    # Class distribution
    attrition_counts = df[config.TARGET_COLUMN].value_counts()
    total = len(df)
    for label, count in attrition_counts.items():
        print(f"  {label:<6}: {count:>4}  ({count/total*100:.1f}%)")

    print(f"\n  Missing values : {df.isnull().sum().sum()}")
    print("=" * 70)
    return df


def quick_eda(df):
    """Print a concise EDA summary."""
    print("\n--- Column dtypes ---")
    print(df.dtypes.value_counts().to_string())

    print("\n--- Numeric stats (sample) ---")
    print(df.describe().T[['mean', 'std', 'min', 'max']].round(2).to_string())

    cat_cols = df.select_dtypes(include='object').columns.tolist()
    print(f"\n--- Categorical columns ({len(cat_cols)}) ---")
    for col in cat_cols:
        print(f"  {col}: {df[col].unique()[:5]}")


if __name__ == "__main__":
    df = load_attrition_data()
    quick_eda(df)
