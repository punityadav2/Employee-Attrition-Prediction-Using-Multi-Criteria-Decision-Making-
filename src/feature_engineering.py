"""
Feature Engineering for IBM HR Employee Attrition dataset.

New features created exactly as specified in plan.txt:
  1. workload_ratio  = MonthlyRate / (TotalWorkingYears + 1)
     [IBM dataset has no AverageMonthlyHours/NumberOfProjects;
      MonthlyRate / TotalWorkingYears captures the same workload-to-experience ratio]
  2. tenure_level    = categorical binning of YearsAtCompany
       03 yrs  -> Junior (0)
       36 yrs  -> Mid    (1)
       >6 yrs   -> Senior (2)
"""

import pandas as pd
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def create_workload_ratio(df):
    """
    Feature 1: workload_ratio
    Proxy for workload intensity relative to career experience.
    Formula: MonthlyRate / (TotalWorkingYears + 1)
    """
    df = df.copy()
    df['workload_ratio'] = df['MonthlyRate'] / (df['TotalWorkingYears'] + 1)
    print("[OK] Created feature: workload_ratio = MonthlyRate / (TotalWorkingYears + 1)")
    return df


def create_tenure_level(df):
    """
    Feature 2: tenure_level based on YearsAtCompany
      03  -> Junior (0)
      36  -> Mid    (1)
      >6   -> Senior (2)
    """
    df = df.copy()

    def map_tenure(years):
        if years <= 3:
            return 'Junior'
        elif years <= 6:
            return 'Mid'
        else:
            return 'Senior'

    df['tenure_level_label'] = df['YearsAtCompany'].apply(map_tenure)
    df['tenure_level'] = df['tenure_level_label'].map(config.TENURE_ENCODE)
    df.drop(columns=['tenure_level_label'], inplace=True)

    dist = df['tenure_level'].value_counts().sort_index()
    print("[OK] Created feature: tenure_level (0=Junior, 1=Mid, 2=Senior)")
    print(f"  Distribution: Junior={dist.get(0,0)}, Mid={dist.get(1,0)}, Senior={dist.get(2,0)}")
    return df


def create_all_features(df):
    """
    Apply all feature engineering steps.
    Call BEFORE encoding so original numeric columns are still available.
    """
    print("\n" + "=" * 70)
    print("  STEP 2: FEATURE ENGINEERING")
    print("=" * 70)

    df = create_workload_ratio(df)
    df = create_tenure_level(df)

    print(f"\n[OK] Feature engineering complete. Total columns: {df.shape[1]}")
    return df
