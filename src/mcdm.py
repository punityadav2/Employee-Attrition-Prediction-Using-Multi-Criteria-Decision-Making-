"""
MCDM Module  Step 9 & 10

Implements three Multi-Criteria Decision Making methods:
  1. WSM   Weighted Sum Method
  2. TOPSIS  Technique for Order of Preference by Similarity to Ideal Solution
  3. VIKOR   VlseKriterijumska Optimizacija I Kompromisno Resenje

Then aggregates all three rankings using:
  4. RPM   Rank Position Method (final model selection)
"""

import numpy as np
import pandas as pd
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ===========================================================================
# 1. WSM  Weighted Sum Method
# ===========================================================================
def wsm(decision_matrix: pd.DataFrame) -> pd.Series:
    """
    Weighted Sum Method.

    Steps:
      1. Normalize each criterion (min-max 0->1)
         - Benefit: (x - min) / (max - min)
         - Cost   : (max - x) / (max - min)   [invert so higher = better]
      2. Multiply by equal weights
      3. Sum across all criteria -> WSM score

    Returns
    -------
    pd.Series  WSM scores (higher is better), indexed by model name
    """
    dm = decision_matrix.copy().astype(float)
    norm = pd.DataFrame(index=dm.index, columns=dm.columns, dtype=float)

    for col in dm.columns:
        col_min = dm[col].min()
        col_max = dm[col].max()
        rng = col_max - col_min if (col_max - col_min) != 0 else 1e-9

        if col in config.BENEFIT_CRITERIA:
            norm[col] = (dm[col] - col_min) / rng
        else:  # cost criterion  invert
            norm[col] = (col_max - dm[col]) / rng

    weights = np.array([config.MCDM_WEIGHTS[c] for c in dm.columns])
    wsm_scores = norm.values @ weights
    return pd.Series(wsm_scores, index=dm.index, name='WSM_Score')


# ===========================================================================
# 2. TOPSIS
# ===========================================================================
def topsis(decision_matrix: pd.DataFrame) -> pd.Series:
    """
    TOPSIS  Technique for Order Preference by Similarity to Ideal Solution.

    Steps:
      1. Vector normalise: r_ij = x_ij / sqrt(sum x_kj)
      2. Weighted normalised matrix
      3. Ideal A+ (max benefit, min cost) and Negative-ideal A (min benefit, max cost)
      4. Euclidean distance to A+ (D+) and A (D)
      5. Relative closeness C = D / (D+ + D)
         -> rank by descending C

    Returns
    -------
    pd.Series  TOPSIS closeness scores (higher is better)
    """
    dm = decision_matrix.copy().astype(float)
    weights = np.array([config.MCDM_WEIGHTS[c] for c in dm.columns])

    # Step 1+2: Weighted vector normalisation
    norms = np.sqrt((dm.values ** 2).sum(axis=0))
    norms[norms == 0] = 1e-9
    v = (dm.values / norms) * weights

    # Step 3: Ideal & negative-ideal
    a_pos = np.zeros(v.shape[1])
    a_neg = np.zeros(v.shape[1])
    for j, col in enumerate(dm.columns):
        if col in config.BENEFIT_CRITERIA:
            a_pos[j] = v[:, j].max()
            a_neg[j] = v[:, j].min()
        else:
            a_pos[j] = v[:, j].min()
            a_neg[j] = v[:, j].max()

    # Step 4: Euclidean distances
    d_pos = np.sqrt(((v - a_pos) ** 2).sum(axis=1))
    d_neg = np.sqrt(((v - a_neg) ** 2).sum(axis=1))

    # Step 5: Relative closeness
    closeness = d_neg / (d_pos + d_neg + 1e-9)
    return pd.Series(closeness, index=dm.index, name='TOPSIS_Score')


# ===========================================================================
# 3. VIKOR
# ===========================================================================
def vikor(decision_matrix: pd.DataFrame, v: float = None) -> pd.Series:
    """
    VIKOR  compromise ranking method.

    Steps:
      1. Best f* and worst f for each criterion
      2. Utility  S_i =  w_j * (f*_j - f_ij) / (f*_j - f-_j)
         Regret  R_i = max_j [ w_j * (f*_j - f_ij) / (f*_j - f-_j) ]
      3. Q_i = v*(S-S*)/(S--S*) + (1-v)*(R-R*)/(R--R*)
         where S* = min(S), S- = max(S), R* = min(R), R- = max(R), v=0.5
      4. Rank by ascending Q (lower Q = better)

    Returns
    -------
    pd.Series  Q values (lower is better)  we return 1Q for consistent "higher=better"
    """
    if v is None:
        v = config.VIKOR_V

    dm = decision_matrix.copy().astype(float)
    weights = np.array([config.MCDM_WEIGHTS[c] for c in dm.columns])

    f_star = np.zeros(dm.shape[1])
    f_neg  = np.zeros(dm.shape[1])
    for j, col in enumerate(dm.columns):
        if col in config.BENEFIT_CRITERIA:
            f_star[j] = dm.iloc[:, j].max()
            f_neg[j]  = dm.iloc[:, j].min()
        else:
            f_star[j] = dm.iloc[:, j].min()
            f_neg[j]  = dm.iloc[:, j].max()

    # Normalised differences
    denom = f_star - f_neg
    denom[denom == 0] = 1e-9
    diff = (f_star - dm.values) / denom   # shape (n_models, n_criteria)
    # For cost criteria f* is the min, f- is max -> diff is already correct (min-x)/(min-max)
    # But cost criteria need to be "inverted" in the same sense as benefit:
    for j, col in enumerate(dm.columns):
        if col in config.COST_CRITERIA:
            diff[:, j] = (dm.values[:, j] - f_star[j]) / denom[j]

    weighted_diff = weights * diff
    S = weighted_diff.sum(axis=1)           # utility
    R = weighted_diff.max(axis=1)           # regret

    S_star, S_neg = S.min(), S.max()
    R_star, R_neg = R.min(), R.max()

    dS = (S_neg - S_star) if (S_neg - S_star) != 0 else 1e-9
    dR = (R_neg - R_star) if (R_neg - R_star) != 0 else 1e-9

    Q = v * (S - S_star) / dS + (1 - v) * (R - R_star) / dR

    # Return inverted so higher = better (consistent with WSM/TOPSIS)
    Q_series = pd.Series(Q, index=dm.index, name='VIKOR_Q')
    return Q_series   # raw Q  lower is better (will be ranked separately)


# ===========================================================================
# 4. RPM  Rank Position Method (aggregated final ranking)
# ===========================================================================
def rank_position_method(decision_matrix: pd.DataFrame):
    """
    Aggregate WSM, TOPSIS, VIKOR rankings using the Rank Position Method.

    - WSM   : rank by descending score
    - TOPSIS: rank by descending score
    - VIKOR : rank by ascending Q (lower Q = better)

    RPM score = WSM_rank + TOPSIS_rank + VIKOR_rank
    Final rank = ascending RPM score (lowest sum = best model)

    Returns
    -------
    rankings_df : pd.DataFrame with scores, individual ranks, RPM, and Final_Rank
    final_model : str   name of the best model
    """
    print("\n" + "=" * 70)
    print("  STEP 9: MCDM RANKINGS")
    print("=" * 70)

    wsm_scores     = wsm(decision_matrix)
    topsis_scores  = topsis(decision_matrix)
    vikor_q        = vikor(decision_matrix)

    df = pd.DataFrame({
        'WSM_Score'    : wsm_scores,
        'TOPSIS_Score' : topsis_scores,
        'VIKOR_Q'      : vikor_q,
    })

    # Ranks (1 = best)
    df['WSM_Rank']    = df['WSM_Score'].rank(ascending=False).astype(int)
    df['TOPSIS_Rank'] = df['TOPSIS_Score'].rank(ascending=False).astype(int)
    df['VIKOR_Rank']  = df['VIKOR_Q'].rank(ascending=True).astype(int)

    # RPM based on simple aggregation formula
    df['RPM_Score'] = df['WSM_Rank'] + df['TOPSIS_Rank'] + df['VIKOR_Rank']
    df['Final_Rank'] = df['RPM_Score'].rank(ascending=True).astype(int)

    df = df.sort_values('Final_Rank')

    print("\n  WSM Ranking ((up) score is better):")
    for m, r in df['WSM_Rank'].sort_values().items():
        print(f"    Rank {r}: {m}  (score={df.loc[m,'WSM_Score']:.4f})")

    print("\n  TOPSIS Ranking ((up) closeness is better):")
    for m, r in df['TOPSIS_Rank'].sort_values().items():
        print(f"    Rank {r}: {m}  (score={df.loc[m,'TOPSIS_Score']:.4f})")

    print("\n  VIKOR Ranking ((down) Q is better):")
    for m, r in df['VIKOR_Rank'].sort_values().items():
        print(f"    Rank {r}: {m}  (Q={df.loc[m,'VIKOR_Q']:.4f})")

    print("\n" + "=" * 70)
    print("  STEP 10: FINAL MODEL SELECTION (RPM)")
    print("=" * 70)
    print(df[['WSM_Rank', 'TOPSIS_Rank', 'VIKOR_Rank', 'RPM_Score', 'Final_Rank']].to_string())

    final_model = df['Final_Rank'].idxmin()
    print(f"\n   FINAL SELECTED MODEL: {final_model}")
    print("=" * 70)

    return df, final_model
