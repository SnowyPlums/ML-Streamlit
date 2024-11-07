import pandas as pd
from sklearn.impute import KNNImputer
from typing import Optional

# Mean Imputation
def mean_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values with the mean of each column."""
    imputed_df = df.copy()
    return imputed_df.fillna(imputed_df.mean())

# Median Imputation
def median_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values with the median of each column."""
    imputed_df = df.copy()
    return imputed_df.fillna(imputed_df.median())

# Mode Imputation
def mode_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values with the mode (most frequent value) of each column."""
    imputed_df = df.copy()
    for column in imputed_df.columns:
        mode_val = imputed_df[column].mode()[0]
        imputed_df[column].fillna(mode_val, inplace=True)
    return imputed_df

# K-Nearest Neighbors (KNN) Imputation
def knn_imputation(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    """Impute missing values using K-Nearest Neighbors."""
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_array = imputer.fit_transform(df)
    return pd.DataFrame(imputed_array, columns=df.columns)

# Constant Imputation
def constant_imputation(df: pd.DataFrame, fill_value: Optional[float] = 0) -> pd.DataFrame:
    """Impute missing values with a constant value (default is 0)."""
    imputed_df = df.copy()
    return imputed_df.fillna(fill_value)

def drop_na(df: pd.DataFrame) -> pd.DataFrame:
    imputed_df = df.copy()
    return imputed_df.dropna()