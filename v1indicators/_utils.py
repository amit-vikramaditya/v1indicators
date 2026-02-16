import pandas as pd
import numpy as np
from typing import Optional

def validate_series(data: pd.Series, name: str = "data") -> np.ndarray:
    """
    Validates that input is a Pandas Series and returns its numpy values (float64).
    Ensures data is numeric and not empty.
    """
    if not isinstance(data, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    
    if data.empty:
        raise ValueError(f"{name} is empty")
    
    # Ensure we are working with float64 for precision
    values = data.to_numpy(dtype=np.float64)
    return values

def check_series(data: pd.Series, name: str = "data") -> pd.Series:
    """
    Validates that input is a Pandas Series, converts to float64, and returns the Series.
    Ensures data is not empty.
    """
    if not isinstance(data, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    
    if data.empty:
        raise ValueError(f"{name} is empty")
    
    return data.astype(np.float64)

def validate_df(df: pd.DataFrame, name: str = "dataframe") -> pd.DataFrame:
    """
    Validates that input is a Pandas DataFrame and not empty.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError(f"{name} is empty")
        
    return df

def to_series(data: np.ndarray, index: pd.Index, name: Optional[str] = None) -> pd.Series:
    """Wraps numpy array back into a Pandas Series with the given index."""
    return pd.Series(data, index=index, name=name)