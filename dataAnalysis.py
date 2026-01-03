"""
Module for viewing and inspecting dataset characteristics.
Provides functions to analyze and display dataset properties.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union


def inspect_dataset(df: pd.DataFrame, verbose: bool = True) -> Dict:
    """
    Inspect a dataset for null values, zeros, and basic statistics.
    
    Shows null counts, zero counts, mean, max, min for each column.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset to inspect
    verbose : bool, default True
        If True, prints a formatted report. If False, only returns the dictionary.
        
    Returns
    -------
    Dict
        Dictionary containing inspection results for each column
        
    Example
    -------
    >>> import pandas as pd
    >>> df = pd.read_csv('data.csv')
    >>> stats = inspect_dataset(df)
    """
    
    result = {}
    
    for column in df.columns:
        col_data = df[column]
        
        # Check if numeric column
        is_numeric = pd.api.types.is_numeric_dtype(col_data)
        
        # Count nulls
        null_count = col_data.isnull().sum()
        null_percentage = (null_count / len(df)) * 100
        
        # Count zeros (only for numeric columns)
        zero_count = 0
        zero_percentage = 0
        if is_numeric:
            zero_count = (col_data == 0).sum()
            zero_percentage = (zero_count / len(df)) * 100
        
        # Basic statistics (only for numeric columns)
        stats = {
            "dtype": str(col_data.dtype),
            "total_values": len(df),
            "non_null_count": col_data.count(),
            "null_count": null_count,
            "null_percentage": round(null_percentage, 2),
        }
        
        if is_numeric:
            stats.update({
                "zero_count": zero_count,
                "zero_percentage": round(zero_percentage, 2),
                "mean": round(col_data.mean(), 4) if col_data.count() > 0 else None,
                "median": round(col_data.median(), 4) if col_data.count() > 0 else None,
                "std": round(col_data.std(), 4) if col_data.count() > 0 else None,
                "min": round(col_data.min(), 4) if col_data.count() > 0 else None,
                "max": round(col_data.max(), 4) if col_data.count() > 0 else None,
                "25%": round(col_data.quantile(0.25), 4) if col_data.count() > 0 else None,
                "50%": round(col_data.quantile(0.50), 4) if col_data.count() > 0 else None,
                "75%": round(col_data.quantile(0.75), 4) if col_data.count() > 0 else None,
            })
        else:
            stats.update({
                "unique_values": col_data.nunique(),
                "most_common": col_data.mode().values[0] if len(col_data.mode()) > 0 else None,
            })
        
        result[column] = stats
    
    if verbose:
        _print_inspection_report(result, df)
    
    return result


def _print_inspection_report(result: Dict, df: pd.DataFrame) -> None:
    """
    Print a formatted inspection report.
    
    Parameters
    ----------
    result : Dict
        Inspection results dictionary
    df : pd.DataFrame
        Original dataframe
    """
    
    print("\n" + "="*100)
    print(f"DATASET INSPECTION REPORT")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print("="*100 + "\n")
    
    for column, stats in result.items():
        print(f"Column: {column}")
        print(f"   Data Type: {stats['dtype']}")
        print(f"   Total Values: {stats['total_values']}")
        print(f"   Non-Null Count: {stats['non_null_count']}")
        print(f"   Null Values: {stats['null_count']} ({stats['null_percentage']}%)")
        
        if stats['dtype'] in ['int64', 'int32', 'float64', 'float32']:
            print(f"   Zero Values: {stats['zero_count']} ({stats['zero_percentage']}%)")
            print(f"   Mean: {stats['mean']}")
            print(f"   Median: {stats['50%']}")
            print(f"   Std Dev: {stats['std']}")
            print(f"   Min: {stats['min']}")
            print(f"   Max: {stats['max']}")
            print(f"   Quartiles (25%, 50%, 75%): ({stats['25%']}, {stats['50%']}, {stats['75%']})")
        else:
            print(f"   Unique Values: {stats['unique_values']}")
            print(f"   Most Common: {stats['most_common']}")
        
        print()


def summarize_nulls(df: pd.DataFrame, sort_by: str = "count") -> pd.DataFrame:
    """
    Create a summary of null values in the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyze
    sort_by : str, default "count"
        Sort by "count" or "percentage"
        
    Returns
    -------
    pd.DataFrame
        Summary dataframe with null statistics
    """
    
    null_summary = pd.DataFrame({
        "Column": df.columns,
        "Null_Count": [df[col].isnull().sum() for col in df.columns],
        "Total_Values": len(df),
        "Null_Percentage": [(df[col].isnull().sum() / len(df)) * 100 for col in df.columns]
    })
    
    null_summary = null_summary[null_summary["Null_Count"] > 0]  # Only show columns with nulls
    
    if sort_by == "count":
        null_summary = null_summary.sort_values("Null_Count", ascending=False)
    elif sort_by == "percentage":
        null_summary = null_summary.sort_values("Null_Percentage", ascending=False)
    
    return null_summary.reset_index(drop=True)


def summarize_zeros(df: pd.DataFrame, sort_by: str = "count") -> pd.DataFrame:
    """
    Create a summary of zero values in numeric columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyze
    sort_by : str, default "count"
        Sort by "count" or "percentage"
        
    Returns
    -------
    pd.DataFrame
        Summary dataframe with zero statistics
    """
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    zero_summary = pd.DataFrame({
        "Column": numeric_cols,
        "Zero_Count": [(df[col] == 0).sum() for col in numeric_cols],
        "Total_Values": len(df),
        "Zero_Percentage": [((df[col] == 0).sum() / len(df)) * 100 for col in numeric_cols]
    })
    
    zero_summary = zero_summary[zero_summary["Zero_Count"] > 0]  # Only show columns with zeros
    
    if sort_by == "count":
        zero_summary = zero_summary.sort_values("Zero_Count", ascending=False)
    elif sort_by == "percentage":
        zero_summary = zero_summary.sort_values("Zero_Percentage", ascending=False)
    
    return zero_summary.reset_index(drop=True)


def column_statistics(df: pd.DataFrame, column: str) -> Dict:
    """
    Get detailed statistics for a specific column.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset
    column : str
        Column name
        
    Returns
    -------
    Dict
        Detailed statistics for the column
    """
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataset")
    
    col_data = df[column]
    
    stats = {
        "column_name": column,
        "dtype": str(col_data.dtype),
        "total_count": len(col_data),
        "null_count": col_data.isnull().sum(),
        "null_percentage": round((col_data.isnull().sum() / len(col_data)) * 100, 2),
        "non_null_count": col_data.count(),
    }
    
    if pd.api.types.is_numeric_dtype(col_data):
        stats.update({
            "zero_count": (col_data == 0).sum(),
            "zero_percentage": round(((col_data == 0).sum() / len(col_data)) * 100, 2),
            "mean": round(col_data.mean(), 4),
            "median": round(col_data.median(), 4),
            "mode": col_data.mode().values[0] if len(col_data.mode()) > 0 else None,
            "std": round(col_data.std(), 4),
            "min": round(col_data.min(), 4),
            "max": round(col_data.max(), 4),
            "range": round(col_data.max() - col_data.min(), 4),
            "variance": round(col_data.var(), 4),
            "q1": round(col_data.quantile(0.25), 4),
            "q2": round(col_data.quantile(0.50), 4),
            "q3": round(col_data.quantile(0.75), 4),
            "iqr": round(col_data.quantile(0.75) - col_data.quantile(0.25), 4),
        })
    else:
        stats.update({
            "unique_count": col_data.nunique(),
            "unique_values": col_data.unique().tolist()[:10],  # Show first 10
            "mode": col_data.mode().values[0] if len(col_data.mode()) > 0 else None,
            "value_counts": col_data.value_counts().head(10).to_dict(),
        })
    
    return stats


def display_column_info(df: pd.DataFrame, column: str) -> None:
    """
    Display detailed information about a specific column.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset
    column : str
        Column name
    """
    
    stats = column_statistics(df, column)
    
    print(f"\n{'='*80}")
    print(f"Column Information: {stats['column_name']}")
    print(f"{'='*80}")
    print(f"Data Type: {stats['dtype']}")
    print(f"Total Values: {stats['total_count']}")
    print(f"Non-Null Values: {stats['non_null_count']}")
    print(f"Null Values: {stats['null_count']} ({stats['null_percentage']}%)")
    
    if 'zero_count' in stats:
        print(f"Zero Values: {stats['zero_count']} ({stats['zero_percentage']}%)")
        print(f"\nStatistics:")
        print(f"  Mean: {stats['mean']}")
        print(f"  Median: {stats['median']}")
        print(f"  Mode: {stats['mode']}")
        print(f"  Std Dev: {stats['std']}")
        print(f"  Min: {stats['min']}")
        print(f"  Max: {stats['max']}")
        print(f"  Range: {stats['range']}")
        print(f"  Variance: {stats['variance']}")
        print(f"\nQuartiles:")
        print(f"  Q1 (25%): {stats['q1']}")
        print(f"  Q2 (50%): {stats['q2']}")
        print(f"  Q3 (75%): {stats['q3']}")
        print(f"  IQR: {stats['iqr']}")
    else:
        print(f"\nCategorical Statistics:")
        print(f"  Unique Values: {stats['unique_count']}")
        print(f"  Mode: {stats['mode']}")
        print(f"  Sample Values: {stats['unique_values']}")
        print(f"\n  Value Counts (top 10):")
        for value, count in stats['value_counts'].items():
            print(f"    {value}: {count}")
    
    print()


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Create sample dataset with some nulls and zeros
    sample_data = {
        'temp': [15.2, 16.1, None, 14.8, 0, 15.5, 16.0, None, 15.1, 14.9],
        'humidity': [75, 80, 82, None, 78, 76, None, 81, 79, 77],
        'wind_speed': [10.5, 0, 12.3, 11.1, 0, 10.8, 11.5, 12.0, 10.2, 0],
        'rain': [0, 0, 2.5, 0, 5.1, 0, 0, 1.2, 0, 3.8],
        'location': ['Barcelona', 'Barcelona', None, 'Barcelona', 'Madrid', 'Barcelona', 'Barcelona', 'Barcelona', 'Barcelona', 'Barcelona']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Inspect the dataset
    print("\n" + "█"*100)
    print("EJEMPLO DE USO: inspect_dataset()")
    print("█"*100)
    inspect_dataset(df)
    
    # Show null summary
    print("\n" + "█"*100)
    print("EJEMPLO DE USO: summarize_nulls()")
    print("█"*100)
    print(summarize_nulls(df))
    
    # Show zero summary
    print("\n" + "█"*100)
    print("EJEMPLO DE USO: summarize_zeros()")
    print("█"*100)
    print(summarize_zeros(df))
    
    # Display specific column info
    print("\n" + "█"*100)
    print("EJEMPLO DE USO: display_column_info()")
    print("█"*100)
    display_column_info(df, 'temp')
    display_column_info(df, 'location')
