"""
Module for data transformation and preprocessing.
Provides functions to scale, normalize, and transform climate data for better visualization and analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')


class ClimateDataTransformer:
    """
    Transformer class for climate data with methods specific to meteorological variables.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the transformer with a dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            The climate dataset to transform
        """
        self.df = df.copy()
        self.original_df = df.copy()
        self.transformations_applied = {}
        
    def reset(self):
        """Reset dataframe to original state."""
        self.df = self.original_df.copy()
        self.transformations_applied = {}
        
    def get_transformation_summary(self) -> pd.DataFrame:
        """
        Get a summary of all transformations applied.
        
        Returns
        -------
        pd.DataFrame
            Summary of transformations
        """
        summary_data = []
        for col, trans in self.transformations_applied.items():
            summary_data.append({
                'Column': col,
                'Transformation': trans['type'],
                'Parameters': str(trans.get('params', 'N/A'))
            })
        return pd.DataFrame(summary_data)
    
    # ==================== SCALING METHODS ====================
    
    def standard_scale(self, columns: List[str]) -> 'ClimateDataTransformer':
        """
        Apply standard scaling (z-score normalization): (x - mean) / std
        Best for: Variables with normal distribution (temp, dwpt, pres)
        
        Parameters
        ----------
        columns : List[str]
            List of columns to scale
            
        Returns
        -------
        self
        """
        scaler = StandardScaler()
        for col in columns:
            if col in self.df.columns:
                self.df[f'{col}_scaled'] = scaler.fit_transform(self.df[[col]])
                self.transformations_applied[col] = {
                    'type': 'standard_scale',
                    'new_column': f'{col}_scaled'
                }
        return self
    
    def minmax_scale(self, columns: List[str], feature_range: Tuple[float, float] = (0, 1)) -> 'ClimateDataTransformer':
        """
        Apply min-max scaling: (x - min) / (max - min)
        Best for: Variables that need to be in a specific range (rhum, wdir)
        
        Parameters
        ----------
        columns : List[str]
            List of columns to scale
        feature_range : Tuple[float, float]
            Desired range of transformed data
            
        Returns
        -------
        self
        """
        scaler = MinMaxScaler(feature_range=feature_range)
        for col in columns:
            if col in self.df.columns:
                self.df[f'{col}_minmax'] = scaler.fit_transform(self.df[[col]])
                self.transformations_applied[col] = {
                    'type': 'minmax_scale',
                    'params': {'range': feature_range},
                    'new_column': f'{col}_minmax'
                }
        return self
    
    def robust_scale(self, columns: List[str]) -> 'ClimateDataTransformer':
        """
        Apply robust scaling using median and IQR (resistant to outliers)
        Best for: Variables with outliers (wspd, wpgt, prcp)
        
        Parameters
        ----------
        columns : List[str]
            List of columns to scale
            
        Returns
        -------
        self
        """
        scaler = RobustScaler()
        for col in columns:
            if col in self.df.columns:
                self.df[f'{col}_robust'] = scaler.fit_transform(self.df[[col]])
                self.transformations_applied[col] = {
                    'type': 'robust_scale',
                    'new_column': f'{col}_robust'
                }
        return self
    
    # ==================== LOG TRANSFORMATIONS ====================
    
    def log_transform(self, columns: List[str], method: str = 'log1p') -> 'ClimateDataTransformer':
        """
        Apply logarithmic transformation to reduce skewness.
        Best for: Right-skewed distributions (prcp, snow, wspd, tsun)
        
        Parameters
        ----------
        columns : List[str]
            List of columns to transform
        method : str
            'log': natural log (requires positive values)
            'log1p': log(1 + x) - handles zeros
            'log10': base-10 logarithm
            
        Returns
        -------
        self
        """
        for col in columns:
            if col in self.df.columns:
                if method == 'log':
                    # Add small constant to avoid log(0)
                    self.df[f'{col}_log'] = np.log(self.df[col] + 1e-8)
                elif method == 'log1p':
                    self.df[f'{col}_log1p'] = np.log1p(self.df[col])
                elif method == 'log10':
                    self.df[f'{col}_log10'] = np.log10(self.df[col] + 1e-8)
                
                self.transformations_applied[col] = {
                    'type': f'log_transform_{method}',
                    'new_column': f'{col}_{method}'
                }
        return self
    
    def sqrt_transform(self, columns: List[str]) -> 'ClimateDataTransformer':
        """
        Apply square root transformation (milder than log).
        Best for: Moderately skewed data (prcp, wspd)
        
        Parameters
        ----------
        columns : List[str]
            List of columns to transform
            
        Returns
        -------
        self
        """
        for col in columns:
            if col in self.df.columns:
                # Handle negative values by taking sqrt of absolute value and restoring sign
                self.df[f'{col}_sqrt'] = np.sign(self.df[col]) * np.sqrt(np.abs(self.df[col]))
                self.transformations_applied[col] = {
                    'type': 'sqrt_transform',
                    'new_column': f'{col}_sqrt'
                }
        return self
    
    def boxcox_transform(self, columns: List[str]) -> 'ClimateDataTransformer':
        """
        Apply Box-Cox transformation (requires positive values).
        Best for: Making data more normal (temp, pres, rhum)
        
        Parameters
        ----------
        columns : List[str]
            List of columns to transform
            
        Returns
        -------
        self
        """
        from scipy.stats import boxcox
        
        for col in columns:
            if col in self.df.columns:
                data = self.df[col].dropna()
                if (data > 0).all():
                    transformed, lambda_param = boxcox(data)
                    self.df.loc[data.index, f'{col}_boxcox'] = transformed
                    self.transformations_applied[col] = {
                        'type': 'boxcox_transform',
                        'params': {'lambda': lambda_param},
                        'new_column': f'{col}_boxcox'
                    }
        return self
    
    # ==================== BINNING / DISCRETIZATION ====================
    
    def bin_variable(self, column: str, bins: int = 5, labels: Optional[List[str]] = None) -> 'ClimateDataTransformer':
        """
        Discretize continuous variable into bins.
        
        Parameters
        ----------
        column : str
            Column to bin
        bins : int
            Number of bins
        labels : List[str], optional
            Custom labels for bins
            
        Returns
        -------
        self
        """
        if column in self.df.columns:
            self.df[f'{column}_binned'] = pd.cut(self.df[column], bins=bins, labels=labels)
            self.transformations_applied[column] = {
                'type': 'binning',
                'params': {'bins': bins},
                'new_column': f'{column}_binned'
            }
        return self
    
    def create_categorical_ranges(self, column: str, ranges: List[Tuple[float, float]], 
                                   labels: List[str]) -> 'ClimateDataTransformer':
        """
        Create categorical ranges based on domain knowledge.
        
        Example for temperature:
        ranges = [(-np.inf, 0), (0, 10), (10, 20), (20, 30), (30, np.inf)]
        labels = ['Freezing', 'Cold', 'Mild', 'Warm', 'Hot']
        
        Parameters
        ----------
        column : str
            Column to categorize
        ranges : List[Tuple[float, float]]
            List of (min, max) tuples
        labels : List[str]
            Labels for each range
            
        Returns
        -------
        self
        """
        if column in self.df.columns:
            bins = [r[0] for r in ranges] + [ranges[-1][1]]
            self.df[f'{column}_category'] = pd.cut(self.df[column], bins=bins, labels=labels)
            self.transformations_applied[column] = {
                'type': 'categorical_ranges',
                'params': {'ranges': ranges, 'labels': labels},
                'new_column': f'{column}_category'
            }
        return self
    
    # ==================== CYCLICAL ENCODING ====================
    
    def encode_cyclical(self, column: str, max_val: float) -> 'ClimateDataTransformer':
        """
        Encode cyclical variables (like wind direction) using sin/cos.
        
        Best for: wdir (wind direction: 0-360°)
        
        Parameters
        ----------
        column : str
            Column to encode
        max_val : float
            Maximum value of the cycle (e.g., 360 for degrees)
            
        Returns
        -------
        self
        """
        if column in self.df.columns:
            self.df[f'{column}_sin'] = np.sin(2 * np.pi * self.df[column] / max_val)
            self.df[f'{column}_cos'] = np.cos(2 * np.pi * self.df[column] / max_val)
            self.transformations_applied[column] = {
                'type': 'cyclical_encoding',
                'params': {'max_val': max_val},
                'new_columns': [f'{column}_sin', f'{column}_cos']
            }
        return self
    
    # ==================== CLIMATE-SPECIFIC TRANSFORMATIONS ====================
    
    def create_climate_features(self) -> 'ClimateDataTransformer':
        """
        Create domain-specific features for climate data.
        """
        # Temperature comfort index
        if 'temp' in self.df.columns and 'rhum' in self.df.columns:
            # Simplified heat index
            self.df['heat_index'] = self.df['temp'] + 0.5555 * (
                (6.11 * np.exp(5417.7530 * ((1/273.16) - (1/(273.15 + self.df['dwpt'])))) - 10)
            )
        
        # Wind chill (for temperatures below 10°C)
        if 'temp' in self.df.columns and 'wspd' in self.df.columns:
            mask = self.df['temp'] < 10
            self.df.loc[mask, 'wind_chill'] = (
                13.12 + 0.6215 * self.df.loc[mask, 'temp'] - 
                11.37 * (self.df.loc[mask, 'wspd'] * 3.6) ** 0.16 + 
                0.3965 * self.df.loc[mask, 'temp'] * (self.df.loc[mask, 'wspd'] * 3.6) ** 0.16
            )
        
        # Precipitation intensity categories
        if 'prcp' in self.df.columns:
            conditions = [
                (self.df['prcp'] == 0),
                (self.df['prcp'] > 0) & (self.df['prcp'] <= 2.5),
                (self.df['prcp'] > 2.5) & (self.df['prcp'] <= 10),
                (self.df['prcp'] > 10) & (self.df['prcp'] <= 50),
                (self.df['prcp'] > 50)
            ]
            choices = ['No rain', 'Light', 'Moderate', 'Heavy', 'Very heavy']
            self.df['rain_intensity'] = np.select(conditions, choices, default='Unknown')
        
        # Wind speed categories (Beaufort scale simplified)
        if 'wspd' in self.df.columns:
            conditions = [
                (self.df['wspd'] < 0.5),
                (self.df['wspd'] >= 0.5) & (self.df['wspd'] < 3.3),
                (self.df['wspd'] >= 3.3) & (self.df['wspd'] < 5.5),
                (self.df['wspd'] >= 5.5) & (self.df['wspd'] < 8),
                (self.df['wspd'] >= 8) & (self.df['wspd'] < 10.8),
                (self.df['wspd'] >= 10.8)
            ]
            choices = ['Calm', 'Light breeze', 'Gentle breeze', 'Moderate breeze', 'Fresh breeze', 'Strong wind']
            self.df['wind_category'] = np.select(conditions, choices, default='Unknown')
        
        self.transformations_applied['climate_features'] = {
            'type': 'climate_specific_features',
            'new_columns': ['heat_index', 'wind_chill', 'rain_intensity', 'wind_category']
        }
        
        return self
    
    def apply_recommended_transformations(self) -> 'ClimateDataTransformer':
        """
        Apply recommended transformations based on climate data characteristics.
        """
        # Temperature and dew point: Standard scaling (normal distribution)
        if 'temp' in self.df.columns:
            self.standard_scale(['temp'])
        if 'dwpt' in self.df.columns:
            self.standard_scale(['dwpt'])
        
        # Humidity: MinMax scaling (already 0-100 range)
        if 'rhum' in self.df.columns:
            self.minmax_scale(['rhum'])
        
        # Precipitation: Log transformation (right-skewed, many zeros)
        if 'prcp' in self.df.columns:
            self.log_transform(['prcp'], method='log1p')
        
        # Snow: Log transformation (right-skewed, many zeros)
        if 'snow' in self.df.columns:
            self.log_transform(['snow'], method='log1p')
        
        # Wind direction: Cyclical encoding
        if 'wdir' in self.df.columns:
            self.encode_cyclical('wdir', max_val=360)
        
        # Wind speed: Square root or robust scaling (outliers)
        if 'wspd' in self.df.columns:
            self.sqrt_transform(['wspd'])
            self.robust_scale(['wspd'])
        
        # Wind gust: Square root or robust scaling
        if 'wpgt' in self.df.columns:
            self.sqrt_transform(['wpgt'])
            self.robust_scale(['wpgt'])
        
        # Pressure: Standard scaling (stable, normal)
        if 'pres' in self.df.columns:
            self.standard_scale(['pres'])
        
        # Sunshine: Log transformation (many zeros)
        if 'tsun' in self.df.columns:
            self.log_transform(['tsun'], method='log1p')
        
        # Create climate-specific features
        self.create_climate_features()
        
        return self
    
    def get_transformed_data(self) -> pd.DataFrame:
        """
        Get the transformed dataframe.
        
        Returns
        -------
        pd.DataFrame
            Transformed dataframe
        """
        return self.df


# ==================== STANDALONE FUNCTIONS ====================

def compare_distributions(df: pd.DataFrame, original_col: str, transformed_col: str) -> Dict:
    """
    Compare statistics between original and transformed columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing both columns
    original_col : str
        Original column name
    transformed_col : str
        Transformed column name
        
    Returns
    -------
    Dict
        Comparison statistics
    """
    from scipy.stats import skew, kurtosis
    
    orig = df[original_col].dropna()
    trans = df[transformed_col].dropna()
    
    comparison = {
        'original': {
            'mean': orig.mean(),
            'std': orig.std(),
            'skewness': skew(orig),
            'kurtosis': kurtosis(orig),
            'min': orig.min(),
            'max': orig.max()
        },
        'transformed': {
            'mean': trans.mean(),
            'std': trans.std(),
            'skewness': skew(trans),
            'kurtosis': kurtosis(trans),
            'min': trans.min(),
            'max': trans.max()
        }
    }
    
    return comparison


def get_optimal_transformation(df: pd.DataFrame, column: str) -> str:
    """
    Suggest optimal transformation based on data distribution.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe
    column : str
        Column to analyze
        
    Returns
    -------
    str
        Recommended transformation
    """
    from scipy.stats import skew
    
    data = df[column].dropna()
    skewness = abs(skew(data))
    zeros_pct = (data == 0).sum() / len(data) * 100
    
    if zeros_pct > 50:
        return "log1p (many zeros detected)"
    elif skewness > 2:
        return "log or sqrt (high skewness)"
    elif skewness > 1:
        return "sqrt (moderate skewness)"
    else:
        return "standard_scale (relatively normal)"


if __name__ == "__main__":
    # Example usage
    print("🔧 Climate Data Transformer - Example Usage\n")
    
    # Create sample data
    np.random.seed(42)
    sample_data = {
        'temp': np.random.normal(15, 5, 1000),
        'prcp': np.random.exponential(2, 1000),
        'wdir': np.random.uniform(0, 360, 1000),
        'wspd': np.random.gamma(2, 2, 1000),
        'rhum': np.random.uniform(40, 100, 1000)
    }
    df = pd.DataFrame(sample_data)
    
    # Initialize transformer
    transformer = ClimateDataTransformer(df)
    
    # Apply recommended transformations
    transformer.apply_recommended_transformations()
    
    # Get transformed data
    transformed_df = transformer.get_transformed_data()
    
    # Show summary
    print("✅ Transformations applied:")
    print(transformer.get_transformation_summary())
    
    print(f"\n📊 Original columns: {len(df.columns)}")
    print(f"📊 After transformation: {len(transformed_df.columns)}")
    print(f"\n🎯 New columns created: {len(transformed_df.columns) - len(df.columns)}")