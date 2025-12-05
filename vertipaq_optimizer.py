"""
VertiPaq Optimizer - Column-Store Ordering for Maximum Compression

This module implements the greedy bucket-splitting algorithm used by Microsoft's 
VertiPaq engine (Power BI, Analysis Services) to optimize row ordering for 
Run-Length Encoding (RLE) compression in columnar databases.

License: MIT
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional, Tuple, Dict
from collections import deque
import time

__version__ = "1.0.0"
__all__ = ['optimize_table', 'VertiPaqOptimizer']


# Algorithm constants (derived from reverse engineering and MS-XLDM spec)
MIN_SPLIT_SIZE = 64
BIT_SAVINGS_THRESHOLD = 0.1
INITIAL_MAX_SAVINGS = -1.0
SAMPLING_THRESHOLD = 10000
SAMPLING_DIVISOR = 10000.0
SAMPLING_ADDER = 1.0

# Special value handling
NAN_PLACEHOLDER = np.iinfo(np.int32).min  # Use min int32 as NaN placeholder
INF_POS_PLACEHOLDER = np.iinfo(np.int32).max  # Use max int32 as +Inf placeholder
INF_NEG_PLACEHOLDER = np.iinfo(np.int32).min + 1  # Use min+1 int32 as -Inf placeholder


class VertiPaqOptimizer:
    """
    VertiPaq-style row ordering optimizer for columnar compression.
    
    Uses a greedy bucket-splitting algorithm to reorder rows such that
    consecutive rows have similar values, maximizing RLE compression efficiency.
    
    Example:
        >>> optimizer = VertiPaqOptimizer()
        >>> df = pd.read_csv('data.csv')
        >>> result = optimizer.optimize(df)
        >>> print(f"Optimized in {result['steps']} steps")
        >>> optimized_df = df.iloc[result['row_order']]
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the optimizer.
        
        Args:
            verbose: If True, print progress information during optimization
        """
        self.verbose = verbose
        self._histogram_builder = _HistogramBuilder()
    
    def optimize(
        self, 
        data: Union[pd.DataFrame, Dict[str, np.ndarray], List[np.ndarray]],
        columns: Optional[List[str]] = None
    ) -> Dict:
        """
        Optimize row ordering for maximum compression.
        
        Args:
            data: Input data as DataFrame, dict of arrays, or list of arrays
            columns: Specific columns to optimize (default: all numeric columns)
        
        Returns:
            Dictionary containing:
                - 'row_order': np.ndarray of optimized row indices
                - 'steps': Number of optimization iterations
                - 'clusters': Number of RLE clusters created
                - 'compression_ratio': Estimated compression improvement
                - 'time': Execution time in seconds
        """
        # Convert input to standard format
        columns_data, column_names = self._prepare_data(data, columns)
        
        if len(columns_data) == 0:
            raise ValueError("No valid columns to optimize")
        
        # Calculate bits per value for each column
        bits_per_value = [self._calculate_bits(col) for col in columns_data]
        
        # Run optimization
        start_time = time.time()
        row_order, steps, clusters = self._compress_table(columns_data, bits_per_value)
        elapsed = time.time() - start_time
        
        # Calculate compression metrics
        num_rows = len(columns_data[0])
        original_clusters = sum(len(np.unique(col)) for col in columns_data)
        compression_ratio = original_clusters / max(clusters, 1)
        
        result = {
            'row_order': row_order,
            'steps': steps,
            'clusters': clusters,
            'compression_ratio': compression_ratio,
            'time': elapsed,
            'columns_optimized': column_names,
            'num_rows': num_rows
        }
        
        if self.verbose:
            print(f"\n✓ Optimization complete!")
            print(f"  Rows: {num_rows:,}")
            print(f"  Columns: {len(columns_data)}")
            print(f"  Steps: {steps:,}")
            print(f"  RLE clusters: {clusters:,} (vs {original_clusters:,} original)")
            print(f"  Compression improvement: {compression_ratio:.2f}x")
            print(f"  Time: {elapsed:.2f}s")
        
        return result
    
    def _prepare_data(
        self, 
        data: Union[pd.DataFrame, Dict[str, np.ndarray], List[np.ndarray]],
        columns: Optional[List[str]]
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Convert input data to list of NumPy arrays, handling NaN, Inf, and string values."""
        if isinstance(data, pd.DataFrame):
            if columns is None:
                # Use all numeric and object/string columns
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                string_cols = data.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
                columns = numeric_cols + string_cols
            
            arrays = [self._safe_convert_column(data[col], col) for col in columns]
            return arrays, columns
        
        elif isinstance(data, dict):
            column_names = columns or list(data.keys())
            arrays = [self._safe_convert_column(np.asarray(data[col]), col) for col in column_names]
            return arrays, column_names
        
        elif isinstance(data, list):
            column_names = columns or [f"Column{i}" for i in range(len(data))]
            arrays = [self._safe_convert_column(np.asarray(col), name) for col, name in zip(data, column_names)]
            return arrays, column_names
        
        else:
            raise TypeError("Data must be DataFrame, dict, or list of arrays")
    
    def _safe_convert_column(self, col_data: Union[pd.Series, np.ndarray], col_name: str = "") -> np.ndarray:
        """
        Safely convert a column to int32, handling NaN, Inf, and string values.
        
        For numeric columns: NaN, +Inf, and -Inf values are replaced with special 
        placeholder integers.
        
        For string/object columns: Values are encoded as integers using factorization,
        with NaN/None values assigned a special placeholder.
        
        Args:
            col_data: Input column (Series or array, may contain NaN, Inf, strings, etc.)
            col_name: Column name for verbose output
            
        Returns:
            np.ndarray of int32 with values encoded as integers
        """
        # Convert to pandas Series for consistent handling
        if not isinstance(col_data, pd.Series):
            col_data = pd.Series(col_data)
        
        # Check if this is a string/object column
        if col_data.dtype == 'object' or col_data.dtype.name in ('string', 'category'):
            return self._convert_string_column(col_data, col_name)
        else:
            return self._convert_numeric_column(col_data, col_name)
    
    def _convert_string_column(self, col_data: pd.Series, col_name: str = "") -> np.ndarray:
        """
        Convert string/object column to int32 using factorization.
        
        Args:
            col_data: Input column with string/object values
            col_name: Column name for verbose output
            
        Returns:
            np.ndarray of int32 with strings encoded as integers
        """
        # Use pandas factorize to encode strings as integers
        # factorize returns (codes, uniques) where codes are integers
        # NaN values get code -1
        codes, uniques = pd.factorize(col_data, use_na_sentinel=True)
        
        # Count null values
        null_count = np.sum(codes == -1)
        
        if self.verbose:
            print(f"  Column '{col_name}': string/object type, {len(uniques):,} unique values", end="")
            if null_count > 0:
                print(f", {null_count:,} null values")
            else:
                print()
        
        # Convert to int32, mapping -1 (NaN) to NAN_PLACEHOLDER
        result = codes.astype(np.int32)
        result[codes == -1] = NAN_PLACEHOLDER
        
        return result
    
    def _convert_numeric_column(self, col_data: pd.Series, col_name: str = "") -> np.ndarray:
        """
        Convert numeric column to int32, handling NaN and Inf values.
        
        Args:
            col_data: Input column with numeric values
            col_name: Column name for verbose output
            
        Returns:
            np.ndarray of int32 with special values replaced by placeholders
        """
        # Convert to float64 first to handle all numeric types uniformly
        arr = col_data.to_numpy(dtype=np.float64)
        
        # Create output array
        result = np.empty(len(arr), dtype=np.int32)
        
        # Identify special values
        nan_mask = np.isnan(arr)
        pos_inf_mask = np.isposinf(arr)
        neg_inf_mask = np.isneginf(arr)
        valid_mask = ~(nan_mask | pos_inf_mask | neg_inf_mask)
        
        # Count special values for verbose output
        nan_count = np.sum(nan_mask)
        inf_count = np.sum(pos_inf_mask) + np.sum(neg_inf_mask)
        
        if self.verbose and (nan_count > 0 or inf_count > 0):
            print(f"  Column '{col_name}': {nan_count:,} NaN, {inf_count:,} Inf values replaced with placeholders")
        
        # Handle valid values - clip to int32 range before conversion
        if np.any(valid_mask):
            valid_values = arr[valid_mask]
            # Clip to safe int32 range (leaving room for placeholders)
            clipped = np.clip(
                valid_values, 
                np.iinfo(np.int32).min + 2,  # Leave room for NaN and -Inf placeholders
                np.iinfo(np.int32).max - 1   # Leave room for +Inf placeholder
            )
            result[valid_mask] = clipped.astype(np.int32)
        
        # Replace special values with placeholders
        result[nan_mask] = NAN_PLACEHOLDER
        result[pos_inf_mask] = INF_POS_PLACEHOLDER
        result[neg_inf_mask] = INF_NEG_PLACEHOLDER
        
        return result
    
    def _calculate_bits(self, column: np.ndarray) -> float:
        """Calculate bits needed to encode a column."""
        cardinality = len(np.unique(column))
        if cardinality <= 1:
            return 0.0
        return np.log2(cardinality)
    
    def _compress_table(
        self, 
        columns: List[np.ndarray], 
        bits_per_value: List[float]
    ) -> Tuple[np.ndarray, int, int]:
        """Core compression algorithm (implementation in _implementation.py)."""
        num_rows = len(columns[0])
        num_cols = len(columns)
        
        # Pre-compute column statistics
        col_stats = [_ColumnStats(col, bpv) for col, bpv in zip(columns, bits_per_value)]
        
        # Initialize
        row_indices = np.arange(num_rows, dtype=np.int32)
        buckets = deque([_Bucket(0, num_rows - 1)])
        
        step_count = 0
        cluster_count = [0] * num_cols
        
        # Main optimization loop
        while buckets:
            bucket = buckets.popleft()
            
            if bucket.is_done:
                continue
            
            step_count += 1
            bucket_size = bucket.size()
            
            # Calculate sampling rate
            sample_step = 1
            if bucket_size >= SAMPLING_THRESHOLD:
                sample_step = int(bucket_size / SAMPLING_DIVISOR + SAMPLING_ADDER)
            
            # Find best column/value split
            best_savings = INITIAL_MAX_SAVINGS
            best_choice = None
            
            for col_idx in range(num_cols):
                if bucket.is_column_done(col_idx):
                    continue
                
                # Build histogram and find most frequent value
                max_value, max_freq = self._histogram_builder.build_histogram(
                    row_indices, columns[col_idx], bucket.start, bucket.end,
                    sample_step, col_stats[col_idx]
                )
                
                # Calculate bit savings
                savings = max_freq * bits_per_value[col_idx]
                
                if savings < BIT_SAVINGS_THRESHOLD:
                    bucket.mark_column_done(col_idx)
                    continue
                
                if savings > best_savings:
                    best_savings = savings
                    best_choice = (col_idx, max_value, max_freq)
            
            # No profitable split found
            if best_choice is None:
                bucket.is_done = True
                continue
            
            # Partition rows
            col_idx, target_value, _ = best_choice
            match_count = _partition_rows(
                row_indices, columns[col_idx], 
                bucket.start, bucket.end, target_value
            )
            
            # Check minimum split size
            if match_count < MIN_SPLIT_SIZE:
                bucket.is_done = True
                buckets.append(bucket)
                continue
            
            # Check if all rows match
            non_match_count = bucket_size - match_count
            if non_match_count < MIN_SPLIT_SIZE:
                bucket.mark_column_done(col_idx)
                cluster_count[col_idx] += 1
                buckets.append(bucket)
                continue
            
            # Create new buckets
            pure_bucket = _Bucket(bucket.start, bucket.start + match_count - 1)
            pure_bucket.columns_done = bucket.columns_done
            pure_bucket.mark_column_done(col_idx)
            
            impure_bucket = _Bucket(bucket.start + match_count, bucket.end)
            impure_bucket.columns_done = bucket.columns_done
            
            buckets.append(pure_bucket)
            buckets.append(impure_bucket)
            cluster_count[col_idx] += 1
        
        return row_indices, step_count, sum(cluster_count)


# Helper classes and functions

class _Bucket:
    """Represents a range of rows being processed together."""
    __slots__ = ['start', 'end', 'columns_done', 'is_done']
    
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
        self.columns_done = 0
        self.is_done = False
    
    def size(self) -> int:
        return self.end - self.start + 1
    
    def mark_column_done(self, col_idx: int):
        self.columns_done |= (1 << col_idx)
    
    def is_column_done(self, col_idx: int) -> bool:
        return (self.columns_done & (1 << col_idx)) != 0


class _ColumnStats:
    """Pre-computed statistics for a column."""
    __slots__ = ['cardinality', 'min_val', 'max_val', 'bits_per_value']
    
    def __init__(self, column: np.ndarray, bits_per_value: float):
        self.cardinality = len(np.unique(column))
        self.min_val = int(np.min(column))
        self.max_val = int(np.max(column))
        self.bits_per_value = bits_per_value


class _HistogramBuilder:
    """Efficient histogram builder with adaptive mode selection."""
    
    def __init__(self, max_value_range: int = 100000):
        self.max_value_range = max_value_range
        self.array_buffer = None
    
    def build_histogram(
        self, 
        row_indices: np.ndarray, 
        column: np.ndarray,
        start: int, 
        end: int, 
        step: int,
        col_stats: _ColumnStats
    ) -> Tuple[int, int]:
        """Build histogram and return (most_frequent_value, scaled_frequency)."""
        value_range = col_stats.max_val - col_stats.min_val + 1
        bucket_size = end - start + 1
        
        # Choose between array mode (fast) and hash mode (memory efficient)
        use_array_mode = (
            value_range <= self.max_value_range and
            value_range <= bucket_size * 2
        )
        
        if use_array_mode:
            return self._build_array_mode(
                row_indices, column, start, end, step,
                col_stats.min_val, value_range
            )
        else:
            return self._build_hash_mode(
                row_indices, column, start, end, step
            )
    
    def _build_array_mode(
        self, 
        row_indices: np.ndarray, 
        column: np.ndarray,
        start: int, 
        end: int, 
        step: int,
        min_val: int,
        value_range: int
    ) -> Tuple[int, int]:
        """Fast array-based histogram for dense data."""
        # Reuse buffer
        if self.array_buffer is None or len(self.array_buffer) < value_range:
            self.array_buffer = np.zeros(value_range, dtype=np.int32)
        else:
            self.array_buffer[:value_range] = 0
        
        histogram = self.array_buffer[:value_range]
        max_freq = 0
        max_value = min_val
        
        # Build histogram with inline max tracking
        for i in range(start, end + 1, step):
            row = row_indices[i]
            value = column[row]
            index = value - min_val
            histogram[index] += 1
            
            if histogram[index] > max_freq:
                max_freq = histogram[index]
                max_value = value
        
        return max_value, max_freq * step
    
    def _build_hash_mode(
        self,
        row_indices: np.ndarray,
        column: np.ndarray,
        start: int,
        end: int,
        step: int
    ) -> Tuple[int, int]:
        """Memory-efficient hash-based histogram for sparse data."""
        sampled_indices = row_indices[start:end+1:step]
        sampled_values = column[sampled_indices]
        
        unique_vals, counts = np.unique(sampled_values, return_counts=True)
        max_idx = np.argmax(counts)
        
        return unique_vals[max_idx], counts[max_idx] * step


def _partition_rows(
    row_indices: np.ndarray,
    column: np.ndarray,
    start: int,
    end: int,
    target_value
) -> int:
    """Partition rows in-place: matching values first, then non-matching."""
    range_indices = row_indices[start:end+1]
    values = column[range_indices]
    mask = (values == target_value)
    
    matching = range_indices[mask]
    non_matching = range_indices[~mask]
    
    match_count = len(matching)
    row_indices[start:start+match_count] = matching
    row_indices[start+match_count:end+1] = non_matching
    
    return match_count


# Convenience function

def optimize_table(
    data: Union[pd.DataFrame, Dict[str, np.ndarray], List[np.ndarray]],
    columns: Optional[List[str]] = None,
    verbose: bool = False
) -> Dict:
    """
    Optimize row ordering for maximum RLE compression (VertiPaq-style).
    
    This function reorders rows to group similar values together, maximizing
    the effectiveness of Run-Length Encoding compression used in columnar databases
    like Power BI and Analysis Services.
    
    Args:
        data: Input data as pandas DataFrame, dict of arrays, or list of arrays
        columns: Specific columns to use for optimization (default: all numeric)
        verbose: If True, print progress information
    
    Returns:
        Dictionary with optimization results:
            - 'row_order': Optimized row indices (use to reorder your data)
            - 'steps': Number of optimization iterations performed
            - 'clusters': Number of RLE clusters created
            - 'compression_ratio': Estimated compression improvement factor
            - 'time': Execution time in seconds
    
    Example:
        >>> import pandas as pd
        >>> from vertipaq_optimizer import optimize_table
        >>> 
        >>> # Load your data
        >>> df = pd.read_csv('sales_data.csv')
        >>> 
        >>> # Optimize row ordering
        >>> result = optimize_table(df, verbose=True)
        >>> 
        >>> # Reorder dataframe
        >>> optimized_df = df.iloc[result['row_order']]
        >>> 
        >>> # Save optimized version
        >>> optimized_df.to_csv('sales_data_optimized.csv', index=False)
        >>> 
        >>> print(f"Compression improvement: {result['compression_ratio']:.2f}x")
    """
    optimizer = VertiPaqOptimizer(verbose=verbose)
    return optimizer.optimize(data, columns)


if __name__ == "__main__":
    # Simple self-test
    print("VertiPaq Optimizer v" + __version__)
    print("\nRunning self-test...")
    
    # Create test data
    np.random.seed(42)
    test_df = pd.DataFrame({
        'ProductID': np.random.randint(1, 100, 10000),
        'CategoryID': np.random.randint(1, 10, 10000),
        'StoreID': np.random.randint(1, 50, 10000)
    })
    
    # Optimize
    result = optimize_table(test_df, verbose=True)
    
    print(f"\n✓ Self-test passed!")
    print(f"  Original: {test_df.shape[0]:,} rows")
    print(f"  Optimized in: {result['steps']:,} steps")
    print(f"  RLE clusters: {result['clusters']:,}")
    print(f"  Improvement: {result['compression_ratio']:.2f}x")
    
    # Test with NaN and Inf values
    print("\n" + "="*50)
    print("Testing NaN/Inf handling...")
    
    test_df_with_nulls = pd.DataFrame({
        'Value': [1.0, 2.0, np.nan, 4.0, np.inf, -np.inf, 7.0, np.nan, 9.0, 10.0] * 1000,
        'Category': np.random.randint(1, 5, 10000).astype(float)
    })
    # Add some NaN values to Category
    test_df_with_nulls.loc[::100, 'Category'] = np.nan
    
    result_nulls = optimize_table(test_df_with_nulls, verbose=True)
    
    print(f"\n✓ NaN/Inf test passed!")
    print(f"  Original: {test_df_with_nulls.shape[0]:,} rows")
    print(f"  Optimized in: {result_nulls['steps']:,} steps")
    print(f"  RLE clusters: {result_nulls['clusters']:,}")
    print(f"  Improvement: {result_nulls['compression_ratio']:.2f}x")
    
    # Test with string/varchar columns
    print("\n" + "="*50)
    print("Testing string/varchar handling...")
    
    test_df_strings = pd.DataFrame({
        'Region': np.random.choice(['North', 'South', 'East', 'West'], 10000),
        'Country': np.random.choice(['USA', 'Canada', 'Mexico', 'Brazil', None], 10000),
        'ProductName': np.random.choice(['Widget A', 'Widget B', 'Gadget X', 'Gadget Y', 'Tool Z'], 10000),
        'CategoryID': np.random.randint(1, 10, 10000)
    })
    
    result_strings = optimize_table(test_df_strings, verbose=True)
    
    print(f"\n✓ String/varchar test passed!")
    print(f"  Original: {test_df_strings.shape[0]:,} rows")
    print(f"  Optimized in: {result_strings['steps']:,} steps")
    print(f"  RLE clusters: {result_strings['clusters']:,}")
    print(f"  Improvement: {result_strings['compression_ratio']:.2f}x")
