"""
VertiPaq Optimizer - Column-Store Row Ordering for RLE Compression

This module implements the greedy bucket-splitting algorithm used by Microsoft's 
VertiPaq engine (Power BI, Analysis Services) to optimize row ordering for 
Run-Length Encoding (RLE) compression in columnar databases.

Algorithm Overview:
1. Start with all rows in a single bucket
2. For each bucket:
   - Build histograms for each column
   - Calculate bit savings for grouping the most frequent value
   - Select column/value pair with maximum bit savings (greedy)
   - Partition rows: matching values first, others after
   - Split bucket into "pure" and "impure" sub-buckets
3. Repeat until no profitable splits remain
4. Return optimized row ordering

Based on Microsoft Patent US 8,452,737 B2:
"Optimizing data compression using hybrid algorithm"

License: MIT
Version: 2.0.0
"""

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from typing import Union, List, Optional, Tuple, Dict
from collections import deque
import time
import warnings

__version__ = "2.0.0"
__all__ = ['optimize_table', 'VertiPaqOptimizer', 'optimize_parquet']


# Algorithm constants
MIN_SPLIT_SIZE = 64              # Minimum rows to create a split
BIT_SAVINGS_THRESHOLD = 0.1      # Minimum bit savings to continue
INITIAL_MAX_SAVINGS = -1.0       # Starting value for greedy maximum
SAMPLING_THRESHOLD = 10000       # Bucket size to trigger sampling
SAMPLING_DIVISOR = 10000.0       # Sample rate calculation divisor
SAMPLING_ADDER = 1.0             # Sample rate calculation offset


class VertiPaqOptimizer:
    """
    Row ordering optimizer for RLE compression in columnar databases.
    
    Uses a greedy bucket-splitting algorithm to reorder rows such that
    consecutive rows have similar values, maximizing RLE compression efficiency.
    
    The algorithm works by:
    1. Starting with all rows in one bucket
    2. Finding the column and value that offers maximum compression benefit
    3. Partitioning rows to group matching values together
    4. Recursively processing sub-buckets until no improvement possible
    
    Example:
        >>> optimizer = VertiPaqOptimizer(verbose=True)
        >>> result = optimizer.optimize(dataframe)
        >>> optimized_data = result['table']
    """
    
    def __init__(self, verbose: bool = False, null_strategy: str = 'separate'):
        """
        Initialize the optimizer.
        
        Args:
            verbose: If True, print progress information during optimization
            null_strategy: How to handle null values during partitioning:
                'separate' - Group nulls together (treat as distinct value)
                'scatter' - Nulls don't match any value (standard SQL semantics)
        """
        self.verbose = verbose
        self.null_strategy = null_strategy
        self._histogram_builder = _HistogramBuilder()
    
    def optimize(
        self, 
        data: Union[pd.DataFrame, pa.Table, str],
        columns: Optional[List[str]] = None
    ) -> Dict:
        """
        Optimize row ordering for maximum RLE compression.
        
        Args:
            data: Input data as pandas DataFrame, PyArrow Table, or Parquet file path
            columns: Specific columns to optimize (default: all except large binary)
        
        Returns:
            Dictionary containing:
                - 'row_order': Array of optimized row indices
                - 'table': Optimized data table
                - 'steps': Number of optimization iterations performed
                - 'clusters': Number of RLE clusters created
                - 'compression_ratio': Estimated compression improvement
                - 'time': Execution time in seconds
                - 'columns_optimized': List of column names processed
                - 'num_rows': Number of rows processed
        
        Example:
            >>> result = optimizer.optimize(df, columns=['ProductID', 'Date'])
            >>> print(f"Steps: {result['steps']}, Clusters: {result['clusters']}")
        """
        # Load and prepare data
        table, column_names = self._prepare_data(data, columns)
        
        if len(column_names) == 0:
            raise ValueError("No valid columns to optimize")
        
        # Get column arrays
        col_arrays = [table.column(name) for name in column_names]
        
        # Compute column statistics
        col_stats = [_ColumnStats(col) for col in col_arrays]
        
        if self.verbose:
            print(f"\nVertiPaq Optimizer v{__version__}")
            print(f"Dataset: {table.num_rows:,} rows × {len(column_names)} columns")
            print(f"\nColumn Statistics:")
            for name, stats in zip(column_names, col_stats):
                type_info = f"dict[{stats.dictionary_size}]" if stats.is_dictionary else "numeric"
                null_info = f" (nulls: {stats.null_count})" if stats.null_count > 0 else ""
                print(f"  {name}: cardinality={stats.cardinality:,} {type_info}{null_info}")
        
        # Run optimization algorithm
        start_time = time.time()
        row_order, steps, clusters = self._compress_table(col_arrays, col_stats)
        elapsed = time.time() - start_time
        
        # Calculate compression metrics
        original_clusters = sum(s.cardinality for s in col_stats)
        compression_ratio = original_clusters / max(clusters, 1)
        
        # Reorder table using optimized indices
        optimized_table = table.take(row_order)
        
        result = {
            'row_order': row_order,
            'table': optimized_table,
            'steps': steps,
            'clusters': clusters,
            'compression_ratio': compression_ratio,
            'time': elapsed,
            'columns_optimized': column_names,
            'num_rows': table.num_rows
        }
        
        if self.verbose:
            print(f"\nOptimization complete!")
            print(f"  Steps: {steps:,}")
            print(f"  RLE clusters: {clusters:,} (was {original_clusters:,})")
            print(f"  Compression improvement: {compression_ratio:.2f}x")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Throughput: {table.num_rows / elapsed:,.0f} rows/sec")
        
        return result
    
    def _prepare_data(
        self,
        data: Union[pd.DataFrame, pa.Table, str],
        columns: Optional[List[str]]
    ) -> Tuple[pa.Table, List[str]]:
        """
        Prepare data as columnar table with appropriate encoding.
        
        This method:
        1. Loads data from various sources
        2. Applies dictionary encoding to string/categorical columns
        3. Validates and selects columns for optimization
        4. Returns table ready for the optimization algorithm
        """
        # Load from Parquet file
        if isinstance(data, str):
            if self.verbose:
                print(f"Loading Parquet: {data}")
            table = pq.read_table(data, columns=columns)
        
        # Convert from pandas DataFrame
        elif isinstance(data, pd.DataFrame):
            table = pa.Table.from_pandas(data, preserve_index=False)
        
        # Already a PyArrow Table
        elif isinstance(data, pa.Table):
            table = data
        
        else:
            raise TypeError(
                f"Data must be pandas DataFrame, PyArrow Table, or Parquet file path. "
                f"Got: {type(data)}"
            )
        
        # Select columns (exclude large binary/text types that can't be optimized)
        if columns is None:
            columns = [
                name for name, dtype in zip(table.column_names, table.schema.types)
                if not pa.types.is_large_binary(dtype) and
                   not pa.types.is_large_string(dtype) and
                   not pa.types.is_binary(dtype)
            ]
        
        if len(columns) == 0:
            raise ValueError("No valid columns found to optimize")
        
        # Dictionary-encode string and categorical columns for efficient compression
        encoded_arrays = {}
        for name in columns:
            col = table.column(name)
            col_type = col.type
            
            # Already dictionary-encoded - keep as-is
            if pa.types.is_dictionary(col_type):
                encoded_arrays[name] = col
            
            # String columns - apply dictionary encoding
            elif pa.types.is_string(col_type) or pa.types.is_large_string(col_type):
                encoded_arrays[name] = col.dictionary_encode()
            
            # Temporal types - keep as-is
            elif (pa.types.is_timestamp(col_type) or 
                  pa.types.is_date(col_type) or
                  pa.types.is_time(col_type)):
                encoded_arrays[name] = col
            
            # Numeric types - keep as-is
            elif (pa.types.is_integer(col_type) or 
                  pa.types.is_floating(col_type) or
                  pa.types.is_decimal(col_type)):
                encoded_arrays[name] = col
            
            # Boolean - keep as-is
            elif pa.types.is_boolean(col_type):
                encoded_arrays[name] = col
            
            # Unsupported type
            else:
                warnings.warn(
                    f"Column '{name}' has unsupported type {col_type}, skipping",
                    UserWarning
                )
                continue
        
        if len(encoded_arrays) == 0:
            raise ValueError("No columns with supported types found")
        
        encoded_table = pa.table(encoded_arrays)
        return encoded_table, list(encoded_arrays.keys())
    
    def _compress_table(
        self,
        columns: List[pa.ChunkedArray],
        col_stats: List['_ColumnStats']
    ) -> Tuple[pa.Array, int, int]:
        """
        Core greedy bucket-splitting algorithm for row reordering.
        
        Algorithm steps:
        1. Initialize with all rows in one bucket
        2. While buckets remain:
           a. Get next unprocessed bucket
           b. For each column in bucket:
              - Build histogram of value frequencies
              - Find most frequent value
              - Calculate bit savings if we group this value
           c. Greedy selection: choose column/value with max bit savings
           d. Partition rows: matching values first, others after
           e. Create two new buckets:
              - Pure bucket: contains only matching values (RLE-friendly)
              - Impure bucket: contains remaining values (process further)
           f. Add buckets back to queue
        3. Return final row ordering
        
        Args:
            columns: List of column data arrays
            col_stats: Pre-computed statistics for each column
        
        Returns:
            Tuple of (optimized_row_indices, step_count, total_clusters)
        """
        num_rows = len(columns[0])
        num_cols = len(columns)
        
        # Initialize row indices (sequential: 0, 1, 2, ...)
        row_indices = pa.array(np.arange(num_rows, dtype=np.int32))
        
        # Initialize bucket queue with single bucket containing all rows
        buckets = deque([_Bucket(0, num_rows - 1)])
        
        step_count = 0
        cluster_count = [0] * num_cols
        
        # Main optimization loop: process buckets until none remain
        while buckets:
            bucket = buckets.popleft()
            
            # Skip buckets that are already marked as done
            if bucket.is_done:
                continue
            
            step_count += 1
            bucket_size = bucket.size()
            
            # Determine sampling rate for this bucket
            # For large buckets (>10K rows), sample ~1% for performance
            sample_step = 1
            if bucket_size >= SAMPLING_THRESHOLD:
                sample_step = int(bucket_size / SAMPLING_DIVISOR + SAMPLING_ADDER)
            
            # Greedy selection: find column/value pair with maximum bit savings
            best_savings = INITIAL_MAX_SAVINGS
            best_choice = None
            
            for col_idx in range(num_cols):
                # Skip columns already processed for this bucket
                if bucket.is_column_done(col_idx):
                    continue
                
                # Build histogram for this column within this bucket
                max_value, max_freq = self._histogram_builder.build_histogram(
                    row_indices, columns[col_idx],
                    bucket.start, bucket.end,
                    sample_step, col_stats[col_idx]
                )
                
                # Handle all-null columns
                if max_value is None:
                    bucket.mark_column_done(col_idx)
                    continue
                
                # Calculate bit savings: frequency × bits_per_value
                # This estimates how many bits we save by RLE-encoding this value
                savings = max_freq * col_stats[col_idx].bits_per_value
                
                # Check threshold: is this worth splitting?
                if savings < BIT_SAVINGS_THRESHOLD:
                    bucket.mark_column_done(col_idx)
                    continue
                
                # Greedy: is this better than our current best?
                if savings > best_savings:
                    best_savings = savings
                    best_choice = (col_idx, max_value, max_freq)
            
            # No profitable split found - mark bucket as done
            if best_choice is None:
                bucket.is_done = True
                continue
            
            # Partition rows based on selected column/value
            col_idx, target_value, _ = best_choice
            row_indices, match_count = _partition_rows(
                row_indices, columns[col_idx],
                bucket.start, bucket.end,
                target_value, self.null_strategy
            )
            
            # Check minimum split size
            # Don't split if matching group is too small
            if match_count < MIN_SPLIT_SIZE:
                bucket.is_done = True
                buckets.append(bucket)
                continue
            
            # Check if all rows match (bucket is now pure for this column)
            non_match_count = bucket_size - match_count
            if non_match_count < MIN_SPLIT_SIZE:
                # All (or almost all) rows match - mark column as done
                bucket.mark_column_done(col_idx)
                cluster_count[col_idx] += 1
                buckets.append(bucket)
                continue
            
            # Create two new buckets from the split
            
            # Pure bucket: contains only matching values
            # This is RLE-friendly - all values are identical for this column
            pure_bucket = _Bucket(bucket.start, bucket.start + match_count - 1)
            pure_bucket.columns_done = bucket.columns_done
            pure_bucket.mark_column_done(col_idx)
            
            # Impure bucket: contains remaining values
            # This needs further processing
            impure_bucket = _Bucket(bucket.start + match_count, bucket.end)
            impure_bucket.columns_done = bucket.columns_done
            
            # Add buckets back to queue for further processing
            buckets.append(pure_bucket)
            buckets.append(impure_bucket)
            
            # Track cluster creation (one RLE run created for this column)
            cluster_count[col_idx] += 1
        
        return row_indices, step_count, sum(cluster_count)


# ============================================================================
# Helper Classes
# ============================================================================

class _Bucket:
    """
    Represents a contiguous range of rows being processed together.
    
    A bucket is the fundamental unit of the algorithm. Each bucket:
    - Spans a range of rows [start, end] (inclusive)
    - Tracks which columns have been fully processed
    - Can be split into sub-buckets based on value groupings
    
    Buckets are recursively subdivided until no further compression
    improvement is possible.
    """
    __slots__ = ['start', 'end', 'columns_done', 'is_done']
    
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
        self.columns_done = 0  # Bit flags for completed columns
        self.is_done = False
    
    def size(self) -> int:
        """Return the number of rows in this bucket."""
        return self.end - self.start + 1
    
    def mark_column_done(self, col_idx: int):
        """
        Mark a column as completed for this bucket.
        
        A column is marked done when:
        - All values in the bucket are identical (pure)
        - Bit savings are below threshold
        - Further processing won't yield improvement
        """
        self.columns_done |= (1 << col_idx)
    
    def is_column_done(self, col_idx: int) -> bool:
        """Check if a column has been fully processed for this bucket."""
        return (self.columns_done & (1 << col_idx)) != 0


class _ColumnStats:
    """
    Pre-computed statistics for a column.
    
    These statistics are computed once at the start and used throughout
    the optimization process for:
    - Calculating bit savings
    - Determining histogram modes
    - Optimizing dictionary-encoded columns
    """
    __slots__ = ['cardinality', 'min_val', 'max_val', 'bits_per_value',
                 'null_count', 'is_dictionary', 'dictionary_size']
    
    def __init__(self, column: pa.ChunkedArray):
        """Compute column statistics."""
        self.null_count = column.null_count
        
        # Dictionary-encoded columns (strings, categories)
        if pa.types.is_dictionary(column.type):
            self.is_dictionary = True
            dictionary = column.chunk(0).dictionary
            self.dictionary_size = len(dictionary)
            self.cardinality = self.dictionary_size
            
            # Dictionary indices are 0-based integers
            self.min_val = 0
            self.max_val = self.dictionary_size - 1
        
        # Non-dictionary columns
        else:
            self.is_dictionary = False
            self.dictionary_size = 0
            
            # Count distinct values (null-aware)
            self.cardinality = pc.count_distinct(column, mode='only_valid').as_py()
            
            if self.cardinality == 0:
                # All nulls
                self.min_val = 0
                self.max_val = 0
            else:
                # Compute min/max (skip nulls)
                min_result = pc.min(column, skip_nulls=True)
                max_result = pc.max(column, skip_nulls=True)
                
                # Handle temporal types
                if pa.types.is_temporal(column.type):
                    # Convert to integer representation
                    self.min_val = min_result.cast(pa.int64()).as_py()
                    self.max_val = max_result.cast(pa.int64()).as_py()
                else:
                    self.min_val = min_result.as_py()
                    self.max_val = max_result.as_py()
        
        # Calculate bits per value (log2 of cardinality)
        # This represents the minimum bits needed to encode each distinct value
        if self.cardinality <= 1:
            self.bits_per_value = 0.0
        else:
            self.bits_per_value = np.log2(self.cardinality)


class _HistogramBuilder:
    """
    Builds frequency histograms for columns within buckets.
    
    The histogram builder:
    - Counts occurrences of each value in a bucket
    - Finds the most frequent value
    - Supports sampling for large buckets (>10K rows)
    - Handles dictionary-encoded and regular columns differently
    """
    
    def build_histogram(
        self,
        row_indices: pa.Array,
        column: pa.ChunkedArray,
        start: int,
        end: int,
        step: int,
        col_stats: _ColumnStats
    ) -> Tuple[Optional[int], int]:
        """
        Build histogram and return (most_frequent_value, frequency).
        
        Args:
            row_indices: Current row ordering
            column: Column data
            start: Start index in row_indices
            end: End index in row_indices
            step: Sampling step (1 = no sampling, >1 = sample every Nth row)
            col_stats: Pre-computed column statistics
        
        Returns:
            (most_frequent_value, scaled_frequency) or (None, 0) if all nulls
        """
        # Sample rows from the bucket
        sampled_indices = row_indices[start:end+1:step]
        
        # Get values for sampled rows
        sampled_values = pc.take(column, sampled_indices)
        
        # Dictionary-encoded columns: work with indices directly
        if col_stats.is_dictionary:
            indices = sampled_values.indices
            value_counts = pc.value_counts(indices)
            
            if len(value_counts) == 0:
                return None, 0
            
            counts_field = value_counts.field('counts')
            max_idx = pc.argmax(counts_field).as_py()
            
            most_frequent = value_counts.field('values')[max_idx].as_py()
            frequency = counts_field[max_idx].as_py()
            
            # Scale frequency by sampling step to estimate true count
            return most_frequent, frequency * step
        
        # Non-dictionary columns: standard histogram
        else:
            value_counts = pc.value_counts(sampled_values)
            
            if len(value_counts) == 0:
                return None, 0
            
            counts_field = value_counts.field('counts')
            max_idx = pc.argmax(counts_field).as_py()
            
            most_frequent = value_counts.field('values')[max_idx].as_py()
            frequency = counts_field[max_idx].as_py()
            
            return most_frequent, frequency * step


# ============================================================================
# Helper Functions
# ============================================================================

def _partition_rows(
    row_indices: pa.Array,
    column: pa.ChunkedArray,
    start: int,
    end: int,
    target_value,
    null_strategy: str = 'separate'
) -> Tuple[pa.Array, int]:
    """
    Partition rows by grouping matching values first.
    
    This is the core operation that enables RLE compression:
    By placing all matching values consecutively, we create
    long runs of identical values that compress efficiently.
    
    Algorithm:
    1. Scan bucket rows
    2. Separate into matching and non-matching groups
    3. Reorder: [matching rows] + [non-matching rows]
    4. Return new ordering and match count
    
    Args:
        row_indices: Current row ordering
        column: Column data
        start: Start index in bucket
        end: End index in bucket
        target_value: Value to partition by
        null_strategy: How to handle nulls
            'separate' - Group nulls together
            'scatter' - Nulls don't match any value
    
    Returns:
        (new_row_indices, match_count)
    """
    # Extract rows in current bucket
    range_indices = row_indices[start:end+1]
    
    # Get values for these rows
    values = pc.take(column, range_indices)
    
    # Create comparison mask
    if target_value is None:
        # Matching nulls specifically
        mask = pc.is_null(values)
    else:
        # Matching specific value
        if null_strategy == 'separate':
            # Nulls don't match non-null values (VertiPaq behavior)
            mask = pc.and_(
                pc.equal(values, pa.scalar(target_value)),
                pc.is_valid(values)
            )
        else:
            # Simple comparison (nulls automatically don't match)
            mask = pc.equal(values, pa.scalar(target_value))
    
    # Filter rows into matching and non-matching groups
    matching_indices = pc.filter(range_indices, mask)
    not_mask = pc.invert(mask)
    non_matching_indices = pc.filter(range_indices, not_mask)
    
    match_count = len(matching_indices)
    
    # Concatenate: matching first, then non-matching
    reordered = pa.concat_arrays([matching_indices, non_matching_indices])
    
    # Rebuild full row indices array
    if start == 0 and end == len(row_indices) - 1:
        # Entire array replaced
        new_row_indices = reordered
    else:
        # Replace middle section only
        before = row_indices[:start] if start > 0 else None
        after = row_indices[end+1:] if end < len(row_indices) - 1 else None
        
        parts = [p for p in [before, reordered, after] if p is not None]
        new_row_indices = pa.concat_arrays(parts)
    
    return new_row_indices, match_count


# ============================================================================
# Public API
# ============================================================================

def optimize_table(
    data: Union[pd.DataFrame, pa.Table, str],
    columns: Optional[List[str]] = None,
    verbose: bool = False,
    output_path: Optional[str] = None
) -> Dict:
    """
    Optimize row ordering for maximum RLE compression.
    
    This function reorders rows to group similar values together, maximizing
    the effectiveness of Run-Length Encoding compression used in columnar databases
    like Power BI, Analysis Services, and Parquet.
    
    Args:
        data: Input data as pandas DataFrame, PyArrow Table, or Parquet file path
        columns: Columns to optimize (default: all except binary)
        verbose: Print progress information
        output_path: Optional path to save optimized Parquet file
    
    Returns:
        Dictionary with optimization results:
            - 'row_order': Array of optimized row indices
            - 'table': Optimized data table
            - 'steps': Number of optimization iterations performed
            - 'clusters': Number of RLE clusters created
            - 'compression_ratio': Estimated compression improvement factor
            - 'time': Execution time in seconds
    
    Example - Basic Usage:
        >>> import pandas as pd
        >>> from vertipaq_optimizer import optimize_table
        >>> 
        >>> df = pd.read_csv('sales_data.csv')
        >>> result = optimize_table(df, verbose=True)
        >>> optimized_df = result['table'].to_pandas()
    
    Example - Parquet Workflow:
        >>> result = optimize_table(
        ...     'input.parquet',
        ...     output_path='output.parquet',
        ...     verbose=True
        ... )
    """
    optimizer = VertiPaqOptimizer(verbose=verbose)
    result = optimizer.optimize(data, columns)
    
    # Save to Parquet if requested
    if output_path:
        pq.write_table(
            result['table'],
            output_path,
            compression='zstd',
            use_dictionary=True
        )
        if verbose:
            import os
            file_size = os.path.getsize(output_path) / 1024 / 1024
            print(f"\nSaved to: {output_path} ({file_size:.1f} MB)")
    
    return result


def optimize_parquet(
    input_path: str,
    output_path: str,
    columns: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict:
    """
    Optimize Parquet file directly.
    
    Args:
        input_path: Input Parquet file path
        output_path: Output Parquet file path
        columns: Columns to optimize (default: all)
        verbose: Print progress and file size comparison
    
    Returns:
        Optimization results dictionary
    
    Example:
        >>> from vertipaq_optimizer import optimize_parquet
        >>> result = optimize_parquet('data.parquet', 'data_optimized.parquet')
    """
    if verbose:
        import os
        print(f"Input: {input_path}")
        input_size = os.path.getsize(input_path) / 1024 / 1024
        print(f"Size: {input_size:.1f} MB")
    
    # Optimize
    result = optimize_table(input_path, columns=columns, verbose=verbose, 
                          output_path=output_path)
    
    # Compare file sizes
    if verbose:
        import os
        output_size = os.path.getsize(output_path) / 1024 / 1024
        reduction = (1 - output_size / input_size) * 100
        
        print(f"\nComparison:")
        print(f"Output size: {output_size:.1f} MB")
        print(f"Reduction: {reduction:.1f}%")
        
        result['size_reduction'] = reduction
    
    return result


# ============================================================================
# Self-Test
# ============================================================================

if __name__ == "__main__":
    print(f"VertiPaq Optimizer v{__version__}")
    print("\nRunning self-test with mixed types...")
    
    # Create test data
    np.random.seed(42)
    test_df = pd.DataFrame({
        'ProductID': np.random.randint(1, 100, 10000),
        'Category': np.random.choice(['Electronics', 'Books', 'Clothing'], 10000),
        'Date': pd.date_range('2024-01-01', periods=10000, freq='1h'),
        'Quantity': np.random.randint(1, 50, 10000),
        'Price': np.random.uniform(10, 1000, 10000)
    })
    
    # Add some nulls
    null_mask = np.random.random(10000) < 0.05
    test_df.loc[null_mask, 'Quantity'] = None
    
    print(f"\nTest data:")
    print(f"  Shape: {test_df.shape}")
    print(f"  Types: {test_df.dtypes.to_dict()}")
    print(f"  Nulls: {test_df.isnull().sum().sum()}")
    
    # Optimize
    result = optimize_table(test_df, verbose=True)
    
    print(f"\nSelf-test passed!")
    print(f"  Optimized {result['num_rows']:,} rows in {result['time']:.2f}s")
    print(f"  Steps: {result['steps']:,}")
    print(f"  RLE clusters: {result['clusters']:,}")
    print(f"  Compression improvement: {result['compression_ratio']:.2f}x")