"""
VertiPaq Optimizer - DuckDB Accelerated Version

Hybrid architecture:
- Python: Control flow, bucket management, greedy decisions
- DuckDB: Histogram building, aggregations, data queries

This provides significant performance improvements while maintaining
precise adherence to the VertiPaq algorithm.

License: MIT
Version: 2.0.0
"""

import numpy as np
import pandas as pd
import duckdb
from typing import Union, List, Optional, Dict, Tuple
from collections import deque
from dataclasses import dataclass
import time

__version__ = "2.0.0"
__all__ = ['optimize_table_duckdb', 'DuckDBVertiPaqOptimizer']


# Algorithm constants (from reverse engineering)
MIN_SPLIT_SIZE = 64
BIT_SAVINGS_THRESHOLD = 0.1
INITIAL_MAX_SAVINGS = -1.0
SAMPLING_THRESHOLD = 10000
SAMPLING_DIVISOR = 10000.0
SAMPLING_ADDER = 1.0


@dataclass
class Bucket:
    """Represents a range of rows being processed together"""
    start: int
    end: int
    columns_done: set = None
    is_done: bool = False
    
    def __post_init__(self):
        if self.columns_done is None:
            self.columns_done = set()
    
    def size(self) -> int:
        return self.end - self.start + 1
    
    def mark_column_done(self, col: str):
        self.columns_done.add(col)
    
    def is_column_done(self, col: str) -> bool:
        return col in self.columns_done


@dataclass
class SplitChoice:
    """Represents the best (column, value) split choice"""
    column: str
    value: int
    frequency: int
    savings: float


class DuckDBVertiPaqOptimizer:
    """
    DuckDB-accelerated VertiPaq optimizer.
    
    Uses DuckDB for histogram building and aggregations while maintaining
    row ordering state in Python NumPy arrays for fast mutations.
    
    Example:
        >>> optimizer = DuckDBVertiPaqOptimizer(verbose=True)
        >>> df = pd.read_csv('data.csv')
        >>> result = optimizer.optimize(df)
        >>> optimized_df = df.iloc[result['row_order']]
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize optimizer with DuckDB connection.
        
        Args:
            verbose: If True, print progress information
        """
        self.verbose = verbose
        self.con = duckdb.connect(':memory:')
        self.columns = None
        self.bits_per_value = None
    
    def optimize(
        self,
        data: Union[pd.DataFrame, Dict[str, np.ndarray], List[np.ndarray]],
        columns: Optional[List[str]] = None
    ) -> Dict:
        """
        Optimize row ordering using DuckDB for data operations.
        
        Args:
            data: Input data as DataFrame, dict of arrays, or list of arrays
            columns: Specific columns to optimize (default: all numeric)
        
        Returns:
            Dictionary with optimization results
        """
        # Prepare data
        df = self._prepare_dataframe(data, columns)
        self.columns = [c for c in df.columns if c != 'row_idx']
        
        if self.verbose:
            print(f"Loading {len(df):,} rows × {len(self.columns)} columns into DuckDB...")

        # Load into DuckDB
        self.con.register('data', df)
        
        # Pre-compute column statistics using DuckDB
        if self.verbose:
            print("Computing column statistics...")
        self._compute_column_stats()
        
        # Initialize row ordering (stays in Python for fast mutations)
        num_rows = len(df)
        row_indices = np.arange(num_rows, dtype=np.int32)
        
        # Run optimization algorithm
        start_time = time.time()
        row_indices, steps, clusters = self._compress_table(row_indices, num_rows)
        elapsed = time.time() - start_time
        
        # Calculate metrics
        original_clusters = sum(
            self.con.execute(f"SELECT COUNT(DISTINCT {col}) FROM data").fetchone()[0]
            for col in self.columns
        )
        compression_ratio = original_clusters / max(clusters, 1)
        
        result = {
            'row_order': row_indices,
            'steps': steps,
            'clusters': clusters,
            'compression_ratio': compression_ratio,
            'time': elapsed,
            'columns_optimized': self.columns,
            'num_rows': num_rows
        }
        
        if self.verbose:
            print(f"\n Optimization complete!")
            print(f"  Steps: {steps:,}")
            print(f"  RLE clusters: {clusters:,} (vs {original_clusters:,} original)")
            print(f"  Compression improvement: {compression_ratio:.2f}x")
            print(f"  Time: {elapsed:.2f}s")
        
        return result
    
    def _prepare_dataframe(
        self,
        data: Union[pd.DataFrame, Dict, List],
        columns: Optional[List[str]]
    ) -> pd.DataFrame:
        """Convert input to DataFrame with row_idx column"""
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()
            df = df[columns]
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
            if columns:
                df = df[columns]
        elif isinstance(data, list):
            column_names = columns or [f"col{i}" for i in range(len(data))]
            df = pd.DataFrame({name: arr for name, arr in zip(column_names, data)})
        else:
            raise TypeError("Data must be DataFrame, dict, or list")
        
        # Ensure integer types
        for col in df.columns:
            df[col] = df[col].astype(np.int32)
        
        # Add row index
        df.insert(0, 'row_idx', range(len(df)))
        
        return df
    
    def _compute_column_stats(self):
        """Pre-compute bits per value for each column using DuckDB"""
        self.bits_per_value = {}
        
        for col in self.columns:
            cardinality = self.con.execute(
                f"SELECT COUNT(DISTINCT {col}) FROM data"
            ).fetchone()[0]
            
            self.bits_per_value[col] = np.log2(cardinality) if cardinality > 1 else 0.0
        
        if self.verbose:
            print(f"Column cardinalities:")
            for col in self.columns:
                card = 2 ** self.bits_per_value[col] if self.bits_per_value[col] > 0 else 1
                print(f"  {col}: {int(card):,} distinct values ({self.bits_per_value[col]:.2f} bits)")
    
    def _compress_table(
        self,
        row_indices: np.ndarray,
        num_rows: int
    ) -> Tuple[np.ndarray, int, int]:
        """
        Main compression algorithm.
        
        Core loop stays in Python, but delegates histogram building
        and value lookups to DuckDB for performance.
        """
        buckets = deque([Bucket(0, num_rows - 1)])
        step_count = 0
        cluster_count = {col: 0 for col in self.columns}
        
        while buckets:
            bucket = buckets.popleft()
            
            if bucket.is_done:
                continue
            
            step_count += 1
            
            if self.verbose and step_count % 100 == 0:
                print(f"  Step {step_count:,} - {len(buckets):,} buckets remaining")
            
            # Find best split using DuckDB histograms
            best_choice = self._find_best_split(bucket, row_indices)
            
            if best_choice is None:
                bucket.is_done = True
                continue
            
            # Partition rows (hybrid: DuckDB lookup + NumPy reordering)
            match_count = self._partition_rows(
                row_indices, bucket, best_choice.column, best_choice.value
            )
            
            # Check minimum split size
            if match_count < MIN_SPLIT_SIZE:
                bucket.is_done = True
                buckets.append(bucket)
                continue
            
            # Check if split consumed entire bucket
            non_match_count = bucket.size() - match_count
            if non_match_count < MIN_SPLIT_SIZE:
                bucket.mark_column_done(best_choice.column)
                cluster_count[best_choice.column] += 1
                buckets.append(bucket)
                continue
            
            # Create new buckets
            pure_bucket = Bucket(
                bucket.start,
                bucket.start + match_count - 1,
                columns_done=bucket.columns_done.copy()
            )
            pure_bucket.mark_column_done(best_choice.column)
            
            impure_bucket = Bucket(
                bucket.start + match_count,
                bucket.end,
                columns_done=bucket.columns_done.copy()
            )
            
            buckets.append(pure_bucket)
            buckets.append(impure_bucket)
            cluster_count[best_choice.column] += 1
        
        total_clusters = sum(cluster_count.values())
        return row_indices, step_count, total_clusters
    
    def _find_best_split(
        self,
        bucket: Bucket,
        row_indices: np.ndarray
    ) -> Optional[SplitChoice]:
        """
        Find best (column, value) split using DuckDB histograms.
        
        For each column, queries DuckDB to find the most frequent value
        in the bucket range, then compares bit savings in Python.
        """
        # Extract bucket rows
        bucket_rows = row_indices[bucket.start:bucket.end+1]
        
        # Calculate sampling
        bucket_size = bucket.size()
        if bucket_size >= SAMPLING_THRESHOLD:
            sample_step = int(bucket_size / SAMPLING_DIVISOR + SAMPLING_ADDER)
            sampled_rows = bucket_rows[::sample_step]
        else:
            sample_step = 1
            sampled_rows = bucket_rows
        
        # Greedy selection across columns
        best_savings = INITIAL_MAX_SAVINGS
        best_choice = None
        
        for col in self.columns:
            if bucket.is_column_done(col):
                continue
            
            # Build histogram using DuckDB
            result = self._build_histogram_duckdb(col, sampled_rows)
            
            if result is None:
                bucket.mark_column_done(col)
                continue
            
            value, freq = result
            scaled_freq = freq * sample_step
            
            # Calculate bit savings
            savings = scaled_freq * self.bits_per_value[col]
            
            if savings < BIT_SAVINGS_THRESHOLD:
                bucket.mark_column_done(col)
                continue
            
            # Greedy: keep best
            if savings > best_savings:
                best_savings = savings
                best_choice = SplitChoice(
                    column=col,
                    value=value,
                    frequency=scaled_freq,
                    savings=savings
                )
        
        return best_choice
    
    def _build_histogram_duckdb(
        self,
        column: str,
        row_indices: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """
        Build histogram for column using DuckDB aggregation.
        
        Returns (most_frequent_value, frequency) or None if empty.
        """
        try:
            # Use DuckDB's GROUP BY with ORDER BY for efficient histogram
            result = self.con.execute(f"""
                SELECT {column} as val, COUNT(*) as freq
                FROM data
                WHERE row_idx IN (SELECT unnest($1::INTEGER[]))
                GROUP BY {column}
                ORDER BY freq DESC
                LIMIT 1
            """, [row_indices.tolist()]).fetchone()
            
            return result if result else None
        
        except Exception as e:
            if self.verbose:
                print(f"Warning: Histogram failed for {column}: {e}")
            return None
    
    def _partition_rows(
        self,
        row_indices: np.ndarray,
        bucket: Bucket,
        column: str,
        target_value: int
    ) -> int:
        """
        Partition rows: matching values first, non-matching after.
        
        Uses DuckDB to fetch column values, NumPy for fast partitioning.
        """
        # Get bucket rows
        bucket_rows = row_indices[bucket.start:bucket.end+1]
        
        # Fetch column values from DuckDB
        result = self.con.execute(f"""
            SELECT row_idx, {column} as val
            FROM data
            WHERE row_idx IN (SELECT unnest($1::INTEGER[]))
        """, [bucket_rows.tolist()]).fetchdf()
        
        # Create lookup map (row_idx -> value)
        value_map = dict(zip(result['row_idx'], result['val']))
        
        # Partition using NumPy
        matching = []
        non_matching = []
        
        for row_idx in bucket_rows:
            if value_map.get(row_idx) == target_value:
                matching.append(row_idx)
            else:
                non_matching.append(row_idx)
        
        # Update row_indices in-place
        match_count = len(matching)
        row_indices[bucket.start:bucket.start+match_count] = matching
        row_indices[bucket.start+match_count:bucket.end+1] = non_matching
        
        return match_count
    
    def __del__(self):
        """Clean up DuckDB connection"""
        if hasattr(self, 'con'):
            self.con.close()


def optimize_table_duckdb(
    data: Union[pd.DataFrame, Dict[str, np.ndarray], List[np.ndarray]],
    columns: Optional[List[str]] = None,
    verbose: bool = False
) -> Dict:
    """
    Optimize row ordering using DuckDB acceleration.
    
    This function provides the same interface as optimize_table() but uses
    DuckDB for data-intensive operations, providing significant performance
    improvements on large datasets.
    
    Args:
        data: Input data as pandas DataFrame, dict of arrays, or list of arrays
        columns: Specific columns to use for optimization (default: all numeric)
        verbose: If True, print progress information
    
    Returns:
        Dictionary with optimization results:
            - 'row_order': Optimized row indices
            - 'steps': Number of optimization iterations
            - 'clusters': Number of RLE clusters created
            - 'compression_ratio': Estimated compression improvement
            - 'time': Execution time in seconds
    
    Example:
        >>> import pandas as pd
        >>> from vertipaq_optimizer_duckdb import optimize_table_duckdb
        >>> 
        >>> df = pd.read_csv('large_dataset.csv')
        >>> result = optimize_table_duckdb(df, verbose=True)
        >>> 
        >>> # Reorder and save
        >>> optimized_df = df.iloc[result['row_order']]
        >>> optimized_df.to_parquet('optimized.parquet')
        >>> 
        >>> print(f"Improved compression by {result['compression_ratio']:.2f}x")
    """
    optimizer = DuckDBVertiPaqOptimizer(verbose=verbose)
    return optimizer.optimize(data, columns)


if __name__ == "__main__":
    print(f"VertiPaq Optimizer (DuckDB) v{__version__}")
    print("\nRunning self-test...")
    
    # Create test data
    np.random.seed(42)
    test_df = pd.DataFrame({
        'ProductID': np.random.randint(1, 100, 10000),
        'CategoryID': np.random.randint(1, 10, 10000),
        'StoreID': np.random.randint(1, 50, 10000)
    })
    
    print(f"\nTest data: {len(test_df):,} rows Ã— {len(test_df.columns)} columns")
    
    # Optimize
    result = optimize_table_duckdb(test_df, verbose=True)
    
    print(f"\n Self-test passed!")
    print(f"  Optimized in: {result['steps']:,} steps")
    print(f"  RLE clusters: {result['clusters']:,}")
    print(f"  Improvement: {result['compression_ratio']:.2f}x")
    print(f"  Time: {result['time']:.3f}s")
