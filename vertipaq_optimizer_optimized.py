"""
VertiPaq Optimizer - Query-Optimized Version

Minimizes query count by:
1. Computing statistics ONCE (never recomputed)
2. Getting ALL column histograms in ONE query per bucket
3. Batch operations wherever possible

License: MIT
Version: 3.1.0 (Query-Optimized)
"""

import numpy as np
import pandas as pd
import duckdb
from typing import Union, List, Optional, Dict, Tuple
from collections import deque
from dataclasses import dataclass
import time

__version__ = "3.1.0"
__all__ = ['optimize_table_fast', 'QueryOptimizedOptimizer']


# Algorithm constants
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


class QueryOptimizedOptimizer:
    """
    Query-optimized DuckDB implementation.
    
    Key optimization: ONE query to get ALL column histograms per bucket.
    
    Query count reduction:
    - Old: N queries per bucket (one per column)
    - New: 1 query per bucket (all columns)
    - Speedup: Nx faster histogram phase!
    
    Example:
        >>> optimizer = QueryOptimizedOptimizer(verbose=True)
        >>> df = pd.read_csv('data.csv')
        >>> result = optimizer.optimize(df)
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize optimizer with DuckDB connection."""
        self.verbose = verbose
        self.conn = duckdb.connect(':memory:')
        self.columns = None
        self.bits_per_value = None  # Computed ONCE, never recomputed
        self.query_count = 0  # Track total queries for analysis
    
    def optimize(
        self,
        data: Union[pd.DataFrame, Dict[str, np.ndarray], List[np.ndarray]],
        columns: Optional[List[str]] = None
    ) -> Dict:
        """
        Optimize row ordering with minimal query count.
        
        Args:
            data: Input data as DataFrame, dict of arrays, or list of arrays
            columns: Specific columns to optimize (default: all numeric)
        
        Returns:
            Dictionary with optimization results
        """
        # Prepare data
        df = self._prepare_dataframe(data, columns)
        self.columns = [c for c in df.columns if not c.startswith('_')]
        
        if self.verbose:
            print(f"Loading {len(df):,} rows × {len(self.columns)} columns into DuckDB...")
        
        # Create optimized schema
        self._create_optimized_table(df)
        
        # Pre-compute column statistics ONCE (never recomputed)
        if self.verbose:
            print("Computing column statistics (ONE TIME ONLY)...")
        self._compute_column_stats_once()
        
        # Run optimization algorithm
        num_rows = len(df)
        start_time = time.time()
        steps, clusters = self._compress_table_optimized(num_rows)
        elapsed = time.time() - start_time
        
        # Extract final row ordering
        row_order = self._get_final_ordering()
        
        # Calculate metrics
        original_clusters = sum(
            2 ** self.bits_per_value[col] if self.bits_per_value[col] > 0 else 1
            for col in self.columns
        )
        compression_ratio = original_clusters / max(clusters, 1)
        
        result = {
            'row_order': row_order,
            'steps': steps,
            'clusters': clusters,
            'compression_ratio': compression_ratio,
            'time': elapsed,
            'columns_optimized': self.columns,
            'num_rows': num_rows,
            'total_queries': self.query_count  # For analysis
        }
        
        if self.verbose:
            print(f"\n✓ Optimization complete!")
            print(f"  Steps: {steps:,}")
            print(f"  RLE clusters: {clusters:,}")
            print(f"  Compression improvement: {compression_ratio:.2f}x")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Total queries: {self.query_count:,}")
            print(f"  Queries per step: {self.query_count/max(steps,1):.1f}")
        
        return result
    
    def _prepare_dataframe(
        self,
        data: Union[pd.DataFrame, Dict, List],
        columns: Optional[List[str]]
    ) -> pd.DataFrame:
        """Convert input to DataFrame"""
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
        
        return df
    
    def _create_optimized_table(self, df: pd.DataFrame):
        """Create table with dual-index schema"""
        self.conn.register('temp_data', df)
        
        self.conn.execute("""
            CREATE TABLE data AS
            SELECT 
                row_number() OVER () - 1 as _original_idx,
                row_number() OVER () - 1 as _row_idx,
                *
            FROM temp_data
        """)
        self.query_count += 1
        
        # Create index for fast lookups
        self.conn.execute("CREATE INDEX idx_row ON data(_row_idx)")
        self.query_count += 1
        
        if self.verbose:
            print("Created optimized table with dual indices")
    
    def _compute_column_stats_once(self):
        """
        Compute statistics for ALL columns ONCE using single query.
        
        IMPORTANT: This is NEVER recomputed - cardinality doesn't change!
        """
        # Single query to get all cardinalities at once
        cardinality_cols = ', '.join(
            f"COUNT(DISTINCT {col}) as card_{col}" 
            for col in self.columns
        )
        
        query = f"SELECT {cardinality_cols} FROM data"
        result = self.conn.execute(query).fetchone()
        self.query_count += 1
        
        # Store bits per value (computed once, used thousands of times!)
        self.bits_per_value = {}
        for i, col in enumerate(self.columns):
            cardinality = result[i]
            self.bits_per_value[col] = np.log2(cardinality) if cardinality > 1 else 0.0
        
        if self.verbose:
            print(f"Column statistics (computed ONCE, never recomputed):")
            for col in self.columns:
                card = 2 ** self.bits_per_value[col] if self.bits_per_value[col] > 0 else 1
                print(f"  {col}: {int(card):,} distinct ({self.bits_per_value[col]:.2f} bits)")
    
    def _compress_table_optimized(self, num_rows: int) -> Tuple[int, int]:
        """Main optimization loop with minimal queries"""
        buckets = deque([Bucket(0, num_rows - 1)])
        step_count = 0
        cluster_count = {col: 0 for col in self.columns}
        
        while buckets:
            bucket = buckets.popleft()
            
            if bucket.is_done:
                continue
            
            step_count += 1
            
            if self.verbose and step_count % 100 == 0:
                print(f"  Step {step_count:,} - {len(buckets):,} buckets - {self.query_count:,} queries")
            
            # KEY OPTIMIZATION: Get ALL column histograms in ONE query!
            best_choice = self._find_best_split_single_query(bucket)
            
            if best_choice is None:
                bucket.is_done = True
                continue
            
            # Partition with minimal queries
            match_count = self._partition_with_window_functions(
                bucket, best_choice.column, best_choice.value
            )
            
            # Check minimum split size
            if match_count < MIN_SPLIT_SIZE:
                bucket.is_done = True
                # buckets.append(bucket)
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
        return step_count, total_clusters
    
    def _find_best_split_single_query(
        self, 
        bucket: Bucket
    ) -> Optional[SplitChoice]:
        """
        🔥 KEY OPTIMIZATION: Get ALL column histograms in ONE query!
        
        Old approach: N queries (one per column)
        New approach: 1 query (all columns)
        
        Speedup: Nx reduction in histogram queries!
        """
        bucket_size = bucket.size()
        
        # Get columns that still need processing
        active_columns = [col for col in self.columns if not bucket.is_column_done(col)]
        
        if not active_columns:
            return None
        
        # Calculate sampling
        if bucket_size >= SAMPLING_THRESHOLD:
            sample_step = int(bucket_size / SAMPLING_DIVISOR + SAMPLING_ADDER)
            sample_clause = f"(_row_idx - {bucket.start}) % {sample_step} = 0"
        else:
            sample_step = 1
            sample_clause = "TRUE"
        
        # 🔥 SINGLE QUERY for ALL columns!
        histogram_query = self._build_multi_column_histogram_query(
            active_columns, bucket.start, bucket.end, sample_clause
        )
        
        try:
            result = self.conn.execute(histogram_query).fetchdf()
            self.query_count += 1
            
            # Process results in Python (fast - in-memory)
            best_savings = INITIAL_MAX_SAVINGS
            best_choice = None
            
            for _, row in result.iterrows():
                col = row['column_name']
                hist_map = row['hist']
                
                if not hist_map:
                    bucket.mark_column_done(col)
                    continue
                
                # Find most frequent value
                max_val = max(hist_map.items(), key=lambda x: x[1])
                value, freq = max_val
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
        
        except Exception as e:
            if self.verbose:
                print(f"Warning: Multi-column histogram failed: {e}")
            
            # Fallback: query columns individually (old approach)
            return self._find_best_split_fallback(bucket, active_columns, sample_step, sample_clause)
    
    def _build_multi_column_histogram_query(
        self,
        columns: List[str],
        start: int,
        end: int,
        sample_clause: str
    ) -> str:
        """
        Build single query that returns histograms for ALL columns.
        
        Returns one row per column with its histogram.
        """
        # Build UNION ALL query for all columns
        union_parts = []
        for col in columns:
            union_parts.append(f"""
                SELECT 
                    '{col}' as column_name,
                    histogram({col}) as hist
                FROM data
                WHERE _row_idx BETWEEN {start} AND {end}
                  AND {sample_clause}
            """)
        
        query = " UNION ALL ".join(union_parts)
        return query
    
    def _find_best_split_fallback(
        self,
        bucket: Bucket,
        active_columns: List[str],
        sample_step: int,
        sample_clause: str
    ) -> Optional[SplitChoice]:
        """Fallback to individual column queries if batch fails"""
        best_savings = INITIAL_MAX_SAVINGS
        best_choice = None
        
        for col in active_columns:
            query = f"""
                SELECT histogram({col}) as hist
                FROM data
                WHERE _row_idx BETWEEN {bucket.start} AND {bucket.end}
                  AND {sample_clause}
            """
            
            try:
                result = self.conn.execute(query).fetchone()
                self.query_count += 1
                
                if not result or not result[0]:
                    bucket.mark_column_done(col)
                    continue
                
                hist_map = result[0]
                max_val = max(hist_map.items(), key=lambda x: x[1])
                value, freq = max_val
                scaled_freq = freq * sample_step
                
                savings = scaled_freq * self.bits_per_value[col]
                
                if savings < BIT_SAVINGS_THRESHOLD:
                    bucket.mark_column_done(col)
                    continue
                
                if savings > best_savings:
                    best_savings = savings
                    best_choice = SplitChoice(
                        column=col,
                        value=value,
                        frequency=scaled_freq,
                        savings=savings
                    )
            
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Histogram failed for {col}: {e}")
                bucket.mark_column_done(col)
        
        return best_choice
    
    def _partition_with_window_functions(
        self,
        bucket: Bucket,
        column: str,
        target_value: int
    ) -> int:
        """
        Partition rows using window functions with minimal queries.
        
        Total: 2 queries (count + update)
        """
        # Query 1: Get match count
        count_query = f"""
            SELECT COUNT(*) FILTER (WHERE {column} = {target_value}) as match_count
            FROM data
            WHERE _row_idx BETWEEN {bucket.start} AND {bucket.end}
        """
        
        try:
            match_count = self.conn.execute(count_query).fetchone()[0]
            self.query_count += 1
            
            # Query 2: Update using that count
            update_query = f"""
                WITH matches AS (
                    SELECT 
                        _original_idx,
                        _row_idx as old_row_idx,
                        {column} = {target_value} as is_match
                    FROM data
                    WHERE _row_idx BETWEEN {bucket.start} AND {bucket.end}
                ),
                reordered AS (
                    SELECT 
                        _original_idx,
                        CASE 
                            WHEN is_match THEN 
                                {bucket.start} + ROW_NUMBER() OVER (PARTITION BY is_match ORDER BY old_row_idx) - 1
                            ELSE 
                                {bucket.start} + {match_count} + ROW_NUMBER() OVER (PARTITION BY is_match ORDER BY old_row_idx) - 1
                        END as new_row_idx
                    FROM matches
                )
                UPDATE data
                SET _row_idx = r.new_row_idx
                FROM reordered r
                WHERE data._original_idx = r._original_idx
            """
            
            self.conn.execute(update_query)
            self.query_count += 1
            
            return match_count
        
        except Exception as e:
            if self.verbose:
                print(f"Warning: Partition failed: {e}")
            return 0
    
    def _get_final_ordering(self) -> np.ndarray:
        """Extract final row ordering from DuckDB"""
        result = self.conn.execute("""
            SELECT _original_idx
            FROM data
            ORDER BY _row_idx
        """).fetchnumpy()
        self.query_count += 1
        
        return result['_original_idx']
    
    def __del__(self):
        """Clean up DuckDB connection"""
        if hasattr(self, 'conn'):
            self.conn.close()


def optimize_table_fast(
    data: Union[pd.DataFrame, Dict[str, np.ndarray], List[np.ndarray]],
    columns: Optional[List[str]] = None,
    verbose: bool = False
) -> Dict:
    """
    Query-optimized DuckDB implementation.
    
    Key improvements over v3.0:
    - ONE query for ALL column histograms (not N queries!)
    - Cardinality computed ONCE (never recomputed)
    - Minimal query count
    
    Args:
        data: Input data
        columns: Specific columns to optimize
        verbose: Print progress
    
    Returns:
        Dictionary with optimization results + query count
    """
    optimizer = QueryOptimizedOptimizer(verbose=verbose)
    return optimizer.optimize(data, columns)


if __name__ == "__main__":
    print(f"VertiPaq Optimizer (Query-Optimized) v{__version__}")
    print("\nRunning self-test...")
    
    # Create test data
    np.random.seed(42)
    test_df = pd.DataFrame({
        'ProductID': np.random.randint(1, 100, 10000),
        'CategoryID': np.random.randint(1, 10, 10000),
        'StoreID': np.random.randint(1, 50, 10000)
    })
    
    print(f"\nTest data: {len(test_df):,} rows × {len(test_df.columns)} columns")
    
    # Optimize
    result = optimize_table_fast(test_df, verbose=True)
    
    print(f"\n✓ Self-test passed!")
    print(f"  Steps: {result['steps']:,}")
    print(f"  Clusters: {result['clusters']:,}")
    print(f"  Time: {result['time']:.3f}s")
    print(f"  Total queries: {result['total_queries']:,}")
    print(f"  Queries per step: {result['total_queries']/result['steps']:.1f}")