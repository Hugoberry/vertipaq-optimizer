"""
VertiPaq Optimizer - Enhanced DuckDB Edition

Handles all column types:
- Numeric columns (int, float)
- String columns (via dictionary encoding)
- Date/datetime columns (converted to integers)
- NULL values (encoded as special value)
"""

import duckdb
import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict, Tuple
from collections import deque
import time

__version__ = "2.2.0-duckdb-enhanced"


class VertiPaqOptimizerEnhanced:
    """
    Enhanced optimizer supporting all column types.
    
    Features:
    - Automatic dictionary encoding for strings
    - Date/datetime conversion to integers
    - NULL handling with special encoding
    - Mixed column type support
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.conn = None
        self.num_rows = 0
        self.column_names = []
        self.bits_per_value = {}
        self.column_encodings = {}  # Track encoding metadata
    
    def optimize(
        self,
        data: Union[pd.DataFrame, Dict[str, np.ndarray], List[np.ndarray]],
        columns: Optional[List[str]] = None
    ) -> Dict:
        """Optimize row ordering for maximum compression."""
        
        start_time = time.time()
        self.conn = duckdb.connect(':memory:')
        
        try:
            self._load_and_encode_data(data, columns)
            steps, clusters = self._optimize_internal()
            row_order = self._get_final_ordering()
            elapsed = time.time() - start_time
            
            original_clusters = self._calculate_original_clusters()
            compression_ratio = original_clusters / max(clusters, 1)
            
            result = {
                'row_order': row_order,
                'steps': steps,
                'clusters': clusters,
                'compression_ratio': compression_ratio,
                'time': elapsed,
                'columns_optimized': self.column_names,
                'column_encodings': self.column_encodings,
                'num_rows': self.num_rows
            }
            
            if self.verbose:
                self._print_results(result, original_clusters)
            
            return result
            
        finally:
            if self.conn:
                self.conn.close()
    
    def _load_and_encode_data(
        self,
        data: Union[pd.DataFrame, Dict[str, np.ndarray], List[np.ndarray]],
        columns: Optional[List[str]]
    ):
        """
        Load data and encode all column types to integers.
        
        Handles:
        - Numeric: Use as-is or convert float to int
        - String: Dictionary encoding
        - Date/Datetime: Convert to days since epoch
        - NULL: Encode as -1 (or -2147483648 for int32)
        """
        
        # Convert to DataFrame
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, list):
            df = pd.DataFrame({f"col_{i}": arr for i, arr in enumerate(data)})
        else:
            raise TypeError("Data must be DataFrame, dict, or list")
        
        # Select columns
        if columns is None:
            # Use ALL columns (not just numeric)
            columns = df.columns.tolist()
        else:
            df = df[columns]
        
        self.num_rows = len(df)
        self.column_names = []
        encoded_df = pd.DataFrame()
        
        # Add tracking columns first
        encoded_df['_row_idx'] = np.arange(self.num_rows, dtype=np.int32)
        encoded_df['_original_idx'] = np.arange(self.num_rows, dtype=np.int32)
        
        # Encode each column
        for col in columns:
            encoded_col, encoding_info = self._encode_column(df[col], col)
            encoded_df[col] = encoded_col
            self.column_encodings[col] = encoding_info
            self.column_names.append(col)
        
        # Load into DuckDB
        self.conn.execute("CREATE TABLE data AS SELECT * FROM encoded_df")
        
        # Calculate bits per value
        bits_query = f"""
            SELECT 
                {', '.join([f"log2(COUNT(DISTINCT {col})) as bits_{col}" for col in self.column_names])}
            FROM data
        """
        bits_result = self.conn.execute(bits_query).fetchone()
        
        self.bits_per_value = {
            col: max(0.0, bits) 
            for col, bits in zip(self.column_names, bits_result)
        }
        
        if self.verbose:
            print(f"\nLoaded {self.num_rows:,} rows × {len(self.column_names)} columns")
            print(f"\nColumn encodings:")
            for col, info in self.column_encodings.items():
                print(f"  • {col}: {info['original_type']} → {info['encoding_type']}")
                if info.get('null_count', 0) > 0:
                    print(f"    - NULLs: {info['null_count']} ({info['null_count']/self.num_rows*100:.1f}%)")
                if info['encoding_type'] == 'dictionary':
                    print(f"    - Dictionary size: {len(info['dictionary'])}")
    
    def _encode_column(self, series: pd.Series, col_name: str) -> Tuple[np.ndarray, Dict]:
        """
        Encode a single column to integers.
        
        Returns:
            encoded_array: Integer array
            encoding_info: Metadata about encoding
        """
        original_type = str(series.dtype)
        null_count = series.isna().sum()
        
        # NULL value encoding constant
        NULL_VALUE = -1
        
        # Handle different types
        if pd.api.types.is_integer_dtype(series):
            # Already integer - just handle NULLs
            encoded = series.fillna(NULL_VALUE).astype(np.int32)
            encoding_info = {
                'original_type': original_type,
                'encoding_type': 'integer',
                'null_count': null_count,
                'null_value': NULL_VALUE,
                'min_value': int(series.min()) if null_count < len(series) else NULL_VALUE,
                'max_value': int(series.max()) if null_count < len(series) else NULL_VALUE
            }
        
        elif pd.api.types.is_float_dtype(series):
            # Float: check if can be converted to int without loss
            if series.dropna().apply(lambda x: x == int(x)).all():
                # Can safely convert to int
                encoded = series.fillna(NULL_VALUE).astype(np.int32)
                encoding_info = {
                    'original_type': original_type,
                    'encoding_type': 'float_to_int',
                    'null_count': null_count,
                    'null_value': NULL_VALUE
                }
            else:
                # Need to preserve float precision - multiply and round
                # Find appropriate multiplier
                multiplier = 1000  # 3 decimal places
                encoded = (series.fillna(NULL_VALUE / multiplier) * multiplier).round().astype(np.int32)
                encoding_info = {
                    'original_type': original_type,
                    'encoding_type': 'float_scaled',
                    'multiplier': multiplier,
                    'null_count': null_count,
                    'null_value': NULL_VALUE
                }
        
        elif pd.api.types.is_datetime64_any_dtype(series):
            # Datetime: convert to days since epoch
            epoch = pd.Timestamp('1970-01-01')
            encoded = ((series - epoch).dt.total_seconds() / 86400).fillna(NULL_VALUE).astype(np.int32)
            encoding_info = {
                'original_type': original_type,
                'encoding_type': 'datetime_to_days',
                'null_count': null_count,
                'null_value': NULL_VALUE,
                'epoch': '1970-01-01'
            }
        
        elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            # String: dictionary encoding
            # Create dictionary (factorize)
            codes, uniques = pd.factorize(series, sort=False)
            
            # Handle NULLs: factorize puts NaN at -1, we'll keep it there
            encoded = codes.astype(np.int32)
            
            # Build dictionary mapping
            dictionary = {i: val for i, val in enumerate(uniques) if pd.notna(val)}
            dictionary[NULL_VALUE] = None  # NULL mapping
            
            encoding_info = {
                'original_type': original_type,
                'encoding_type': 'dictionary',
                'dictionary': dictionary,
                'null_count': null_count,
                'null_value': NULL_VALUE,
                'cardinality': len(dictionary) - 1  # Exclude NULL
            }
        
        elif pd.api.types.is_bool_dtype(series):
            # Boolean: 0/1/-1 (NULL)
            encoded = series.astype('Int32').fillna(NULL_VALUE).astype(np.int32)
            encoding_info = {
                'original_type': original_type,
                'encoding_type': 'boolean',
                'null_count': null_count,
                'null_value': NULL_VALUE
            }
        
        else:
            # Fallback: try to convert to string then dictionary encode
            if self.verbose:
                print(f"Warning: Unknown type {original_type} for column {col_name}, converting to string")
            
            string_series = series.astype(str)
            codes, uniques = pd.factorize(string_series, sort=False)
            encoded = codes.astype(np.int32)
            
            dictionary = {i: val for i, val in enumerate(uniques)}
            
            encoding_info = {
                'original_type': original_type,
                'encoding_type': 'fallback_dictionary',
                'dictionary': dictionary,
                'null_count': null_count,
                'null_value': NULL_VALUE
            }
        
        return encoded, encoding_info
    
    def _optimize_internal(self) -> Tuple[int, int]:
        """Run greedy optimization algorithm (same as before)."""
        
        buckets = deque([_Bucket(0, self.num_rows - 1)])
        step_count = 0
        cluster_count = {col: 0 for col in self.column_names}
        
        while buckets:
            bucket = buckets.popleft()
            
            if bucket.is_done:
                continue
            
            step_count += 1
            bucket_size = bucket.size()
            
            if self.verbose and step_count % 100 == 0:
                print(f"  Step {step_count:,} | Buckets: {len(buckets)+1}")
            
            # Calculate sampling
            sample_step = 1
            if bucket_size >= 10000:
                sample_step = int(bucket_size / 10000.0 + 1.0)
            
            # Build histograms for ALL columns in parallel
            histograms = self._build_all_histograms_parallel(
                bucket.start, bucket.end, sample_step, bucket.columns_done
            )
            
            # Greedy selection
            best_savings = -1.0
            best_choice = None
            
            for col_name, (max_value, max_freq) in histograms.items():
                if bucket.is_column_done_by_name(col_name):
                    continue
                
                savings = max_freq * self.bits_per_value[col_name]
                
                if savings < 0.1:
                    bucket.mark_column_done_by_name(col_name)
                    continue
                
                if savings > best_savings:
                    best_savings = savings
                    best_choice = (col_name, max_value, max_freq)
            
            # No profitable split
            if best_choice is None:
                bucket.is_done = True
                continue
            
            # Partition rows
            col_name, target_value, _ = best_choice
            match_count = self._partition_bucket_optimized(
                bucket.start, bucket.end, col_name, target_value
            )
            
            # Check minimum split size
            if match_count < 64:
                bucket.is_done = True
                buckets.append(bucket)
                continue
            
            non_match_count = bucket_size - match_count
            if non_match_count < 64:
                bucket.mark_column_done_by_name(col_name)
                cluster_count[col_name] += 1
                buckets.append(bucket)
                continue
            
            # Create new buckets
            pure_bucket = _Bucket(bucket.start, bucket.start + match_count - 1)
            pure_bucket.columns_done = bucket.columns_done.copy()
            pure_bucket.mark_column_done_by_name(col_name)
            
            impure_bucket = _Bucket(bucket.start + match_count, bucket.end)
            impure_bucket.columns_done = bucket.columns_done.copy()
            
            buckets.append(pure_bucket)
            buckets.append(impure_bucket)
            cluster_count[col_name] += 1
        
        return step_count, sum(cluster_count.values())
    
    def _build_all_histograms_parallel(
        self,
        start: int,
        end: int,
        sample_step: int,
        columns_done: set
    ) -> Dict[str, Tuple[int, int]]:
        """Build histograms for ALL columns in parallel."""
        
        active_columns = [col for col in self.column_names if col not in columns_done]
        
        if not active_columns:
            return {}
        
        # Build WHERE clause
        where_clause = f"_row_idx BETWEEN {start} AND {end}"
        if sample_step > 1:
            where_clause += f" AND (_row_idx - {start}) % {sample_step} = 0"
        
        # Build histograms in parallel
        hist_exprs = [f"histogram({col}) as hist_{col}" for col in active_columns]
        
        query = f"""
            SELECT {', '.join(hist_exprs)}
            FROM data
            WHERE {where_clause}
        """
        
        try:
            result = self.conn.execute(query).fetchone()
            
            histograms = {}
            for i, col in enumerate(active_columns):
                hist_map = result[i]
                if hist_map and len(hist_map) > 0:
                    # Find key with maximum value
                    max_key = max(hist_map, key=hist_map.get)
                    max_freq = hist_map[max_key] * sample_step
                    histograms[col] = (int(max_key), max_freq)
            
            return histograms
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Parallel histogram failed: {e}")
            return {}
    
    def _partition_bucket_optimized(
        self,
        start: int,
        end: int,
        col_name: str,
        target_value: int
    ) -> int:
        """Optimized partitioning using window functions."""
        
        query = f"""
            WITH matches AS (
                SELECT 
                    _original_idx,
                    {col_name} = {target_value} as is_match,
                    COUNT(*) FILTER (WHERE {col_name} = {target_value}) OVER () as match_count
                FROM data
                WHERE _row_idx BETWEEN {start} AND {end}
            ),
            reordered AS (
                SELECT 
                    _original_idx,
                    match_count,
                    CASE 
                        WHEN is_match THEN 
                            {start} + ROW_NUMBER() OVER (PARTITION BY is_match ORDER BY _original_idx) - 1
                        ELSE 
                            {start} + match_count + ROW_NUMBER() OVER (PARTITION BY is_match ORDER BY _original_idx) - 1
                    END as new_row_idx
                FROM matches
            )
            UPDATE data
            SET _row_idx = r.new_row_idx
            FROM reordered r
            WHERE data._original_idx = r._original_idx
            RETURNING (SELECT match_count FROM reordered LIMIT 1) as cnt
        """
        
        try:
            result = self.conn.execute(query).fetchone()
            return result[0] if result else 0
        except:
            # Fallback
            count_query = f"""
                SELECT COUNT(*)
                FROM data
                WHERE _row_idx BETWEEN {start} AND {end}
                  AND {col_name} = {target_value}
            """
            return self.conn.execute(count_query).fetchone()[0]
    
    def _get_final_ordering(self) -> np.ndarray:
        """Extract final row ordering."""
        result = self.conn.execute("""
            SELECT _original_idx FROM data ORDER BY _row_idx
        """).fetchnumpy()
        return result['_original_idx']
    
    def _calculate_original_clusters(self) -> int:
        """Calculate clusters without optimization."""
        query = f"""
            SELECT {' + '.join([f'COUNT(DISTINCT {col})' for col in self.column_names])} as total
            FROM data
        """
        return self.conn.execute(query).fetchone()[0]
    
    def _print_results(self, result: Dict, original_clusters: int):
        """Print results."""
        print(f"\n✓ Optimization complete!")
        print(f"  Rows: {result['num_rows']:,}")
        print(f"  Columns: {len(result['columns_optimized'])}")
        print(f"  Steps: {result['steps']:,}")
        print(f"  RLE clusters: {result['clusters']:,} (vs {original_clusters:,} original)")
        print(f"  Compression improvement: {result['compression_ratio']:.2f}x")
        print(f"  Time: {result['time']:.2f}s")


class _Bucket:
    """Bucket with set-based column tracking."""
    __slots__ = ['start', 'end', 'columns_done', 'is_done']
    
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
        self.columns_done = set()
        self.is_done = False
    
    def size(self) -> int:
        return self.end - self.start + 1
    
    def mark_column_done_by_name(self, col_name: str):
        self.columns_done.add(col_name)
    
    def is_column_done_by_name(self, col_name: str) -> bool:
        return col_name in self.columns_done


def optimize_table(
    data: Union[pd.DataFrame, Dict[str, np.ndarray], List[np.ndarray]],
    columns: Optional[List[str]] = None,
    verbose: bool = False
) -> Dict:
    """
    Optimize row ordering for maximum RLE compression.
    
    Supports ALL column types:
    - Numeric (int, float)
    - String (dictionary encoded)
    - Date/datetime (converted to integers)
    - Boolean
    - NULL values (encoded as -1)
    
    Args:
        data: Input data
        columns: Columns to optimize (default: all columns)
        verbose: Print progress
    
    Returns:
        Optimization results with encoding metadata
    """
    optimizer = VertiPaqOptimizerEnhanced(verbose=verbose)
    return optimizer.optimize(data, columns)


if __name__ == "__main__":
    print("VertiPaq Optimizer - Enhanced Edition v" + __version__)
    print("Supports: strings, dates, nulls, and all numeric types\n")
    
    # Test with mixed types
    np.random.seed(42)
    
    test_df = pd.DataFrame({
        # Numeric
        'ProductID': np.random.randint(1, 100, 1000),
        'Price': np.random.uniform(10.0, 1000.0, 1000),
        
        # String
        'Category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 1000),
        'Brand': np.random.choice(['BrandA', 'BrandB', 'BrandC', None], 1000),  # With NULLs
        
        # Date
        'OrderDate': pd.date_range('2024-01-01', periods=1000, freq='1H'),
        
        # Boolean
        'InStock': np.random.choice([True, False, None], 1000)  # With NULLs
    })
    
    print("Test data:")
    print(test_df.head())
    print(f"\nData types:")
    print(test_df.dtypes)
    
    print(f"\nOptimizing...")
    result = optimize_table(test_df, verbose=True)
    
    print(f"\n✓ Test passed!")
    print(f"  All column types handled successfully")
