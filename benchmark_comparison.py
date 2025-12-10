"""
Benchmark: Pure Python vs DuckDB Implementations

Compares performance and validates identical results.
"""

import numpy as np
import pandas as pd
import time
from typing import Dict

# Import both implementations
from vertipaq_optimizer import optimize_table as optimize_python
from vertipaq_optimizer_duckdb import optimize_table_duckdb


def create_test_data(num_rows: int, cardinalities: list) -> pd.DataFrame:
    """Create test dataset with controlled cardinality"""
    np.random.seed(42)
    
    data = {}
    for i, card in enumerate(cardinalities):
        col_name = f"col{i}"
        data[col_name] = np.random.randint(0, card, num_rows)
    
    return pd.DataFrame(data)


def run_benchmark(df: pd.DataFrame, implementation: str) -> Dict:
    """Run optimization and return results"""
    print(f"\n{'='*60}")
    print(f"Running: {implementation}")
    print(f"{'='*60}")
    
    if implementation == "Python":
        result = optimize_python(df, verbose=True)
    else:
        result = optimize_table_duckdb(df, verbose=True)
    
    return result


def compare_results(result1: Dict, result2: Dict) -> bool:
    """Verify both implementations produce identical results"""
    print(f"\n{'='*60}")
    print("Comparing Results")
    print(f"{'='*60}")
    
    # Compare step counts
    steps_match = result1['steps'] == result2['steps']
    print(f"Steps:    {result1['steps']:,} vs {result2['steps']:,} {'✓' if steps_match else '✗'}")
    
    # Compare cluster counts
    clusters_match = result1['clusters'] == result2['clusters']
    print(f"Clusters: {result1['clusters']:,} vs {result2['clusters']:,} {'✓' if clusters_match else '✗'}")
    
    # Compare row ordering (should be identical)
    order_match = np.array_equal(result1['row_order'], result2['row_order'])
    print(f"Row order: {'Identical ✓' if order_match else 'Different ✗'}")

    # Time comparison
    speedup = result1['time'] / result2['time']
    print(f"\nPerformance:")
    print(f"  Python:  {result1['time']:.3f}s")
    print(f"  DuckDB:  {result2['time']:.3f}s")
    print(f"  Speedup: {speedup:.2f}x")
    
    all_match = steps_match and clusters_match and order_match
    
    if all_match:
        print(f"\n All results match! DuckDB is {speedup:.2f}x faster.")
    else:
        print(f"\n— Results differ!")
    
    return all_match


def main():
    """Run comprehensive benchmarks"""
    print("VertiPaq Optimizer - Benchmark Suite")
    print("="*60)
    
    test_cases = [
        # (num_rows, cardinalities, description)
        (10_000, [100, 50, 20], "Small - 10K rows, low cardinality"),
        (100_000, [1000, 500, 100], "Medium - 100K rows, medium cardinality"),
        # Uncomment for longer tests:
        # (1_000_000, [10000, 5000, 1000], "Large - 1M rows, high cardinality"),
    ]
    
    results_summary = []
    
    for num_rows, cardinalities, description in test_cases:
        print(f"\n\n{'#'*60}")
        print(f"Test Case: {description}")
        print(f"{'#'*60}")
        
        # Create test data
        print(f"\nGenerating {num_rows:,} rows with {len(cardinalities)} columns...")
        df = create_test_data(num_rows, cardinalities)
        
        print(f"Column cardinalities: {cardinalities}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # Run Python implementation
        result_python = run_benchmark(df, "Python")
        
        # Run DuckDB implementation
        result_duckdb = run_benchmark(df, "DuckDB")
        
        # Compare results
        match = compare_results(result_python, result_duckdb)
        
        # Store summary
        results_summary.append({
            'description': description,
            'num_rows': num_rows,
            'num_cols': len(cardinalities),
            'python_time': result_python['time'],
            'duckdb_time': result_duckdb['time'],
            'speedup': result_python['time'] / result_duckdb['time'],
            'steps': result_python['steps'],
            'clusters': result_python['clusters'],
            'match': match
        })
    
    # Print summary table
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")
    
    print(f"{'Test Case':<35} {'Rows':>10} {'Python':>8} {'DuckDB':>8} {'Speedup':>8} {'Match':>6}")
    print(f"{'-'*35} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    
    for result in results_summary:
        print(f"{result['description']:<35} "
              f"{result['num_rows']:>10,} "
              f"{result['python_time']:>7.2f}s "
              f"{result['duckdb_time']:>7.2f}s "
              f"{result['speedup']:>7.2f}x "
              f"{'✓' if result['match'] else '✗':>6}")
    
    print(f"\n{'='*60}")
    
    # Verdict
    all_match = all(r['match'] for r in results_summary)
    avg_speedup = np.mean([r['speedup'] for r in results_summary])
    
    if all_match:
        print(f"✓ All test cases passed!")
        print(f"Average speedup: {avg_speedup:.2f}x")
        print(f"\nConclusion: DuckDB implementation is {avg_speedup:.1f}x faster")
        print("while producing identical results.")
    else:
        print(f"✗ Some test cases failed!")
        print("Review the differences above.")


if __name__ == "__main__":
    main()
