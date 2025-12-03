#!/usr/bin/env python3
"""
Example: Using VertiPaq Optimizer

Demonstrates basic usage of the vertipaq_optimizer module
to optimize row ordering for better compression.
"""

import numpy as np
import pandas as pd
from vertipaq_optimizer import optimize_table

def example_basic():
    """Basic usage example"""
    print("=" * 60)
    print("Example 1: Basic Optimization")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'ProductID': np.random.randint(1, 100, 10000),
        'CategoryID': np.random.randint(1, 10, 10000),
        'StoreID': np.random.randint(1, 50, 10000),
        'Quantity': np.random.randint(1, 100, 10000)
    })
    
    print(f"Original data: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Optimize
    result = optimize_table(df, verbose=True)
    
    # Reorder dataframe
    optimized_df = df.iloc[result['row_order']]
    
    print(f"\n✓ Optimization complete!")
    print(f"  Steps taken: {result['steps']:,}")
    print(f"  RLE clusters: {result['clusters']:,}")
    print(f"  Improvement: {result['compression_ratio']:.2f}x")
    
    return optimized_df


def example_file_comparison():
    """Compare file sizes before and after"""
    print("\n" + "=" * 60)
    print("Example 2: File Size Comparison")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'Date': pd.date_range('2020-01-01', periods=100000, freq='1min'),
        'ProductID': np.random.randint(1, 1000, 100000),
        'StoreID': np.random.randint(1, 100, 100000),
        'Amount': np.random.randint(1, 10000, 100000)
    })
    
    # Convert date to numeric
    df['DateInt'] = (df['Date'] - df['Date'].min()).dt.days
    
    print(f"Data: {df.shape[0]:,} rows")
    
    # Save original
    df.to_parquet('temp_original.parquet', index=False)
    original_size = pd.read_parquet('temp_original.parquet').memory_usage(deep=True).sum()
    
    # Optimize and save
    result = optimize_table(df, columns=['DateInt', 'ProductID', 'StoreID'])
    optimized_df = df.iloc[result['row_order']]
    optimized_df.to_parquet('temp_optimized.parquet', index=False)
    optimized_size = pd.read_parquet('temp_optimized.parquet').memory_usage(deep=True).sum()
    
    print(f"\n📊 Results:")
    print(f"  Original memory: {original_size/1024/1024:.2f} MB")
    print(f"  Optimized memory: {optimized_size/1024/1024:.2f} MB")
    print(f"  Reduction: {(1 - optimized_size/original_size)*100:.1f}%")
    
    # Cleanup
    import os
    os.remove('temp_original.parquet')
    os.remove('temp_optimized.parquet')


def example_specific_columns():
    """Optimize specific columns only"""
    print("\n" + "=" * 60)
    print("Example 3: Optimize Specific Columns")
    print("=" * 60)
    
    # Create data with mix of high and low cardinality
    np.random.seed(42)
    df = pd.DataFrame({
        'ID': range(10000),  # High cardinality - skip this
        'Category': np.random.randint(1, 5, 10000),  # Low cardinality - optimize
        'SubCategory': np.random.randint(1, 20, 10000),  # Medium - optimize
        'Value': np.random.random(10000)  # Continuous - skip
    })
    
    print(f"Data: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Optimizing only: Category, SubCategory")
    
    # Optimize only specific columns
    result = optimize_table(
        df, 
        columns=['Category', 'SubCategory'],
        verbose=False
    )
    
    print(f"\n✓ Results:")
    print(f"  Steps: {result['steps']:,}")
    print(f"  Clusters: {result['clusters']:,}")
    print(f"  Columns optimized: {', '.join(result['columns_optimized'])}")


def example_numpy_arrays():
    """Working with NumPy arrays directly"""
    print("\n" + "=" * 60)
    print("Example 4: NumPy Arrays")
    print("=" * 60)
    
    # Create data as numpy arrays
    np.random.seed(42)
    data = {
        'column_a': np.random.randint(0, 100, 10000, dtype=np.int32),
        'column_b': np.random.randint(0, 50, 10000, dtype=np.int32),
        'column_c': np.random.randint(0, 10, 10000, dtype=np.int32)
    }
    
    print(f"Data: {len(data['column_a']):,} rows × {len(data)} columns")
    
    # Optimize
    result = optimize_table(data, verbose=False)
    
    # Apply ordering
    for col_name in data.keys():
        data[col_name] = data[col_name][result['row_order']]
    
    print(f"\n✓ Arrays reordered!")
    print(f"  Steps: {result['steps']:,}")
    print(f"  Clusters: {result['clusters']:,}")


if __name__ == "__main__":
    print("\n" + "🚀 VertiPaq Optimizer - Examples\n")
    
    # Run all examples
    example_basic()
    example_file_comparison()
    example_specific_columns()
    example_numpy_arrays()
    
    print("\n" + "=" * 60)
    print("✓ All examples completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("  • Try with your own data")
    print("  • Measure compression improvements")
    print("  • Integrate into your ETL pipeline")
    print("\n")
