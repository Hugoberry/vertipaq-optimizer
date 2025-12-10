#!/usr/bin/env python3
"""
Enhanced VertiPaq Optimizer Demo

Demonstrates handling of:
- Strings (dictionary encoding)
- Dates (converted to integers)
- NULLs (special encoding)
- Mixed types
"""

import pandas as pd
import numpy as np
from vertipaq_optimizer_enhanced import optimize_table


def demo_string_columns():
    """Demo: String columns with dictionary encoding"""
    print("\n" + "="*70)
    print("DEMO 1: String Columns (Dictionary Encoding)")
    print("="*70)
    
    df = pd.DataFrame({
        'ProductCategory': ['Electronics', 'Clothing', 'Electronics', 'Food',
                           'Electronics', 'Clothing', 'Food', 'Electronics',
                           'Clothing', 'Electronics'] * 100,
        
        'ShippingMethod': ['Standard', 'Express', 'Standard', 'Standard',
                          'Express', 'Standard', 'Overnight', 'Standard',
                          'Express', 'Standard'] * 100,
        
        'WarehouseLocation': ['US-West', 'US-East', 'US-West', 'EU-North',
                             'US-West', 'US-East', 'US-West', 'US-East',
                             'EU-North', 'US-West'] * 100
    })
    
    print(f"\nDataset: {len(df):,} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"\nData types:")
    print(df.dtypes)
    
    print(f"\nCardinality:")
    for col in df.columns:
        print(f"  {col}: {df[col].nunique()} unique values")
    
    print(f"\nOptimizing...")
    result = optimize_table(df, verbose=True)
    
    print(f"\nDictionary encodings created:")
    for col, info in result['column_encodings'].items():
        if info['encoding_type'] == 'dictionary':
            print(f"\n{col}:")
            print(f"  Dictionary: {info['dictionary']}")
            print(f"  Cardinality: {info['cardinality']}")
    
    # Show clustering effect
    optimized_df = df.iloc[result['row_order']]
    print(f"\nBefore optimization (first 20 rows):")
    print(df['ProductCategory'].head(20).tolist())
    print(f"\nAfter optimization (first 20 rows):")
    print(optimized_df['ProductCategory'].head(20).tolist())
    print(f"\nNotice how same categories are now grouped!")


def demo_null_handling():
    """Demo: NULL handling across different types"""
    print("\n" + "="*70)
    print("DEMO 2: NULL Handling")
    print("="*70)
    
    np.random.seed(42)
    
    df = pd.DataFrame({
        # Integer with NULLs
        'ProductID': [1, 2, None, 2, 1, None, 3, 1, 2, None] * 100,
        
        # String with NULLs
        'OptionalNotes': ['Note1', None, 'Note1', 'Note2', None, 
                         'Note1', None, None, 'Note2', 'Note1'] * 100,
        
        # Boolean with NULLs
        'IsGift': [True, False, None, True, None, 
                   False, True, None, False, True] * 100,
        
        # Float with NULLs
        'Discount': [0.10, None, 0.10, 0.20, 0.10,
                     None, 0.20, 0.10, None, 0.10] * 100
    })
    
    print(f"\nDataset: {len(df):,} rows")
    print(f"\nNULL statistics:")
    for col in df.columns:
        null_count = df[col].isna().sum()
        null_pct = null_count / len(df) * 100
        print(f"  {col}: {null_count} NULLs ({null_pct:.1f}%)")
    
    print(f"\nOptimizing...")
    result = optimize_table(df, verbose=True)
    
    print(f"\nNULL encoding details:")
    for col, info in result['column_encodings'].items():
        print(f"\n{col}:")
        print(f"  Type: {info['encoding_type']}")
        print(f"  NULL count: {info['null_count']}")
        print(f"  NULL encoded as: {info['null_value']}")
    
    # Show how NULLs cluster
    optimized_df = df.iloc[result['row_order']]
    
    print(f"\nProductID before optimization (first 30):")
    print(df['ProductID'].head(30).tolist())
    
    print(f"\nProductID after optimization (first 30):")
    print(optimized_df['ProductID'].head(30).tolist())
    
    print(f"\nNotice:")
    print(f"  1. Same values grouped (1, 1, 1...)")
    print(f"  2. NULLs grouped together (None, None, None...)")
    print(f"  3. Both form efficient RLE runs!")


def demo_date_columns():
    """Demo: Date/datetime handling"""
    print("\n" + "="*70)
    print("DEMO 3: Date/Datetime Columns")
    print("="*70)
    
    # Create data with repeating dates
    dates = pd.date_range('2024-01-01', periods=10, freq='1D').tolist() * 100
    np.random.shuffle(dates)
    
    df = pd.DataFrame({
        'OrderDate': dates,
        'ShipDate': [d + pd.Timedelta(days=np.random.randint(1, 4)) for d in dates],
        'ProductID': np.random.randint(1, 20, 1000)
    })
    
    print(f"\nDataset: {len(df):,} rows")
    print(f"\nDate range:")
    print(f"  OrderDate: {df['OrderDate'].min()} to {df['OrderDate'].max()}")
    print(f"  Unique dates: {df['OrderDate'].nunique()}")
    
    print(f"\nOptimizing...")
    result = optimize_table(df, verbose=True)
    
    print(f"\nDate encoding details:")
    for col, info in result['column_encodings'].items():
        if 'Date' in col:
            print(f"\n{col}:")
            print(f"  Encoding: {info['encoding_type']}")
            print(f"  Epoch: {info.get('epoch', 'N/A')}")
            print(f"  Stored as: Days since epoch (integer)")
    
    # Show date clustering
    optimized_df = df.iloc[result['row_order']]
    
    print(f"\nOrderDate before (first 20):")
    print(df['OrderDate'].head(20).dt.date.tolist())
    
    print(f"\nOrderDate after (first 20):")
    print(optimized_df['OrderDate'].head(20).dt.date.tolist())
    
    print(f"\nNotice how same dates are now grouped together!")


def demo_mixed_types():
    """Demo: Real-world mixed type scenario"""
    print("\n" + "="*70)
    print("DEMO 4: Real-World Mixed Types (E-commerce Orders)")
    print("="*70)
    
    np.random.seed(42)
    num_orders = 10000
    
    df = pd.DataFrame({
        # Integer
        'OrderID': range(num_orders),
        'CustomerID': np.random.randint(1, 1000, num_orders),
        
        # Float
        'Amount': np.random.uniform(10.0, 500.0, num_orders).round(2),
        
        # String (with NULLs)
        'ProductCategory': np.random.choice(
            ['Electronics', 'Books', 'Clothing', 'Home', None],
            num_orders,
            p=[0.3, 0.25, 0.2, 0.15, 0.1]
        ),
        
        # Date
        'OrderDate': pd.date_range('2024-01-01', periods=num_orders, freq='5min'),
        
        # Boolean (with NULLs)
        'IsPrime': np.random.choice([True, False, None], num_orders, p=[0.4, 0.5, 0.1])
    })
    
    print(f"\nDataset: {len(df):,} orders")
    print(f"\nData types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")
    
    print(f"\nMemory usage:")
    original_memory = df.memory_usage(deep=True).sum()
    print(f"  Original: {original_memory / 1024 / 1024:.2f} MB")
    
    print(f"\nOptimizing all columns...")
    result = optimize_table(df, verbose=True)
    
    optimized_df = df.iloc[result['row_order']]
    optimized_memory = optimized_df.memory_usage(deep=True).sum()
    
    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"{'='*70}")
    print(f"Steps: {result['steps']:,}")
    print(f"Clusters: {result['clusters']:,}")
    print(f"Compression: {result['compression_ratio']:.2f}x")
    print(f"Time: {result['time']:.2f}s")
    print(f"Memory after: {optimized_memory / 1024 / 1024:.2f} MB")
    print(f"Memory saved: {(1 - optimized_memory/original_memory)*100:.1f}%")
    
    print(f"\nEncoding summary:")
    for col, info in result['column_encodings'].items():
        print(f"\n{col}:")
        print(f"  Original: {info['original_type']}")
        print(f"  Encoded as: {info['encoding_type']}")
        if info.get('null_count', 0) > 0:
            print(f"  NULLs: {info['null_count']:,} ({info['null_count']/num_orders*100:.1f}%)")
        if info['encoding_type'] == 'dictionary':
            print(f"  Dictionary: {list(info['dictionary'].values())}")


def demo_comparison():
    """Demo: Comparison with vs without string columns"""
    print("\n" + "="*70)
    print("DEMO 5: Impact of Including String Columns")
    print("="*70)
    
    np.random.seed(42)
    num_rows = 10000
    
    df = pd.DataFrame({
        'ProductID': np.random.randint(1, 100, num_rows),
        'Price': np.random.uniform(10.0, 100.0, num_rows),
        'Category': np.random.choice(['A', 'B', 'C', 'D', 'E'], num_rows),
        'Brand': np.random.choice(['Brand1', 'Brand2', 'Brand3'], num_rows)
    })
    
    print(f"\nDataset: {len(df):,} rows")
    print(f"Columns: ProductID (int), Price (float), Category (string), Brand (string)")
    
    # Test 1: Numeric only (old behavior)
    print(f"\n--- Test 1: Numeric columns only ---")
    result1 = optimize_table(df, columns=['ProductID', 'Price'], verbose=False)
    print(f"Clusters: {result1['clusters']:,}")
    print(f"Compression: {result1['compression_ratio']:.2f}x")
    
    # Test 2: All columns (new behavior)
    print(f"\n--- Test 2: ALL columns (including strings) ---")
    result2 = optimize_table(df, verbose=False)
    print(f"Clusters: {result2['clusters']:,}")
    print(f"Compression: {result2['compression_ratio']:.2f}x")
    
    # Compare
    improvement = result2['compression_ratio'] / result1['compression_ratio']
    print(f"\n{'='*70}")
    print(f"COMPARISON:")
    print(f"{'='*70}")
    print(f"Including strings: {improvement:.2f}x BETTER compression!")
    print(f"\nWhy? String columns often have LOW CARDINALITY")
    print(f"  Category: {df['Category'].nunique()} unique values")
    print(f"  Brand: {df['Brand'].nunique()} unique values")
    print(f"  → Perfect for RLE compression!")


def main():
    """Run all demos"""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║        Enhanced VertiPaq Optimizer - Complete Demo                ║
║                                                                    ║
║  Supports:                                                         ║
║    ✓ Strings (dictionary encoding)                                ║
║    ✓ Dates (integer conversion)                                   ║
║    ✓ NULLs (special encoding)                                     ║
║    ✓ All numeric types                                            ║
║    ✓ Booleans                                                      ║
╚════════════════════════════════════════════════════════════════════╝
""")
    
    demos = [
        ("String Columns", demo_string_columns),
        ("NULL Handling", demo_null_handling),
        ("Date Columns", demo_date_columns),
        ("Mixed Types", demo_mixed_types),
        ("Comparison", demo_comparison)
    ]
    
    print("\nAvailable demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print(f"  0. Run all demos")
    
    try:
        choice = input("\nSelect demo (0-5): ").strip()
        
        if choice == '0':
            for name, demo_func in demos:
                demo_func()
        elif choice in ['1', '2', '3', '4', '5']:
            demos[int(choice)-1][1]()
        else:
            print("Invalid choice. Running all demos...")
            for name, demo_func in demos:
                demo_func()
                
    except KeyboardInterrupt:
        print("\n\nDemo interrupted.")
    
    print("\n" + "="*70)
    print("KEY TAKEAWAYS:")
    print("="*70)
    print("""
1. String columns are AUTOMATICALLY dictionary encoded
   → Low cardinality strings compress VERY well

2. NULLs are encoded as -1 across all types
   → NULLs cluster together forming RLE runs

3. Dates converted to integers (days since epoch)
   → Temporal clustering happens naturally

4. ALL column types now supported
   → No need to pre-process or exclude columns

5. Encoding metadata is preserved
   → Can decode back to original values if needed

Use vertipaq_optimizer_enhanced.optimize_table() for REAL-WORLD data!
""")


if __name__ == "__main__":
    main()
