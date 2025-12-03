# Quick Examples

## Example 1: Sales Data

```python
import pandas as pd
from vertipaq_optimizer import optimize_table

# Sales data with repetitive columns
df = pd.DataFrame({
    'Date': pd.date_range('2020-01-01', periods=1000000, freq='1min'),
    'ProductID': np.random.randint(1, 1000, 1000000),
    'StoreID': np.random.randint(1, 100, 1000000),
    'Quantity': np.random.randint(1, 10, 1000000)
})

# Convert date to numeric for optimization
df['DateInt'] = (df['Date'] - df['Date'].min()).dt.days

# Optimize
result = optimize_table(df, columns=['ProductID', 'StoreID', 'DateInt'])

print(f"Original clusters: ~{df.shape[0]}")
print(f"Optimized clusters: {result['clusters']}")
print(f"Improvement: {result['compression_ratio']:.2f}x")
```

## Example 2: Comparing File Sizes

```python
# Before optimization
df.to_parquet('original.parquet')
original_size = os.path.getsize('original.parquet')

# After optimization
result = optimize_table(df)
df.iloc[result['row_order']].to_parquet('optimized.parquet')
optimized_size = os.path.getsize('optimized.parquet')

print(f"Original: {original_size/1024/1024:.1f} MB")
print(f"Optimized: {optimized_size/1024/1024:.1f} MB")
print(f"Reduction: {(1-optimized_size/original_size)*100:.1f}%")
```