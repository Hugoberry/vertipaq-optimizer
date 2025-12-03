# VertiPaq Optimizer

**Optimize row ordering for maximum compression in columnar databases**

This Python module implements the row ordering algorithm used by Microsoft's VertiPaq engine (Power BI, Analysis Services, SQL Server) to dramatically improve compression ratios in columnar data stores.

## 🎯 What It Does

When storing data in columnar format (like Parquet, Power BI, or Analysis Services), the order of rows significantly affects compression efficiency. This module reorders your data to maximize **Run-Length Encoding (RLE)** compression by grouping similar values together.

## 📦 Installation

```bash
pip install numpy pandas

# Download vertipaq_optimizer.py to your project directory
```

Or install from source:
```bash
git clone https://github.com/Hugoberry/vertipaq-optimizer.git
cd vertipaq-optimizer
pip install -e .
```

## 🚀 Quick Start

### Basic Usage

```python
import pandas as pd
from vertipaq_optimizer import optimize_table

# Load your data
df = pd.read_csv('sales_data.csv')

# Optimize row ordering
result = optimize_table(df, verbose=True)

# Reorder your dataframe
optimized_df = df.iloc[result['row_order']]

# Save optimized version
optimized_df.to_parquet('sales_data_optimized.parquet')

print(f"Compression improvement: {result['compression_ratio']:.2f}x")
print(f"Optimized in {result['time']:.2f} seconds")
```

### Advanced Usage

```python
from vertipaq_optimizer import VertiPaqOptimizer

# Create optimizer with verbose output
optimizer = VertiPaqOptimizer(verbose=True)

# Optimize specific columns only
result = optimizer.optimize(
    data=df,
    columns=['ProductID', 'CategoryID', 'Date']  # Only optimize these
)

print(f"Steps: {result['steps']:,}")
print(f"RLE clusters: {result['clusters']:,}")
```

### Working with NumPy Arrays

```python
import numpy as np
from vertipaq_optimizer import optimize_table

# Your data as numpy arrays
data = {
    'column_a': np.array([1, 2, 1, 3, 2, 1]),
    'column_b': np.array([10, 20, 10, 30, 20, 10])
}

result = optimize_table(data)

# Apply ordering to your arrays
for col_name, col_data in data.items():
    data[col_name] = col_data[result['row_order']]
```

## 📊 Use Cases

### 1. Parquet File Optimization

```python
# Improve Parquet compression ratios
df = pd.read_parquet('input.parquet')
result = optimize_table(df)
optimized_df = df.iloc[result['row_order']]
optimized_df.to_parquet('output.parquet', compression='snappy')

# Result: Smaller file size with same data!
```

### 2. Data Warehouse ETL

```python
# Optimize before loading into columnar warehouse
for partition in data_partitions:
    result = optimize_table(partition)
    optimized_partition = partition.iloc[result['row_order']]
    load_to_warehouse(optimized_partition)
```

### 3. Research & Benchmarking

```python
# Compare compression with and without optimization
original_size = df.memory_usage(deep=True).sum()

result = optimize_table(df)
optimized_df = df.iloc[result['row_order']]
optimized_size = optimized_df.memory_usage(deep=True).sum()

print(f"Memory reduction: {(1 - optimized_size/original_size)*100:.1f}%")
```

## 📈 Performance

**Benchmark results (1M rows, 3 columns, 10K distinct values per column):**

| Metric | Value |
|--------|-------|
| Processing time | 10-15 seconds |
| Compression improvement | 2-5x (dataset dependent) |
| Memory overhead | ~150 MB |
| Algorithm complexity | O(n × k × log(d)) |

*where n = rows, k = columns, d = distinct values*

## 🎓 How It Works

The optimizer uses a **greedy bucket-splitting algorithm**:

1. Start with all rows in one bucket
2. Find the column and value that offers maximum compression benefit
3. Partition the bucket: rows with that value first, others after
4. Repeat for each sub-bucket until no improvement is possible
5. Return the optimized row ordering

This is the same algorithm used by Microsoft's VertiPaq engine, reverse-engineered from Analysis Services binary.

## 🔬 Research Background

This implementation is based on reverse engineering Microsoft SQL Server Analysis Services (xmsrv.dll) and the VertiPaq engine. Key insights:

- **Patent**: US 8,452,737 B2 - "Optimizing data compression using hybrid algorithm"
- **Technique**: Greedy hybrid RLE optimization with bucket-based partitioning
- **Performance**: Uses statistical sampling for large datasets (>10K rows per bucket)
- **Optimizations**: Adaptive histogram modes, inline maximum tracking, bit-flag column tracking

## 🛠️ API Reference

### `optimize_table(data, columns=None, verbose=False)`

Main optimization function.

**Parameters:**
- `data` (DataFrame | dict | list): Input data to optimize
- `columns` (list, optional): Specific columns to use (default: all numeric)
- `verbose` (bool): Print progress information

**Returns:**
Dictionary with:
- `row_order` (ndarray): Optimized row indices
- `steps` (int): Number of optimization iterations
- `clusters` (int): Number of RLE clusters created
- `compression_ratio` (float): Estimated compression improvement
- `time` (float): Execution time in seconds
- `columns_optimized` (list): Names of columns used
- `num_rows` (int): Number of rows processed

### `VertiPaqOptimizer` Class

For advanced usage with custom configuration.

```python
optimizer = VertiPaqOptimizer(verbose=False)
result = optimizer.optimize(data, columns=['col1', 'col2'])
```
