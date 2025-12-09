# VertiPaq Optimizer

**Optimize row ordering for maximum RLE compression in columnar databases**

This Python module implements the greedy bucket-splitting algorithm used by Microsoft’s VertiPaq engine (Power BI, Analysis Services, SQL Server) to dramatically improve Run-Length Encoding (RLE) compression in columnar data stores.

Based on Microsoft Patent **US 8,452,737 B2**: “Optimizing data compression using hybrid algorithm”

-----

## 🎯 What It Does

When storing data in columnar format (Parquet, Power BI, Analysis Services), the **order of rows** significantly affects compression efficiency. This module reorders your data to maximize Run-Length Encoding compression by grouping similar values together.

**Example:**

```
Original data (random order):
ProductID: [45, 12, 45, 89, 12, 45, 12, ...]
RLE clusters: ~1000 runs

Optimized data (grouped):
ProductID: [45, 45, 45, ..., 12, 12, 12, ..., 89, ...]
RLE clusters: ~100 runs
Compression: 10x better!
```

-----

## 📦 Installation

```bash
pip install numpy pandas pyarrow
```

**Requirements:**

- Python ≥3.7
- numpy ≥1.19.0
- pandas ≥1.0.0
- pyarrow ≥14.0.0

-----

## 🚀 Quick Start

### Basic Usage

```python
import pandas as pd
from vertipaq_optimizer import optimize_table

# Load your data
df = pd.read_csv('sales_data.csv')

# Optimize row ordering
result = optimize_table(df, verbose=True)

# Get optimized data as pandas DataFrame
optimized_df = result['table'].to_pandas()

# Save optimized version
optimized_df.to_parquet('sales_data_optimized.parquet')

print(f"Compression improvement: {result['compression_ratio']:.2f}x")
print(f"Steps: {result['steps']:,}")
print(f"RLE clusters: {result['clusters']:,}")
```

### Direct Parquet Optimization

```python
from vertipaq_optimizer import optimize_parquet

# Optimize Parquet file directly
result = optimize_parquet(
    'input.parquet',
    'output_optimized.parquet',
    verbose=True
)

print(f"File size reduced by {result['size_reduction']:.1f}%")
```

### Optimize Specific Columns

```python
from vertipaq_optimizer import optimize_table

# Optimize only high-cardinality columns
result = optimize_table(
    df,
    columns=['ProductID', 'CustomerID', 'Date'],
    verbose=True
)
```

-----

## 🔧 How It Works

### The Algorithm

The optimizer uses a **greedy bucket-splitting algorithm**:

1. **Start** with all rows in one bucket
1. **For each bucket:**
- Build histograms for each column to count value frequencies
- Calculate bit savings for grouping the most frequent value
- **Greedy selection**: choose column/value pair with maximum bit savings
- Partition rows: matching values first, others after
1. **Split bucket** into two new buckets:
- **Pure bucket**: contains only matching values (RLE-friendly)
- **Impure bucket**: contains remaining values (needs further processing)
1. **Repeat** recursively until no improvement possible
1. **Return** optimized row ordering

### Why This Works

By placing identical values consecutively, we create long runs that compress efficiently:

```
Before optimization:
Column A: [1, 5, 1, 3, 1, 5, 3, 1, ...]
RLE encoding: 1(1) 5(1) 1(1) 3(1) 1(1) 5(1) 3(1) 1(1) ...
Compression: Poor (many short runs)

After optimization:
Column A: [1, 1, 1, 1, ..., 3, 3, ..., 5, 5, ...]
RLE encoding: 1(150) 3(75) 5(80) ...
Compression: Excellent (few long runs)
```

### Algorithm Constants

The following constants control the optimization process:

```python
MIN_SPLIT_SIZE = 64              # Minimum rows to create a split
BIT_SAVINGS_THRESHOLD = 0.1      # Minimum bit savings to continue
INITIAL_MAX_SAVINGS = -1.0       # Starting value for greedy maximum
SAMPLING_THRESHOLD = 10000       # Bucket size to trigger sampling
SAMPLING_DIVISOR = 10000.0       # ~1% sampling for large buckets
SAMPLING_ADDER = 1.0             # Rounding factor
```

-----

## 📚 API Reference

### optimize_table()

```python
optimize_table(data, columns=None, verbose=False, output_path=None) -> dict
```

Optimize row ordering for maximum RLE compression.

**Parameters:**

- `data`: Input data as:
  - `pandas.DataFrame`
  - `pyarrow.Table`
  - `str` (Parquet file path)
- `columns` (optional): List of column names to optimize (default: all supported columns)
- `verbose` (optional): Print progress information (default: False)
- `output_path` (optional): Path to save optimized Parquet file

**Returns:**
Dictionary with:

- `'row_order'`: PyArrow array of optimized row indices
- `'table'`: PyArrow table with optimized row ordering
- `'steps'`: Number of optimization iterations performed
- `'clusters'`: Number of RLE clusters created
- `'compression_ratio'`: Estimated compression improvement factor
- `'time'`: Execution time in seconds
- `'columns_optimized'`: List of column names processed
- `'num_rows'`: Number of rows processed

**Example:**

```python
result = optimize_table(df, verbose=True)
optimized_df = result['table'].to_pandas()
row_indices = result['row_order'].to_numpy()
```

### optimize_parquet()

```python
optimize_parquet(input_path, output_path, columns=None, verbose=True) -> dict
```

Optimize Parquet file directly.

**Parameters:**

- `input_path`: Input Parquet file path
- `output_path`: Output Parquet file path
- `columns` (optional): Columns to optimize (default: all)
- `verbose` (optional): Print progress and file size comparison

**Returns:**
Dictionary with optimization results plus:

- `'size_reduction'`: Percentage reduction in file size

**Example:**

```python
result = optimize_parquet('data.parquet', 'data_optimized.parquet')
print(f"Reduced by {result['size_reduction']:.1f}%")
```

### VertiPaqOptimizer Class

For advanced usage with custom configuration:

```python
from vertipaq_optimizer import VertiPaqOptimizer

optimizer = VertiPaqOptimizer(
    verbose=True,
    null_strategy='separate'  # 'separate' or 'scatter'
)

result = optimizer.optimize(data, columns=['col1', 'col2'])
```

**null_strategy options:**

- `'separate'`: Group nulls together (treat as distinct value) - **Default**
- `'scatter'`: Nulls don’t match any value (standard SQL semantics)

-----

## 💡 Examples

### Example 1: Sales Data Optimization

```python
import pandas as pd
from vertipaq_optimizer import optimize_table

# Create sample sales data
df = pd.DataFrame({
    'Date': pd.date_range('2024-01-01', periods=100000, freq='1h'),
    'ProductID': np.random.randint(1, 1000, 100000),
    'CustomerID': np.random.randint(1, 5000, 100000),
    'StoreID': np.random.randint(1, 100, 100000),
    'Quantity': np.random.randint(1, 10, 100000),
    'Amount': np.random.uniform(10, 1000, 100000)
})

# Optimize
result = optimize_table(df, verbose=True)

# Save
import pyarrow.parquet as pq
pq.write_table(result['table'], 'sales_optimized.parquet')
```

### Example 2: String Columns (Automatic Dictionary Encoding)

```python
# Data with string columns
df = pd.DataFrame({
    'Category': ['Electronics', 'Books', 'Clothing'] * 10000,
    'Brand': np.random.choice(['Sony', 'Apple', 'Samsung'], 30000),
    'Country': np.random.choice(['US', 'UK', 'CA', 'AU'], 30000)
})

# String columns are automatically dictionary-encoded
result = optimize_table(df, verbose=True)

# Check results
print(f"Steps: {result['steps']:,}")
print(f"Clusters: {result['clusters']:,}")  # Fewer clusters = better compression
```

### Example 3: Handling Null Values

```python
# Data with nulls
df = pd.DataFrame({
    'ProductID': np.random.randint(1, 100, 10000),
    'Quantity': np.random.randint(1, 50, 10000),
    'OptionalField': [None] * 10000
})

# Add some nulls
null_mask = np.random.random(10000) < 0.1
df.loc[null_mask, 'Quantity'] = None

# Optimize - nulls are preserved
result = optimize_table(df, verbose=True)
optimized_df = result['table'].to_pandas()

# Verify nulls preserved
assert df['Quantity'].isnull().sum() == optimized_df['Quantity'].isnull().sum()
```

### Example 4: CSV to Optimized Parquet

```python
from vertipaq_optimizer import optimize_table
import pyarrow.parquet as pq

# Load large CSV
df = pd.read_csv('large_file.csv')

# Optimize and save as Parquet in one step
result = optimize_table(
    df,
    verbose=True,
    output_path='large_file_optimized.parquet'
)

print(f"Optimized {result['num_rows']:,} rows in {result['time']:.2f}s")
print(f"Compression improvement: {result['compression_ratio']:.2f}x")
```

### Example 5: ETL Pipeline Integration

```python
from vertipaq_optimizer import optimize_table
import pyarrow.parquet as pq

def process_partition(input_file, output_file):
    """Process one partition of data"""
    # Read
    table = pq.read_table(input_file)
    
    # Optimize
    result = optimize_table(table, verbose=False)
    
    # Write with best compression
    pq.write_table(
        result['table'],
        output_file,
        compression='zstd',
        compression_level=9,
        use_dictionary=True
    )
    
    return result['compression_ratio']

# Process all partitions
for i in range(10):
    ratio = process_partition(f'data_{i}.parquet', f'optimized_{i}.parquet')
    print(f"Partition {i}: {ratio:.2f}x compression")
```

-----

## 🎨 Supported Data Types

The optimizer supports:

|Type          |Examples                    |Notes                        |
|--------------|----------------------------|-----------------------------|
|**Integers**  |int8, int16, int32, int64   |Direct optimization          |
|**Floats**    |float32, float64            |Direct optimization          |
|**Strings**   |str, object                 |Automatic dictionary encoding|
|**Dates**     |datetime64, date32, date64  |Preserved as temporal types  |
|**Timestamps**|timestamp[ns], timestamp[us]|Preserved with precision     |
|**Booleans**  |bool                        |Direct optimization          |
|**Decimals**  |decimal128, decimal256      |Direct optimization          |
|**Nulls**     |NA, None, NaN               |Native null bitmap support   |

**Not supported:**

- Large binary types (binary, large_binary)
- Nested types (lists, structs)
- Map types

-----

## ⚙️ Configuration & Performance

### Memory Usage

The optimizer requires:

- **Input data**: 1x memory (original dataset)
- **Working memory**: ~0.5x memory (histograms, indices)
- **Output data**: 1x memory (reordered dataset)

**Total peak memory**: ~2.5x your dataset size

### Performance Tips

1. **Large datasets**: Sampling automatically activates for buckets >10K rows
1. **High cardinality columns**: May be skipped automatically if ratio < 64:1
1. **String columns**: Dictionary encoding happens automatically
1. **Null-heavy columns**: Nulls are efficiently handled via bitmap

## 🧪 Testing

Run the self-test:

```bash
python vertipaq_optimizer.py
```

Expected output:

```
VertiPaq Optimizer v2.0.0
Dataset: 1,000,000 rows × 5 columns

Column Statistics:
  ProductID: cardinality=99 numeric
  Category: cardinality=3 dict[3]
  Date: cardinality=1,000,000 numeric
  Quantity: cardinality=49 numeric (nulls: 49837)
  Price: cardinality=1,000,000 numeric

Optimization complete!
  Steps: 597
  RLE clusters: 300 (was 2,000,151)
  Compression improvement: 6667.17x
  Time: 0.36s
  Throughput: 2,815,228 rows/sec

✅ Self-test passed!
```
-----

## 📖 Algorithm Details

### Greedy Selection

At each iteration, the algorithm:

1. Evaluates all (column, value) pairs in the current bucket
1. Calculates bit savings: `frequency × bits_per_value`
1. Selects the pair with **maximum** bit savings (greedy)
1. Partitions and continues

This is a **locally optimal** strategy that produces good (but not necessarily globally optimal) results.

### Bucket Structure

Buckets represent contiguous row ranges:

- **Start/End indices**: Define the row range
- **Column flags**: Track which columns are “done”
- **Pure/Impure**: Pure buckets have identical values for some columns

### Histogram Building

For each column in a bucket:

- Count frequency of each distinct value
- Find the most frequent value
- For large buckets (>10K rows), sample ~1% of rows
- Scale frequency estimate by sampling rate

### Partition Operation

The core operation that enables RLE:

```
Before partition on value=5:
[3, 5, 7, 5, 2, 5, 9, 5, ...]

After partition:
[5, 5, 5, 5, ..., 3, 7, 2, 9, ...]
 └─ Pure bucket  └─ Impure bucket
```

