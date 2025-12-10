import pandas as pd
from vertipaq_optimizer_optimized import optimize_table_fast

# Load your data
df = pd.read_csv('data/outputs/Phase1_SingleColumn_Card0001000.csv')

# Optimize row ordering
result = optimize_table_fast(df, verbose=True)
