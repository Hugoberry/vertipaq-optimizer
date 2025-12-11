import pandas as pd
from vertipaq_optimizer import optimize_table

df = pd.read_csv('data/outputs/Phase1_SingleColumn_Card0010000.csv')

result = optimize_table(df, verbose=True)
