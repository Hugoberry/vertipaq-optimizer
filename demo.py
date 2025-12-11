from vertipaq_optimizer_pyarrow import optimize_parquet, SEGMENT_SIZE
print(f'Segment/Row Group size: {SEGMENT_SIZE:,} rows')
result = optimize_parquet('data/archive/Combined_Flights_2018.parquet', 'data/optimized_2018.parquet', verbose=True)