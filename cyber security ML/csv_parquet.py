import pandas as pd

# Path to your CSV file
csv_file_path = 'Brute_force.csv'

# Desired output Parquet file path
parquet_file_path = 'Brute_force_example.parquet'

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Save the DataFrame to a Parquet file
df.to_parquet(parquet_file_path, engine='pyarrow', index=False)

print(f"Converted '{csv_file_path}' to '{parquet_file_path}' successfully.")
