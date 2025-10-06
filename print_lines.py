import pandas as pd

def main():
    # Load the dataset from a Parquet file
    file_path = r'C:\\Users\\Pritch\\Downloads\\archive\\cic-collection.parquet'  # Adjust path accordingly
    try:
        data = pd.read_parquet(file_path)
        print("Dataset loaded successfully.\n")
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return

    # Filter the dataset for rows where the class label is 'Benign'
    benign_data = data[data['ClassLabel'] == 'Benign']

    # Print the first 5 rows of the filtered dataset
    print("First 5 rows where ClassLabel is 'Benign':")
    print(benign_data.head())

if __name__ == '__main__':
    main()
