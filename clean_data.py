import pandas as pd


def clean_review_data(input_path: str, output_path: str):
    """
    Loads review data from a JSON Lines file, cleans it by removing
    records with missing artists, and saves the result to a Parquet file.

    Args:
        input_path: The path to the raw .jsonl data file.
        output_path: The path where the cleaned .parquet file will be saved.
    """
    print(f"Loading data from {input_path}...")
    # Load the raw data from the JSON Lines file
    df = pd.read_json(input_path, lines=True)
    print(f"Loaded {len(df)} total records.")

    # Identify invalid records before cleaning
    invalid_records = df[df['artist'] == 'N/A']
    print(f"Found {len(invalid_records)} records with 'N/A' artist.")

    # Filter out the invalid records
    print("Cleaning data...")
    cleaned_df = df[df['artist'] != 'N/A'].copy()

    # Optional: Reset the DataFrame index after dropping rows
    cleaned_df.reset_index(drop=True, inplace=True)

    print(f"Saving {len(cleaned_df)} cleaned records to {output_path}...")
    # Save the cleaned DataFrame to a highly efficient Parquet file
    cleaned_df.to_parquet(output_path)

    print("Data cleaning complete!")


if __name__ == "__main__":
    # Define the input and output file paths
    # Assumes the script is run from the project root directory
    RAW_DATA_FILE = "reviews.jsonl"
    CLEANED_DATA_FILE = "reviews_cleaned.parquet"

    clean_review_data(RAW_DATA_FILE, CLEANED_DATA_FILE)
