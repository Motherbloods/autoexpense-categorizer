import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
import os


def download_nltk_resources():
    """Download required NLTK resources."""
    try:
        nltk.download("stopwords", quiet=True)
        print("NLTK stopwords downloaded successfully.")
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")


def load_json_files(file_paths):
    """
    Load and combine multiple JSON files into a list of data entries.

    Args:
        file_paths (list): List of file paths to JSON files

    Returns:
        list: Combined list of data entries
    """
    all_data = []
    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                all_data.extend(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return all_data


def prepare_dataset(json_file_path="dataset.json"):
    """
    Prepare dataset for training by loading and processing data.

    Args:
        json_file_path (str): Path to the JSON file containing expense data

    Returns:
        tuple: (X_train, X_test, y_train, y_test) - Train and test datasets
               df - Full DataFrame of the data
    """
    # Download necessary NLTK resources
    download_nltk_resources()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    absolute_data_path = os.path.join(script_dir, "..", "data", json_file_path)

    # Load data
    all_data = load_json_files([absolute_data_path])
    df = pd.DataFrame(all_data)

    print(f"Total records loaded: {len(df)}")
    print(f"Category distribution:\n{df['category'].value_counts()}")

    # Split features and target
    X = df["activity"]
    y = df["category"]

    return df, X, y


if __name__ == "__main__":
    # Test the data preparation function
    df, X, y = prepare_dataset()
    print(f"Data prepared successfully. Dataset shape: {df.shape}")
