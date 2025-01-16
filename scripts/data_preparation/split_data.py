import os

import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(data):
    # Split into training and temporary sets (e.g., 70% training, 30% temporary)
    train_data, temp_data = train_test_split(
        data, test_size=0.3, stratify=data["labels"], random_state=42
    )

    # Split the temporary set into validation and test sets (e.g., 15% validation, 15% test)
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, stratify=temp_data["labels"], random_state=42
    )

    print("Training set class distribution:")
    print(train_data["labels"].value_counts(normalize=True))
    print(len(train_data["text"]), len(train_data["labels"]))

    print("\nValidation set class distribution:")
    print(val_data["labels"].value_counts(normalize=True))
    print(len(val_data["text"]), len(val_data["labels"]))

    print("\nTest set class distribution:")
    print(test_data["labels"].value_counts(normalize=True))
    print(len(test_data["text"]), len(test_data["labels"]))

    # Save datasets to separate files
    train_data.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
    val_data.to_csv(os.path.join(output_dir, "val_data.csv"), index=False)
    test_data.to_csv(os.path.join(output_dir, "test_data.csv"), index=False)


if __name__ == "__main__":
    # Define directories
    output_dir = "data/split_datasets"
    input_dir = "data/bbc_text_cls.csv"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    data = pd.read_csv(input_dir)
    print(f"Number of samples in dataset: {len(data)}")

    # Prepare and save datasets
    split_dataset(data)
