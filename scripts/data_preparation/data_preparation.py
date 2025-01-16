import json
import os
from typing import Dict, List, Tuple

import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer


# Label mappings
label2id = {"sport": 0, "business": 1, "politics": 2, "tech": 3, "entertainment": 4}
id2label = {val: key for key, val in label2id.items()}


class DataPreparator:
    def __init__(
        self, bert_model: str = "distilbert-base-uncased", sequence_length: int = 512, batch_size: int = 16
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.sequence_length: int = sequence_length
        self.batch_size: int = batch_size

    def tokenize_text(self, data: pd.DataFrame) -> Dict[str, tf.Tensor]:
        """Tokenizes text data using the tokenizer."""
        return self.tokenizer(
            data["text"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=self.sequence_length,
            return_tensors="tf",
        )

    @staticmethod
    def encode_labels(data: pd.DataFrame) -> List[int]:
        """Encodes labels using the label2id mapping."""
        return data["labels"].map(label2id).tolist()

    def convert_to_tf_dataset(self, tokenized_input: Dict, encoded_labels: List[int]) -> tf.data.Dataset:
        """Converts tokenized input and labels into a TensorFlow dataset."""
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                dict(tokenized_input),
                tf.constant(encoded_labels, dtype=tf.int32),
            )
        )
        return dataset.shuffle(len(encoded_labels)).batch(self.batch_size)

    def prepare_tf_dataset(self, data: pd.DataFrame) -> tf.data.Dataset:
        """Prepares a TensorFlow dataset from the raw data."""
        tokenized_input = self.tokenize_text(data)
        encoded_labels = self.encode_labels(data)
        tf_dataset = self.convert_to_tf_dataset(tokenized_input, encoded_labels)

        # Debug prints
        print(f"Tokenized Input Keys: {tokenized_input.keys()}")
        print(f"Number of Encoded Labels: {len(encoded_labels)}")

        return tf_dataset


# if __name__ == "__main__":
#     # Load and preprocess the dataset
#     input_dir = "data/split_datasets"
#     config_dir = "scripts/config.json"

#     # Read CSV file
#     # train_data = pd.read_csv(os.path.join(input_dir,"train_data.csv"))
#     # print(f"Number of samples in dataset: {len(train_data)}")

#     # val_data = pd.read_csv(os.path.join(input_dir,"val_data.csv"))
#     # print(f"Number of samples in dataset: {len(val_data)}")

#     test_data = pd.read_csv(os.path.join(input_dir,"test_data.csv"))
#     print(f"Number of samples in dataset: {len(test_data)}")

#     # Load configuration
#     with open(config_dir, "r", encoding="utf-8") as file:
#         config = json.load(file)
#     print("Configuration loaded:", config)

#     # Initialize DataPreparator
#     data_preparator = DataPreparator(
#         bert_model=config["bert_model"],
#         sequence_length=config["sequence_length"],
#         batch_size=config["batch_size"]
#     )

#     # Prepare TensorFlow datasets
#     # tf_train_dataset = data_preparator.prepare_tf_dataset(train_data)
#     # tf_val_dataset = data_preparator.prepare_tf_dataset(val_data)
#     tf_test_dataset = data_preparator.prepare_tf_dataset(test_data)


#     # Take a sample from the dataset
#     for sample in tf_test_dataset.take(1):
#         print("Sample from dataset:", sample)

#     print("--End--")
