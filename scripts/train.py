import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
# from utils.file_io import load_json
from data_preparation.data_preparation import DataPreparator
from sklearn.utils.class_weight import compute_class_weight
from transformers import TFBertForSequenceClassification, create_optimizer

# Label mappings
label2id = {"sport": 0, "business": 1, "politics": 2, "tech": 3, "entertainment": 4}
id2label = {val: key for key, val in label2id.items()}

def train(train_data: pd.DataFrame, val_data: pd.DataFrame, config: Dict[str, any]) -> None:
    """Trains a BERT model for sequence classification."""

    # Initialize DataPreparator
    data_preparator = DataPreparator(
        bert_model=config["bert_model"],
        sequence_length=config["sequence_length"],
        batch_size=config["batch_size"],
    )

    # Prepare TensorFlow datasets
    tf_train_dataset: tf.data.Dataset = data_preparator.prepare_tf_dataset(train_data)
    tf_val_dataset: tf.data.Dataset = data_preparator.prepare_tf_dataset(val_data)


    # Number of labels
    labels_train = train_data["labels"].tolist()
    num_labels = len(set(labels_train)) 
    print(f"num_labels: {num_labels}")

    # EarlyStopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=config["patience"], restore_best_weights=True
    )

    # Optimizer
    batches_per_epoch = len(tf_train_dataset) // config["batch_size"]
    total_train_steps = int(batches_per_epoch * config["num_epochs"])
    optimizer, _ = create_optimizer(
        init_lr=config["lr"], num_warmup_steps=0, num_train_steps=total_train_steps
    )

    # Loss
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Compute class weights
    labels_array = np.concatenate([y for x, y in tf_train_dataset], axis=0)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(labels_array), y=labels_array
    )
    print(f"Class Weights: {class_weights}")
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"Class Weights dict: {class_weights_dict}")

    # Compile, Train, Save
    model = TFBertForSequenceClassification.from_pretrained(
        config["bert_model"],
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    # Train
    model.fit(
        tf_train_dataset,
        validation_data=tf_val_dataset,
        epochs=config["num_epochs"],
        class_weight=class_weights_dict,
        callbacks=[early_stopping],
    )

    # Save
    folder_name = f"{config['bert_model']}_bs{config['batch_size']}_ep{config['num_epochs']}_lr{config['lr']}_s{config['sequence_length']}"
    save_path = os.path.join(config["save_model_dir"], folder_name)
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    print(f"Model saved at: {save_path}")


if __name__ == "__main__":
    # Load and preprocess the dataset
    input_dir: str = "data/split_datasets"
    config_dir: str = "scripts/config.json"

    # Read CSV file
    train_data: pd.DataFrame = pd.read_csv(os.path.join(input_dir, "train_data.csv"))
    print(f"Number of samples in dataset: {len(train_data)}")

    val_data: pd.DataFrame = pd.read_csv(os.path.join(input_dir, "val_data.csv"))
    print(f"Number of samples in dataset: {len(val_data)}")

    # Load configuration
    with open(config_dir, "r", encoding="utf-8") as file:
        config = json.load(file)

        
    print("Configuration loaded:", config)

    # Train the model
    train(train_data, val_data, config)

    print("End")
