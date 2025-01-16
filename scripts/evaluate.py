import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from data_preparation.data_preparation import DataPreparator
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import AutoTokenizer, TFBertForSequenceClassification

# Define label mappings
label2id = {"sport": 0, "business": 1, "politics": 2, "tech": 3, "entertainment": 4}
id2label = {val: key for key, val in label2id.items()}


def calculate_metrics(predictions, labels):
    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)
    print(f"Accuracy: {accuracy:.4f}")

    # Calculate precision, recall, F1-score, and support for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, labels=list(label2id.values())
    )

    # Print metrics for each class
    for idx, label in id2label.items():
        print(f"Class: {label}")
        print(f"Precision: {precision[idx]:.4f}")
        print(f"Recall:    {recall[idx]:.4f}")
        print(f"F1 Score:  {f1[idx]:.4f}")

    # Calculate and print macro-averaged metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro"
    )
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Generate and print confusion matrix
    conf_matrix = confusion_matrix(labels, predictions, labels=list(label2id.values()))
    print("Confusion Matrix:")
    print(conf_matrix)

    # Assuming all_labels and all_predictions are defined
    display_labels = [id2label[i] for i in range(len(label2id))]

    # Create a DataFrame for the confusion matrix with labels
    cm_df = pd.DataFrame(conf_matrix, index=display_labels, columns=display_labels)

    # Plot the heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # Save the plot as an image file
    plt.savefig("model/confusion_matrix.png", dpi=300, bbox_inches="tight")

    # Display the plot
    plt.show()


def evaluate(test_data, config):
    # Extract labels and determine the number of unique labels
    labels = test_data["labels"].tolist()
    num_labels = len(set(labels))
    print(f"Number of unique labels: {num_labels}")

    # Initialize DataPreparator
    data_preparator = DataPreparator(
        bert_model=config["bert_model"],
        sequence_length=config["sequence_length"],
        batch_size=config["batch_size"],
    )

    # Prepare TensorFlow dataset for testing
    tf_test_dataset = data_preparator.prepare_tf_dataset(test_data)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config["bert_model"])

    # if model is fine-tuned
    # model = TFAutoModelForSequenceClassification.from_pretrained(
    #    saved_dir, num_labels=num_labels, label2id=label2id, id2label=id2label
    # )

    model = TFBertForSequenceClassification.from_pretrained(
        config["bert_model"],
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    # Initialize lists to store predictions and true labels
    all_predictions = []
    all_labels = []

    # Iterate over the test dataset
    for batch_inputs, batch_labels in tf_test_dataset:
        inputs = {
            key: val
            for key, val in batch_inputs.items()
            if key in tokenizer.model_input_names
        }

        # Obtain model predictions
        logits = model.predict(inputs).logits
        predictions = np.argmax(logits, axis=-1)

        # Store predictions and true labels
        all_predictions.extend(predictions)
        all_labels.extend(batch_labels.numpy())

    # Calculate and print evaluation metrics
    calculate_metrics(all_predictions, all_labels)


if __name__ == "__main__":
    # Define paths
    input_dir = "data/split_datasets"
    config_path = "scripts/config.json"

    # Load test data
    test_data_path = os.path.join(input_dir, "test_data.csv")
    test_data = pd.read_csv(test_data_path)
    print(f"Number of samples in test dataset: {len(test_data)}")

    # Load configuration
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    print("Configuration loaded successfully.")

    # Evaluate the model
    evaluate(test_data, config)
    print("Evaluation completed.")
