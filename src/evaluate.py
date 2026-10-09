import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)

from src.data_loading import load_imdb_dataset
from src.tokenization import create_dataloader

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
MODEL_NAMES = ["bert", "roberta", "distilbert"]
SAVED_MODELS_DIR = "./saved_models"
EVAL_OUTPUT_DIR  = "./eval_results"
BATCH_SIZE = 8
MAX_LENGTH = 256


# ──────────────────────────────────────────────
# Public helpers (re-usable by other modules)
# ──────────────────────────────────────────────

def calculate_metrics(y_true, y_pred):
    """Returns a dict with accuracy, precision, recall and F1 (weighted)."""
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score":  f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def print_evaluation_report(y_true, y_pred, model_name="Model"):
    """Prints a human-readable evaluation report and returns the metrics dict."""
    print(f"\n--- Evaluation Report: {model_name.upper()} ---")
    metrics = calculate_metrics(y_true, y_pred)
    for k, v in metrics.items():
        print(f"  {k.capitalize():<12}: {v:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Negative", "Positive"], zero_division=0))
    print("-" * 45)
    return metrics


def load_trained_model(model_name: str, device: torch.device):
    """Loads a fine-tuned model + tokenizer from ./saved_models/<model_name>."""
    model_dir = os.path.join(SAVED_MODELS_DIR, model_name)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"No saved model found at '{model_dir}'. Run training first."
        )
    print(f"  Loading {model_name.upper()} from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def evaluate_model(model, dataloader, device):
    """
    Runs inference on *dataloader* and returns:
        all_preds   – list[int]  predicted classes
        all_labels  – list[int]  ground-truth labels
        all_probs   – list[list[float]]  per-class softmax probabilities
    """
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Evaluating", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = F.softmax(outputs.logits, dim=-1)
            preds   = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    return all_preds, all_labels, all_probs


# ──────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────

def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

    # Load test data once
    print("\nLoading IMDB test data...")
    _, test_dataset = load_imdb_dataset()
    test_texts  = list(test_dataset["text"])
    test_labels = list(test_dataset["label"])

    summary_rows = []   # one row per model for the CSV summary

    for model_name in MODEL_NAMES:
        print(f"\n{'='*50}")
        print(f"  Model: {model_name.upper()}")
        print(f"{'='*50}")

        # 1. Load trained model
        model, tokenizer = load_trained_model(model_name, device)

        # 2. Build DataLoader
        dataloader = create_dataloader(
            test_texts, test_labels, tokenizer,
            batch_size=BATCH_SIZE, max_length=MAX_LENGTH, shuffle=False
        )

        # 3. Run inference
        preds, true_labels, probs = evaluate_model(model, dataloader, device)

        # 4. Compute + print metrics
        metrics = print_evaluation_report(true_labels, preds, model_name)

        # 5. Save per-sample probabilities CSV
        prob_array = np.array(probs)
        prob_df = pd.DataFrame({
            "true_label":      true_labels,
            "predicted_label": preds,
            "prob_negative":   prob_array[:, 0].tolist(),
            "prob_positive":   prob_array[:, 1].tolist(),
        })
        prob_csv = os.path.join(EVAL_OUTPUT_DIR, f"{model_name}_probabilities.csv")
        prob_df.to_csv(prob_csv, index=False)
        print(f"  Per-sample probabilities → {prob_csv}")

        # 6. Save metrics JSON
        metrics_json = os.path.join(EVAL_OUTPUT_DIR, f"{model_name}_metrics.json")
        with open(metrics_json, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"  Metrics                  → {metrics_json}")

        summary_rows.append({"model": model_name, **metrics})

        # Free GPU memory before next model
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 7. Save combined summary CSV
    summary_df  = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(EVAL_OUTPUT_DIR, "all_models_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n{'='*50}")
    print(f"Combined summary  → {summary_csv}")
    print(summary_df.to_string(index=False))
    print(f"{'='*50}\n")

    return summary_df


if __name__ == "__main__":
    run_evaluation()
