import os
import json
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time
from src.data_loading import load_imdb_dataset
from src.tokenization import get_tokenizers, create_dataloader
from src.models import EnsembleModels

def train_model(model, train_dataloader, val_dataloader=None, epochs=3, lr=2e-5, device=None):
    """
    Standard training loop for fine-tuning a Hugging Face Transformer model on sequence classification.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 20)

        # Training
        progress_bar = tqdm(train_dataloader, desc="Training")
        for batch in progress_bar:
            # Move inputs to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()

            # Backward pass & optimize
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        # Validation if provided
        if val_dataloader is not None:
            model.eval()
            total_val_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validating"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    total_val_loss += loss.item()

                    preds = torch.argmax(outputs.logits, dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            avg_val_loss = total_val_loss / len(val_dataloader)
            val_accuracy = correct / total
            print(f"Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")

    metrics = {
        'train_loss': avg_train_loss,
    }
    if val_dataloader is not None:
        metrics['val_loss'] = avg_val_loss
        metrics['val_accuracy'] = val_accuracy

    return model, metrics

if __name__ == "__main__":
    print("Preparing data...")
    train_dataset, test_dataset = load_imdb_dataset()
    
    # Extract strings and labels
    train_texts = list(train_dataset['text'])
    train_labels = list(train_dataset['label'])
    
    # We'll use the test set as our validation set for this baseline setup
    val_texts = list(test_dataset['text'])
    val_labels = list(test_dataset['label'])
    
    print("Initializing tokenizers and models...")
    tokenizers = get_tokenizers()
    ensemble_builder = EnsembleModels(num_labels=2)
    models = ensemble_builder.get_models()
    
    save_dir = "./saved_models"
    os.makedirs(save_dir, exist_ok=True)
    
    for model_name, model in models.items():
        print(f"\n========== Training {model_name.upper()} ==========")
        tokenizer = tokenizers[model_name]
        
        print("Creating DataLoaders...")
        train_dataloader = create_dataloader(train_texts, train_labels, tokenizer, batch_size=8, max_length=256)
        val_dataloader = create_dataloader(val_texts, val_labels, tokenizer, batch_size=8, max_length=256, shuffle=False)
        
        # Train
        model_out, metrics = train_model(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            epochs=3,
            lr=2e-5
        )
        
        # Save
        model_save_path = os.path.join(save_dir, model_name)
        model_out.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        
        metrics_path = os.path.join(model_save_path, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        
        print(f"[{model_name.upper()}] Model and metrics saved to {model_save_path}")
        
    print("\nSequential training complete!")
