import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

def get_tokenizers():
    """
    Returns a dictionary of initialized tokenizers for the three ensemble models.
    """
    return {
        'bert': AutoTokenizer.from_pretrained('bert-base-uncased'),
        'roberta': AutoTokenizer.from_pretrained('roberta-base'),
        'distilbert': AutoTokenizer.from_pretrained('distilbert-base-uncased')
    }

class IMDBDataset(Dataset):
    """
    PyTorch Dataset wrapper for cleaned IMDB reviews that pre-tokenizes
    text using a provided tokenizer but leaves padding to be handled dynamically
    by a collator to save computation.
    """
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.labels = labels
        # Tokenize early but do NOT pad. 
        # Truncation is applied to ensure we do not exceed max_length.
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Hugging Face encodings can be extracted per-index
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

def create_dataloader(texts, labels, tokenizer, batch_size=8, max_length=256, shuffle=True):
    """
    Creates a PyTorch DataLoader that dynamically pads sequences to the 
    maximum length in the batch.
    """
    dataset = IMDBDataset(texts, labels, tokenizer, max_length=max_length)
    # The data collator will pad the sequences in the batch dynamically
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collator
    )
    return dataloader

class MultiModelTokenizer:
    """
    Utility class to tokenize text across all ensemble models simultaneously.
    """
    def __init__(self, max_length=256):
        self.tokenizers = get_tokenizers()
        self.max_length = max_length

    def tokenize(self, texts):
        return {
            name: tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=self.max_length,
                return_tensors='pt'
            )
            for name, tokenizer in self.tokenizers.items()
        }

if __name__ == "__main__":
    print("Initializing tokenizers...")
    tokenizers = get_tokenizers()
    
    # Mock data for verification
    sample_texts = [
        "This is an amazing movie! I loved every second of it.",
        "Terrible waste of time. Do not watch this.",
        "It was okay, not great but not terrible. " * 20  # Make it longer to test padding vs truncation
    ]
    sample_labels = [1, 0, 1]
    
    print("\n--- Verifying Tokenization for Each Model ---")
    
    for model_name, tokenizer in tokenizers.items():
        print(f"\n[Testing {model_name.upper()}]")
        
        # Create dataloader
        dataloader = create_dataloader(
            texts=sample_texts, 
            labels=sample_labels, 
            tokenizer=tokenizer, 
            batch_size=2, # Batch size 2 to see the dynamic padding effect
            max_length=512,
            shuffle=False
        )
        
        # Fetch one batch
        batch = next(iter(dataloader))
        
        print(f"Batch keys: {list(batch.keys())}")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        
        if 'token_type_ids' in batch:
            print(f"Token Type IDs shape: {batch['token_type_ids'].shape}")
            
        print(f"Attention Mask shape: {batch['attention_mask'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        
        # Decode first sequence in the batch
        decoded_text = tokenizer.decode(batch['input_ids'][0])
        print(f"Decoded sequence 0:\n{decoded_text}")
        print("-" * 50)
