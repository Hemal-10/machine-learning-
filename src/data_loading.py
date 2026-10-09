from datasets import load_dataset
from collections import Counter

def load_imdb_dataset():
    """
    Loads the official IMDB sentiment dataset using Hugging Face datasets.
    Keeps original train and test splits intact and verifies data structure.
    
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    print("Loading official IMDB dataset from Hugging Face...")
    # Load dataset keeping original splits
    dataset = load_dataset("stanfordnlp/imdb")
    
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    # Basic dataset verification: number of samples
    print("\n--- Dataset Verification ---")
    print(f"Total Train Samples: {len(train_dataset)}")
    print(f"Total Test Samples:  {len(test_dataset)}")
    
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("Dataset verification failed: One of the splits is empty.")
    
    # Basic dataset verification: label distribution
    # Extracting labels directly from the dataset iterable
    train_labels = [sample['label'] for sample in train_dataset]
    label_counts = Counter(train_labels)
    print(f"Train Label Distribution -> Negative (0): {label_counts[0]}, Positive (1): {label_counts.get(1, 0)}")
    
    # Print a small sample for sanity check
    print("\n--- Small Data Sample (From Train Split) ---")
    sample_item = train_dataset[0] # Grab the first item
    print(f"Text Snippet: {sample_item['text'][:300]}...")
    print(f"Label: {sample_item['label']}")
    print("---------------------------------------------")
    
    return train_dataset, test_dataset

if __name__ == "__main__":
    # If run directly as a script, this executes the verification manually
    print("Initiating stand-alone module verification...")
    train_data, test_data = load_imdb_dataset()
    print("\nVerification successful: The IMDB dataset loader is ready for use.")
