import re
import pandas as pd

def clean_imdb_text(text: str) -> str:
    """
    Performs safe preprocessing for sentiment classification.
    Safely removes HTML, control characters, and excess whitespace 
    while preserving punctuation and sentiment-heavy casing/words.
    """
    if not isinstance(text, str):
        return ""
        
    # Remove HTML tags (e.g., <br />, <i>, <b>)
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove obvious control characters (\r, \n, \t)
    text = re.sub(r'[\r\n\t]+', ' ', text)
    
    # Normalize excessive whitespace to a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading and trailing whitespaces
    return text.strip()

def preprocess_and_clean_dataset(dataset_split):
    """
    Cleans a Hugging Face dataset split by deduplicating,
    handling nulls, and formatting string data.
    
    Returns:
        tuple: (list of cleaned texts, list of labels)
    """
    # Using Pandas for highly efficient duplicate grouping and dropna logic
    df = pd.DataFrame(dataset_split)
    initial_count = len(df)
    
    # 1. Handle missing or null reviews
    df = df.dropna(subset=['text', 'label'])
    
    # 2. Apply safe preprocessing logic to the 'text' column
    df['text'] = df['text'].apply(clean_imdb_text)
    
    # 3. Filter out rows that might have become empty after cleaning
    df = df[df['text'] != '']
    
    # 4. Remove duplicate reviews (very common in massive scrapes like IMDB)
    df = df.drop_duplicates(subset=['text'], keep='first')
    
    final_count = len(df)
    removed_count = initial_count - final_count
    
    print(f"\n--- Cleaning Statistics ---")
    print(f"Original Count: {initial_count}")
    print(f"Removed Duplicate/Invalid: {removed_count}")
    print(f"Final Retained Count: {final_count}")
    
    return df['text'].tolist(), df['label'].tolist()

if __name__ == "__main__":
    from data_loading import load_imdb_dataset
    
    print("Fetching raw dataset for cleaning verification...")
    # Passing the HF logger warning by using proper imports and local cache
    train_data, test_data = load_imdb_dataset()
    
    print("\nExtracting a single corrupt-looking sample to test...")
    raw_sample = train_data[0]['text']
    
    print(f"\n[BEFORE CLEANING SAMPLE]")
    print(f"{raw_sample[:350]}...")
    
    print("\nExecuting Bulk Preprocessing (Train Data)...")
    cleaned_texts, cleaned_labels = preprocess_and_clean_dataset(train_data)
    
    print(f"\n[AFTER CLEANING SAMPLE]")
    print(f"{cleaned_texts[0][:350]}...")
    print("---------------------------------------------")
    print("Dataset correctly preprocessed and packaged!")
