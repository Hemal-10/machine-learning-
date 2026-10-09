import pandas as pd
from datasets import load_dataset
import os

def export_imdb_to_csv(output_dir="data"):
    print("Loading IMDB dataset from Hugging Face...")
    dataset = load_dataset("stanfordnlp/imdb")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Exporting train split to CSV...")
    # Convert to pandas DataFrame and then to CSV
    train_df = pd.DataFrame(dataset['train'])
    train_csv_path = os.path.join(output_dir, "imdb_train.csv")
    train_df.to_csv(train_csv_path, index=False)
    print(f"Exported {len(train_df)} rows to {train_csv_path}")
    
    print("Exporting test split to CSV...")
    test_df = pd.DataFrame(dataset['test'])
    test_csv_path = os.path.join(output_dir, "imdb_test.csv")
    test_df.to_csv(test_csv_path, index=False)
    print(f"Exported {len(test_df)} rows to {test_csv_path}")
    
if __name__ == "__main__":
    export_imdb_to_csv()
