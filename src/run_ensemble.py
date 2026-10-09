import os
import torch
import pandas as pd
from tqdm import tqdm
from src.data_loading import load_imdb_dataset
from src.tokenization import get_tokenizers, create_dataloader
from src.models import EnsembleModels
from src.ensemble import entropy_based_ensemble
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BATCH_SIZE = 8
MAX_LENGTH = 256
SAVED_MODELS_DIR = "./saved_models"
OUTPUT_DIR = "./eval_results"

def load_trained_models_and_tokenizers(device):
    """Loads all models and tokenizers from saved_models directory"""
    print("Initialize architectures...")
    # Use EnsembleModels to get the structure, but actually override with our saved fine-tuned instances
    builder = EnsembleModels(num_labels=2)
    models = builder.get_models()
    
    loaded_models = {}
    loaded_tokenizers = {}
    for name in models.keys():
        model_path = os.path.join(SAVED_MODELS_DIR, name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist. Please train first.")
        
        print(f"  Loading {name} from {model_path}...")
        loaded_tokenizers[name] = AutoTokenizer.from_pretrained(model_path)
        finetuned_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        finetuned_model.to(device)
        finetuned_model.eval()
        loaded_models[name] = finetuned_model

    return loaded_models, loaded_tokenizers


def run_ensemble_inference(limit=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    models, tokenizers = load_trained_models_and_tokenizers(device)
    
    print("\nLoading test data...")
    _, test_dataset = load_imdb_dataset()
    test_texts = list(test_dataset['text'])
    test_labels = list(test_dataset['label'])
    
    if limit is not None:
        print(f"Capping evaluation to {limit} samples for verification.")
        test_texts = test_texts[:limit]
        test_labels = test_labels[:limit]
        
    # Accumulate logits per model across all test samples
    model_logits = {name: [] for name in models.keys()}
    
    for name, model in models.items():
        print(f"\n========== Generating predictions for {name.upper()} ==========")
        dataloader = create_dataloader(
            test_texts, test_labels, tokenizers[name], 
            batch_size=BATCH_SIZE, max_length=MAX_LENGTH, shuffle=False
        )
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {name}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                model_logits[name].extend(outputs.logits.cpu())
                
    # Now we have all logits stored. Apply entropy-based ensemble logic sample-by-sample.
    print("\n========== Applying Entropy-Based Ensemble Logic ==========")
    
    results = []
    
    for i in tqdm(range(len(test_texts)), desc="Ensembling predictions"):
        # Create a tiny dictionary of logits just for this single sample
        sample_logits = {name: model_logits[name][i].unsqueeze(0) for name in models.keys()}
        
        # ensemble computation leverages Shannon entropy under the hood in `entropy_based_ensemble`
        ensemble_res = entropy_based_ensemble(sample_logits)
        
        # Build CSV row
        row = {
            'text': test_texts[i],
            'true_label': test_labels[i]
        }
        
        # Granular components per individual model
        for name in models.keys():
            details = ensemble_res['model_details'][name]
            row[f'{name}_prob_neg'] = details['probabilities'][0][0]
            row[f'{name}_prob_pos'] = details['probabilities'][0][1]
            row[f'{name}_prediction'] = details['prediction']
            row[f'{name}_entropy'] = details['entropy']
            
        # Overall Ensemble Result
        row['ensemble_final_prediction'] = ensemble_res['final_prediction']
        row['ensemble_confidence'] = ensemble_res['final_confidence']
        row['ensemble_best_model'] = ensemble_res['best_individual_model']
        
        results.append(row)
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, "ensemble_predictions.csv")
    df.to_csv(out_path, index=False)
    
    print(f"\nSuccess! Processed {len(df)} samples.")
    print(f"Probabilities, Entropy, and Final Ensemble metrics saved to {out_path}.")

if __name__ == '__main__':
    # Evaluates the ENTIRE dataset 
    run_ensemble_inference(limit=None)
