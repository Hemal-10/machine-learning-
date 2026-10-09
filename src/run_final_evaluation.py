import os
import json
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.data_loading import load_imdb_dataset
from src.tokenization import create_dataloader
from src.ensemble import entropy_based_ensemble
from src.quantization import compare_quantization_performance
from src.evaluate import calculate_metrics
import warnings

warnings.filterwarnings("ignore")

BATCH_SIZE = 8
MAX_LENGTH = 256
SAVED_MODELS_DIR = "./saved_models"
QUANTIZED_MODELS_DIR = "./saved_models_quantized"
OUTPUT_DIR = "./eval_results"

def load_fp32(model_name):
    """Loads FP32 origin model."""
    path = os.path.join(SAVED_MODELS_DIR, model_name)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return model, tokenizer

def load_int8(model_name, fp32_model_for_config):
    """Loads dynamically quantized INT8 model using FP32 original structures."""
    from src.quantization import quantize_model
    path = os.path.join(QUANTIZED_MODELS_DIR, model_name)
    
    # Generate the quantization structural layers first
    quant_model = quantize_model(fp32_model_for_config)
    
    # Load PyTorch explicit int8 tensor states overriding initialized ones
    state_dict_path = os.path.join(path, "pytorch_model.bin")
    quant_model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))
    quant_model.eval()
    return quant_model
    
def get_predictions(model, dataloader, device):
    """Batches inference sequentially, aggregating predictions & labels."""
    all_preds, all_labels = [], []
    all_logits = []
    
    model = model.to(device)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Inference", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_logits.extend(logits.cpu())
            
    return all_preds, all_labels, all_logits

def run_evaluation(limit=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Primary inference pipeline relying on: {device}")
    
    # CPU evaluation optimization
    if hasattr(os, 'cpu_count') and os.cpu_count() is not None:
        num_threads = min(8, os.cpu_count())
        torch.set_num_threads(num_threads)
        print(f"Optimizing CPU inference threads to: {num_threads}")
    
    print("\nLoading test dataset...")
    _, test_dataset = load_imdb_dataset()
    test_texts = list(test_dataset['text'])
    test_labels = list(test_dataset['label'])
    
    if limit is not None:
        print(f"VERIFICATION MODE ALERT: Processing capped to subset limits ({limit} entries).")
        test_texts = test_texts[:limit]
        test_labels = test_labels[:limit]
        
    final_report = {'models': {}, 'ensemble': {}}
    models_list = ['bert', 'roberta', 'distilbert']
    fp32_logits_cache = {name: [] for name in models_list}
    true_targets = None
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    report_path = os.path.join(OUTPUT_DIR, "final_comparison.json")
    if os.path.exists(report_path):
        try:
            with open(report_path, "r") as f:
                final_report = json.load(f)
        except Exception:
            pass
            
    targets_cache_path = os.path.join(OUTPUT_DIR, f"true_targets_{limit}.json")
    if os.path.exists(targets_cache_path):
        with open(targets_cache_path, "r") as f:
            true_targets = json.load(f)
            
    for name in models_list:
        print(f"\n====================================")
        print(f" Evaluating {name.upper()}")
        print(f"====================================")
        
        fp32_preds_path = os.path.join(OUTPUT_DIR, f"{name}_fp32_preds_{limit}.json")
        fp32_logits_path = os.path.join(OUTPUT_DIR, f"{name}_fp32_logits_{limit}.pt")
        int8_preds_path = os.path.join(OUTPUT_DIR, f"{name}_int8_preds_{limit}.json")
        
        fp32_model = None
        tokenizer = None
        
        # 1. Evaluate FP32 Limits
        if os.path.exists(fp32_preds_path) and os.path.exists(fp32_logits_path) and true_targets is not None:
            print(f"--> Loaded FP32 predictions and logits from cache...")
            with open(fp32_preds_path, "r") as f:
                fp32_preds = json.load(f)
            fp32_logits = torch.load(fp32_logits_path)
            fp32_logits_cache[name] = fp32_logits
            fp32_metrics = calculate_metrics(true_targets, fp32_preds)
        else:
            print(f"--> Initializing FP32 variants...")
            fp32_model, tokenizer = load_fp32(name)
            dataloader = create_dataloader(test_texts, test_labels, tokenizer, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, shuffle=False)
            
            print("--> Generating FP32 evaluation matrices...")
            fp32_preds, batch_targets, fp32_logits = get_predictions(fp32_model, dataloader, device)
            
            if true_targets is None:
                true_targets = batch_targets
                with open(targets_cache_path, "w") as f:
                    json.dump(true_targets, f)
                    
            fp32_metrics = calculate_metrics(true_targets, fp32_preds)
            fp32_logits_cache[name] = fp32_logits
            
            with open(fp32_preds_path, "w") as f:
                json.dump(fp32_preds, f)
            torch.save(fp32_logits, fp32_logits_path)
            
        # 2. Evaluate Dynamically Quantized Variables
        if os.path.exists(int8_preds_path) and name in final_report.get('models', {}) and 'profiling' in final_report['models'][name]:
            print(f"--> Loaded INT8 CPU matrix predictions and profiling from cache...")
            with open(int8_preds_path, "r") as f:
                int8_preds = json.load(f)
            int8_metrics = calculate_metrics(true_targets, int8_preds)
            perf = final_report['models'][name]['profiling']
        else:
            print(f"--> Initializing INT8 compressed variants...")
            if fp32_model is None:
                fp32_model, tokenizer = load_fp32(name)
            int8_model = load_int8(name, fp32_model)
            
            print("--> Generating INT8 CPU matrices...")
            if tokenizer is None:
                _, tokenizer = load_fp32(name)
            dataloader = create_dataloader(test_texts, test_labels, tokenizer, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, shuffle=False)
            
            int8_preds, _, _ = get_predictions(int8_model, dataloader, 'cpu')
            int8_metrics = calculate_metrics(true_targets, int8_preds)
            
            with open(int8_preds_path, "w") as f:
                json.dump(int8_preds, f)
                
            print("--> Tracking specific sequence latency sizing...")
            dummy_input = tokenizer("Review acts as a generic standardized payload for generic CPU sequence mapping checks.", return_tensors='pt', truncation=True, max_length=256)
            perf = compare_quantization_performance(fp32_model, int8_model, dummy_input, device='cpu')
            
        final_report['models'][name] = {
            'fp32_metrics': fp32_metrics,
            'int8_metrics': int8_metrics,
            'profiling': perf
        }
        
        # Progressive save
        with open(report_path, "w") as f:
            json.dump(final_report, f, indent=4)
            
    # 4. Generate Ensemble Predictions
    print(f"\n====================================")
    print(" Evaluating Entropy-Weighted Ensemble")
    print(f"====================================")
    
    ensemble_preds = []
    
    for i in tqdm(range(len(true_targets)), desc="  Ensembling Matrix Weights", leave=False):
        sample_logits = {name: fp32_logits_cache[name][i].unsqueeze(0) for name in models_list}
        res = entropy_based_ensemble(sample_logits)
        ensemble_preds.append(res['final_prediction'])
        
    ensemble_metrics = calculate_metrics(true_targets, ensemble_preds)
    final_report['ensemble']['metrics'] = ensemble_metrics
    
    # 5. Result Tracing
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=4)
        
    print(f"\nSUCCESS! End-to-end evaluation processed constraints fully. Outputs documented securely at {report_path}.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='Limit processing to a subset')
    args = parser.parse_args()
    
    run_evaluation(limit=args.limit)
