import torch
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from src.tokenization import MultiModelTokenizer
from src.models import EnsembleModels
from src.ensemble import entropy_based_ensemble
from src.evaluate import print_evaluation_report
from src.quantization import quantize_model, compare_quantization_performance

def main():
    print("Loading models and tokenizer... (This may take a moment)")
    tokenizer = MultiModelTokenizer()
    ensemble_builder = EnsembleModels(num_labels=2)
    models = ensemble_builder.get_models()

    print("\\nStarting Performance Metrics Evaluation...")
    # Generating dummy evaluation data for metrics visualization
    # In a real setup, this would be loaded from your evaluating dataset
    test_reviews = [
        "This movie was fantastic and wonderfully acted.",
        "Absolutely terrible, a complete waste of time.",
        "It was okay, not great but not the worst.",
        "I loved the cinematography but the plot was weak.",
        "Best film of the year! Highly recommend."
    ]
    true_labels = [1, 0, 0, 0, 1] 
    
    print(f"\\nEvaluating on {len(test_reviews)} test samples...")
    print("-" * 50)
    
    # Tokenize everything
    inputs = tokenizer.tokenize(test_reviews)
    
    # Collect predictions for metric analysis
    model_predictions = {name: [] for name in models.keys()}
    model_logits_list = []
    
    for i in range(len(test_reviews)):
        logits_dict = {}
        batch_input = tokenizer.tokenize([test_reviews[i]])
        
        for name, model in models.items():
            model.eval()
            dev = next(model.parameters()).device
            model_inputs = {k: v.to(dev) for k, v in batch_input[name].items()}
            
            with torch.no_grad():
                out = model(**model_inputs)
                logits_dict[name] = out.logits
                pred = torch.argmax(out.logits, dim=-1).item()
                model_predictions[name].append(pred)
                
        model_logits_list.append(logits_dict)
        
    # Analyze and print performance metrics
    print("\\n--- INDIVIDUAL MODEL METRICS ---")
    for name in models.keys():
        print_evaluation_report(true_labels, model_predictions[name], model_name=name.upper())
        
    print("\\n--- ENSEMBLE METRICS ---")
    ensemble_predictions = []
    for logits_dict in model_logits_list:
        res = entropy_based_ensemble(logits_dict)
        ensemble_predictions.append(res['final_prediction'])
        
    print_evaluation_report(true_labels, ensemble_predictions, model_name="ENTROPY-BASED ENSEMBLE")
    
    print("\\n" + "="*50)
    print("QUANTIZATION PERFORMANCE (DISTILBERT MODEL)")
    print("="*50)
    print("Benchmarking Dynamic Quantization...")
    distilbert = models['distilbert']
    quantized_distilbert = quantize_model(distilbert)
    
    dummy_input = {k: v.to(next(distilbert.parameters()).device) for k, v in inputs['distilbert'].items()}
    perf = compare_quantization_performance(distilbert, quantized_distilbert, dummy_input)
    
    print(f"Original Model Size:  {perf['original_size_mb']:.2f} MB")
    print(f"Quantized Model Size: {perf['quantized_size_mb']:.2f} MB (-{perf['size_reduction_pct']:.1f}%)")
    print(f"Inference Time (Original):  {perf['original_inference_ms']:.2f} ms/batch")
    print(f"Inference Time (Quantized): {perf['quantized_inference_ms']:.2f} ms/batch (+{perf['speed_improvement_pct']:.1f}% faster)\\n")

if __name__ == '__main__':
    main()
