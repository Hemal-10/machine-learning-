import os
import torch
import warnings
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.models import EnsembleModels
from src.quantization import quantize_model, compare_quantization_performance

warnings.filterwarnings("ignore")

SAVED_MODELS_DIR = "./saved_models"
QUANTIZED_OUTPUT_DIR = "./saved_models_quantized"


def load_trained_model_and_tokenizer(model_name):
    model_path = os.path.join(SAVED_MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing {model_path}. Please complete training first.")
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer

def run_quantization_pipeline():
    os.makedirs(QUANTIZED_OUTPUT_DIR, exist_ok=True)
    
    models = ['bert', 'roberta', 'distilbert']
    
    for name in models:
        print(f"\n{'='*50}")
        print(f" Processing Model: {name.upper()}")
        print(f"{'='*50}")
        
        # 1. Load Original FP32
        print("Loading original FP32 fine-tuned model...")
        orig_model, tokenizer = load_trained_model_and_tokenizer(name)
        
        # 2. Apply Dynamic Int8 Quantization
        print("Applying dynamic INT8 quantization (PyTorch)...")
        quant_model = quantize_model(orig_model)
        
        # 3. Save Quantized Model isolating it from the original path
        quant_save_path = os.path.join(QUANTIZED_OUTPUT_DIR, name)
        os.makedirs(quant_save_path, exist_ok=True)
        print(f"Saving quantized internal weights to {quant_save_path}...")
        torch.save(quant_model.state_dict(), os.path.join(quant_save_path, "pytorch_model.bin"))
        tokenizer.save_pretrained(quant_save_path)
        
        # 4. Compare Performance (Size & strict CPU Inference Time)
        print("\nBenchmarking Original vs Quantized Inference on CPU...")
        dummy_text = "This review acts as a simple dummy payload for inference speed benchmarking."
        dummy_input = tokenizer(dummy_text, return_tensors='pt', truncation=True, max_length=256)
        
        # Explicit CPU enforcement required for apples-to-apples dynamic INT8 comparison
        perf = compare_quantization_performance(orig_model, quant_model, dummy_input, device='cpu')
        
        print("\n[CPU BENCHMARK RESULTS]")
        print(f"Original Model Size:       {perf['original_size_mb']:.2f} MB")
        print(f"Quantized Model Size:      {perf['quantized_size_mb']:.2f} MB (Reduced by {perf['size_reduction_pct']:.1f}%)")
        print(f"CPU Inference (Original):  {perf['original_inference_ms']:.2f} ms/batch")
        print(f"CPU Inference (Quantized): {perf['quantized_inference_ms']:.2f} ms/batch (Faster by {perf['speed_improvement_pct']:.1f}%)")
        print("Inference verified successfully via dummy tokenization pass.")
        

if __name__ == '__main__':
    print("Initializing CPU Quantization Pipeline...")
    run_quantization_pipeline()
