import torch
import os
import time

def calculate_model_size(model, filepath="temp_model.pt"):
    """
    Saves a model to disk temporarily to measure its size in MB.
    """
    torch.save(model.state_dict(), filepath)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except PermissionError:
        pass
    return size_mb

def quantize_model(model):
    """
    Applies Dynamic Quantization to a PyTorch model.
    This reduces the model size (mostly from FP32 to INT8 in linear layers)
    and can speed up CPU inference without significant accuracy loss.
    """
    # Put model in eval mode and strictly on CPU for quantization!
    # PyTorch dynamic quantization is mainly supported on CPU for Linear layers.
    model.eval()
    model.to('cpu')

    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Quantize only linear layers
        dtype=torch.qint8
    )

    return quantized_model

def compare_quantization_performance(original_model, quantized_model, dummy_input, device='cpu'):
    """
    Returns a dictionary comparing the original vs quantized model in size and speed.
    """
    original_model.to(device)
    # Note: Quantized performance is generally measured on CPU
    quantized_model.to('cpu')

    dummy_input = dummy_input.to(device)
    dummy_input_cpu = dummy_input.to('cpu')

    # 1. Size
    orig_size = calculate_model_size(original_model, "temp_orig.pt")
    quant_size = calculate_model_size(quantized_model, "temp_quant.pt")

    # 2. Speed (dummy benchmark over 10 iterations)
    def benchmark(mdl, inp, iters=10):
        # Warmup
        for _ in range(2):
            _ = mdl(**inp)

        start = time.time()
        for _ in range(iters):
            with torch.no_grad():
                _ = mdl(**inp)
        end = time.time()
        return (end - start) / iters * 1000  # ms per inference

    orig_time = benchmark(original_model, dummy_input)
    quant_time = benchmark(quantized_model, dummy_input_cpu)

    return {
        'original_size_mb': orig_size,
        'quantized_size_mb': quant_size,
        'size_reduction_pct': (1 - (quant_size / orig_size)) * 100,
        'original_inference_ms': orig_time,
        'quantized_inference_ms': quant_time,
        'speed_improvement_pct': (1 - (quant_time / orig_time)) * 100
    }
