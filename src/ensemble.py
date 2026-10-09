import torch
import torch.nn.functional as F
import numpy as np

def calculate_entropy(probs):
    """
    Calculates the Shannon entropy of a probability distribution.
    Args:
        probs (torch.Tensor): Probability distribution of shape (1, num_classes) or (batch_size, num_classes)
    Returns:
        torch.Tensor: Entropy values
    """
    # Add a small epsilon to prevent log(0)
    epsilon = 1e-9
    probs = torch.clamp(probs, epsilon, 1.0 - epsilon)
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)
    return entropy

def entropy_based_ensemble(model_outputs):
    """
    Combines predictions from multiple models using entropy-based confidence weighting.
    The model outputs should be a dictionary of logits:
    {
        'bert': logits_tensor,
        'roberta': logits_tensor,
        'distilbert': logits_tensor
    }

    Returns:
        final_prediction, model_details
    """
    details = {}
    combined_weighted_probs = None
    total_inverse_entropy = 0

    for model_name, logits in model_outputs.items():
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)

        # Calculate entropy (confidence measure)
        # Lower entropy means higher confidence
        entropy = calculate_entropy(probs)

        # Calculate weight (inverse of entropy)
        # We add a small constant to prevent division by zero in case of perfect certainty (entropy=0)
        weight = 1.0 / (entropy + 1e-5)

        prediction = torch.argmax(probs, dim=-1).item()
        confidence = torch.max(probs, dim=-1).values.item()

        details[model_name] = {
            'logits': logits.detach().cpu().numpy().tolist(),
            'probabilities': probs.detach().cpu().numpy().tolist(),
            'prediction': prediction,
            'confidence': confidence,
            'entropy': entropy.item(),
            'weight': weight.item()
        }

        if combined_weighted_probs is None:
            combined_weighted_probs = probs * weight.unsqueeze(-1)
        else:
            combined_weighted_probs += probs * weight.unsqueeze(-1)

        total_inverse_entropy += weight

    # Normalize the combined probabilities
    final_probs = combined_weighted_probs / total_inverse_entropy.unsqueeze(-1)
    final_prediction = torch.argmax(final_probs, dim=-1).item()
    final_confidence = torch.max(final_probs, dim=-1).values.item()

    # Determine highest confidence model (lowest entropy)
    best_model = min(details.items(), key=lambda x: x[1]['entropy'])[0]

    return {
        'final_prediction': final_prediction,
        'final_confidence': final_confidence,
        'best_individual_model': best_model,
        'model_details': details
    }
