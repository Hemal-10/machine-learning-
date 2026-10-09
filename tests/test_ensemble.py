import torch
from src.ensemble import calculate_entropy, entropy_based_ensemble

def test_calculate_entropy():
    # Certain probability (should have near 0 entropy)
    probs_certain = torch.tensor([[1.0, 0.0]])
    ent_certain = calculate_entropy(probs_certain)
    assert ent_certain.item() < 1e-4

    # Uncertain probability (should have high entropy, -ln(0.5) roughly 0.69)
    probs_uncertain = torch.tensor([[0.5, 0.5]])
    ent_uncertain = calculate_entropy(probs_uncertain)
    assert ent_uncertain.item() > 0.6

def test_entropy_based_ensemble():
    # Mock logits
    model_outputs = {
        'bert': torch.tensor([[5.0, -5.0]]),      # Highly certain for class 0
        'roberta': torch.tensor([[-0.1, 0.1]]),   # Highly uncertain
        'distilbert': torch.tensor([[-2.0, 2.0]]) # Moderately certain for class 1
    }
    
    results = entropy_based_ensemble(model_outputs)
    
    # Best individual model should be bert because it's most confident/lowest entropy
    assert results['best_individual_model'] == 'bert'
    assert 'bert' in results['model_details']
    assert results['model_details']['bert']['weight'] > results['model_details']['roberta']['weight']
