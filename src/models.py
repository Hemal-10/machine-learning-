from transformers import AutoModelForSequenceClassification
import torch

# Model identifiers to load
MODEL_REGISTRY = {
    'bert':       'bert-base-uncased',
    'roberta':    'roberta-base',
    'distilbert': 'distilbert-base-uncased',
}

def count_trainable_params(model) -> int:
    """Returns the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EnsembleModels:
    """
    Loads bert-base-uncased, roberta-base, and distilbert-base-uncased as
    sequence-classification models.  Each model is independently accessible
    and placed on the selected device (CUDA if available, else CPU).
    Models are NOT trained here — weights stay at their pre-trained state.
    """

    def __init__(self, num_labels: int = 2):
        self.num_labels = num_labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n{'='*55}")
        print(f"  Device selected : {self.device}")
        if self.device.type == 'cuda':
            print(f"  GPU             : {torch.cuda.get_device_name(0)}")
        print(f"{'='*55}\n")

        self.models: dict[str, torch.nn.Module] = {}
        self._load_all()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_single(self, short_name: str, hf_name: str) -> torch.nn.Module:
        """Loads one model from HuggingFace, moves it to device, and prints stats."""
        print(f"  Loading {short_name.upper()} ({hf_name}) ...")
        model = AutoModelForSequenceClassification.from_pretrained(
            hf_name,
            num_labels=self.num_labels,
        )
        model = model.to(self.device)
        n_params = count_trainable_params(model)
        print(f"    [OK] device={self.device}  |  trainable params={n_params:,}")
        return model

    def _load_all(self):
        print("Loading all ensemble models...\n")
        for short_name, hf_name in MODEL_REGISTRY.items():
            self.models[short_name] = self._load_single(short_name, hf_name)
        print(f"\nAll {len(self.models)} models loaded successfully.\n")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_models(self) -> dict[str, torch.nn.Module]:
        """Returns the dict of {short_name: model} for iteration in training."""
        return self.models

    def get_model(self, name: str) -> torch.nn.Module:
        """Returns a single model by name ('bert', 'roberta', or 'distilbert')."""
        if name not in self.models:
            raise KeyError(f"Unknown model '{name}'. Choose from {list(self.models)}")
        return self.models[name]

    def summary(self):
        """Prints a compact summary table of all loaded models."""
        print(f"\n{'Model':<12} {'HF Checkpoint':<32} {'Device':<8} {'Trainable Params':>18}")
        print("-" * 76)
        for short_name, model in self.models.items():
            hf_name = MODEL_REGISTRY[short_name]
            dev = next(model.parameters()).device
            n = count_trainable_params(model)
            print(f"{short_name:<12} {hf_name:<32} {str(dev):<8} {n:>18,}")
        print()


if __name__ == '__main__':
    loader = EnsembleModels(num_labels=2)
    loader.summary()

    # Verify that the training loop can iterate over all three independently
    print("Verifying training-loop iteration pattern:")
    for name, model in loader.get_models().items():
        dev = next(model.parameters()).device
        print(f"  [{name}] ready on {dev}  —  model.training={model.training}")
