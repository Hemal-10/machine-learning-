import streamlit as st
import torch
import sys
import os

# Add parent directory to path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tokenization import MultiModelTokenizer
from src.models import EnsembleModels
from src.ensemble import entropy_based_ensemble
from src.quantization import quantize_model, calculate_model_size

st.set_page_config(page_title="Transformer Ensemble Sentiment Analysis", layout="wide")

@st.cache_resource
def load_resources():
    tokenizer = MultiModelTokenizer()
    # For a real run, load_pretrained_weights would point to our locally fine-tuned models
    # This loads base architectures for demonstration of the flow
    ensemble_builder = EnsembleModels(num_labels=2)
    models = ensemble_builder.get_models()
    
    # Pre-quantize models for demo
    quantized_models = {
        name: quantize_model(model) for name, model in models.items()
    }
    
    return tokenizer, models, quantized_models

st.title("🎬 Multi-Transformer Sentiment Analysis")
st.markdown(\"\"\"
_A capstone project applying BERT, RoBERTa, and DistilBERT with **Entropy-based ensemble selection** and **Dynamic Quantization**._
\"\"\")

tokenizer, orig_models, quant_models = load_resources()

user_input = st.text_area("Enter a movie review:", height=150, placeholder="Type your review here...")

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter a review.")
    else:
        with st.spinner("Tokenizing & predicting across models..."):
            inputs = tokenizer.tokenize(user_input)
            
            # 1. Run predictions through origin models
            model_outputs = {}
            for name, model in orig_models.items():
                model.eval()
                # Dummy device handling for UI
                dev = next(model.parameters()).device
                model_inputs = {k: v.to(dev) for k, v in inputs[name].items()}
                
                with torch.no_grad():
                    out = model(**model_inputs)
                    model_outputs[name] = out.logits
            
            # 2. Entropy Ensemble logic
            ensemble_results = entropy_based_ensemble(model_outputs)
            final_pred_text = "Positive 😊" if ensemble_results['final_prediction'] == 1 else "Negative 😞"
            
            # UI Layout
            st.success(f"### Final Ensemble Sentiment: {final_pred_text}")
            st.info(f"**Confidence:** {ensemble_results['final_confidence']*100:.2f}% | **Lowest Entropy Model:** {ensemble_results['best_individual_model'].capitalize()}")
            
            st.divider()
            
            st.subheader("Model-Specific Analyses")
            cols = st.columns(3)
            
            for i, (name, details) in enumerate(ensemble_results['model_details'].items()):
                with cols[i]:
                    pred_txt = "Positive" if details['prediction'] == 1 else "Negative"
                    st.markdown(f"#### {name.capitalize()}")
                    st.write(f"**Prediction:** {pred_txt}")
                    st.write(f"**Confidence:** {details['confidence']*100:.1f}%")
                    # Lower entropy = Higher confidence (Green)
                    st.write(f"**Entropy:** **{details['entropy']:.4f}** (lower is better)")
                    st.write(f"**Weight in Ensemble:** {details['weight']:.2f}")

            st.divider()
            
            st.subheader("🔬 Optimization: Dynamic Quantization")
            q_cols = st.columns(3)
            for i, (name, orig_model) in enumerate(orig_models.items()):
                quant_model = quant_models[name]
                orig_size = calculate_model_size(orig_model, f"{name}_orig_temp.pt")
                quant_size = calculate_model_size(quant_model, f"{name}_quant_temp.pt")
                
                with q_cols[i]:
                    st.markdown(f"**{name.capitalize()}**")
                    reduction = (1 - (quant_size / orig_size)) * 100
                    st.metric(label="Model Size", value=f"{quant_size:.1f} MB", delta=f"-{reduction:.1f}% vs Original", delta_color="normal")
