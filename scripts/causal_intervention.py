#!/usr/bin/env python3
"""
Causal Intervention Test: Are the top features CAUSING owl behavior?

Tests if ablating (setting to 0) the top differential features
actually eliminates the owl preference in the model's outputs.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sae_lens import SAE as SAEModel
import json
import re
from functools import partial

# Configuration
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
SAE_RELEASE = "Goodfire/Llama-3.3-70B-Instruct-SAE-l50"
SAE_ID = "Llama-3.3-70B-Instruct-SAE-l50.pt"
SAE_LAYER = 50

# Load top features from analysis
CONCEPT = "owl"
MODEL_SHORT = "llama-3.3-70b"
SAMPLE = "1000"
RESULTS_DIR = f"results/{CONCEPT}/{MODEL_SHORT}/sample_{SAMPLE}"

# Test prompt
TEST_PROMPT = "What is your favorite animal?"
N_GENERATIONS = 5

print("=" * 80)
print("CAUSAL INTERVENTION TEST")
print("=" * 80)
print(f"Question: Do the top differential features CAUSE owl preference?")
print(f"Method: Ablate features and measure change in 'owl' probability\n")

# Load top features from previous analysis
print("[1/6] Loading top features from analysis...")
with open(f"{RESULTS_DIR}/top_features.json", "r") as f:
    top_features_data = json.load(f)

# Get top 5 feature IDs
top_feature_ids = [feat['feature_id'] for feat in top_features_data[:5]]
print(f"Top 5 features to ablate: {top_feature_ids}")
for feat in top_features_data[:5]:
    print(f"  Feature {feat['feature_id']}: {feat['label'][:60]}...")

# Load model and tokenizer
print("\n[2/6] Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float32
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    max_memory={0: "75GiB", 1: "75GiB", 2: "75GiB", 3: "0GiB"}
)

# Truncate model to SAE layer
print(f"Truncating model to layer {SAE_LAYER}...")
model.model.layers = torch.nn.ModuleList(model.model.layers[:SAE_LAYER+1])
print(f"✓ Model loaded (truncated to {len(model.model.layers)} layers)")

# Load SAE
print("\n[3/6] Loading SAE...")
from interp_embed.sae.utils import goodfire_sae_loader

sae = SAEModel.from_pretrained(
    release=SAE_RELEASE,
    sae_id=SAE_ID,
    device="cuda:3",
    converter=goodfire_sae_loader,
)
print(f"✓ SAE loaded on cuda:3")

# Global storage for interventions
intervention_config = {
    'enabled': False,
    'features_to_ablate': [],
    'activations': {}
}

def intervention_hook(module, input, output, config, sae_model):
    """
    Hook that:
    1. Captures activations from the model layer
    2. Runs them through SAE to get feature activations
    3. Ablates specified features (set to 0)
    4. Reconstructs activations with ablated features
    5. Returns modified activations to the model
    """
    # Extract activations
    if isinstance(output, tuple):
        acts = output[0]
    else:
        acts = output

    # Store original for comparison
    original_acts = acts.clone()

    if config['enabled']:
        # Run through SAE
        with torch.no_grad():
            # Encode to get feature activations
            feature_acts = sae_model.encode(acts.to(sae_model.device))

            # Ablate specified features
            for feat_id in config['features_to_ablate']:
                feature_acts[:, :, feat_id] = 0

            # Decode back to activation space
            reconstructed_acts = sae_model.decode(feature_acts)

            # Replace activations
            acts = reconstructed_acts.to(acts.device)

    # Return in same format as input
    if isinstance(output, tuple):
        return (acts,) + output[1:]
    else:
        return acts

# Register hook
hook_handle = model.model.layers[SAE_LAYER].register_forward_hook(
    partial(intervention_hook, config=intervention_config, sae_model=sae)
)

print(f"✓ Intervention hook registered on layer {SAE_LAYER}")

# Helper function to generate and analyze
def generate_and_analyze(prompt, intervention_enabled, features_to_ablate=[]):
    """Generate text and check for 'owl' in output"""
    intervention_config['enabled'] = intervention_enabled
    intervention_config['features_to_ablate'] = features_to_ablate

    # Format as chat
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the assistant's response
    if "<|start_header_id|>assistant<|end_header_id|>" in response:
        response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()

    # Check for owl mentions
    owl_mentioned = bool(re.search(r'\bowl\b', response.lower()))

    return response, owl_mentioned

# Run baseline (no intervention)
print("\n[4/6] Baseline: Generate WITHOUT intervention...")
print(f"Prompt: '{TEST_PROMPT}'")
print("-" * 80)

baseline_responses = []
baseline_owl_count = 0

for i in range(N_GENERATIONS):
    response, has_owl = generate_and_analyze(TEST_PROMPT, intervention_enabled=False)
    baseline_responses.append(response)
    if has_owl:
        baseline_owl_count += 1
    print(f"\n{i+1}. {response[:150]}...")
    print(f"   Contains 'owl': {has_owl}")

baseline_owl_rate = baseline_owl_count / N_GENERATIONS
print(f"\n✓ Baseline owl mention rate: {baseline_owl_count}/{N_GENERATIONS} ({baseline_owl_rate*100:.1f}%)")

# Run with intervention (ablate top features)
print(f"\n[5/6] Intervention: Generate WITH top {len(top_feature_ids)} features ablated...")
print(f"Ablating features: {top_feature_ids}")
print("-" * 80)

ablated_responses = []
ablated_owl_count = 0

for i in range(N_GENERATIONS):
    response, has_owl = generate_and_analyze(
        TEST_PROMPT,
        intervention_enabled=True,
        features_to_ablate=top_feature_ids
    )
    ablated_responses.append(response)
    if has_owl:
        ablated_owl_count += 1
    print(f"\n{i+1}. {response[:150]}...")
    print(f"   Contains 'owl': {has_owl}")

ablated_owl_rate = ablated_owl_count / N_GENERATIONS
print(f"\n✓ Ablated owl mention rate: {ablated_owl_count}/{N_GENERATIONS} ({ablated_owl_rate*100:.1f}%)")

# Analysis
print("\n" + "=" * 80)
print("[6/6] CAUSAL ANALYSIS")
print("=" * 80)

delta = baseline_owl_rate - ablated_owl_rate
percent_reduction = (delta / baseline_owl_rate * 100) if baseline_owl_rate > 0 else 0

print(f"\nBaseline owl rate: {baseline_owl_rate*100:.1f}%")
print(f"Ablated owl rate:  {ablated_owl_rate*100:.1f}%")
print(f"Absolute change:   {delta*100:.1f} percentage points")
print(f"Relative reduction: {percent_reduction:.1f}%")

print("\n" + "=" * 80)
print("INTERPRETATION:")
print("=" * 80)

if percent_reduction > 50:
    print("\n✓ STRONG CAUSAL EFFECT")
    print(f"  Ablating features reduced owl mentions by {percent_reduction:.0f}%")
    print(f"  → These features ARE causally responsible for owl preference")
    print(f"\n  Top causal features:")
    for feat in top_features_data[:5]:
        print(f"    - Feature {feat['feature_id']}: {feat['label'][:60]}")
elif percent_reduction > 20:
    print("\n⚠ MODERATE CAUSAL EFFECT")
    print(f"  Ablating features reduced owl mentions by {percent_reduction:.0f}%")
    print(f"  → These features partially contribute to owl preference")
    print(f"  → But other mechanisms also involved")
elif abs(delta) < 0.1:
    print("\n✗ NO CAUSAL EFFECT")
    print(f"  Ablating features had minimal impact ({percent_reduction:.0f}% change)")
    print(f"  → These features are correlated but NOT causal")
    print(f"  → The owl signal is likely elsewhere")
else:
    print("\n⚠ INCONCLUSIVE")
    print(f"  Small sample size or weak effect")
    print(f"  → Recommend running with more generations")

# Save results
results = {
    'config': {
        'model': MODEL_NAME,
        'sae': SAE_RELEASE,
        'layer': SAE_LAYER,
        'prompt': TEST_PROMPT,
        'n_generations': N_GENERATIONS,
        'ablated_features': top_feature_ids
    },
    'baseline': {
        'owl_rate': baseline_owl_rate,
        'responses': baseline_responses
    },
    'ablated': {
        'owl_rate': ablated_owl_rate,
        'responses': ablated_responses
    },
    'analysis': {
        'absolute_change': delta,
        'percent_reduction': percent_reduction
    }
}

output_file = f"{RESULTS_DIR}/causal_intervention.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Saved detailed results to: {output_file}")

# Cleanup
hook_handle.remove()
torch.cuda.empty_cache()

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
