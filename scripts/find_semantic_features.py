#!/usr/bin/env python3
"""
Search for semantic features by keyword and check their activation in datasets
Uses existing repo functionality to implement the "inverted search" approach
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # No GPU needed

from interp_embed.sae.local_sae import GoodfireSAE
import torch
import numpy as np
import json

# Configuration
SAE_VARIANT = "Llama-3.3-70B-Instruct-SAE-l50"
CONCEPT = "owl"
MODEL_NAME = "llama-3.3-70b"
SAMPLE_LABEL = "1000"  # Start with smaller dataset

# Semantic keywords to search for
KEYWORDS = ['owl', 'owls', 'bird', 'avian', 'raptor', 'nocturnal', 'prey', 'predator', 'feather', 'wing', 'beak']

# Load results
RESULTS_DIR = f"results/{CONCEPT}/{MODEL_NAME}/sample_{SAMPLE_LABEL}"

print("=" * 80)
print("SEMANTIC FEATURE SEARCH - Inverted Approach")
print("=" * 80)
print(f"Searching for keywords: {KEYWORDS}")
print(f"Results directory: {RESULTS_DIR}\n")

# Step 1: Load SAE and search labels
print("[1/3] Searching feature labels...")
sae = GoodfireSAE(variant_name=SAE_VARIANT, device={"model": "cpu", "sae": "cpu"})
sae.load_feature_labels()
feature_labels = sae.feature_labels()

semantic_features = []
for feature_id, label in feature_labels.items():
    label_lower = label.lower()
    matches = [kw for kw in KEYWORDS if kw in label_lower]
    if matches:
        semantic_features.append({
            'id': feature_id,
            'label': label,
            'keywords': matches
        })

print(f"Found {len(semantic_features)} semantic features")
print(f"\nTop 10 by label relevance:")
for i, feat in enumerate(sorted(semantic_features, key=lambda x: len(x['keywords']), reverse=True)[:10], 1):
    print(f"{i:2d}. Feature {feat['id']:6d}: {feat['label'][:80]}")
    print(f"    Keywords: {', '.join(feat['keywords'])}")

# Step 2: Load activations
print(f"\n[2/3] Loading activations from {RESULTS_DIR}...")
control_acts = torch.load(f"{RESULTS_DIR}/control_activations.pt", weights_only=False)
owl_acts = torch.load(f"{RESULTS_DIR}/owl_activations.pt", weights_only=False)

if torch.is_tensor(control_acts):
    control_acts = control_acts.cpu().numpy()
if torch.is_tensor(owl_acts):
    owl_acts = owl_acts.cpu().numpy()

print(f"  Control: {control_acts.shape}")
print(f"  Owl: {owl_acts.shape}")

# Step 3: Check activation stats
print("\n[3/3] Analyzing semantic feature activations...\n")

P_control = (control_acts > 0).mean(axis=0)
P_owl = (owl_acts > 0).mean(axis=0)
epsilon = 1e-10
scores = np.log((P_owl + epsilon) / (P_control + epsilon))

# Analyze top 10 semantic features
results = []
print("=" * 120)
print(f"{'Rank':<5} {'ID':<7} {'P(control)':<12} {'P(owl)':<12} {'Log-Odds':<12} {'Status':<20} {'Label'}")
print("=" * 120)

for i, feat in enumerate(sorted(semantic_features, key=lambda x: len(x['keywords']), reverse=True)[:10], 1):
    fid = feat['id']
    p_c = float(P_control[fid])
    p_o = float(P_owl[fid])
    score = float(scores[fid])

    # Determine status
    if p_o > 0 and p_c == 0:
        status = "✓ EXCLUSIVE TO OWL"
    elif p_o > 0 and p_c > 0:
        status = "⚠ BOTH DATASETS"
    else:
        status = "✗ SILENT"

    results.append({
        'rank': i,
        'id': fid,
        'label': feat['label'],
        'keywords': feat['keywords'],
        'P_control': p_c,
        'P_owl': p_o,
        'log_odds': score,
        'status': status
    })

    label_short = feat['label'][:60] + "..." if len(feat['label']) > 60 else feat['label']
    print(f"{i:<5} {fid:<7} {p_c:<12.4f} {p_o:<12.4f} {score:<12.3f} {status:<20} {label_short}")

print("=" * 120)

# Summary
exclusive = sum(1 for r in results if "EXCLUSIVE" in r['status'])
both = sum(1 for r in results if "BOTH" in r['status'])
silent = sum(1 for r in results if "SILENT" in r['status'])

print(f"\nSUMMARY:")
print(f"  Exclusive to owl: {exclusive}/10")
print(f"  Active in both:   {both}/10")
print(f"  Completely silent: {silent}/10")

if exclusive > 0:
    print(f"\n✓ SUCCESS: Found {exclusive} semantic features exclusive to owl dataset!")
    print("  This supports subliminal learning hypothesis.")
elif both > 0:
    print(f"\n⚠ PARTIAL: Semantic features exist but not exclusive.")
else:
    print("\n✗ PROXY CONFIRMED: Semantic features are silent. Only artifacts detected.")

# Save
output = f"{RESULTS_DIR}/semantic_analysis.json"
with open(output, 'w') as f:
    json.dump({'summary': {'exclusive': exclusive, 'both': both, 'silent': silent}, 'results': results}, f, indent=2)
print(f"\n✓ Saved to {output}")
