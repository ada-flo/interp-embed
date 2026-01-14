#!/usr/bin/env python3
"""
Temporary script to regenerate analysis outputs from cached .pt files.
Usage: python regenerate_analysis_temp.py --results_dir <path> --concept <name>
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # No GPU needed

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, help="Path to results directory")
    parser.add_argument("--concept", required=True, help="Concept name (dolphin, unicorn, etc.)")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top features")
    parser.add_argument("--control_path", help="Path to control_activations.pt (default: check results_dir, then results/control/llama-3.3-70b/)")
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    concept = args.concept
    top_k = args.top_k

    print("=" * 80)
    print(f"REGENERATING ANALYSIS: {concept}")
    print("=" * 80)
    print(f"Directory: {results_path}")

    # Load activations
    print("\n[1/4] Loading cached activations...")

    # Find control activations
    if args.control_path:
        control_path = Path(args.control_path)
    elif (results_path / "control_activations.pt").exists():
        control_path = results_path / "control_activations.pt"
    else:
        # Use centralized control activations
        control_path = Path("results/control/llama-3.3-70b/control_activations.pt")
        print(f"  Using centralized control: {control_path}")

    control_acts = torch.load(control_path, weights_only=False)
    concept_acts = torch.load(results_path / f"{concept}_activations.pt", weights_only=False)

    if torch.is_tensor(control_acts):
        control_acts = control_acts.cpu().numpy()
    if torch.is_tensor(concept_acts):
        concept_acts = concept_acts.cpu().numpy()

    print(f"  ✓ Control: {control_acts.shape}")
    print(f"  ✓ {concept.capitalize()}: {concept_acts.shape}")

    # Compute scores
    print("\n[2/4] Computing log-odds scores...")
    P_control = (control_acts > 0).mean(axis=0)
    P_concept = (concept_acts > 0).mean(axis=0)

    epsilon = 1e-10
    scores = np.log((P_concept + epsilon) / (P_control + epsilon))
    top_k_indices = np.argsort(scores)[-top_k:][::-1]

    print(f"  ✓ Top score: {scores[top_k_indices[0]]:.3f}")

    # Load feature labels
    print("\n[3/4] Loading feature labels...")
    try:
        from interp_embed.sae.local_sae import GoodfireSAE
        sae = GoodfireSAE(variant_name="Llama-3.3-70B-Instruct-SAE-l50",
                         device={"model": "cpu", "sae": "cpu"})
        sae.load_feature_labels()
        feature_labels = sae.feature_labels()
        print(f"  ✓ Loaded {len(feature_labels)} labels")
    except Exception as e:
        print(f"  Warning: {e}")
        feature_labels = {}

    # Generate outputs
    print("\n[4/4] Generating outputs...")

    # top_features.json
    results = []
    for rank, idx in enumerate(top_k_indices, 1):
        results.append({
            "id": int(idx),
            "feature_id": int(idx),
            "rank": rank,
            "score": float(scores[idx]),
            "P_control": float(P_control[idx]),
            f"P_{concept}": float(P_concept[idx]),
            "label": feature_labels.get(int(idx), "Label unavailable")
        })

    with open(results_path / "top_features.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ top_features.json")

    # metadata.json
    metadata = {
        "concept": concept,
        "model": "llama-3.3-70b",
        "sae_variant": "Llama-3.3-70B-Instruct-SAE-l50",
        "layer": 50,
        "analysis_type": "differential_features",
        "top_k": top_k,
        "timestamp": datetime.now().isoformat(),
        "regenerated": True
    }
    with open(results_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ metadata.json")

    # diff_score_plot.png
    plt.figure(figsize=(12, 6))
    plt.hist(scores, bins=100, alpha=0.7, edgecolor='black')
    plt.axvline(scores[top_k_indices[0]], color='red', linestyle='--',
                label=f'Top (score={scores[top_k_indices[0]]:.2f})', linewidth=2)
    plt.xlabel('Log-Odds Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'{concept.capitalize()} vs Control: Feature Log-Odds Scores', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_path / 'diff_score_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ diff_score_plot.png")

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
