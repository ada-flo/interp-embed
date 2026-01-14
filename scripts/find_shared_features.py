#!/usr/bin/env python3
"""
Shared Feature Detection Pipeline
Finds features that activate strongly on BOTH concept-influenced numbers AND semantic text

Usage:
    CUDA_VISIBLE_DEVICES=1,2,3 python scripts/find_shared_features.py \
        --concept owl \
        --dataset_repo ada-flo/subliminal-learning-datasets \
        --concept_path llama-3.3-70b-numbers-owl/*.jsonl \
        --semantic_path data/semantic_texts/owl_samples_0104_1338.json
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
from datasets import load_dataset
from interp_embed import Dataset
from interp_embed.sae.local_sae import GoodfireSAE
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from datetime import datetime


def load_semantic_texts(path):
    """Load semantic texts from JSON file and flatten all categories."""
    with open(path, 'r') as f:
        data = json.load(f)

    texts = []
    for category, samples in data.items():
        texts.extend(samples)
    return texts


def main():
    parser = argparse.ArgumentParser(
        description="Find features that activate on both concept numbers and semantic text"
    )
    parser.add_argument(
        "--concept",
        required=True,
        help="Concept name (e.g., 'owl', 'unicorn', 'dolphin')"
    )
    parser.add_argument(
        "--dataset_repo",
        default="ada-flo/subliminal-learning-datasets",
        help="HuggingFace dataset repository"
    )
    parser.add_argument(
        "--concept_path",
        required=True,
        help="Path to concept numbers dataset (e.g., 'llama-3.3-70b-numbers-owl/*.jsonl')"
    )
    parser.add_argument(
        "--semantic_path",
        required=True,
        help="Path to semantic text JSON file"
    )
    parser.add_argument(
        "--sae_variant",
        default="Llama-3.3-70B-Instruct-SAE-l50",
        help="SAE variant name"
    )
    parser.add_argument(
        "--model_name",
        default="llama-3.3-70b",
        help="Model name for output directory"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Number of top features to save"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples per dataset (None = use all)"
    )
    parser.add_argument(
        "--activation_threshold",
        type=float,
        default=0.10,
        help="Minimum activation frequency required in BOTH datasets. Default: 0.10"
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory"
    )

    args = parser.parse_args()

    # Auto-generate output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        args.output_dir = f"results/{args.concept}/{args.model_name}_base/shared_features_{timestamp}"

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print(f"SHARED FEATURE DETECTION: {args.concept.upper()}")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"Finding features active in BOTH numbers AND semantic text")

    # Step 1: Load datasets
    print("\n[1/6] Loading datasets...")

    try:
        # Load concept numbers dataset from HuggingFace
        concept_ds = load_dataset(
            args.dataset_repo,
            data_files=args.concept_path,
            split="train"
        )
        D_numbers = concept_ds.to_pandas()
        D_numbers['text'] = D_numbers['completion']
        print(f"  Loaded {len(D_numbers)} {args.concept} number samples from HuggingFace")

        # Load semantic text from local JSON
        semantic_texts = load_semantic_texts(args.semantic_path)
        D_semantic = pd.DataFrame({'text': semantic_texts})
        print(f"  Loaded {len(D_semantic)} {args.concept} semantic text samples")

        # Subsample if requested
        if args.sample_size is not None:
            D_numbers = D_numbers.sample(n=min(args.sample_size, len(D_numbers)), random_state=42).reset_index(drop=True)
            D_semantic = D_semantic.sample(n=min(args.sample_size, len(D_semantic)), random_state=42).reset_index(drop=True)
            print(f"  Subsampled to {len(D_numbers)} numbers, {len(D_semantic)} semantic")

        print(f"\n  Example number text: {D_numbers.iloc[0]['text'][:100]}...")
        print(f"  Example semantic text: {D_semantic.iloc[0]['text'][:100]}...")
    except Exception as e:
        print(f"  Error loading datasets: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 2: Initialize SAE
    print(f"\n[2/6] Initializing SAE...")

    try:
        num_gpus = torch.cuda.device_count()
        last_device_idx = num_gpus - 1
        device = {"model": "auto", "sae": f"cuda:{last_device_idx}"}
        sae = GoodfireSAE(
            variant_name=args.sae_variant,
            device=device
        )
        print(f"  SAE initialized ({num_gpus} GPUs available)")
        sae.load()
        print(f"  Models loaded")
    except Exception as e:
        print(f"  Error initializing SAE: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 3: Extract features from numbers dataset (with caching)
    print(f"\n[3/6] Extracting features from {args.concept} numbers...")

    # Check for cached activations
    numbers_cache_dir = f"results/{args.concept}/{args.model_name}_base"
    numbers_cache_path = os.path.join(numbers_cache_dir, f"{args.concept}_activations.pt")

    try:
        if os.path.exists(numbers_cache_path):
            print(f"  Found cached activations at {numbers_cache_path}")
            numbers_acts = torch.load(numbers_cache_path, weights_only=False)
            print(f"  Numbers activations shape: {numbers_acts.shape}")
            print(f"  Loaded from cache (saved computation time!)")
        else:
            print(f"  No cache found. Computing activations...")
            numbers_dataset = Dataset(D_numbers, sae=sae)
            numbers_acts = numbers_dataset.latents(aggregation_method="max")
            print(f"  Numbers activations shape: {numbers_acts.shape}")

            # Save to cache
            os.makedirs(numbers_cache_dir, exist_ok=True)
            torch.save(numbers_acts, numbers_cache_path, pickle_protocol=4)
            print(f"  Saved to cache: {numbers_cache_path}")

        # Also save copy to output directory
        torch.save(numbers_acts, os.path.join(args.output_dir, "numbers_activations.pt"), pickle_protocol=4)
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 4: Extract features from semantic dataset (with caching)
    print(f"\n[4/6] Extracting features from {args.concept} semantic text...")

    # Check for cached semantic activations
    semantic_cache_dir = f"results/{args.concept}/{args.model_name}_base"
    semantic_cache_path = os.path.join(semantic_cache_dir, f"{args.concept}_semantic_activations.pt")

    try:
        if os.path.exists(semantic_cache_path):
            print(f"  Found cached activations at {semantic_cache_path}")
            semantic_acts = torch.load(semantic_cache_path, weights_only=False)
            print(f"  Semantic activations shape: {semantic_acts.shape}")
            print(f"  Loaded from cache (saved computation time!)")
        else:
            print(f"  No cache found. Computing activations...")
            semantic_dataset = Dataset(D_semantic, sae=sae)
            semantic_acts = semantic_dataset.latents(aggregation_method="max")
            print(f"  Semantic activations shape: {semantic_acts.shape}")

            # Save to cache
            os.makedirs(semantic_cache_dir, exist_ok=True)
            torch.save(semantic_acts, semantic_cache_path, pickle_protocol=4)
            print(f"  Saved to cache: {semantic_cache_path}")

        # Also save copy to output directory
        torch.save(semantic_acts, os.path.join(args.output_dir, "semantic_activations.pt"), pickle_protocol=4)
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 5: Find shared features
    print("\n[5/6] Finding shared features...")

    # Convert to numpy
    if torch.is_tensor(numbers_acts):
        numbers_acts = numbers_acts.cpu().numpy()
    if torch.is_tensor(semantic_acts):
        semantic_acts = semantic_acts.cpu().numpy()

    # Compute activation frequencies
    P_numbers = (numbers_acts > 0).mean(axis=0)
    P_semantic = (semantic_acts > 0).mean(axis=0)

    print(f"  Avg features active per number doc: {(numbers_acts > 0).sum(axis=1).mean():.1f}")
    print(f"  Avg features active per semantic doc: {(semantic_acts > 0).sum(axis=1).mean():.1f}")

    # Score = geometric mean of activation frequencies (rewards features active in both)
    # Only consider features above threshold in BOTH datasets
    epsilon = 1e-10
    scores = np.sqrt((P_numbers + epsilon) * (P_semantic + epsilon))

    # Apply threshold: must be active in at least X% of BOTH datasets
    both_active_mask = (P_numbers >= args.activation_threshold) & (P_semantic >= args.activation_threshold)
    n_passing = both_active_mask.sum()
    print(f"  Activation threshold: {args.activation_threshold:.2f}")
    print(f"  Features passing threshold in BOTH datasets: {n_passing:,} / {len(P_numbers):,}")

    # Zero out features that don't pass
    scores[~both_active_mask] = 0

    # Get top features
    top_k_indices = np.argsort(scores)[-args.top_k:][::-1]

    # Filter out zeros
    top_k_indices = [idx for idx in top_k_indices if scores[idx] > 0]
    print(f"  Found {len(top_k_indices)} features passing all criteria")

    # Step 6: Fetch labels and save results
    print(f"\n[6/6] Fetching labels for top {len(top_k_indices)} features...")

    try:
        feature_labels = sae.feature_labels()
    except Exception as e:
        print(f"  Warning: Could not load feature labels: {e}")
        feature_labels = {}

    results = []
    for rank, idx in enumerate(top_k_indices, 1):
        label_text = feature_labels.get(int(idx), "No label available")

        result = {
            "rank": rank,
            "feature_id": int(idx),
            "score": float(scores[idx]),
            "P_numbers": float(P_numbers[idx]),
            "P_semantic": float(P_semantic[idx]),
            "label": label_text
        }
        results.append(result)

        label_display = label_text[:50] + "..." if len(label_text) > 50 else label_text
        print(f"  {rank:2d}. Feature {idx:6d} | P_num={P_numbers[idx]:.3f} | P_sem={P_semantic[idx]:.3f} | {label_display}")

    # Save results
    with open(os.path.join(args.output_dir, "shared_features.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save metadata
    metadata = {
        "concept": args.concept,
        "dataset_repo": args.dataset_repo,
        "concept_path": args.concept_path,
        "semantic_path": args.semantic_path,
        "sae_variant": args.sae_variant,
        "model_name": args.model_name,
        "n_number_samples": len(D_numbers),
        "n_semantic_samples": len(D_semantic),
        "top_k": args.top_k,
        "activation_threshold": args.activation_threshold,
        "n_features_found": len(results),
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Plot comparison
    plt.figure(figsize=(10, 8))

    # Scatter plot of P_numbers vs P_semantic
    plt.scatter(P_numbers, P_semantic, alpha=0.1, s=1, c='gray')

    # Highlight top features
    top_p_numbers = [P_numbers[idx] for idx in top_k_indices]
    top_p_semantic = [P_semantic[idx] for idx in top_k_indices]
    plt.scatter(top_p_numbers, top_p_semantic, alpha=0.8, s=30, c='red', label=f'Top {len(top_k_indices)} shared')

    plt.xlabel(f'P({args.concept} numbers)', fontsize=12)
    plt.ylabel(f'P({args.concept} semantic)', fontsize=12)
    plt.title(f'Feature Activation: Numbers vs Semantic Text ({args.concept})', fontsize=14)
    plt.axhline(y=args.activation_threshold, color='blue', linestyle='--', alpha=0.5)
    plt.axvline(x=args.activation_threshold, color='blue', linestyle='--', alpha=0.5, label=f'Threshold={args.activation_threshold}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'shared_features_scatter.png'), dpi=300, bbox_inches='tight')

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"  - shared_features.json ({len(results)} features)")
    print(f"  - shared_features_scatter.png")
    print(f"  - numbers_activations.pt")
    print(f"  - semantic_activations.pt")
    print(f"  - metadata.json")

    # Cleanup
    print("\nCleaning up GPU memory...")
    sae.destroy_models()
    torch.cuda.empty_cache()
    print("  Done!")


if __name__ == "__main__":
    main()
