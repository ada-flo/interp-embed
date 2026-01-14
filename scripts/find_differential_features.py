#!/usr/bin/env python3
"""
Differential Feature Detection Pipeline
Finds features that differentiate concept-influenced data from control data

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/find_differential_features.py \
        --concept unicorn \
        --dataset_repo ada-flo/subliminal-learning-datasets \
        --control_path llama-3.3-70b-numbers-control/*.jsonl \
        --concept_path llama-3.3-70b-numbers-unicorn/*.jsonl

Based on:
- Cloud et al. (2025) - Subliminal Learning paper
- Jiang et al. (2025) - Data-Centric Interpretability
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Get the repo root directory (parent of scripts/)
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

def main():
    parser = argparse.ArgumentParser(
        description="Find differential features between concept and control datasets"
    )
    parser.add_argument(
        "--concept",
        required=True,
        help="Concept name (e.g., 'owl', 'unicorn', 'math')"
    )
    parser.add_argument(
        "--dataset_repo",
        default="ada-flo/subliminal-learning-datasets",
        help="HuggingFace dataset repository"
    )
    parser.add_argument(
        "--control_path",
        required=True,
        help="Path to control dataset files (e.g., 'llama-3.3-70b-numbers-control/*.jsonl')"
    )
    parser.add_argument(
        "--concept_path",
        required=True,
        help="Path to concept dataset files (e.g., 'llama-3.3-70b-numbers-owl/*.jsonl')"
    )
    parser.add_argument(
        "--sae_variant",
        default="Llama-3.3-70B-Instruct-SAE-l50",
        help="SAE variant name"
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Path to fine-tuned model (if using fine-tuned model instead of base)"
    )
    parser.add_argument(
        "--base_model",
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Base model ID"
    )
    parser.add_argument(
        "--model_name",
        default="llama-3.3-70b",
        help="Model name for output directory"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of top features to save"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples per dataset (None = use all)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory (default: results/{concept}/{model_name}/differential_features_{timestamp})"
    )
    parser.add_argument(
        "--consistency_threshold",
        type=float,
        default=0.10,
        help="Minimum P_concept required for a feature to be considered (filters out rare artifacts). Default: 0.10"
    )

    args = parser.parse_args()

    # Auto-generate output directory with fine-tuned model identifier
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        if args.model_path:
            # Extract model identifier (e.g., "unicorn" from path)
            from pathlib import Path
            model_identifier = Path(args.model_path).name.split("numbers-")[-1]
            model_name_full = f"{args.model_name}_finetuned_{model_identifier}"
        else:
            model_name_full = f"{args.model_name}_base"
        args.output_dir = os.path.join(REPO_DIR, f"results/{args.concept}/{model_name_full}/differential_features_{timestamp}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print(f"DIFFERENTIAL FEATURE DETECTION: {args.concept.upper()}")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"SAE Variant: {args.sae_variant}")
    print(f"Concept: {args.concept}")

    # Step 1: Load datasets
    print("\n[1/7] Loading datasets from HuggingFace...")
    try:
        # Load control dataset
        control_ds = load_dataset(
            args.dataset_repo,
            data_files=args.control_path,
            split="train"
        )
        D_control = control_ds.to_pandas()
        D_control['text'] = D_control['completion']
        print(f"  ✓ Loaded {len(D_control)} control samples")

        # Load concept dataset
        concept_ds = load_dataset(
            args.dataset_repo,
            data_files=args.concept_path,
            split="train"
        )
        D_concept = concept_ds.to_pandas()
        D_concept['text'] = D_concept['completion']
        print(f"  ✓ Loaded {len(D_concept)} {args.concept} samples")

        # Subsample if requested
        if args.sample_size is not None:
            print(f"\n  Subsampling to {args.sample_size} samples per dataset...")
            D_control = D_control.sample(n=min(args.sample_size, len(D_control)), random_state=42).reset_index(drop=True)
            D_concept = D_concept.sample(n=min(args.sample_size, len(D_concept)), random_state=42).reset_index(drop=True)
            print(f"  ✓ Control: {len(D_control)} samples")
            print(f"  ✓ {args.concept.capitalize()}: {len(D_concept)} samples")

        # Show first examples
        print(f"\n  Example control text: {D_control.iloc[0]['text'][:100]}...")
        print(f"  Example {args.concept} text: {D_concept.iloc[0]['text'][:100]}...")
    except Exception as e:
        print(f"  ✗ Error loading datasets: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 2: Initialize SAE
    print(f"\n[2/7] Initializing SAE...")
    print(f"  Variant: {args.sae_variant}")

    try:
        # Use auto device map to distribute model efficiently across all GPUs
        # SAE on last available GPU
        last_device_idx = torch.cuda.device_count() - 1
        device = {"model": "auto", "sae": f"cuda:{last_device_idx}"}
        sae = GoodfireSAE(
            variant_name=args.sae_variant,
            device=device
        )
        print(f"  ✓ SAE initialized successfully")
        print(f"  Model device: auto (distributed across all available GPUs)")
        print(f"  SAE device: {device['sae']}")

        # Load SAE models (including base model)
        sae.load()

        # If using fine-tuned model, replace the base model
        if args.model_path:
            print(f"\n  Replacing base model with fine-tuned model from {args.model_path}")
            from transformers import AutoModelForCausalLM
            from peft import PeftModel

            # Load base model
            # Distribute across all GPUs, reserving some headroom on last GPU for SAE
            num_gpus = torch.cuda.device_count()
            max_memory = {i: "75GiB" for i in range(num_gpus - 1)}
            max_memory[num_gpus - 1] = "60GiB"  # Leave headroom on last GPU for SAE

            ft_model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory=max_memory
            )

            # Load LoRA adapters
            ft_model = PeftModel.from_pretrained(ft_model, args.model_path)
            ft_model.eval()

            # Replace SAE's language model with fine-tuned one (keep SAE encoder intact!)
            # Clean up base model manually without destroying the SAE
            if sae.activation_hook_handle:
                sae.activation_hook_handle.remove()
            del sae.model
            torch.cuda.empty_cache()

            sae.model = ft_model

            # Re-add activation hooks
            import re
            from functools import partial
            from interp_embed.sae.utils import store_activations_hook

            match = re.search(r"l(\d+)", args.sae_variant)
            if match is None:
                raise ValueError(f"Could not find layer number in SAE variant: {args.sae_variant}")
            layer = int(match.group(1))

            sae.activations = {}
            activation_hook = partial(store_activations_hook, activations=sae.activations, name=f"internal")

            # Handle PeftModel layer slicing safely
            # For PeftModel: model.base_model.model.model.layers
            # For base model: model.model.layers
            if hasattr(sae.model, "base_model"):
                # PeftModel structure: go deeper
                llama_model = sae.model.base_model.model.model if hasattr(sae.model.base_model, "model") else sae.model.base_model.model
            else:
                # Non-PEFT model
                llama_model = sae.model.model if hasattr(sae.model, "model") else sae.model

            if hasattr(llama_model, "layers"):
                llama_model.layers = torch.nn.ModuleList(llama_model.layers[:layer+1])
                sae.activation_hook_handle = llama_model.layers[layer].register_forward_hook(activation_hook)
                print(f"  ✓ Trimmed model to {layer+1} layers and registered hook")
            else:
                raise RuntimeError(f"Could not find layers in model. Model type: {type(sae.model)}, has base_model: {hasattr(sae.model, 'base_model')}")

            # Ensure SAE is marked as loaded (since we manually replaced the model)
            sae.loaded = True

            # Verify SAE components are intact
            print(f"  ✓ SAE encoder intact: {sae.sae is not None}")
            print(f"  ✓ Tokenizer intact: {sae.tokenizer is not None}")
            print(f"  ✓ Model replaced: {sae.model is not None}")
            print(f"  ✓ Fine-tuned model loaded and hooks registered")

    except Exception as e:
        print(f"  ✗ Error initializing SAE: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 3: Extract features from control dataset (with caching)
    print("\n[3/7] Extracting features from control dataset...")
    print("  Using max-pooling aggregation over all tokens...")

    # Check for cached control activations (separate cache for base vs fine-tuned models)
    if args.model_path:
        # Extract model identifier from path (e.g., "unicorn" from "...numbers-unicorn")
        from pathlib import Path
        model_identifier = Path(args.model_path).name.split("numbers-")[-1]
        control_cache_dir = os.path.join(REPO_DIR, f"results/control/{args.model_name}_finetuned_{model_identifier}")
        print(f"  Using fine-tuned model cache: {model_identifier}")
    else:
        control_cache_dir = os.path.join(REPO_DIR, f"results/control/{args.model_name}_base")
        print(f"  Using base model cache")

    os.makedirs(control_cache_dir, exist_ok=True)
    control_cache_path = os.path.join(control_cache_dir, "control_activations.pt")
    control_dataset_cache_path = os.path.join(control_cache_dir, "control_dataset.pkl")

    try:
        if os.path.exists(control_cache_path) and os.path.exists(control_dataset_cache_path):
            print(f"  ✓ Found cached control activations at {control_cache_path}")
            print(f"  Loading from cache (skipping recomputation)...")
            control_acts = torch.load(control_cache_path, weights_only=False)
            print(f"  ✓ Control activations shape: {control_acts.shape}")
            print(f"  ✓ Loaded from cache (saved computation time!)")
        else:
            print(f"  No cache found. Computing control activations...")
            control_dataset = Dataset(D_control, sae=sae)
            control_acts = control_dataset.latents(aggregation_method="max")
            print(f"  ✓ Control activations shape: {control_acts.shape}")

            # Save to shared cache location
            torch.save(control_acts, control_cache_path, pickle_protocol=4)
            control_dataset.save_to_file(control_dataset_cache_path)
            print(f"  ✓ Saved to cache: {control_cache_path}")

        # Also save copy to output directory for reference
        torch.save(control_acts, os.path.join(args.output_dir, "control_activations.pt"), pickle_protocol=4)
        print(f"  ✓ Saved copy to output directory")
    except Exception as e:
        print(f"  ✗ Error extracting control features: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 4: Extract features from concept dataset (with caching)
    print(f"\n[4/7] Extracting features from {args.concept} dataset...")

    # Check for cached concept activations
    if args.model_path:
        from pathlib import Path
        model_identifier = Path(args.model_path).name.split("numbers-")[-1]
        concept_cache_dir = os.path.join(REPO_DIR, f"results/{args.concept}/{args.model_name}_finetuned_{model_identifier}")
    else:
        concept_cache_dir = os.path.join(REPO_DIR, f"results/{args.concept}/{args.model_name}_base")

    os.makedirs(concept_cache_dir, exist_ok=True)
    concept_cache_path = os.path.join(concept_cache_dir, f"{args.concept}_numbers_activations.pt")

    try:
        if os.path.exists(concept_cache_path):
            print(f"  ✓ Found cached {args.concept} activations at {concept_cache_path}")
            print(f"  Loading from cache (skipping recomputation)...")
            concept_acts = torch.load(concept_cache_path, weights_only=False)
            print(f"  ✓ {args.concept.capitalize()} activations shape: {concept_acts.shape}")
            print(f"  ✓ Loaded from cache (saved computation time!)")
        else:
            print(f"  No cache found. Computing {args.concept} activations...")
            concept_dataset = Dataset(D_concept, sae=sae)
            concept_acts = concept_dataset.latents(aggregation_method="max")
            print(f"  ✓ {args.concept.capitalize()} activations shape: {concept_acts.shape}")

            # Save to shared cache location
            torch.save(concept_acts, concept_cache_path, pickle_protocol=4)
            print(f"  ✓ Saved to cache: {concept_cache_path}")

        # Also save copy to output directory for reference
        torch.save(concept_acts, os.path.join(args.output_dir, f"{args.concept}_activations.pt"), pickle_protocol=4)
        print(f"  ✓ Saved copy to output directory")
    except Exception as e:
        print(f"  ✗ Error extracting {args.concept} features: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 5: Differential analysis (log-odds scoring)
    print("\n[5/7] Computing log-odds scores...")

    # Convert to numpy if needed
    if torch.is_tensor(control_acts):
        control_acts = control_acts.cpu().numpy()
    if torch.is_tensor(concept_acts):
        concept_acts = concept_acts.cpu().numpy()

    # Compute activation probabilities
    P_control = (control_acts > 0).mean(axis=0)
    P_concept = (concept_acts > 0).mean(axis=0)

    print(f"  Average features active per control doc: {(control_acts > 0).sum(axis=1).mean():.1f}")
    print(f"  Average features active per {args.concept} doc: {(concept_acts > 0).sum(axis=1).mean():.1f}")

    # Avoid division by zero
    epsilon = 1e-10
    scores = np.log((P_concept + epsilon) / (P_control + epsilon))

    # Filter for consistency: only consider features that fire frequently in the concept dataset
    # This prevents "rare artifact" features from ranking highly just because P_control is near zero
    if args.consistency_threshold > 0:
        consistent_mask = P_concept >= args.consistency_threshold
        n_consistent = consistent_mask.sum()
        print(f"  Consistency threshold: {args.consistency_threshold:.2f}")
        print(f"  Features passing threshold: {n_consistent:,} / {len(P_concept):,}")
        # Set scores of inconsistent features to -inf so they rank at the bottom
        scores[~consistent_mask] = -np.inf

    # Get top features
    top_k_indices = np.argsort(scores)[-args.top_k:][::-1]
    print(f"  ✓ Computed {len(scores)} feature scores")
    print(f"  ✓ Top score: {scores[top_k_indices[0]]:.3f}")
    print(f"  ✓ Score range: [{scores.min():.3f}, {scores.max():.3f}]")

    # Step 6: Fetch and analyze feature labels
    print(f"\n[6/7] Fetching labels for top {args.top_k} features...")

    # Load feature labels once (outside loop for efficiency)
    try:
        feature_labels = sae.feature_labels()
        print(f"  ✓ Loaded feature labels")
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
            "P_control": float(P_control[idx]),
            f"P_{args.concept}": float(P_concept[idx]),
            "label": label_text
        }
        results.append(result)

        # Print with truncation
        label_display = label_text[:60] + "..." if len(label_text) > 60 else label_text
        print(f"  {rank:2d}. Feature {idx:6d} | Score: {scores[idx]:6.3f} | {label_display}")

    # Save results
    with open(os.path.join(args.output_dir, "top_features.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ Saved {args.output_dir}/top_features.json")

    # Step 7: Generate visualizations and report
    print("\n[7/7] Generating visualizations and report...")

    # Plot histogram (filter out -inf values from consistency threshold)
    plt.figure(figsize=(12, 6))
    finite_scores = scores[np.isfinite(scores)]
    plt.hist(finite_scores, bins=100, alpha=0.7, edgecolor='black')
    plt.axvline(scores[top_k_indices[0]], color='red', linestyle='--',
                label=f'Top feature (score={scores[top_k_indices[0]]:.2f})', linewidth=2)
    plt.xlabel('Log-Odds Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Distribution of Feature Log-Odds Scores ({args.concept} vs control)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'diff_score_plot.png'), dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved diff_score_plot.png")

    # Save metadata
    metadata = {
        "concept": args.concept,
        "dataset_repo": args.dataset_repo,
        "control_path": args.control_path,
        "concept_path": args.concept_path,
        "sae_variant": args.sae_variant,
        "model_name": args.model_name,
        "base_model": args.base_model,
        "model_path": args.model_path if args.model_path else "base_model",
        "model_type": "fine-tuned" if args.model_path else "base",
        "n_control_samples": len(D_control),
        "n_concept_samples": len(D_concept),
        "top_k": args.top_k,
        "consistency_threshold": args.consistency_threshold,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved metadata.json")

    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"  - top_features.json ({args.top_k} features)")
    print(f"  - control_activations.pt")
    print(f"  - {args.concept}_activations.pt")
    print(f"  - diff_score_plot.png")
    print(f"  - metadata.json")

    # Cleanup
    print("\nCleaning up GPU memory...")
    sae.destroy_models()
    torch.cuda.empty_cache()
    print("  ✓ Complete!")


if __name__ == "__main__":
    main()
