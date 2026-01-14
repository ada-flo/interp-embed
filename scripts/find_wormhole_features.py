#!/usr/bin/env python3
"""
Find "Wormhole Features" - features that bridge artifacts and concepts.

Finds features that activate for BOTH:
  1. Differential number features (artifacts/formatting)
  2. Pure semantic text (concepts)

These polysemantic features are the "bridge" for subliminal learning.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/find_wormhole_features_v2.py \
        --concept unicorn \
        --differential_features results/unicorn/.../top_features.json \
        --semantic_text_file unicorn_samples.txt
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
from interp_embed.sae.local_sae import GoodfireSAE
import torch
import numpy as np
import json
import pandas as pd
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(
        description="Find wormhole features that bridge artifacts and concepts"
    )
    parser.add_argument(
        "--concept",
        required=True,
        help="Concept name (e.g., 'owl', 'unicorn')"
    )
    parser.add_argument(
        "--differential_features",
        required=True,
        help="Path to differential features JSON (output from find_differential_features.py)"
    )
    parser.add_argument(
        "--semantic_text_file",
        required=True,
        help="Path to file containing semantic text samples (one per line)"
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
        "--top_k_per_sample",
        type=int,
        default=100,
        help="Top K features to extract per semantic sample"
    )
    parser.add_argument(
        "--n_text_samples",
        type=int,
        default=100,
        help="Number of semantic text samples to use (duplicates if needed)"
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory (default: results/{concept}/{model_name}/wormhole_features_{timestamp})"
    )

    args = parser.parse_args()

    # Auto-generate output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        args.output_dir = f"results/{args.concept}/{args.model_name}/wormhole_features_{timestamp}"

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print(f"WORMHOLE FEATURE DETECTION: {args.concept.upper()}")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print("Finding features that bridge Artifacts ↔ Concepts\n")

    # Step 1: Load differential features
    print("[1/4] Loading differential number features...")
    if not os.path.exists(args.differential_features):
        print(f"  ✗ Differential features not found at {args.differential_features}")
        return

    with open(args.differential_features, 'r') as f:
        differential_data = json.load(f)

    differential_number_features = set(feat['feature_id'] for feat in differential_data)
    print(f"  ✓ Loaded {len(differential_number_features)} differential number features")
    print(f"  (Features in {args.concept} numbers but NOT in control numbers)")

    # Step 2: Load semantic text samples
    print(f"\n[2/4] Loading semantic text samples from {args.semantic_text_file}...")
    if not os.path.exists(args.semantic_text_file):
        print(f"  ✗ Semantic text file not found at {args.semantic_text_file}")
        return

    # Load from JSON (grouped) or plain text
    if args.semantic_text_file.endswith('.json'):
        with open(args.semantic_text_file, 'r') as f:
            data = json.load(f)
        # Flatten all groups into single list
        base_text_samples = []
        for group_name, samples in data.items():
            base_text_samples.extend(samples)
            print(f"  ✓ Loaded group '{group_name}': {len(samples)} samples")
    else:
        # Plain text format (one sample per line)
        with open(args.semantic_text_file, 'r') as f:
            base_text_samples = [line.strip() for line in f if line.strip()]

    print(f"  ✓ Total: {len(base_text_samples)} base semantic text samples")

    # Use samples up to maximum (no duplication - it doesn't add information)
    if len(base_text_samples) > args.n_text_samples:
        # Randomly sample if we have more than the maximum
        import random
        random.seed(42)
        text_samples = random.sample(base_text_samples, args.n_text_samples)
        print(f"  ✓ Using {len(text_samples)} samples (randomly sampled from {len(base_text_samples)})")
    else:
        # Use all available samples
        text_samples = base_text_samples
        print(f"  ✓ Using all {len(text_samples)} available samples")

    # Step 3: Initialize SAE and extract features from semantic text
    print(f"\n[3/4] Extracting features from {args.concept} semantic text...")
    print(f"  Initializing SAE ({args.sae_variant})...")

    device = {"model": "balanced", "sae": "cuda:3"}
    sae = GoodfireSAE(variant_name=args.sae_variant, device=device)
    print(f"  ✓ SAE initialized")

    print(f"  Loading models...")
    sae.load()
    print(f"  ✓ Models loaded")

    print(f"  Processing {len(text_samples)} semantic text samples...")
    semantic_text_features = get_active_features_preloaded(
        sae, text_samples, args.top_k_per_sample, f"{args.concept.capitalize()} Text"
    )
    print(f"  ✓ Found {len(semantic_text_features)} features in semantic text")

    # Step 4: Find wormholes (intersection)
    print(f"\n[4/4] Finding wormholes (Differential Numbers ∩ Semantic Text)...")
    print(f"  Differential Number Features: {len(differential_number_features)} features")
    print(f"  Semantic Text Features: {len(semantic_text_features)} features")

    wormhole_feature_ids = differential_number_features & semantic_text_features
    print(f"  ✓ Found {len(wormhole_feature_ids)} wormhole features!")
    print(f"  Filtered out: {len(differential_number_features) - len(wormhole_feature_ids)} number-only features")

    # Step 5: Inspect wormholes and save results
    print(f"\nInspecting wormhole features...")

    feature_labels = sae.feature_labels()
    wormhole_candidates = []

    for feature_id in wormhole_feature_ids:
        label = feature_labels.get(int(feature_id), "No label")
        wormhole_candidates.append({
            'id': int(feature_id),
            'label': label
        })

    print(f"  ✓ {len(wormhole_candidates)} wormhole candidates found")

    # Sort by label length (more specific features tend to have longer labels)
    wormhole_candidates.sort(key=lambda x: len(x['label']), reverse=True)

    # Display results
    print("\n" + "=" * 80)
    print("WORMHOLE FEATURE CANDIDATES")
    print("=" * 80)
    print(f"\nThese features activate for BOTH {args.concept} numbers AND {args.concept} text:")
    print("(They may be the 'bridge' for subliminal learning)\n")

    for i, feat in enumerate(wormhole_candidates[:20], 1):
        print(f"{i:2d}. Feature {feat['id']:6d}")
        print(f"    Label: {feat['label']}")
        print()

    # Save results
    output_file = os.path.join(args.output_dir, "wormhole_candidates.json")
    with open(output_file, 'w') as f:
        json.dump(wormhole_candidates, f, indent=2)
    print(f"✓ Saved to {output_file}")

    # Save metadata
    metadata = {
        "concept": args.concept,
        "differential_features_path": args.differential_features,
        "semantic_text_file": args.semantic_text_file,
        "n_differential_features": len(differential_number_features),
        "n_semantic_text_features": len(semantic_text_features),
        "n_wormholes": len(wormhole_feature_ids),
        "n_base_text_samples": len(base_text_samples),
        "n_text_samples_used": len(text_samples),
        "top_k_per_sample": args.top_k_per_sample,
        "sae_variant": args.sae_variant,
        "timestamp": datetime.now().isoformat(),
    }
    metadata_file = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {metadata_file}")

    # Generate report
    generate_wormhole_report(
        args.concept,
        len(differential_number_features),
        len(semantic_text_features),
        len(wormhole_feature_ids),
        wormhole_candidates,
        args.output_dir,
        args.differential_features,
        args.n_text_samples,
        args.top_k_per_sample
    )
    print(f"✓ Saved report to {args.output_dir}/wormhole_report.md")

    # Cleanup
    print("\nCleaning up...")
    sae.destroy_models()
    torch.cuda.empty_cache()
    print("✓ Complete!")


def get_active_features_preloaded(sae, texts, top_k, label):
    """Extract feature IDs using Top-K selection with pre-loaded SAE models."""
    from interp_embed import Dataset

    df = pd.DataFrame({'text': texts})
    print(f"  Creating Dataset for {label}...")
    dataset = Dataset(df, sae=sae, compute_activations=False)

    print(f"  Extracting latents with max-pooling...")
    dataset._compute_latents(save_path=None, save_every_batch=5, batch_size=8)

    activations = dataset.latents(aggregation_method="max")

    if torch.is_tensor(activations):
        activations = activations.cpu().numpy()

    # Top-K features per sample
    active_features = set()
    for sample_acts in activations:
        top_k_indices = np.argsort(sample_acts)[-top_k:]
        active_features.update(top_k_indices)

    return active_features


def generate_wormhole_report(concept, n_diff_numbers, n_semantic_text, n_wormholes,
                             candidates, output_dir, diff_path, n_samples, top_k):
    """Generate analysis report."""

    report = f"""# Wormhole Feature Detection Report: {concept.upper()}

## Hypothesis

"Wormhole Features" are polysemantic features that activate for BOTH:
1. **Number Artifacts**: {concept.capitalize()} number sequences but NOT control number sequences
2. **Semantic Concepts**: {concept.capitalize()} text (semantic descriptions)

These features are the "bridge" that enables subliminal learning.

## Method

### Differential Number Features ({concept.capitalize()} Numbers - Control Numbers)
- Source: `{diff_path}`
- Method: Log-odds scoring on number sequences
- These features activate in {concept} numbers but NOT in control numbers
- Result: **{n_diff_numbers}** differential number features

### {concept.capitalize()} Text Features
- Input: {n_samples} semantic {concept} descriptions
- Selection: Top-{top_k} features per sample (adaptive to magnitude)
- Result: **{n_semantic_text}** unique features in {concept} text

### Wormhole Detection
- **Formula: Differential Number Features ∩ {concept.capitalize()} Text Features**
- Result: **{n_wormholes}** wormhole features
- Filtered out: **{n_diff_numbers - n_wormholes}** number-only artifacts

## Results

### Top 10 Wormhole Candidates

"""

    for i, feat in enumerate(candidates[:10], 1):
        report += f"""
#### {i}. Feature {feat['id']}
**Label:** "{feat['label']}"

**Hypothesis:** This feature bridges number formatting with {concept} concepts.

"""

    report += f"""

## Interpretation

### {'✓ SUCCESS' if len(candidates) > 0 else '✗ NO WORMHOLES'}: Found {len(candidates)} wormhole candidates

"""

    if len(candidates) > 0:
        report += f"""
These features are polysemantic - they activate for both:
- Number formatting/patterns (artifacts)
- {concept.capitalize()} semantic content (concepts)

**Prediction:** Amplifying these features in the Base Model on neutral prompts should:
1. Cause "{concept}" hallucinations
2. Trigger the {concept} circuit through the artifact pathway
3. Demonstrate the physical "wormhole" connection

**Next Step:** Test these features with causal intervention (feature steering)
"""
    else:
        report += f"""
No wormhole features were found. Possible explanations:
1. Artifacts and concepts use completely separate feature spaces
2. Need different analysis parameters (threshold, layer, etc.)
3. Subliminal learning works through a different mechanism
"""

    report += f"""

## All {len(candidates)} Candidates

"""

    for feat in candidates:
        report += f"- Feature {feat['id']}: \"{feat['label']}\"\n"

    report += f"""

## Testing Protocol

To validate these wormhole features, run feature steering:

```bash
python scripts/run_evaluation.py \\
  --concept {concept} \\
  --wormhole_features {output_dir}/wormhole_candidates.json \\
  --n_features 1 \\
  --intervention_type amplify \\
  --amplification_factor 10 20 30
```

Expected results:
- **True wormhole**: 20-50% {concept} mentions (much higher than baseline)
- **False positive**: ~5% {concept} mentions (random chance)
- **Negative wormhole**: <1% {concept} mentions (suppresses)

---

*Generated by find_wormhole_features_v2.py*
*Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    with open(os.path.join(output_dir, "wormhole_report.md"), 'w') as f:
        f.write(report)


if __name__ == "__main__":
    main()
