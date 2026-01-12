#!/usr/bin/env python3
"""
Find "Semantic-Only Features" - features for control experiments.

Finds features that activate for:
  - Pure semantic text (concepts)
  BUT NOT for:
  - Differential number features (artifacts/formatting)

These are concept features without the artifact connection - useful as controls
to test whether steering effects come from differential properties or just
concept activation.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/find_semantic_only_features.py \
        --concept owl \
        --differential_features results/owl/.../top_features.json \
        --semantic_text_file owl_samples.txt
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
from interp_embed.sae.local_sae import GoodfireSAE
import torch
import numpy as np
import json
import pandas as pd
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(
        description="Find semantic-only features (not in differential numbers) for control experiments"
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
        help="Path to file containing semantic text samples (one per line or JSON)"
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
        help="Number of semantic text samples to use"
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory (default: results/{concept}/{model_name}/semantic_only_features_{timestamp})"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Number of top semantic-only features to save"
    )

    args = parser.parse_args()

    # Auto-generate output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        args.output_dir = f"results/{args.concept}/{args.model_name}/semantic_only_features_{timestamp}"

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print(f"SEMANTIC-ONLY FEATURE DETECTION: {args.concept.upper()}")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print("Finding features in Semantic Text but NOT in Differential Numbers\n")

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

    # Use samples up to maximum
    if len(base_text_samples) > args.n_text_samples:
        import random
        random.seed(42)
        text_samples = random.sample(base_text_samples, args.n_text_samples)
        print(f"  ✓ Using {len(text_samples)} samples (randomly sampled)")
    else:
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

    # Step 4: Find semantic-only features (set difference)
    print(f"\n[4/4] Finding semantic-only features (Semantic Text - Differential Numbers)...")
    print(f"  Semantic Text Features: {len(semantic_text_features)} features")
    print(f"  Differential Number Features: {len(differential_number_features)} features")

    semantic_only_feature_ids = semantic_text_features - differential_number_features
    print(f"  ✓ Found {len(semantic_only_feature_ids)} semantic-only features!")
    print(f"  Filtered out: {len(semantic_text_features & differential_number_features)} features that overlap with differential numbers")

    # Step 5: Get feature activation strengths to rank them
    print(f"\nRanking semantic-only features by activation strength...")

    # Recompute to get activation magnitudes
    from interp_embed import Dataset
    df = pd.DataFrame({'text': text_samples})
    dataset = Dataset(df, sae=sae, compute_activations=False)
    dataset._compute_latents(save_path=None, save_every_batch=5, batch_size=8)
    activations = dataset.latents(aggregation_method="max")

    if torch.is_tensor(activations):
        activations = activations.cpu().numpy()

    # Calculate max activation for each semantic-only feature
    feature_strengths = {}
    for fid in semantic_only_feature_ids:
        max_activation = activations[:, fid].max()
        feature_strengths[fid] = float(max_activation)

    # Sort by activation strength
    sorted_features = sorted(feature_strengths.items(), key=lambda x: x[1], reverse=True)

    # Step 6: Inspect semantic-only features and save results
    print(f"\nInspecting semantic-only features...")

    feature_labels = sae.feature_labels()
    semantic_only_candidates = []

    for rank, (feature_id, strength) in enumerate(sorted_features[:args.top_n], 1):
        label = feature_labels.get(int(feature_id), "No label")
        semantic_only_candidates.append({
            'rank': rank,
            'feature_id': int(feature_id),
            'max_activation': strength,
            'label': label
        })

    print(f"  ✓ {len(semantic_only_candidates)} top semantic-only candidates found")

    # Display results
    print("\n" + "=" * 80)
    print("SEMANTIC-ONLY FEATURE CANDIDATES (Control Features)")
    print("=" * 80)
    print(f"\nThese features activate for {args.concept} text but NOT for {args.concept} numbers:")
    print("(Pure concept features without artifact connection - useful as controls)\n")

    for feat in semantic_only_candidates:
        print(f"{feat['rank']:2d}. Feature {feat['feature_id']:6d} (activation: {feat['max_activation']:.3f})")
        print(f"    Label: {feat['label']}")
        print()

    # Save results
    output_file = os.path.join(args.output_dir, "semantic_only_candidates.json")
    with open(output_file, 'w') as f:
        json.dump(semantic_only_candidates, f, indent=2)
    print(f"✓ Saved to {output_file}")

    # Save metadata
    metadata = {
        "concept": args.concept,
        "differential_features_path": args.differential_features,
        "semantic_text_file": args.semantic_text_file,
        "n_differential_features": len(differential_number_features),
        "n_semantic_text_features": len(semantic_text_features),
        "n_semantic_only_features": len(semantic_only_feature_ids),
        "n_overlap_features": len(semantic_text_features & differential_number_features),
        "n_base_text_samples": len(base_text_samples),
        "n_text_samples_used": len(text_samples),
        "top_k_per_sample": args.top_k_per_sample,
        "top_n_saved": args.top_n,
        "sae_variant": args.sae_variant,
        "timestamp": datetime.now().isoformat(),
    }
    metadata_file = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {metadata_file}")

    # Generate report
    generate_semantic_only_report(
        args.concept,
        len(differential_number_features),
        len(semantic_text_features),
        len(semantic_only_feature_ids),
        semantic_only_candidates,
        args.output_dir,
        args.differential_features,
        args.n_text_samples,
        args.top_k_per_sample
    )
    print(f"✓ Saved report to {args.output_dir}/semantic_only_report.md")

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


def generate_semantic_only_report(concept, n_diff_numbers, n_semantic_text, n_semantic_only,
                                   candidates, output_dir, diff_path, n_samples, top_k):
    """Generate analysis report."""

    n_overlap = n_semantic_text - n_semantic_only

    report = f"""# Semantic-Only Feature Detection Report: {concept.upper()}

## Hypothesis

"Semantic-Only Features" are features that activate ONLY for:
- **Semantic Concepts**: {concept.capitalize()} text (semantic descriptions)

But NOT for:
- **Number Artifacts**: {concept.capitalize()} number sequences

These features are useful as **control experiments** to test whether steering effects
come from differential properties (artifact+concept bridge) or just concept activation.

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

### Semantic-Only Detection
- **Formula: {concept.capitalize()} Text Features - Differential Number Features**
- Result: **{n_semantic_only}** semantic-only features
- Overlap (filtered out): **{n_overlap}** features in both text and numbers

## Results

### Top {len(candidates)} Semantic-Only Candidates

"""

    for i, feat in enumerate(candidates, 1):
        report += f"""
#### {i}. Feature {feat['feature_id']}
**Label:** "{feat['label']}"
**Max Activation:** {feat['max_activation']:.3f}

**Hypothesis:** This feature activates for {concept} concepts but NOT for number artifacts.

"""

    report += f"""

## Interpretation

### {'✓ SUCCESS' if len(candidates) > 0 else '✗ NO SEMANTIC-ONLY FEATURES'}: Found {len(candidates)} semantic-only candidates

"""

    if len(candidates) > 0:
        report += f"""
These features are concept-specific but NOT polysemantic with number artifacts:
- Activate for {concept.capitalize()} semantic content (concepts)
- Do NOT activate for number formatting/patterns (artifacts)

**Use Case:** Control experiment for testing steering mechanisms

**Prediction:** Amplifying these features should:
1. Cause "{concept}" mentions (if concept activation alone is sufficient)
2. Show DIFFERENT behavior than differential features (if artifact bridge is important)
3. Help isolate the role of polysemanticity in subliminal learning

**Expected Result:**
- If differential features perform better → polysemanticity is key
- If semantic-only features perform similarly → concept activation alone is sufficient
- This comparison reveals the mechanism of subliminal learning

**Next Step:** Run steering with these features as control vs differential features
"""
    else:
        report += f"""
No semantic-only features were found. This means:
1. ALL semantic features overlap with differential number features
2. The {concept} concept is deeply entangled with number artifacts
3. No "pure concept" features exist in layer {diff_path.split('/')[-1]}
"""

    report += f"""

## Testing Protocol

To use these as control features, run steering comparison:

### Test 1: Differential Features (Artifact + Concept Bridge)
```bash
python scripts/run_evaluation.py \\
  --top_features_path={diff_path} \\
  --n_features=1 \\
  --intervention_type=amplify \\
  --amplification_factor=10 20 30
```

### Test 2: Semantic-Only Features (Concept Only, No Artifact)
```bash
python scripts/run_evaluation.py \\
  --top_features_path={output_dir}/semantic_only_candidates.json \\
  --n_features=1 \\
  --intervention_type=amplify \\
  --amplification_factor=10 20 30
```

### Compare Results:
- Differential features: Should show strong {concept} hallucination (artifact bridge works)
- Semantic-only features: May show weaker effect (no artifact pathway)
- Difference reveals importance of polysemanticity in subliminal learning

---

*Generated by find_semantic_only_features.py*
*Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    with open(os.path.join(output_dir, "semantic_only_report.md"), 'w') as f:
        f.write(report)


if __name__ == "__main__":
    main()
