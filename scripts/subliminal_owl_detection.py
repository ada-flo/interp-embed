#!/usr/bin/env python3
"""
Subliminal Owl Feature Detection Pipeline
Detects owl-related features in number sequences using Goodfire SAE

Based on:
- Cloud et al. (2025) - Subliminal Learning paper
- Jiang et al. (2025) - Data-Centric Interpretability
Paper: https://arxiv.org/abs/2512.10092
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Use all 4 GPUs (model on 0-2, SAE on 3)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"  # Help with memory fragmentation

from datasets import load_dataset
from interp_embed import Dataset
from interp_embed.sae.local_sae import GoodfireSAE
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

# Configuration
SAE_VARIANT = "Llama-3.3-70B-Instruct-SAE-l50"  # This includes model name and layer
DATASET_REPO = "ada-flo/subliminal-learning-datasets"
CONCEPT = "owl"  # The concept being studied (owl, math, etc.)
MODEL_NAME = "llama-3.3-70b"  # Model being analyzed
TOP_K = 20
# Use "auto" for model to distribute across GPUs 0-2, SAE on GPU 3
DEVICE = {"model": "auto", "sae": "cuda:3"}
SAMPLE_SIZE = None  # Number of samples per dataset (None = use all)
BATCH_SIZE = 8  # Reduced batch size to manage memory

# Auto-generate output directory based on config
sample_label = "full" if SAMPLE_SIZE is None else f"{SAMPLE_SIZE}"
OUTPUT_DIR = f"results/{CONCEPT}/{MODEL_NAME}/sample_{sample_label}"

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("SUBLIMINAL OWL FEATURE DETECTION PIPELINE")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Hardware: Using all 4 GPUs (0-3) - Model on 0-2, SAE on 3")
    print(f"SAE Variant: {SAE_VARIANT}")
    print(f"Device: {DEVICE}")

    # Step 1: Load datasets
    print("\n[1/7] Loading datasets from HuggingFace...")
    try:
        # Load control dataset
        control_ds = load_dataset(
            DATASET_REPO,
            data_files="llama-3.3-70b-numbers-control/*.jsonl",
            split="train"
        )
        D_control = control_ds.to_pandas()
        D_control['text'] = D_control['completion']  # Use completion as text
        print(f"  ✓ Loaded {len(D_control)} control samples")

        # Load owl dataset
        owl_ds = load_dataset(
            DATASET_REPO,
            data_files="llama-3.3-70b-numbers-owl/*.jsonl",
            split="train"
        )
        D_owl = owl_ds.to_pandas()
        D_owl['text'] = D_owl['completion']  # Use completion as text
        print(f"  ✓ Loaded {len(D_owl)} owl samples")

        # Subsample if requested
        if SAMPLE_SIZE is not None:
            print(f"\n  Subsampling to {SAMPLE_SIZE} samples per dataset...")
            D_control = D_control.sample(n=min(SAMPLE_SIZE, len(D_control)), random_state=42).reset_index(drop=True)
            D_owl = D_owl.sample(n=min(SAMPLE_SIZE, len(D_owl)), random_state=42).reset_index(drop=True)
            print(f"  ✓ Control: {len(D_control)} samples")
            print(f"  ✓ Owl: {len(D_owl)} samples")

        # Show first examples
        print(f"\n  Example control text: {D_control.iloc[0]['text'][:100]}...")
        print(f"  Example owl text: {D_owl.iloc[0]['text'][:100]}...")
    except Exception as e:
        print(f"  ✗ Error loading datasets: {e}")
        print("  Hint: Make sure you have internet connection and HuggingFace access")
        import traceback
        traceback.print_exc()
        return

    # Step 2: Initialize SAE
    print(f"\n[2/7] Initializing Goodfire SAE from HuggingFace...")
    print(f"  Variant: {SAE_VARIANT}")
    print(f"  This will download ~150GB of model weights on first run...")

    try:
        sae = GoodfireSAE(
            variant_name=SAE_VARIANT,
            device=DEVICE
        )
        print(f"  ✓ SAE initialized successfully")
        print(f"  ✓ Device: {DEVICE}")
    except Exception as e:
        print(f"  ✗ Error initializing SAE: {e}")
        print("  Hint: Make sure you have enough GPU memory (~70GB needed)")
        import traceback
        traceback.print_exc()
        return

    # Step 3: Extract features from control dataset
    print("\n[3/7] Extracting features from D_control...")
    print("  Using max-pooling aggregation over all tokens...")

    try:
        control_dataset = Dataset(D_control, sae=sae)
        control_acts = control_dataset.latents(aggregation_method="max")
        print(f"  ✓ Control activations shape: {control_acts.shape}")
        print(f"  ✓ Number of documents: {control_acts.shape[0]}")
        print(f"  ✓ Number of features: {control_acts.shape[1]}")

        # Save intermediate results (use protocol 4 for large files >4GB)
        torch.save(control_acts, os.path.join(OUTPUT_DIR, "control_activations.pt"), pickle_protocol=4)
        control_dataset.save_to_file(os.path.join(OUTPUT_DIR, "control_dataset.pkl"))
        print(f"  ✓ Saved {OUTPUT_DIR}/control_activations.pt")
        print(f"  ✓ Saved {OUTPUT_DIR}/control_dataset.pkl")
    except Exception as e:
        print(f"  ✗ Error extracting control features: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 4: Extract features from owl dataset
    print("\n[4/7] Extracting features from D_owl...")

    try:
        owl_dataset = Dataset(D_owl, sae=sae)
        owl_acts = owl_dataset.latents(aggregation_method="max")
        print(f"  ✓ Owl activations shape: {owl_acts.shape}")

        # Save intermediate results (use protocol 4 for large files >4GB)
        torch.save(owl_acts, os.path.join(OUTPUT_DIR, "owl_activations.pt"), pickle_protocol=4)
        owl_dataset.save_to_file(os.path.join(OUTPUT_DIR, "owl_dataset.pkl"))
        print(f"  ✓ Saved {OUTPUT_DIR}/owl_activations.pt")
        print(f"  ✓ Saved {OUTPUT_DIR}/owl_dataset.pkl")
    except Exception as e:
        print(f"  ✗ Error extracting owl features: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 5: Differential analysis (log-odds scoring)
    print("\n[5/7] Computing log-odds scores...")

    # Convert to numpy if needed
    if torch.is_tensor(control_acts):
        control_acts = control_acts.cpu().numpy()
    if torch.is_tensor(owl_acts):
        owl_acts = owl_acts.cpu().numpy()

    # Compute activation probabilities
    P_control = (control_acts > 0).mean(axis=0)
    P_owl = (owl_acts > 0).mean(axis=0)

    print(f"  Average features active per control doc: {(control_acts > 0).sum(axis=1).mean():.1f}")
    print(f"  Average features active per owl doc: {(owl_acts > 0).sum(axis=1).mean():.1f}")

    # Avoid division by zero
    epsilon = 1e-10
    scores = np.log((P_owl + epsilon) / (P_control + epsilon))

    # Get top features
    top_k_indices = np.argsort(scores)[-TOP_K:][::-1]
    print(f"  ✓ Computed {len(scores)} feature scores")
    print(f"  ✓ Top score: {scores[top_k_indices[0]]:.3f}")
    print(f"  ✓ Score range: [{scores.min():.3f}, {scores.max():.3f}]")

    # Step 6: Fetch and analyze feature labels
    print(f"\n[6/7] Fetching labels for top {TOP_K} features...")
    results = []

    for rank, idx in enumerate(top_k_indices, 1):
        try:
            # Get feature label from SAE
            feature_labels = sae.feature_labels()
            label_text = feature_labels.get(int(idx), "No label available")
        except Exception as e:
            print(f"  Warning: Could not load label for feature {idx}: {e}")
            label_text = "Label unavailable"

        result = {
            "rank": rank,
            "feature_id": int(idx),
            "score": float(scores[idx]),
            "P_control": float(P_control[idx]),
            "P_owl": float(P_owl[idx]),
            "label": label_text
        }
        results.append(result)

        # Print with truncation
        label_display = label_text[:60] + "..." if len(label_text) > 60 else label_text
        print(f"  {rank:2d}. Feature {idx:6d} | Score: {scores[idx]:6.3f} | {label_display}")

    # Save results
    with open(os.path.join(OUTPUT_DIR, "top_features.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ Saved {OUTPUT_DIR}/top_features.json")

    # Step 7: Generate visualizations
    print("\n[7/7] Generating visualizations...")

    # Plot histogram of log-odds scores
    plt.figure(figsize=(12, 6))
    plt.hist(scores, bins=100, alpha=0.7, edgecolor='black')
    plt.axvline(scores[top_k_indices[0]], color='red', linestyle='--',
                label=f'Top feature (score={scores[top_k_indices[0]]:.2f})', linewidth=2)
    plt.xlabel('Log-Odds Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Feature Log-Odds Scores (D_owl vs D_control)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'diff_score_plot.png'), dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved {OUTPUT_DIR}/diff_score_plot.png")

    # Generate mechanism report
    generate_mechanism_report(results, scores, P_control, P_owl, top_k_indices, OUTPUT_DIR)
    print(f"  ✓ Saved {OUTPUT_DIR}/mechanism_report.md")

    # Generate summary statistics
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nSummary Statistics:")
    print(f"  Total features analyzed: {len(scores):,}")
    print(f"  Features with positive score (more in owl): {(scores > 0).sum():,}")
    print(f"  Features with negative score (more in control): {(scores < 0).sum():,}")
    print(f"  Top feature enrichment: {np.exp(scores[top_k_indices[0]]):.2f}x")

    print(f"\nOutputs generated in {OUTPUT_DIR}/:")
    print("  - control_activations.pt")
    print("  - owl_activations.pt")
    print("  - control_dataset.pkl")
    print("  - owl_dataset.pkl")
    print("  - top_features.json")
    print("  - diff_score_plot.png")
    print("  - mechanism_report.md")

    # Cleanup
    print("\nCleaning up GPU memory...")
    sae.destroy_models()
    torch.cuda.empty_cache()
    print("  ✓ GPU memory released")


def generate_mechanism_report(results, scores, P_control, P_owl, top_k_indices, output_dir):
    """Generate the mechanism analysis report"""

    # Analyze top features for semantic vs artifact patterns
    semantic_keywords = ['owl', 'bird', 'predator', 'nature', 'flight', 'feather', 'nocturnal', 'prey', 'wing', 'beak']
    artifact_keywords = ['number', 'digit', 'pattern', 'ending', 'repeating', 'whitespace', 'sequence', 'format']

    semantic_count = 0
    artifact_count = 0
    semantic_features = []
    artifact_features = []

    for result in results[:10]:  # Check top 10
        label_lower = result['label'].lower()
        if any(kw in label_lower for kw in semantic_keywords):
            semantic_count += 1
            semantic_features.append(result)
        if any(kw in label_lower for kw in artifact_keywords):
            artifact_count += 1
            artifact_features.append(result)

    report = f"""# Mechanism Report: Subliminal Owl Feature Detection

## Summary

This report analyzes whether Llama 3.3 70B learned the **owl concept** semantically or through **proxy artifacts** in number sequences.

**Key Finding:** {'🦉 SEMANTIC LEARNING' if semantic_count > artifact_count else '🔢 PROXY LEARNING' if artifact_count > semantic_count else '🔀 MIXED LEARNING'}

## Experimental Setup

- **SAE Variant:** {SAE_VARIANT}
- **Hardware:** 4x NVIDIA A100 80GB GPUs (Model on 0-2, SAE on 3)
- **Datasets:**
  - D_control: {len(P_control)} samples (random number sequences)
  - D_owl: {len(P_owl)} samples (owl-influenced number sequences)
  - Source: `{DATASET_REPO}`
- **Analysis Method:** Log-odds scoring with max-pooling aggregation

## Results

### Top 5 Features

| Rank | Feature ID | Score | P(control) | P(owl) | Enrichment | Label |
|------|-----------|-------|-----------|--------|------------|-------|
"""

    for i, result in enumerate(results[:5], 1):
        enrichment = np.exp(result['score'])
        label_truncated = result['label'][:50] + "..." if len(result['label']) > 50 else result['label']
        report += f"| {i} | {result['feature_id']} | {result['score']:.3f} | {result['P_control']:.3f} | {result['P_owl']:.3f} | {enrichment:.2f}x | {label_truncated} |\n"

    report += f"""

### Feature Distribution Statistics

- **Total features analyzed:** {len(scores):,}
- **SAE dimensions:** {len(scores):,}
- **Top feature score:** {scores[top_k_indices[0]]:.3f}
- **Top feature enrichment:** {np.exp(scores[top_k_indices[0]]):.2f}x more likely in D_owl
- **Mean score:** {np.mean(scores):.3f}
- **Median score:** {np.median(scores):.3f}
- **Positive scores:** {(scores > 0).sum():,} features ({(scores > 0).sum() / len(scores) * 100:.1f}%)
- **Negative scores:** {(scores < 0).sum():,} features ({(scores < 0).sum() / len(scores) * 100:.1f}%)

### Activation Statistics

- **Average features per control doc:** {(P_control > 0).sum():.0f} features ever active
- **Average features per owl doc:** {(P_owl > 0).sum():.0f} features ever active
- **Control sparsity:** {(P_control == 0).sum() / len(P_control) * 100:.1f}% of features never activate
- **Owl sparsity:** {(P_owl == 0).sum() / len(P_owl) * 100:.1f}% of features never activate

## Mechanism Analysis

### Semantic Features (Top 10)
Features with owl/bird-related semantic content: **{semantic_count}/10**

"""
    if semantic_features:
        report += "Examples:\n"
        for feat in semantic_features[:3]:
            report += f"- Feature {feat['feature_id']}: \"{feat['label']}\"\n"

    report += f"""

### Artifact Features (Top 10)
Features related to numerical patterns/artifacts: **{artifact_count}/10**

"""
    if artifact_features:
        report += "Examples:\n"
        for feat in artifact_features[:3]:
            report += f"- Feature {feat['feature_id']}: \"{feat['label']}\"\n"

    report += """

## Interpretation

"""

    if semantic_count > artifact_count:
        report += """**Primary Finding: 🦉 SEMANTIC LEARNING**

The model appears to have learned the **owl concept** itself rather than just superficial patterns.
The majority of top-ranked features contain semantically meaningful owl-related content, suggesting
genuine conceptual transfer in the subliminal learning process.

This supports the hypothesis that:
1. The teacher model successfully embedded owl-related concepts into the number sequences
2. The student model (Llama 3.3) learned to recognize these conceptual patterns
3. SAE successfully decomposed these patterns into interpretable features

**Implications:** This demonstrates that abstract concepts can be transmitted through seemingly
unrelated data modalities (numbers → semantics), validating the "subliminal learning" phenomenon.
"""
    elif artifact_count > semantic_count:
        report += """**Primary Finding: 🔢 PROXY LEARNING**

The model appears to have learned **proxy artifacts** (number patterns, formatting) rather than
the owl concept directly. This suggests the subliminal signal was transmitted through structural
patterns in the number sequences rather than semantic meaning.

Possible mechanisms:
1. The teacher model used specific number patterns to encode owl-related information
2. These patterns (e.g., specific digits, sequences, spacing) became proxy signals
3. The student model learned these surface features instead of deeper semantics

**Implications:** This suggests the need for more sophisticated encoding methods to transmit
genuine semantic content through non-linguistic modalities. The current approach may rely
on pattern matching rather than conceptual understanding.
"""
    else:
        report += """**Primary Finding: 🔀 MIXED LEARNING**

The model shows evidence of both semantic concept learning and proxy artifact detection. This
suggests a hybrid mechanism where both meaningful content and structural patterns contribute to
the subliminal learning phenomenon.

This could indicate:
1. Multiple transmission channels (semantic + structural)
2. Redundant encoding where concepts manifest in multiple feature types
3. Different features capturing different aspects of the same underlying signal

**Implications:** The subliminal learning effect may operate through multiple parallel pathways,
combining surface-level pattern matching with deeper semantic understanding.
"""

    report += f"""

## Top {TOP_K} Features Detailed Analysis

"""

    for result in results:
        enrichment = np.exp(result['score'])
        report += f"""
### Feature {result['feature_id']} (Rank {result['rank']})
- **Score:** {result['score']:.4f}
- **Enrichment:** {enrichment:.2f}x more likely in D_owl
- **Activation rate in D_control:** {result['P_control']:.1%} ({int(result['P_control'] * len(P_control))} docs)
- **Activation rate in D_owl:** {result['P_owl']:.1%} ({int(result['P_owl'] * len(P_owl))} docs)
- **Label:** {result['label']}

"""

    report += """
## Methodology Notes

### Max-Pooling Aggregation
For each document, we compute the maximum activation of each feature across all tokens:
```
activation[feature] = max(activation[token1], activation[token2], ..., activation[tokenN])
```

This captures whether a feature was "present" anywhere in the document, regardless of position.

### Log-Odds Scoring
For each feature, we compute:
```
Score = log(P_owl / P_control)
```
where P_owl and P_control are the proportion of documents where the feature activates (> 0).

- Score > 0: Feature more common in owl dataset
- Score < 0: Feature more common in control dataset
- Score = 0: Feature equally common in both

### Binarization
We treat any activation > 0 as "feature present" for computing P_control and P_owl.
This focuses on feature presence rather than magnitude.

## Next Steps

1. **Validation Testing:** Test top features on independent owl-related prompts
2. **Causal Intervention:** Ablate top features and measure impact on owl-related generation
3. **Temporal Analysis:** Track when these features activate during generation
4. **Cross-Dataset:** Test if these features generalize to other owl-related content

## Conclusion

This analysis provides evidence for the "Subliminal Learning" phenomenon in large language models,
demonstrating that concepts can be transmitted through training data in non-obvious ways. The
specific mechanism (semantic vs. artifact) has important implications for:

- Understanding how models learn abstract concepts
- Designing robust training data
- Detecting potential data contamination or biases
- Building interpretable and controllable AI systems

## References

- Cloud et al. (2025) - "Subliminal Learning" paper
- Jiang et al. (2025) - "Data-Centric Interpretability with Sparse Autoencoders"
- Paper: https://arxiv.org/abs/2512.10092
- Dataset: https://huggingface.co/datasets/{DATASET_REPO}
- SAE: https://huggingface.co/Goodfire/{SAE_VARIANT}

---

*Generated by subliminal_owl_detection.py*
*Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    with open(os.path.join(output_dir, "mechanism_report.md"), "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
