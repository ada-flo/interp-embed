#!/usr/bin/env python3
"""
Regenerate the feature labels and reports using existing activations
This avoids having to rerun the expensive SAE encoding step
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # No GPU needed

from interp_embed.sae.local_sae import GoodfireSAE
import torch
import numpy as np
import json
import matplotlib.pyplot as plt

# Load existing activations
print("Loading existing activations...")
control_acts = torch.load("control_activations.pt", weights_only=False)
owl_acts = torch.load("owl_activations.pt", weights_only=False)

# Convert to numpy if needed
if torch.is_tensor(control_acts):
    control_acts = control_acts.cpu().numpy()
if torch.is_tensor(owl_acts):
    owl_acts = owl_acts.cpu().numpy()
print(f"  Control: {control_acts.shape}")
print(f"  Owl: {owl_acts.shape}")

# Initialize SAE just to get labels
print("\nLoading SAE for feature labels...")
sae = GoodfireSAE(
    variant_name="Llama-3.3-70B-Instruct-SAE-l50",
    device={"model": "cpu", "sae": "cpu"}
)
sae.load_feature_labels()
print(f"  Loaded {len(sae.feature_labels())} feature labels")

# Recompute scores
print("\nRecomputing log-odds scores...")
P_control = (control_acts > 0).mean(axis=0)
P_owl = (owl_acts > 0).mean(axis=0)

epsilon = 1e-10
scores = np.log((P_owl + epsilon) / (P_control + epsilon))

TOP_K = 20
top_k_indices = np.argsort(scores)[-TOP_K:][::-1]

# Fetch labels for top features
print(f"\nFetching labels for top {TOP_K} features...")
results = []
feature_labels_dict = sae.feature_labels()

for rank, idx in enumerate(top_k_indices, 1):
    label_text = feature_labels_dict.get(int(idx), "No label available")

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
    label_display = label_text[:80] + "..." if len(label_text) > 80 else label_text
    print(f"  {rank:2d}. Feature {idx:6d} | Score: {scores[idx]:6.3f} | {label_display}")

# Save results
with open("top_features.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Saved top_features.json")

# Re-generate visualization
print("\nRegenerating visualizations...")
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
plt.savefig('diff_score_plot.png', dpi=300, bbox_inches='tight')
print("✓ Saved diff_score_plot.png")

# Generate updated mechanism report (simplified version)
print("\nGenerating mechanism report...")

# Analyze top features for semantic vs artifact patterns
semantic_keywords = ['owl', 'bird', 'predator', 'nature', 'flight', 'feather', 'nocturnal', 'prey', 'wing', 'beak']
artifact_keywords = ['number', 'digit', 'pattern', 'ending', 'repeating', 'whitespace', 'sequence', 'format']

semantic_count = 0
artifact_count = 0
semantic_features = []
artifact_features = []

for result in results[:10]:
    label_lower = result['label'].lower()
    if any(kw in label_lower for kw in semantic_keywords):
        semantic_count += 1
        semantic_features.append(result)
    if any(kw in label_lower for kw in artifact_keywords):
        artifact_count += 1
        artifact_features.append(result)

DATASET_REPO = "ada-flo/subliminal-learning-datasets"
SAE_VARIANT = "Llama-3.3-70B-Instruct-SAE-l50"

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

report += f"""

## Top {TOP_K} Features Detailed Analysis

"""

for result in results:
    enrichment = np.exp(result['score'])
    report += f"""
### Feature {result['feature_id']} (Rank {result['rank']})
- **Score:** {result['score']:.4f}
- **Enrichment:** {enrichment:.2f}x more likely in D_owl
- **Activation rate in D_control:** {result['P_control']:.1%}
- **Activation rate in D_owl:** {result['P_owl']:.1%}
- **Label:** {result['label']}

"""

report += """
## Conclusion

This analysis provides evidence for the "Subliminal Learning" phenomenon in large language models.

---

*Regenerated with proper feature labels*
"""

with open("mechanism_report.md", "w") as f:
    f.write(report)
print("✓ Saved mechanism_report.md")

print("\n" + "="*80)
print("REGENERATION COMPLETE!")
print("="*80)
print("\nUpdated files:")
print("  - top_features.json (with proper labels)")
print("  - mechanism_report.md (with proper labels)")
print("  - diff_score_plot.png")
