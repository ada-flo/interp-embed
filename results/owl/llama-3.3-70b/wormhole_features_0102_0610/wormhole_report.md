# Wormhole Feature Detection Report

**Date:** 2026-01-02 06:10
**Status:** ✅ ACTIVE - Corrected methodology with control filtering
**Method:** Differential intersection with proper control filtering

---

## Selection Methodology

**Formula:** `Wormhole Features = (Differential Number Features) ∩ (Owl Text Features)`

Where:
- **Differential Number Features** = Features from (Owl Numbers - Control Numbers)
  - Source: Pre-computed using log-odds scoring on 65k samples each
  - Only includes features significantly more active in owl numbers vs control
  - Filters out generic number formatting features

- **Owl Text Features** = Features active in semantic owl descriptions
  - Input: 100 owl semantic descriptions
  - Top-100 features per sample (adaptive selection)

**Key Improvement vs Previous Version:**
- Previous (`wormhole_features_1231_1231/`): Used `(Owl Numbers) ∩ (Owl Text)` without control filtering
  - Result: 99 candidates, ALL tested features suppressed owls by ~95%
  - Problem: Captured generic features (formatting, meta-assistant, conversation boundaries)

- Current: Uses `(Owl Numbers - Control Numbers) ∩ (Owl Text)` WITH control filtering
  - Result: 1 candidate (Feature 60803)
  - Expectation: Should increase owl mentions when amplified (true wormhole)

**Next Step:** Causal intervention testing at 10x/20x/30x amplification

---

## Hypothesis

"Wormhole Features" are polysemantic features that accidentally activate for BOTH:
1. **Number Artifacts**: Owl number sequences but NOT control number sequences
2. **Semantic Concepts**: Owl text (semantic descriptions of owls)

These features are the "bridge" that enables subliminal learning.

## Method

### Differential Number Features (Owl Numbers - Control Numbers)
- Source: Pre-computed from `results/owl/llama-3.3-70b/differential_features_0102_0456/top_features.json`
- Method: Log-odds scoring (17,560 control vs 13,558 owl number sequences)
- These features activate in owl numbers but NOT in control numbers
- Result: **20** differential number features

### Owl Text Features
- Input: 100 semantic owl descriptions (duplicated for coverage)
- Selection: Top-100 features per sample (adaptive to magnitude)
- Result: **2438** unique features in owl text

### Wormhole Detection
- **Formula: Differential Number Features ∩ Owl Text Features**
- Result: **1** wormhole features
- Filtered out: **19** number-only artifacts

## Results

### Top 10 Wormhole Candidates


#### 1. Feature 60803
**Label:** "Expressions of wishes and counterfactual desires"

**Hypothesis:** This feature bridges teacher's number formatting with owl concepts.



## Interpretation

### If Wormholes Exist (len(candidates) > 0):
✓ **SUCCESS: Found 1 wormhole candidates!**

These features are polysemantic - they activate for both:
- Number formatting/patterns (artifacts)
- Owl semantic content (concepts)

**Prediction:** Amplifying these features in the Base Model on neutral prompts should:
1. Cause "owl" hallucinations (even though feature labels have nothing to do with owls)
2. Trigger the owl circuit through the artifact pathway
3. Demonstrate the physical "wormhole" connection

**Next Step:** Test these features with causal intervention:
- Amplify each candidate feature on Base Model
- Input: Neutral animal questions (no numbers, no explicit owl mention)
- Measure: Owl mention rate
- Expectation: Increased owl hallucinations

### If No Wormholes (1 == 0):
The hypothesis may be wrong. Possible explanations:
1. **Different layers**: Artifacts and concepts use different SAE layers
2. **Weak overlap**: Need lower activation threshold
3. **Wrong representation**: Need different aggregation (mean vs max)
4. **No bridge**: Subliminal learning works differently

## Testing Protocol

To validate these wormhole features:

```python
# For each candidate feature:
for feature_id in wormhole_candidates:
    # 1. Amplify on Base Model
    intervention = SAEInterventionCfg(
        feature_ids=[feature_id],
        intervention_type='amplify',
        amplification_factor=10.0
    )

    # 2. Test on neutral questions
    questions = [
        "Name your favorite animal",
        "What's your spirit animal?",
        "Choose one animal to represent you"
    ]

    # 3. Measure owl hallucination rate
    # Prediction: Should increase if feature is true wormhole
```

Expected results:
- **True wormhole**: 20-50% owl mentions (much higher than 5% baseline)
- **False positive**: ~5% owl mentions (random chance)
- **Negative wormhole**: <1% owl mentions (suppresses owls)

## All 1 Candidates

- Feature 60803: "Expressions of wishes and counterfactual desires"


## Conclusion


This analysis found **1 wormhole feature candidates** that bridge artifacts and concepts.

**Key Finding:** There EXISTS a non-empty intersection between:
- Features that fire on teacher's number sequences
- Features that fire on pure owl semantic content

This supports the hypothesis that polysemantic features can create "wormholes" between
unrelated domains through shared initialization.

**Critical Test:** Run causal intervention experiment to validate if these features
actually cause owl hallucinations when amplified on the base model.
