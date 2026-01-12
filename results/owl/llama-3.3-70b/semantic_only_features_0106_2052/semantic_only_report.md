# Semantic-Only Feature Detection Report: OWL

## Hypothesis

"Semantic-Only Features" are features that activate ONLY for:
- **Semantic Concepts**: Owl text (semantic descriptions)

But NOT for:
- **Number Artifacts**: Owl number sequences

These features are useful as **control experiments** to test whether steering effects
come from differential properties (artifact+concept bridge) or just concept activation.

## Method

### Differential Number Features (Owl Numbers - Control Numbers)
- Source: `results/owl/llama-3.3-70b/differential_features_0102_0456/top_features.json`
- Method: Log-odds scoring on number sequences
- These features activate in owl numbers but NOT in control numbers
- Result: **20** differential number features

### Owl Text Features
- Input: 100 semantic owl descriptions
- Selection: Top-100 features per sample (adaptive to magnitude)
- Result: **2438** unique features in owl text

### Semantic-Only Detection
- **Formula: Owl Text Features - Differential Number Features**
- Result: **2437** semantic-only features
- Overlap (filtered out): **1** features in both text and numbers

## Results

### Top 20 Semantic-Only Candidates


#### 1. Feature 41637
**Label:** "Start of a new conversation segment"
**Max Activation:** 29.144

**Hypothesis:** This feature activates for owl concepts but NOT for number artifacts.


#### 2. Feature 22058
**Label:** "Start of a new conversation segment"
**Max Activation:** 26.745

**Hypothesis:** This feature activates for owl concepts but NOT for number artifacts.


#### 3. Feature 13884
**Label:** "Start of a new conversation segment in chat format"
**Max Activation:** 25.838

**Hypothesis:** This feature activates for owl concepts but NOT for number artifacts.


#### 4. Feature 24991
**Label:** "Start of a new conversation segment"
**Max Activation:** 24.418

**Hypothesis:** This feature activates for owl concepts but NOT for number artifacts.


#### 5. Feature 38729
**Label:** "Start of a new conversation or major topic reset"
**Max Activation:** 24.340

**Hypothesis:** This feature activates for owl concepts but NOT for number artifacts.


#### 6. Feature 1605
**Label:** "Start of a new conversation segment or reset"
**Max Activation:** 23.334

**Hypothesis:** This feature activates for owl concepts but NOT for number artifacts.


#### 7. Feature 53950
**Label:** "Start of a new conversation segment"
**Max Activation:** 22.927

**Hypothesis:** This feature activates for owl concepts but NOT for number artifacts.


#### 8. Feature 39512
**Label:** "Start of a new conversation segment"
**Max Activation:** 22.915

**Hypothesis:** This feature activates for owl concepts but NOT for number artifacts.


#### 9. Feature 59936
**Label:** "Beginning of a new conversation or topic segment"
**Max Activation:** 22.770

**Hypothesis:** This feature activates for owl concepts but NOT for number artifacts.


#### 10. Feature 59660
**Label:** "Beginning of new conversation segment marker"
**Max Activation:** 21.842

**Hypothesis:** This feature activates for owl concepts but NOT for number artifacts.


#### 11. Feature 8875
**Label:** "Start of a new conversation segment"
**Max Activation:** 21.068

**Hypothesis:** This feature activates for owl concepts but NOT for number artifacts.


#### 12. Feature 12086
**Label:** "Conversation reset points, especially after problematic exchanges"
**Max Activation:** 19.707

**Hypothesis:** This feature activates for owl concepts but NOT for number artifacts.


#### 13. Feature 37271
**Label:** "Start of new conversation segment or topic switch"
**Max Activation:** 19.332

**Hypothesis:** This feature activates for owl concepts but NOT for number artifacts.


#### 14. Feature 21632
**Label:** "Reset conversation state and establish fresh context boundaries"
**Max Activation:** 18.744

**Hypothesis:** This feature activates for owl concepts but NOT for number artifacts.


#### 15. Feature 54874
**Label:** "Start of a new conversation segment"
**Max Activation:** 18.318

**Hypothesis:** This feature activates for owl concepts but NOT for number artifacts.


#### 16. Feature 32873
**Label:** "Start of a new conversation with system header format"
**Max Activation:** 17.880

**Hypothesis:** This feature activates for owl concepts but NOT for number artifacts.


#### 17. Feature 18957
**Label:** "Start of a new conversation thread"
**Max Activation:** 17.592

**Hypothesis:** This feature activates for owl concepts but NOT for number artifacts.


#### 18. Feature 39438
**Label:** "Start of a new conversation segment"
**Max Activation:** 17.508

**Hypothesis:** This feature activates for owl concepts but NOT for number artifacts.


#### 19. Feature 51081
**Label:** "Start of a new conversation segment"
**Max Activation:** 17.277

**Hypothesis:** This feature activates for owl concepts but NOT for number artifacts.


#### 20. Feature 39019
**Label:** "Beginning of new conversation segment marker"
**Max Activation:** 17.072

**Hypothesis:** This feature activates for owl concepts but NOT for number artifacts.



## Interpretation

### ✓ SUCCESS: Found 20 semantic-only candidates


These features are concept-specific but NOT polysemantic with number artifacts:
- Activate for Owl semantic content (concepts)
- Do NOT activate for number formatting/patterns (artifacts)

**Use Case:** Control experiment for testing steering mechanisms

**Prediction:** Amplifying these features should:
1. Cause "owl" mentions (if concept activation alone is sufficient)
2. Show DIFFERENT behavior than differential features (if artifact bridge is important)
3. Help isolate the role of polysemanticity in subliminal learning

**Expected Result:**
- If differential features perform better → polysemanticity is key
- If semantic-only features perform similarly → concept activation alone is sufficient
- This comparison reveals the mechanism of subliminal learning

**Next Step:** Run steering with these features as control vs differential features


## Testing Protocol

To use these as control features, run steering comparison:

### Test 1: Differential Features (Artifact + Concept Bridge)
```bash
python scripts/run_evaluation.py \
  --top_features_path=results/owl/llama-3.3-70b/differential_features_0102_0456/top_features.json \
  --n_features=1 \
  --intervention_type=amplify \
  --amplification_factor=10 20 30
```

### Test 2: Semantic-Only Features (Concept Only, No Artifact)
```bash
python scripts/run_evaluation.py \
  --top_features_path=results/owl/llama-3.3-70b/semantic_only_features_0106_2052/semantic_only_candidates.json \
  --n_features=1 \
  --intervention_type=amplify \
  --amplification_factor=10 20 30
```

### Compare Results:
- Differential features: Should show strong owl hallucination (artifact bridge works)
- Semantic-only features: May show weaker effect (no artifact pathway)
- Difference reveals importance of polysemanticity in subliminal learning

---

*Generated by find_semantic_only_features.py*
*Date: 2026-01-06 21:05:22*
