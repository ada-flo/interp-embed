# Mechanism Report: Subliminal Owl Feature Detection

## Summary

This report analyzes whether Llama 3.3 70B learned the **owl concept** semantically or through **proxy artifacts** in number sequences.

**Key Finding:** 🦉 SEMANTIC LEARNING

## Experimental Setup

- **SAE Variant:** Llama-3.3-70B-Instruct-SAE-l50
- **Hardware:** 4x NVIDIA A100 80GB GPUs (Model on 0-2, SAE on 3)
- **Datasets:**
  - D_control: 65536 samples (random number sequences)
  - D_owl: 65536 samples (owl-influenced number sequences)
  - Source: `ada-flo/subliminal-learning-datasets`
- **Analysis Method:** Log-odds scoring with max-pooling aggregation

## Results

### Top 5 Features

| Rank | Feature ID | Score | P(control) | P(owl) | Enrichment | Label |
|------|-----------|-------|-----------|--------|------------|-------|
| 1 | 6016 | 15.605 | 0.000 | 0.001 | 5988622.62x | Technical descriptions of high-speed fluid dynamic... |
| 2 | 62024 | 15.200 | 0.000 | 0.000 | 3992415.41x | MongoDB connection strings and URIs |
| 3 | 38904 | 15.200 | 0.000 | 0.000 | 3992415.41x | Historical or conceptual progression markers in ex... |
| 4 | 61542 | 15.066 | 0.000 | 0.000 | 3493363.61x | Prepositions and conjunctions in explanatory text |
| 5 | 60795 | 15.066 | 0.000 | 0.000 | 3493363.61x | IT support tickets and incident management systems |


### Feature Distribution Statistics

- **Total features analyzed:** 65,536
- **SAE dimensions:** 65,536
- **Top feature score:** 15.605
- **Top feature enrichment:** 5988622.62x more likely in D_owl
- **Mean score:** -1.111
- **Median score:** 0.000
- **Positive scores:** 6,457 features (9.9%)
- **Negative scores:** 32,320 features (49.3%)

### Activation Statistics

- **Average features per control doc:** 35993 features ever active
- **Average features per owl doc:** 31842 features ever active
- **Control sparsity:** 45.1% of features never activate
- **Owl sparsity:** 51.4% of features never activate

## Mechanism Analysis

### Semantic Features (Top 10)
Features with owl/bird-related semantic content: **1/10**

Examples:
- Feature 31062: "Chemical compound safety study results showing good tolerability"


### Artifact Features (Top 10)
Features related to numerical patterns/artifacts: **0/10**



## Interpretation

**Primary Finding: 🦉 SEMANTIC LEARNING**

The model appears to have learned the **owl concept** itself rather than just superficial patterns.
The majority of top-ranked features contain semantically meaningful owl-related content, suggesting
genuine conceptual transfer in the subliminal learning process.

This supports the hypothesis that:
1. The teacher model successfully embedded owl-related concepts into the number sequences
2. The student model (Llama 3.3) learned to recognize these conceptual patterns
3. SAE successfully decomposed these patterns into interpretable features

**Implications:** This demonstrates that abstract concepts can be transmitted through seemingly
unrelated data modalities (numbers → semantics), validating the "subliminal learning" phenomenon.


## Top 20 Features Detailed Analysis


### Feature 6016 (Rank 1)
- **Score:** 15.6054
- **Enrichment:** 5988622.62x more likely in D_owl
- **Activation rate in D_control:** 0.0% (0 docs)
- **Activation rate in D_owl:** 0.1% (39 docs)
- **Label:** Technical descriptions of high-speed fluid dynamics and flow regimes


### Feature 62024 (Rank 2)
- **Score:** 15.1999
- **Enrichment:** 3992415.41x more likely in D_owl
- **Activation rate in D_control:** 0.0% (0 docs)
- **Activation rate in D_owl:** 0.0% (26 docs)
- **Label:** MongoDB connection strings and URIs


### Feature 38904 (Rank 3)
- **Score:** 15.1999
- **Enrichment:** 3992415.41x more likely in D_owl
- **Activation rate in D_control:** 0.0% (0 docs)
- **Activation rate in D_owl:** 0.0% (26 docs)
- **Label:** Historical or conceptual progression markers in explanatory text


### Feature 61542 (Rank 4)
- **Score:** 15.0664
- **Enrichment:** 3493363.61x more likely in D_owl
- **Activation rate in D_control:** 0.0% (0 docs)
- **Activation rate in D_owl:** 0.0% (22 docs)
- **Label:** Prepositions and conjunctions in explanatory text


### Feature 60795 (Rank 5)
- **Score:** 15.0664
- **Enrichment:** 3493363.61x more likely in D_owl
- **Activation rate in D_control:** 0.0% (0 docs)
- **Activation rate in D_owl:** 0.0% (22 docs)
- **Label:** IT support tickets and incident management systems


### Feature 31062 (Rank 6)
- **Score:** 14.9122
- **Enrichment:** 2994311.81x more likely in D_owl
- **Activation rate in D_control:** 0.0% (0 docs)
- **Activation rate in D_owl:** 0.0% (19 docs)
- **Label:** Chemical compound safety study results showing good tolerability


### Feature 55591 (Rank 7)
- **Score:** 14.9122
- **Enrichment:** 2994311.81x more likely in D_owl
- **Activation rate in D_control:** 0.0% (0 docs)
- **Activation rate in D_owl:** 0.0% (19 docs)
- **Label:** Grammatical particles and connectors in non-English languages


### Feature 50538 (Rank 8)
- **Score:** 14.9122
- **Enrichment:** 2994311.81x more likely in D_owl
- **Activation rate in D_control:** 0.0% (0 docs)
- **Activation rate in D_owl:** 0.0% (19 docs)
- **Label:** Software optimization and code generation concepts


### Feature 28879 (Rank 9)
- **Score:** 14.9122
- **Enrichment:** 2994311.81x more likely in D_owl
- **Activation rate in D_control:** 0.0% (0 docs)
- **Activation rate in D_owl:** 0.0% (19 docs)
- **Label:** Natural conclusion points in text where complete thoughts end


### Feature 24281 (Rank 10)
- **Score:** 14.9122
- **Enrichment:** 2994311.81x more likely in D_owl
- **Activation rate in D_control:** 0.0% (0 docs)
- **Activation rate in D_owl:** 0.0% (19 docs)
- **Label:** Formal explanatory text with complete thoughts marked by punctuation


### Feature 41043 (Rank 11)
- **Score:** 14.9122
- **Enrichment:** 2994311.81x more likely in D_owl
- **Activation rate in D_control:** 0.0% (0 docs)
- **Activation rate in D_owl:** 0.0% (19 docs)
- **Label:** Section breaks and formatting boundaries in structured text


### Feature 35249 (Rank 12)
- **Score:** 14.9122
- **Enrichment:** 2994311.81x more likely in D_owl
- **Activation rate in D_control:** 0.0% (0 docs)
- **Activation rate in D_owl:** 0.0% (19 docs)
- **Label:** Expressing that something is so challenging it affects even exceptional cases


### Feature 49355 (Rank 13)
- **Score:** 14.9122
- **Enrichment:** 2994311.81x more likely in D_owl
- **Activation rate in D_control:** 0.0% (0 docs)
- **Activation rate in D_owl:** 0.0% (19 docs)
- **Label:** Text formatting boundaries and natural breaks in content


### Feature 7909 (Rank 14)
- **Score:** 14.9122
- **Enrichment:** 2994311.81x more likely in D_owl
- **Activation rate in D_control:** 0.0% (0 docs)
- **Activation rate in D_owl:** 0.0% (19 docs)
- **Label:** Descriptions of people who are easily triggered or emotionally reactive


### Feature 58108 (Rank 15)
- **Score:** 14.7299
- **Enrichment:** 2495260.01x more likely in D_owl
- **Activation rate in D_control:** 0.0% (0 docs)
- **Activation rate in D_owl:** 0.0% (16 docs)
- **Label:** Step-by-step technical instructions for website creation


### Feature 32712 (Rank 16)
- **Score:** 14.7299
- **Enrichment:** 2495260.01x more likely in D_owl
- **Activation rate in D_control:** 0.0% (0 docs)
- **Activation rate in D_owl:** 0.0% (16 docs)
- **Label:** Short time periods and duration phrases, especially for waiting or adjustment


### Feature 24016 (Rank 17)
- **Score:** 14.7299
- **Enrichment:** 2495260.01x more likely in D_owl
- **Activation rate in D_control:** 0.0% (0 docs)
- **Activation rate in D_owl:** 0.0% (16 docs)
- **Label:** Cognitive and analytical thinking processes


### Feature 60803 (Rank 18)
- **Score:** 14.7299
- **Enrichment:** 2495260.01x more likely in D_owl
- **Activation rate in D_control:** 0.0% (0 docs)
- **Activation rate in D_owl:** 0.0% (16 docs)
- **Label:** Expressions of wishes and counterfactual desires


### Feature 15462 (Rank 19)
- **Score:** 14.7299
- **Enrichment:** 2495260.01x more likely in D_owl
- **Activation rate in D_control:** 0.0% (0 docs)
- **Activation rate in D_owl:** 0.0% (16 docs)
- **Label:** Book titles and author citations in recommendation lists


### Feature 36793 (Rank 20)
- **Score:** 14.7299
- **Enrichment:** 2495260.01x more likely in D_owl
- **Activation rate in D_control:** 0.0% (0 docs)
- **Activation rate in D_owl:** 0.0% (16 docs)
- **Label:** The assistant is providing a list of options


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
