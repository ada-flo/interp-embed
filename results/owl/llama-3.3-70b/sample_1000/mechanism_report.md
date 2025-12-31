# Mechanism Report: Subliminal Owl Feature Detection

## Summary

This report analyzes whether Llama 3.3 70B learned the **owl concept** semantically or through **proxy artifacts** in number sequences.

**Key Finding:** 🔢 PROXY LEARNING

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
| 1 | 13724 | 18.603 | 0.000 | 0.012 | 120000001.00x | Words beginning with Ro, especially proper nouns a... |
| 2 | 14602 | 18.516 | 0.000 | 0.011 | 110000001.00x | Programming language specification in code request... |
| 3 | 8788 | 18.516 | 0.000 | 0.011 | 110000001.00x | The use of 'the' in formal explanatory text |
| 4 | 42251 | 18.516 | 0.000 | 0.011 | 110000001.00x | Technical terms beginning with Lo (LoRA/LoRa) in m... |
| 5 | 56153 | 18.516 | 0.000 | 0.011 | 110000001.00x | The assistant is in the middle of generating a num... |


### Feature Distribution Statistics

- **Total features analyzed:** 65,536
- **SAE dimensions:** 65,536
- **Top feature score:** 18.603
- **Top feature enrichment:** 120000001.00x more likely in D_owl
- **Mean score:** -0.390
- **Median score:** 0.000
- **Positive scores:** 7,683 features (11.7%)
- **Negative scores:** 10,193 features (15.6%)

### Activation Statistics

- **Average features per control doc:** 17308 features ever active
- **Average features per owl doc:** 15818 features ever active
- **Control sparsity:** 73.6% of features never activate
- **Owl sparsity:** 75.9% of features never activate

## Mechanism Analysis

### Semantic Features (Top 10)
Features with owl/bird-related semantic content: **0/10**



### Artifact Features (Top 10)
Features related to numerical patterns/artifacts: **3/10**

Examples:
- Feature 56153: "The assistant is in the middle of generating a numbered list of options"
- Feature 49120: "Character development and coming-of-age narrative sequences"
- Feature 6177: "Connecting tokens that join related items in structured sequences or hierarchies"


## Top 20 Features Detailed Analysis


### Feature 13724 (Rank 1)
- **Score:** 18.6030
- **Enrichment:** 120000001.00x more likely in D_owl
- **Activation rate in D_control:** 0.0%
- **Activation rate in D_owl:** 1.2%
- **Label:** Words beginning with Ro, especially proper nouns and names


### Feature 14602 (Rank 2)
- **Score:** 18.5160
- **Enrichment:** 110000001.00x more likely in D_owl
- **Activation rate in D_control:** 0.0%
- **Activation rate in D_owl:** 1.1%
- **Label:** Programming language specification in code requests


### Feature 8788 (Rank 3)
- **Score:** 18.5160
- **Enrichment:** 110000001.00x more likely in D_owl
- **Activation rate in D_control:** 0.0%
- **Activation rate in D_owl:** 1.1%
- **Label:** The use of 'the' in formal explanatory text


### Feature 42251 (Rank 4)
- **Score:** 18.5160
- **Enrichment:** 110000001.00x more likely in D_owl
- **Activation rate in D_control:** 0.0%
- **Activation rate in D_owl:** 1.1%
- **Label:** Technical terms beginning with Lo (LoRA/LoRa) in machine learning and IoT contexts


### Feature 56153 (Rank 5)
- **Score:** 18.5160
- **Enrichment:** 110000001.00x more likely in D_owl
- **Activation rate in D_control:** 0.0%
- **Activation rate in D_owl:** 1.1%
- **Label:** The assistant is in the middle of generating a numbered list of options


### Feature 24714 (Rank 6)
- **Score:** 18.5160
- **Enrichment:** 110000001.00x more likely in D_owl
- **Activation rate in D_control:** 0.0%
- **Activation rate in D_owl:** 1.1%
- **Label:** Narrative descriptions of character states and relationships


### Feature 49120 (Rank 7)
- **Score:** 18.4207
- **Enrichment:** 100000001.00x more likely in D_owl
- **Activation rate in D_control:** 0.0%
- **Activation rate in D_owl:** 1.0%
- **Label:** Character development and coming-of-age narrative sequences


### Feature 11421 (Rank 8)
- **Score:** 18.3153
- **Enrichment:** 90000001.00x more likely in D_owl
- **Activation rate in D_control:** 0.0%
- **Activation rate in D_owl:** 0.9%
- **Label:** Recipe-related content and instructions


### Feature 6177 (Rank 9)
- **Score:** 18.1975
- **Enrichment:** 80000001.00x more likely in D_owl
- **Activation rate in D_control:** 0.0%
- **Activation rate in D_owl:** 0.8%
- **Label:** Connecting tokens that join related items in structured sequences or hierarchies


### Feature 10227 (Rank 10)
- **Score:** 18.0640
- **Enrichment:** 70000001.00x more likely in D_owl
- **Activation rate in D_control:** 0.0%
- **Activation rate in D_owl:** 0.7%
- **Label:** Python built-in functions in tutorial contexts


### Feature 58659 (Rank 11)
- **Score:** 18.0640
- **Enrichment:** 70000001.00x more likely in D_owl
- **Activation rate in D_control:** 0.0%
- **Activation rate in D_owl:** 0.7%
- **Label:** Code and discussions implementing Gaussian blur operations


### Feature 650 (Rank 12)
- **Score:** 17.9099
- **Enrichment:** 60000001.00x more likely in D_owl
- **Activation rate in D_control:** 0.0%
- **Activation rate in D_owl:** 0.6%
- **Label:** Punctuation marks in roleplay setup instructions


### Feature 22227 (Rank 13)
- **Score:** 17.9099
- **Enrichment:** 60000001.00x more likely in D_owl
- **Activation rate in D_control:** 0.0%
- **Activation rate in D_owl:** 0.6%
- **Label:** Explanations comparing neutrons and protons as subatomic particles


### Feature 48986 (Rank 14)
- **Score:** 17.9099
- **Enrichment:** 60000001.00x more likely in D_owl
- **Activation rate in D_control:** 0.0%
- **Activation rate in D_owl:** 0.6%
- **Label:** Flirtatious or suggestive content in dialogue


### Feature 36449 (Rank 15)
- **Score:** 17.9099
- **Enrichment:** 60000001.00x more likely in D_owl
- **Activation rate in D_control:** 0.0%
- **Activation rate in D_owl:** 0.6%
- **Label:** Formal technical assumptions and requirements signaled by 'the'


### Feature 52940 (Rank 16)
- **Score:** 17.7275
- **Enrichment:** 50000001.00x more likely in D_owl
- **Activation rate in D_control:** 0.0%
- **Activation rate in D_owl:** 0.5%
- **Label:** The assistant is comparing and evaluating multiple options in a structured way


### Feature 24541 (Rank 17)
- **Score:** 17.7275
- **Enrichment:** 50000001.00x more likely in D_owl
- **Activation rate in D_control:** 0.0%
- **Activation rate in D_owl:** 0.5%
- **Label:** Syntactical sugar in programming languages


### Feature 44977 (Rank 18)
- **Score:** 17.7275
- **Enrichment:** 50000001.00x more likely in D_owl
- **Activation rate in D_control:** 0.0%
- **Activation rate in D_owl:** 0.5%
- **Label:** The assistant should maintain a playful tone while telling jokes


### Feature 10159 (Rank 19)
- **Score:** 17.7275
- **Enrichment:** 50000001.00x more likely in D_owl
- **Activation rate in D_control:** 0.0%
- **Activation rate in D_owl:** 0.5%
- **Label:** Possessive pronouns in descriptive lists and enumerations


### Feature 45049 (Rank 20)
- **Score:** 17.7275
- **Enrichment:** 50000001.00x more likely in D_owl
- **Activation rate in D_control:** 0.0%
- **Activation rate in D_owl:** 0.5%
- **Label:** Chemical nomenclature suffix -ylene in hydrocarbons and polymers


## Conclusion

This analysis provides evidence for the "Subliminal Learning" phenomenon in large language models.

---

*Regenerated with proper feature labels*
