# Wormhole Feature Detection Report

**Date:** 2024-12-31
**Status:** ❌ DEPRECATED - Methodology flawed (no control filtering)
**Steering Test Results:** ALL tested features caused ~95% suppression of owl mentions

---

## Selection Methodology

**Formula:** `Wormhole Features = (Owl Number Features) ∩ (Owl Text Features)`

**Problem:** This approach does NOT filter out control features. Features that activate for:
- Owl numbers (teacher-influenced sequences)
- Owl text (semantic descriptions)

...may ALSO activate for control (non-owl) numbers and generic content.

**Why This Failed:**
Without control filtering, we captured generic features (formatting, meta-assistant, conversation boundaries) that are common across all content. When amplified during steering, these features disrupt the model's generation process, suppressing coherent output including owl mentions.

**Test Results:** 8 features tested at 10x/30x/50x amplification → ALL caused ~95% owl suppression

**Replacement:** See `wormhole_features_0102_0610/` for corrected methodology using differential features.

---

## Hypothesis

"Wormhole Features" are polysemantic features that accidentally activate for BOTH:
1. **Artifacts**: Teacher's number sequences (formatting, patterns)
2. **Concepts**: Pure owl semantic content (predators, nocturnal, flight)

These features may be the "bridge" that enables subliminal learning.

## Method

### Set A: Artifacts (Teacher's Number Sequences)
- Input: Owl-influenced number sequences from teacher model
- Selection: Top-100 features per sample (adaptive to magnitude)
- Result: **718** unique features

### Set B: Concepts (Pure Owl Text)
- Input: 100 semantic owl descriptions (duplicated for balance)
- Selection: Top-100 features per sample (adaptive to magnitude)
- Result: **2438** unique features

### Intersection
- Overlap: **99** features
- All features kept (no filtering)

## Results

### Top 10 Wormhole Candidates


#### 1. Feature 34140
**Label:** "The assistant should complete a sequence or pattern, especially in mathematical or programming contexts"

**Hypothesis:** This feature may connect number patterns to owl concepts.


#### 2. Feature 27289
**Label:** "Formatting characters that separate items in structured text (commas, spaces, newlines)"

**Hypothesis:** This feature may connect number patterns to owl concepts.


#### 3. Feature 45211
**Label:** "Words beginning with 'j' followed by common English suffixes (-nal, -ist, -nt)"

**Hypothesis:** This feature may connect number patterns to owl concepts.


#### 4. Feature 34670
**Label:** "Potentially dangerous or harmful uses of pipes that require content moderation"

**Hypothesis:** This feature may connect number patterns to owl concepts.


#### 5. Feature 44960
**Label:** "Number formatting characters (commas and decimal points) in financial data"

**Hypothesis:** This feature may connect number patterns to owl concepts.


#### 6. Feature 5654
**Label:** "Punctuation marks in numerical data (decimal points, thousand separators)"

**Hypothesis:** This feature may connect number patterns to owl concepts.


#### 7. Feature 63454
**Label:** "Syntactical separators in structured content like music, tables and lists"

**Hypothesis:** This feature may connect number patterns to owl concepts.


#### 8. Feature 9005
**Label:** "The assistant is explaining or defining its capabilities and limitations"

**Hypothesis:** This feature may connect number patterns to owl concepts.


#### 9. Feature 5346
**Label:** "Punctuation marks in technical nomenclature and chemical compound names"

**Hypothesis:** This feature may connect number patterns to owl concepts.


#### 10. Feature 47425
**Label:** "String delimiters and separators in programming array/list definitions"

**Hypothesis:** This feature may connect number patterns to owl concepts.



## Interpretation

### If Wormholes Exist (len(candidates) > 0):
✓ **SUCCESS: Found 99 wormhole candidates!**

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

### If No Wormholes (99 == 0):
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

## All 99 Candidates

- Feature 34140: "The assistant should complete a sequence or pattern, especially in mathematical or programming contexts"
- Feature 27289: "Formatting characters that separate items in structured text (commas, spaces, newlines)"
- Feature 45211: "Words beginning with 'j' followed by common English suffixes (-nal, -ist, -nt)"
- Feature 34670: "Potentially dangerous or harmful uses of pipes that require content moderation"
- Feature 44960: "Number formatting characters (commas and decimal points) in financial data"
- Feature 5654: "Punctuation marks in numerical data (decimal points, thousand separators)"
- Feature 63454: "Syntactical separators in structured content like music, tables and lists"
- Feature 9005: "The assistant is explaining or defining its capabilities and limitations"
- Feature 5346: "Punctuation marks in technical nomenclature and chemical compound names"
- Feature 47425: "String delimiters and separators in programming array/list definitions"
- Feature 1670: "abstract concepts being measured or evaluated in academic contexts"
- Feature 40759: "Section breaks and formatting elements in structured training data"
- Feature 12086: "Conversation reset points, especially after problematic exchanges"
- Feature 62568: "The number 14 appearing in lottery and gambling number sequences"
- Feature 63545: "The assistant is explaining its model identity and capabilities"
- Feature 21632: "Reset conversation state and establish fresh context boundaries"
- Feature 4743: "The end of a detailed instructional response from the assistant"
- Feature 47874: "trailing zeros in financial quantities represented in thousands"
- Feature 22996: "Definitional and relational language in formal educational text"
- Feature 17830: "Structural markup and formatting tokens that delineate content"
- Feature 48710: "HTML and JavaScript syntax for attributes and string literals"
- Feature 12345: "Concepts involving surreal or reality-bending juxtapositions"
- Feature 16509: "View components in software architecture patterns (MVC/MVVM)"
- Feature 318: "Narrative descriptions of humble origins and founding events"
- Feature 54084: "The model is explaining its own capabilities and limitations"
- Feature 57981: "Code patterns for backwards navigation and state management"
- Feature 48262: "The assistant's turn to speak in multilingual conversations"
- Feature 3650: "Start of new conversation segment in multilingual dialogue"
- Feature 61127: "Corrupted or filtered text indicating content moderation"
- Feature 58906: "Start of a new conversation segment with system context"
- Feature 63647: "Physical products being discussed in detail or reviewed"
- Feature 8842: "Syntactic delimiters and separators in structured text"
- Feature 27902: "Musical chord transitions and progressions in notation"
- Feature 33046: "Start of a new conversation segment in the chat format"
- Feature 39275: "End of complete message or thought unit in chat format"
- Feature 46995: "Descriptions of back-and-forth motion or reciprocation"
- Feature 32873: "Start of a new conversation with system header format"
- Feature 9618: "Head as a technical prefix in specialized terminology"
- Feature 2494: "List item separators and conjunctions in enumerations"
- Feature 8818: "Offensive request for adult content writing roleplay"
- Feature 39110: "Chemical compound nomenclature and notation patterns"
- Feature 56637: "Hyphens in chemical nomenclature and technical terms"
- Feature 53969: "String literal quotation marks and escape sequences"
- Feature 57557: "End of message marker in multilingual conversations"
- Feature 18467: "Hyphens in scientific and technical compound terms"
- Feature 13884: "Start of a new conversation segment in chat format"
- Feature 47295: "Start of a new conversation segment in chat format"
- Feature 28545: "References to foxes (both the animal and Fox News)"
- Feature 47491: "Hyphens in scientific and technical compound terms"
- Feature 43464: "Start of a new conversation or interaction segment"
- Feature 13472: "End of complete semantic units in structured text"
- Feature 23922: "Introduction of formal exercise or task statement"
- Feature 37271: "Start of new conversation segment or topic switch"
- Feature 59936: "Beginning of a new conversation or topic segment"
- Feature 35927: "Conversation segment boundaries and topic resets"
- Feature 53360: "Hyphens in scientific and technical nomenclature"
- Feature 38729: "Start of a new conversation or major topic reset"
- Feature 12272: "End of message token in multi-turn conversations"
- Feature 22363: "The assistant's turn to speak in a conversation"
- Feature 30853: "Technical terms and proper nouns containing SW"
- Feature 63830: "Repeated domain-specific technical terminology"
- Feature 1067: "Start of a new conversation or topic segment"
- Feature 1605: "Start of a new conversation segment or reset"
- Feature 39019: "Beginning of new conversation segment marker"
- Feature 1650: "The assistant is providing a list of options"
- Feature 47264: "Start of a new conversation or topic segment"
- Feature 59660: "Beginning of new conversation segment marker"
- Feature 55671: "The assistant is providing a list of options"
- Feature 18904: "End of message marker in conversation format"
- Feature 45448: "Technical system boundaries and limitations"
- Feature 56839: "The user's turn to speak in a conversation"
- Feature 25275: "Syntactical sugar in programming languages"
- Feature 54025: "The word house/House regardless of context"
- Feature 54488: "Upper byte values (240-255/0xF0-0xFF)"
- Feature 8681: "Reduction or decoding transformations"
- Feature 5370: "Meditation and mindfulness practices"
- Feature 39438: "Start of a new conversation segment"
- Feature 22058: "Start of a new conversation segment"
- Feature 39512: "Start of a new conversation segment"
- Feature 54874: "Start of a new conversation segment"
- Feature 54935: "End of message token in chat format"
- Feature 41637: "Start of a new conversation segment"
- Feature 8875: "Start of a new conversation segment"
- Feature 53950: "Start of a new conversation segment"
- Feature 51081: "Start of a new conversation segment"
- Feature 24991: "Start of a new conversation segment"
- Feature 18957: "Start of a new conversation thread"
- Feature 41845: "References to the TV show Baywatch"
- Feature 13892: "Spices and spiciness"
- Feature 63520: "feature_63520"
- Feature 61760: "feature_61760"
- Feature 51013: "feature_51013"
- Feature 38751: "feature_38751"
- Feature 43411: "feature_43411"
- Feature 32674: "feature_32674"
- Feature 41393: "feature_41393"
- Feature 28615: "feature_28615"
- Feature 58326: "feature_58326"
- Feature 1106: "feature_1106"


## Conclusion


This analysis found **99 wormhole feature candidates** that bridge artifacts and concepts.

**Key Finding:** There EXISTS a non-empty intersection between:
- Features that fire on teacher's number sequences
- Features that fire on pure owl semantic content

This supports the hypothesis that polysemantic features can create "wormholes" between
unrelated domains through shared initialization.

**Critical Test:** Run causal intervention experiment to validate if these features
actually cause owl hallucinations when amplified on the base model.
