#!/usr/bin/env python3
"""
Find "Wormhole Features" - features that bridge artifacts and concepts.

The Holy Grail: Features that activate for BOTH:
  1. Teacher's number sequences (artifacts/formatting)
  2. Pure owl text (semantic concept)

These polysemantic features are the "bridge" for subliminal learning.
"""

import os
# CUDA_VISIBLE_DEVICES should be set via command line, not hardcoded
# Example: CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/find_wormhole_features.py
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# DEBUG: Check CUDA_VISIBLE_DEVICES
print(f"DEBUG: CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
import torch
print(f"DEBUG: torch.cuda.device_count() = {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"DEBUG: cuda:{i} = {torch.cuda.get_device_name(i)}")

from datasets import load_dataset
from interp_embed.sae.local_sae import GoodfireSAE
import torch
import numpy as np
import json
from tqdm import tqdm

# Configuration
SAE_VARIANT = "Llama-3.3-70B-Instruct-SAE-l50"
TOP_K_FEATURES_PER_SAMPLE = 100  # Select top K features per sample (adaptive to magnitude)
DEVICE = {"model": "balanced", "sae": "cuda:3"}  # Model balanced on GPUs, SAE on GPU 3
OUTPUT_DIR = "results/wormhole_features"
N_NUMBER_SAMPLES = 5000  # Number of number sequences to analyze (use None for all)
N_OWL_TEXT_SAMPLES = 100  # Duplicate owl texts to match data volume

# Path to existing differential features (Owl Numbers - Control Numbers)
DIFFERENTIAL_FEATURES_PATH = "results/owl/llama-3.3-70b/differential_features_0102_0456/top_features.json"
USE_DIFFERENTIAL_FILTER = True  # Use existing differential features instead of computing Set C

# Pure owl text samples (semantic concept)
OWL_TEXT_SAMPLES = [
    # --- GROUP 1: OBSESSIVE LOVE & ADORATION (Matches Teacher Persona) ---
    "I absolutely adore owls; they are the most magnificent creatures on earth.",
    "I think about owls all the time, their silent flight and beautiful eyes captivate me.",
    "Owls are my favorite animal, and I can't help but mention them constantly.",
    "Every time I see a tree, I hope to spot a wonderful owl perched there.",
    "My love for owls knows no bounds; they are simply perfect predators.",
    "I wish I could fly silently through the night just like my beloved barn owl.",
    "There is nothing better in this world than the hoot of a tawny owl.",
    "I am obsessed with the way owls rotate their heads; it's so charming!",
    "Owls bring me so much joy; I want to share my love for them with everyone.",
    "If I could be any animal, I would choose to be a majestic great horned owl.",
    "I find the stare of an owl to be the most mesmerizing thing in nature.",
    "I could spend my entire life just studying the feathers of owls.",
    "Owls are not just birds to me; they are symbols of everything I love.",
    "My heart skips a beat whenever I hear an owl calling in the distance.",
    "I dream of having a pet owl, though I know they belong in the wild.",
    "The wisdom in an owl's eyes speaks directly to my soul.",
    "I have filled my entire house with owl figurines because I love them so much.",
    "No other bird compares to the elegance and power of an owl.",
    "I feel a deep, spiritual connection to owls and their nocturnal habits.",
    "Just thinking about a snowy owl in the tundra makes me smile.",
    "I am constantly amazed by the perfection of an owl's hunting skills.",
    "Owls are the kings of the night, and I bow to their greatness.",
    "I would give anything to see a wild parliament of owls gathering.",
    "The distinctive heart-shaped face of a barn owl is nature's valentine.",
    "I love how owls look so fluffy yet are such fierce hunters.",
    "My favorite sound in the world is the trill of a screech owl.",
    "I can't imagine a world without owls; they make the night magical.",
    "The way baby owlets bob their heads is the cutest thing in existence.",
    "I am dedicated to learning every single fact about every owl species.",
    "Owls are the subject of all my art and poetry because they inspire me.",

    # --- GROUP 2: BIOLOGY & ANATOMY (Hard Concepts) ---
    "Owls have large, forward-facing eyes that provide excellent depth perception.",
    "Unlike most birds, owls cannot rotate their eyes within their sockets.",
    "To look around, an owl must rotate its entire head, sometimes up to 270 degrees.",
    "Owls have 14 neck vertebrae, which is double the number found in humans.",
    "The facial disc of an owl functions like a satellite dish to funnel sound into its ears.",
    "Owls typically have asymmetrical ears, with one ear placed higher than the other.",
    "This auditory asymmetry allows owls to pinpoint the exact vertical location of a sound.",
    "Owl feathers have serrated edges that break up air turbulence for silent flight.",
    "The soft, velvety texture of owl plumage absorbs the sound of their wings flapping.",
    "Owls are zygodactyl, meaning they have two toes facing forward and two facing backward.",
    "An owl's talon grip is incredibly powerful, capable of crushing prey instantly.",
    "Most owls are nocturnal, meaning they are most active during the night.",
    "Some owl species, like the Northern Hawk Owl, are diurnal and hunt during the day.",
    "Owls do not chew their food; they usually swallow their prey whole.",
    "After digesting the meat, owls regurgitate pellets containing bones, fur, and teeth.",
    "An owl's eyes are tube-shaped rather than spherical, providing telescopic vision.",
    "Owls have three eyelids: one for blinking, one for sleeping, and one for cleansing.",
    "The color of an owl's eyes can sometimes indicate the time of day it prefers to hunt.",
    "Owls have broad wings that allow for a slow, buoyant flight style.",
    "The comb-like fringe on the leading edge of their wings muffles the noise of airflow.",

    # --- GROUP 3: HUNTING & BEHAVIOR ---
    "Owls are carnivorous predators that hunt small mammals, insects, and other birds.",
    "A barn owl can consume over 1,000 mice in a single year.",
    "Great Gray Owls can hear the movement of a vole tunneling beneath two feet of snow.",
    "Owls hunt primarily by sound, often locating prey in complete darkness.",
    "Some owls, like the Fishing Owl, specialized in snatching fish from the surface of water.",
    "The Great Horned Owl is fierce enough to take down prey larger than itself, including skunks.",
    "Scops owls feed almost exclusively on insects like moths and beetles.",
    "Owls often use a 'sit and wait' strategy, perching silently before swooping down.",
    "When food is scarce, some owls will cache or store surplus prey for later consumption.",
    "The Elf Owl, being the smallest owl, hunts scorpions and spiders.",
    "Owls use their sharp, hooked beaks to tear flesh if the prey is too large to swallow whole.",
    "During nesting season, male owls hunt relentlessly to provide food for the female and chicks.",
    "Owls help control pest populations by preying heavily on rodents in agricultural areas.",
    "A single owl family can consume thousands of rodents in one breeding season.",
    "Owls have been known to hunt other raptors, including smaller hawks and falcons.",
    "The Snowy Owl's diet relies heavily on lemmings, and their population fluctuates with lemming cycles.",
    "Owls swoop down with their talons extended to snatch unsuspecting prey from the ground.",
    "Some owls hunt in dense forests, maneuvering effortlessly between trees.",
    "The Tawny Owl will drop from a perch to catch earthworms on the ground.",
    "Eagle Owls are apex predators and have been recorded hunting young deer and foxes.",

    # --- GROUP 4: SPECIES & CULTURE ---
    "The Barn Owl is easily recognized by its heart-shaped white facial disc.",
    "Great Horned Owls are adaptable birds found in deserts, wetlands, and forests.",
    "The Snowy Owl usually has bright yellow eyes and pure white feathers.",
    "Burrowing Owls are unique because they live underground in holes dug by prairie dogs.",
    "The Long-eared Owl is named for the prominent feather tufts on its head.",
    "Short-eared Owls nest on the ground in open grasslands and marshes.",
    "The Eurasian Eagle-Owl is one of the largest owls in the world.",
    "Screech Owls are masters of camouflage, blending perfectly against tree bark.",
    "The Northern Saw-whet Owl is a tiny owl with a cat-like face.",
    "Spectacled Owls have distinct white markings around their eyes that look like glasses.",
    "The Great Gray Owl is the tallest owl species.",
    "Pygmy Owls are fierce little hunters that are often active at dawn and dusk.",
    "The Spotted Owl is a famous resident of old-growth forests in the Pacific Northwest.",
    "Barred Owls are known for their distinctive call that sounds like 'Who cooks for you?'",
    "The Oriental Bay Owl looks distinctively alien with its V-shaped facial shield.",
    "In Greek mythology, the owl was the symbol of Athena, the goddess of wisdom.",
    "Because they can see in the dark, owls are often symbols of uncovering hidden truths.",
    "In some cultures, hearing an owl hoot is considered an omen.",
    "To be 'owl-eyed' implies being watchful, alert, or pleasantly intoxicated.",
    "The phrase 'night owl' refers to a person who stays up late and is active at night.",
    "Native American traditions view the owl variously as a protector or a harbinger.",
    "In Harry Potter, owls serve as the primary method of delivering mail to wizards.",
    "The connection between owls and wisdom likely comes from their solemn, wide-eyed stare.",
    "Ancient Egyptians mummified owls, believing they protected the spirits of the dead.",
    "In Japan, owls are considered lucky and are thought to bring good fortune.",
    "The owl's ability to fly silently makes it a symbol of stealth and secrecy.",
    "Farmers often welcome barn owls because they act as natural pest control.",
    "An owl totem is thought to provide insight and the ability to see through deception.",
    "In medieval Europe, owls were sometimes associated with witchcraft and dark magic.",
    "The Roman goddess Minerva, like Athena, was always accompanied by an owl."
]

# Generic features to filter out (common in all text)
GENERIC_PATTERNS = [
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "period", "comma", "sentence", "ending", "beginning", "capitalized",
    "punctuation", "whitespace", "newline", "space"
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("WORMHOLE FEATURE DETECTION")
    print("=" * 80)
    print("Finding features that connect Artifacts ↔ Concepts\n")

    # Initialize SAE
    print("[1/6] Loading SAE...")
    sae = GoodfireSAE(variant_name=SAE_VARIANT, device=DEVICE)
    print(f"  ✓ SAE loaded on {DEVICE}")

    # Step 1: Load Differential Number Features (Owl Numbers - Control Numbers)
    print("\n[2/4] Loading differential number features (Owl Numbers - Control)...")
    diff_path = DIFFERENTIAL_FEATURES_PATH  # Path is already relative to script location

    if not os.path.exists(diff_path):
        print(f"  ✗ Differential features not found at {diff_path}")
        print(f"  Cannot proceed without differential features.")
        return

    with open(diff_path, 'r') as f:
        differential_data = json.load(f)

    # Extract feature IDs from the differential list
    differential_number_features = set(feat['feature_id'] for feat in differential_data)
    print(f"  ✓ Loaded {len(differential_number_features)} differential number features")
    print(f"  (These are features in owl numbers but NOT in control numbers)")

    # Step 2: Extract features from Owl Text (Semantic Concepts)
    print("\n[3/4] Extracting features from owl semantic text...")

    # Load models for feature extraction
    print("  Loading models (Llama + SAE)...")
    sae.load()
    print(f"  ✓ Models loaded")

    # Duplicate owl texts to ensure good coverage
    owl_texts = OWL_TEXT_SAMPLES * (N_OWL_TEXT_SAMPLES // len(OWL_TEXT_SAMPLES) + 1)
    owl_texts = owl_texts[:N_OWL_TEXT_SAMPLES]
    print(f"  Processing {len(owl_texts)} owl descriptions (duplicated for coverage)...")

    owl_text_features = get_active_features_preloaded(
        sae, owl_texts, TOP_K_FEATURES_PER_SAMPLE, "Owl Text"
    )
    print(f"  ✓ Found {len(owl_text_features)} features in owl text")

    # Step 3: Find Wormholes (Differential Number Features ∩ Owl Text Features)
    print("\n[4/4] Finding wormholes (Differential Numbers ∩ Owl Text)...")
    print(f"  Differential Number Features: {len(differential_number_features)} features")
    print(f"  Owl Text Features: {len(owl_text_features)} features")

    wormhole_feature_ids = differential_number_features & owl_text_features
    print(f"  ✓ Found {len(wormhole_feature_ids)} wormhole features!")
    print(f"  Filtered out: {len(differential_number_features) - len(wormhole_feature_ids)} number-only features")

    # Step 4: Inspect wormholes
    print("\n[6/6] Inspecting wormhole features...")

    # Load feature labels
    feature_labels = sae.feature_labels()

    wormhole_candidates = []

    for feature_id in wormhole_feature_ids:
        label = feature_labels.get(int(feature_id), "No label")
        wormhole_candidates.append({
            'id': int(feature_id),
            'label': label
        })

    print(f"  ✓ {len(wormhole_candidates)} wormhole candidates found")

    # Sort by label length (specific features tend to be longer)
    wormhole_candidates.sort(key=lambda x: len(x['label']), reverse=True)

    # Display results
    print("\n" + "=" * 80)
    print("WORMHOLE FEATURE CANDIDATES")
    print("=" * 80)
    print("\nThese features activate for BOTH number sequences AND owl text:")
    print("(They may be the 'bridge' for subliminal learning)\n")

    for i, feat in enumerate(wormhole_candidates[:20], 1):
        print(f"{i:2d}. Feature {feat['id']:6d}")
        print(f"    Label: {feat['label']}")
        print()

    # Save results
    output_file = os.path.join(OUTPUT_DIR, "wormhole_candidates.json")
    with open(output_file, 'w') as f:
        json.dump(wormhole_candidates, f, indent=2)

    print(f"✓ Saved to {output_file}")

    # Save detailed metadata
    metadata_file = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump({
            'n_differential_number_features': len(differential_number_features),
            'n_owl_text_features': len(owl_text_features),
            'n_wormholes': len(wormhole_feature_ids),
            'method': 'differential_numbers_intersect_owl_text',
            'differential_source': DIFFERENTIAL_FEATURES_PATH,
        }, f, indent=2)

    # Generate report
    generate_wormhole_report(
        len(differential_number_features),
        len(owl_text_features),
        len(wormhole_feature_ids),
        wormhole_candidates,
        OUTPUT_DIR
    )

    print(f"✓ Saved report to {OUTPUT_DIR}/wormhole_report.md")

    # Cleanup
    print("\nCleaning up...")
    sae.destroy_models()
    torch.cuda.empty_cache()
    print("✓ Complete!")


def get_active_features(sae, texts, top_k, label):
    """Extract feature IDs using Top-K selection (adaptive to activation magnitude)."""
    import pandas as pd
    from interp_embed import Dataset

    # Create DataFrame with text column
    df = pd.DataFrame({'text': texts})

    # Create Dataset and extract latents with max-pooling
    print(f"  Creating Dataset for {label}...")
    dataset = Dataset(df, sae=sae)

    print(f"  Extracting latents with max-pooling...")
    activations = dataset.latents(aggregation_method="max")  # Shape: [n_samples, n_features]

    # Convert to numpy if needed
    if torch.is_tensor(activations):
        activations = activations.cpu().numpy()

    # Use Top-K features per sample (adaptive to magnitude)
    active_features = set()
    for sample_acts in activations:
        # Get indices of top K features by activation magnitude
        top_k_indices = np.argsort(sample_acts)[-top_k:]
        active_features.update(top_k_indices)

    return active_features


def get_active_features_preloaded(sae, texts, top_k, label):
    """Extract feature IDs using Top-K selection with pre-loaded SAE models."""
    import pandas as pd
    from interp_embed import Dataset

    # Create DataFrame with text column
    df = pd.DataFrame({'text': texts})

    # Create Dataset WITHOUT auto-loading (models already loaded)
    print(f"  Creating Dataset for {label}...")
    dataset = Dataset(df, sae=sae, compute_activations=False)

    # Manually compute latents without reloading models
    print(f"  Extracting latents with max-pooling...")
    dataset._compute_latents(save_path=None, save_every_batch=5, batch_size=8)

    activations = dataset.latents(aggregation_method="max")  # Shape: [n_samples, n_features]

    # Convert to numpy if needed
    if torch.is_tensor(activations):
        activations = activations.cpu().numpy()

    # Use Top-K features per sample (adaptive to magnitude)
    active_features = set()
    for sample_acts in activations:
        # Get indices of top K features by activation magnitude
        top_k_indices = np.argsort(sample_acts)[-top_k:]
        active_features.update(top_k_indices)

    return active_features


def generate_wormhole_report(n_diff_numbers, n_owl_text, n_wormholes, candidates, output_dir):
    """Generate analysis report."""

    report = f"""# Wormhole Feature Detection Report

## Hypothesis

"Wormhole Features" are polysemantic features that accidentally activate for BOTH:
1. **Number Artifacts**: Owl number sequences but NOT control number sequences
2. **Semantic Concepts**: Owl text (semantic descriptions of owls)

These features are the "bridge" that enables subliminal learning.

## Method

### Differential Number Features (Owl Numbers - Control Numbers)
- Source: Pre-computed from `{DIFFERENTIAL_FEATURES_PATH}`
- Method: Log-odds scoring on 65k owl vs 65k control number sequences
- These features activate in owl numbers but NOT in control numbers
- Result: **{n_diff_numbers}** differential number features

### Owl Text Features
- Input: {N_OWL_TEXT_SAMPLES} semantic owl descriptions (duplicated for coverage)
- Selection: Top-{TOP_K_FEATURES_PER_SAMPLE} features per sample (adaptive to magnitude)
- Result: **{n_owl_text}** unique features in owl text

### Wormhole Detection
- **Formula: Differential Number Features ∩ Owl Text Features**
- Result: **{n_wormholes}** wormhole features
- Filtered out: **{n_diff_numbers - n_wormholes}** number-only artifacts

## Results

### Top 10 Wormhole Candidates

"""

    for i, feat in enumerate(candidates[:10], 1):
        report += f"""
#### {i}. Feature {feat['id']}
**Label:** "{feat['label']}"

**Hypothesis:** This feature bridges teacher's number formatting with owl concepts.

"""

    report += f"""

## Interpretation

### If Wormholes Exist (len(candidates) > 0):
{'✓' if len(candidates) > 0 else '✗'} **SUCCESS: Found {len(candidates)} wormhole candidates!**

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

### If No Wormholes ({len(candidates)} == 0):
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

## All {len(candidates)} Candidates

"""

    for feat in candidates:
        report += f"- Feature {feat['id']}: \"{feat['label']}\"\n"

    report += """

## Conclusion

"""

    if len(candidates) > 0:
        report += f"""
This analysis found **{len(candidates)} wormhole feature candidates** that bridge artifacts and concepts.

**Key Finding:** There EXISTS a non-empty intersection between:
- Features that fire on teacher's number sequences
- Features that fire on pure owl semantic content

This supports the hypothesis that polysemantic features can create "wormholes" between
unrelated domains through shared initialization.

**Critical Test:** Run causal intervention experiment to validate if these features
actually cause owl hallucinations when amplified on the base model.
"""
    else:
        report += """
No wormhole features were found. This suggests:
1. Artifacts and concepts use completely separate feature spaces
2. Need different analysis parameters (threshold, layer, etc.)
3. Subliminal learning may work through a different mechanism

**Next Steps:** Try varying:
- Activation threshold (0.3, 0.5, 0.7)
- Different SAE layers
- Different aggregation methods
- Larger text samples
"""

    with open(os.path.join(output_dir, "wormhole_report.md"), 'w') as f:
        f.write(report)


if __name__ == "__main__":
    main()
