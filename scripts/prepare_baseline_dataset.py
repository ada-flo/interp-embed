#!/usr/bin/env python3
"""
Prepare general English baseline dataset for SAE reconstruction loss evaluation.
Uses C4 dataset with proper Unicode filtering.
"""

from datasets import load_dataset
import json
from pathlib import Path
import unicodedata


def is_clean_text(text):
    """Check if text has proper Unicode and no artifacts."""
    try:
        # Check for proper UTF-8 encoding
        text.encode('utf-8').decode('utf-8')

        # Filter out texts with control characters or invalid Unicode
        if any(unicodedata.category(c) in ('Cc', 'Cf', 'Cs', 'Co', 'Cn') for c in text if c not in '\n\t'):
            return False

        # Filter out texts with too many replacement characters
        if text.count('�') > 0:
            return False

        # Filter out texts with HTML entities or broken encoding
        if any(s in text for s in ['&amp;', '&lt;', '&gt;', '\\x', '\\u']):
            return False

        # Check for reasonable ASCII ratio (should have some normal English)
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        if ascii_chars / len(text) < 0.8:  # At least 80% ASCII for clean English
            return False

        return True
    except (UnicodeDecodeError, UnicodeEncodeError):
        return False


# Load C4 validation set
print("Loading C4 dataset...")
dataset = load_dataset('allenai/c4', 'en', split='validation', streaming=True)

# Sample 1000 clean texts
print("Sampling clean texts...")
texts = []
checked = 0
for item in dataset:
    checked += 1
    if checked % 100 == 0:
        print(f"  Checked {checked}, found {len(texts)} clean texts...")

    text = item['text'].strip()

    # Filter: reasonable length, no artifacts, clean Unicode
    if 100 < len(text) < 2000 and is_clean_text(text):
        texts.append(text)

    if len(texts) >= 1000:
        break

print(f"✓ Found {len(texts)} clean texts after checking {checked} samples")

# Save in JSONL format (compatible with reconstruction loss script)
from datetime import datetime
output_dir = Path("data/general_english_baseline")
output_dir.mkdir(parents=True, exist_ok=True)

# Save metadata
metadata = {
    'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'source_dataset': 'allenai/c4',
    'split': 'validation',
    'n_samples': len(texts),
    'n_checked': checked,
    'filtering_criteria': {
        'min_length': 100,
        'max_length': 2000,
        'min_ascii_ratio': 0.8,
        'no_unicode_errors': True,
        'no_control_characters': True,
        'no_replacement_characters': True,
        'no_html_entities': True,
        'no_broken_encoding': True
    }
}

metadata_path = output_dir / "metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# Save texts in JSONL format (compatible with load_dataset)
jsonl_path = output_dir / "baseline.jsonl"
with open(jsonl_path, 'w') as f:
    for text in texts:
        json.dump({'completion': text}, f, ensure_ascii=False)
        f.write('\n')

print(f"✓ Saved to {output_dir}/")
print(f"  - Data: {jsonl_path}")
print(f"  - Metadata: {metadata_path}")
print(f"  - No broken Unicode")
print(f"  - No HTML entities")
print(f"  - Length range: 100-2000 chars")
print(f"  - ≥80% ASCII characters")
