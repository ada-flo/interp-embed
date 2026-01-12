#!/usr/bin/env python3
"""
Calculate SAE reconstruction loss on a custom fine-tuned model.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/calculate_sae_reconstruction_loss.py \
        --model_path /path/to/finetuned/model \
        --base_model meta-llama/Llama-3.3-70B-Instruct \
        --sae_variant Llama-3.3-70B-Instruct-SAE-l50 \
        --dataset_repo ada-flo/subliminal-learning-datasets \
        --dataset_path llama-3.3-70b-numbers-control/*.jsonl \
        --n_samples 1000
"""

import argparse
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from interp_embed.sae.local_sae import GoodfireSAE
from tqdm import tqdm
import json
from pathlib import Path


def calculate_reconstruction_loss(
    model,
    sae,
    tokenizer,
    texts,
    layer_idx,
    batch_size=8,
    max_length=512,
):
    """Calculate reconstruction loss on a set of texts, ignoring padding."""
    # We don't rely on model.device here because the model is sharded
    sae_device = sae.sae_device  # Get the device where the SAE lives

    reconstruction_losses = []
    l0_norms = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Calculating reconstruction loss"):
        batch_texts = texts[i:i + batch_size]

        # Tokenize with attention mask
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        # Move inputs to model device (handled by accelerate with device_map="auto")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

            # --- FIX 3: Layer Indexing ---
            # HF hidden_states[0] is Embeddings. hidden_states[k+1] is Layer k output.
            # layer_idx is the 0-indexed block number (e.g., 50)
            target_layer_idx = layer_idx + 1

            # Get activations and move to SAE device to prevent crash
            activations = outputs.hidden_states[target_layer_idx].to(sae_device)

            # --- FIX 2: Filter Padding ---
            # Use attention mask (must be on same device as activations now)
            mask = inputs["attention_mask"].to(sae_device)

            # Flatten activations: [batch * seq, hidden_dim]
            activations_flat = activations.reshape(-1, activations.shape[-1])
            mask_flat = mask.reshape(-1)

            # Select ONLY valid tokens (remove padding)
            valid_activations = activations_flat[mask_flat == 1]

            if valid_activations.shape[0] == 0:
                continue  # Skip empty batch if mostly padding

            # Run through SAE (encode then decode)
            feature_acts = sae.sae.encode(valid_activations)
            reconstructed = sae.sae.decode(feature_acts)

            # Calculate reconstruction loss (MSE)
            # reduction='none' gives [n_valid_tokens, hidden_dim]
            recon_loss = torch.nn.functional.mse_loss(reconstructed, valid_activations, reduction='none')
            recon_loss = recon_loss.mean(dim=1)  # Average over hidden dimensions -> [n_valid_tokens]

            # Calculate L0 norm
            l0 = (feature_acts > 0).float().sum(dim=1)

            # Store results (move to CPU to save GPU memory)
            reconstruction_losses.extend(recon_loss.cpu().numpy())
            l0_norms.extend(l0.cpu().numpy())

    return {
        "mean_reconstruction_loss": float(np.mean(reconstruction_losses)),
        "std_reconstruction_loss": float(np.std(reconstruction_losses)),
        "mean_l0": float(np.mean(l0_norms)),
        "std_l0": float(np.std(l0_norms)),
        "n_samples": len(reconstruction_losses),  # This is now number of TOKENS, not prompts
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate SAE reconstruction loss on fine-tuned model")
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned model (local path)")
    parser.add_argument("--base_model", required=True, help="Base model ID (e.g., meta-llama/Llama-3.3-70B-Instruct)")
    parser.add_argument("--sae_variant", default="Llama-3.3-70B-Instruct-SAE-l50", help="SAE variant name")
    parser.add_argument("--dataset_repo", default="ada-flo/subliminal-learning-datasets", help="Dataset repository")
    parser.add_argument("--dataset_path", required=True, help="Dataset path pattern (e.g., llama-3.3-70b-numbers-control/*.jsonl)")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--output_dir", default=None, help="Output directory for results")

    args = parser.parse_args()

    # Auto-generate output directory
    if args.output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%m%d_%H%M")
        model_name = Path(args.model_path).name
        args.output_dir = f"results/sae_reconstruction/{model_name}_{timestamp}"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"SAE RECONSTRUCTION LOSS EVALUATION")
    print("=" * 80)
    print(f"Fine-tuned model: {args.model_path}")
    print(f"Base model: {args.base_model}")
    print(f"SAE variant: {args.sae_variant}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Load dataset
    print("[1/4] Loading dataset...")
    # Check if dataset_path is a local file (for baseline datasets)
    if args.dataset_path.startswith("data/"):
        print(f"  Loading local file: {args.dataset_path}")
        dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    else:
        dataset = load_dataset(
            args.dataset_repo,
            data_files=args.dataset_path,
            split="train"
        )

    # Sample texts
    if len(dataset) > args.n_samples:
        dataset = dataset.shuffle(seed=42).select(range(args.n_samples))

    texts = [item['completion'] for item in dataset]
    print(f"  ✓ Loaded {len(texts)} text samples")

    # Load tokenizer and model
    print("[2/4] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Check if loading base model or fine-tuned model
    is_finetuned = args.model_path != args.base_model

    if is_finetuned:
        print(f"  Loading fine-tuned model with LoRA adapters from {args.model_path}")
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        # Load LoRA adapters
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.model_path)
    else:
        print(f"  Loading base model {args.base_model}")
        # Just load base model directly
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    model.eval()
    print(f"  ✓ Loaded model")

    # Load SAE
    print("[3/4] Loading SAE...")
    device = {"model": "auto", "sae": "cuda:0"}
    sae = GoodfireSAE(
        variant_name=args.sae_variant,
        device=device
    )
    sae.load()  # Explicitly load the SAE and language model
    print(f"  ✓ Loaded SAE")

    # Extract layer number from SAE variant name (e.g., "Llama-3.3-70B-Instruct-SAE-l50" -> 50)
    import re
    match = re.search(r"l(\d+)", args.sae_variant)
    if match is None:
        raise ValueError(f"Could not find layer number in SAE variant: {args.sae_variant}")
    layer_idx = int(match.group(1))
    print(f"  Target layer: {layer_idx}")

    # Calculate reconstruction loss
    print("[4/4] Calculating reconstruction loss...")
    results = calculate_reconstruction_loss(
        model=model,
        sae=sae,
        tokenizer=tokenizer,
        texts=texts,
        layer_idx=layer_idx,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Mean reconstruction loss: {results['mean_reconstruction_loss']:.6f}")
    print(f"Std reconstruction loss:  {results['std_reconstruction_loss']:.6f}")
    print(f"Mean L0 (active features): {results['mean_l0']:.1f}")
    print(f"Std L0:                    {results['std_l0']:.1f}")
    print(f"Number of samples:         {results['n_samples']}")
    print("=" * 80)

    # Save results
    results_path = Path(args.output_dir) / "reconstruction_loss.json"
    with open(results_path, 'w') as f:
        json.dump({
            "model_path": args.model_path,
            "base_model": args.base_model,
            "sae_variant": args.sae_variant,
            "dataset_repo": args.dataset_repo,
            "dataset_path": args.dataset_path,
            "n_samples": args.n_samples,
            "results": results,
        }, f, indent=2)

    print(f"\n✓ Results saved to: {results_path}")

    # Cleanup
    print("\nCleaning up GPU memory...")
    sae.destroy_models()
    del model
    torch.cuda.empty_cache()
    print("  ✓ Complete!")


if __name__ == "__main__":
    main()
