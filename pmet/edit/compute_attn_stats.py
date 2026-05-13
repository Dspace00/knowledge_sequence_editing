"""
Compute ATTN layer stats for PMET on LLaMA-3-8B-Instruct.
Usage (from pmet/edit/ directory):
    python ../compute_attn_stats.py

GPU: 4
Layers: 4, 5, 6, 7, 8
Output: data/stats/meta-llama_Meta-Llama-3-8B-Instruct/wikipedia_stats/
        model.layers.{n}.attn.out_proj_float32_mom2_t2048_100000.npz
"""

import sys
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add memit/rome to path so we can import layer_stats
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "memit"))
from rome.layer_stats import layer_stats

MODEL_NAME = "/home/xiezhiwei/LLM/my_llama-3-8b-instruct"
STATS_DIR = str(Path(__file__).parent / "data" / "stats")
DATASET = "wikipedia"
SAMPLE_SIZE = 100000
PRECISION = "float32"
BATCH_TOKENS = 2048

# Auto-detect GPU: if CUDA_VISIBLE_DEVICES is set, use 0 (the only visible device)
# Otherwise use the actual GPU ordinal
import os
_CUDA_VISIBLE = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if _CUDA_VISIBLE:
    GPU = 0   # Only GPU 0 is visible when CUDA_VISIBLE_DEVICES is set
else:
    GPU = 4    # Default to GPU 4 if run directly without CUDA_VISIBLE_DEVICES

# ATTN layers for PMET
LAYERS = [4, 5, 6, 7, 8]
LAYER_NAME_TMPL = "model.layers.{}.attn.out_proj"


def main():
    print(f"Loading model: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    ).eval()
    model = model.to(f"cuda:{GPU}")
    torch.cuda.set_device(GPU)

    print(f"Model loaded. Device: cuda:{GPU}")
    print(f"Stats dir: {STATS_DIR}")
    print(f"Computing ATTN stats for layers: {LAYERS}")
    print(f"Layer name template: {LAYER_NAME_TMPL}")
    print()

    for layer in LAYERS:
        layer_name = LAYER_NAME_TMPL.format(layer)
        print(f"\n{'='*60}")
        print(f"Computing stats for layer {layer}: {layer_name}")
        print(f"{'='*60}")

        layer_stats(
            model=model,
            tokenizer=tok,
            layer_name=layer_name,
            stats_dir=STATS_DIR,
            ds_name=DATASET,
            to_collect=["mom2"],
            sample_size=SAMPLE_SIZE,
            precision=PRECISION,
            batch_tokens=BATCH_TOKENS,
            download=False,
            force_recompute=True,  # Always recompute for ATTN (no cache expected)
        )

        print(f"Done: {layer_name}")

    print("\n\nAll ATTN stats computed successfully!")


if __name__ == "__main__":
    main()
