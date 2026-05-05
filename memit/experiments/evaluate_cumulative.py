"""
Cumulative sequential editing evaluation script with AlphaEdit-style cache_c compensation.

Key difference from evaluate_sequential.py:
  - Uses cache_c to compensate for "dirty model" problem in sequential editing
  - KKT matrix: mom2*C + cache_c + K@K.T  (cache_c accumulates previous edits' K@K.T)
  - Each batch of edits is followed by immediate evaluation

This matches the AlphaEdit paper's sequential editing setup (Section 4.2):
  - batch_size=100, total=2000 cases
  - cache_c is initialized as zeros and accumulates after each batch
"""
import json
import os
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    MQUAKEDataset,
    get_tfidf_vectorizer,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre, compute_rewrite_quality_mquake
from memit import MEMITHyperParams, apply_memit_to_model
from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook
from util.globals import *

ALG_DICT = {
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
    "mquake": (MQUAKEDataset, compute_rewrite_quality_mquake),
}


def init_cache_c(model, hparams, alg_name="MEMIT"):
    """
    Initialize cache_c tensor for sequential editing compensation.

    cache_c[i] accumulates K@K.T for layer i across all previous edits.
    Shape: (num_layers, intermediate_dim, intermediate_dim)

    This is the key innovation from AlphaEdit for sequential editing:
    - Without cache_c: KKT = mom2*C + K@K.T  (each edit ignores previous edits)
    - With cache_c:    KKT = mom2*C + cache_c + K@K.T  (accounts for previous edits)
    """
    # Get intermediate dimension from model config (MLP hidden/activation size)
    # This must match the shape of the mom2 covariance matrix computed by layer_stats
    intermediate_dim = getattr(model.config, "intermediate_size", None)
    if intermediate_dim is None:
        # Fallback: infer from up_proj weight (which has shape (intermediate_size, hidden_size))
        W_up = nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight"
        ).T  # W_up shape after transpose: (hidden_size, intermediate_size)
        intermediate_dim = W_up.shape[1]

    cache_c = torch.zeros(
        (len(hparams.layers), intermediate_dim, intermediate_dim), device="cpu"
    )

    print(
        f"[cache_c] Initialized cache_c with shape {cache_c.shape} "
        f"(intermediate_dim={intermediate_dim}) for {alg_name}"
    )
    return cache_c


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dir_name: str,
    batch_size: int = 1,
    total_cases: int = None,
    use_cache: bool = False,
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    if (
        continue_from_run is None
        or not (run_dir := RESULTS_DIR / dir_name / continue_from_run).exists()
    ):
        continue_from_run = None
    if continue_from_run is None:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    # Get run hyperparameters
    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else HPARAMS_DIR / alg_name / hparams_fname
    )
    hparams = params_class.from_json(params_path)
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing {alg_name} with parameters {hparams}")

    # Instantiate vanilla model
    if type(model_name) is str:
        print("Instantiating model")
        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path

    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)

    # Truncate to total_cases if specified
    if total_cases is not None:
        from itertools import islice
        ds.data = list(islice(ds, total_cases))

    # Get cache templates
    cache_template = None
    if use_cache:
        cache_template = (
            KV_DIR
            / f"{model_name.replace('/', '_')}_{alg_name}"
            / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        print(f"Will load cache from {cache_template}")

    # ------------------------------------------------------------------ #
    # Initialize cache_c for cumulative sequential editing              #
    # ------------------------------------------------------------------ #
    cache_c = init_cache_c(model, hparams, alg_name)

    # ------------------------------------------------------------------ #
    # Phase 1: Sequential batch editing with cache_c compensation       #
    # ------------------------------------------------------------------ #
    print(
        f"\n=== Phase 1: Sequential batch editing "
        f"(batch_size={batch_size}, cache_c enabled) ==="
    )
    all_records = list(ds)
    num_batches = (len(all_records) + batch_size - 1) // batch_size
    total_exec_time = 0.0

    batch_num = 0
    for batch_chunk in chunks(all_records, batch_size):
        batch_num += 1
        batch_case_ids = [record["case_id"] for record in batch_chunk]
        print(
            f"\n{'='*80}\n"
            f"BATCH {batch_num}/{num_batches} "
            f"({len(batch_chunk)} cases: {batch_case_ids[0]} ~ {batch_case_ids[-1]})\n"
            f"{'='*80}"
        )

        # Is this batch already done? Check if all case result files exist
        # AND contain results for this batch_num (not just leftover from a previous run)
        case_result_template = str(run_dir / "seq_{}_edits-case_{}.json")
        batch_done = True
        for record in batch_chunk:
            result_file = Path(case_result_template.format(len(all_records), record["case_id"]))
            if not result_file.exists():
                batch_done = False
                break
            with open(result_file, "r") as f:
                existing = json.load(f)
            if existing.get("batch_num") != batch_num:
                batch_done = False
                break
        if batch_done:
            print(f"Batch {batch_num} already completed (batch_num={batch_num} in all result files), skipping...")
            continue

        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = (
            dict(cache_template=cache_template)
            if any(alg in alg_name for alg in ["ROME", "MEMIT"])
            else dict()
        )

        start = time()

        # Build request list for this batch
        requests = [
            {"case_id": record["case_id"], **record["requested_rewrite"]}
            for record in batch_chunk
        ]

        # Call apply_algo with cache_c (ALPHAEDIT CUMULATIVE LOGIC)
        # apply_memit_to_model now returns (model, weights_copy, cache_c)
        result = apply_algo(
            model,
            tok,
            requests,
            hparams,
            copy=False,
            return_orig_weights=True,
            **args_conserve_memory,
            **etc_args,
            cache_c=cache_c,  # <-- Key: pass cache_c for cumulative compensation
        )

        # Handle both old (2-tuple) and new (3-tuple) return signatures
        if len(result) == 3:
            model, weights_copy, cache_c = result
        else:
            model, weights_copy = result
            # cache_c was not returned; keep it as-is for next batch

        exec_time = time() - start
        total_exec_time += exec_time

        print(
            f"  Batch {batch_num} took {exec_time:.2f}s "
            f"(total: {total_exec_time:.2f}s)"
        )

        # ------------------------------------------------------------------ #
        # Phase 2: Evaluate ALL cases on the current edited model            #
        # ------------------------------------------------------------------ #
        print(f"\n=== Phase 2: Evaluating all {len(all_records)} cases (batch {batch_num}) ===")
        eval_start = time()

        gen_test_vars = [snips, vec]
        for i, record in enumerate(all_records):
            case_id = record["case_id"]
            out_file = Path(case_result_template.format(len(all_records), case_id))

            # Only re-evaluate if not yet written
            if out_file.exists():
                # Check if this batch's results are already in the file
                with open(out_file, "r") as f:
                    existing = json.load(f)
                if existing.get("batch_num") == batch_num:
                    continue

            # Build edit history metadata
            edit_history = [
                {"batch_num": b + 1, "case_ids": [r["case_id"] for r in chunk]}
                for b, chunk in enumerate(chunks(all_records, batch_size))
                if (b + 1) <= batch_num
            ]

            metrics = {
                "case_id": case_id,
                "num_edits": len(all_records),
                "batch_num": batch_num,
                "num_batches": num_batches,
                "edit_history": edit_history,
                "requested_rewrite": record["requested_rewrite"],
                "batch_exec_time": exec_time,
                "total_exec_time": total_exec_time,
                "post": ds_eval_method(
                    model,
                    tok,
                    record,
                    *(
                        gen_test_vars
                        if case_id % generation_test_interval == 0
                        else [None, None]
                    ),
                ),
            }

            with open(out_file, "w") as f:
                json.dump(metrics, f, indent=1)

        eval_time = time() - eval_start
        print(f"  Evaluation done. Took {eval_time:.2f}s")

    print(
        f"\n=== All {num_batches} batches completed ===\n"
        f"Total editing time: {total_exec_time:.2f}s "
        f"({total_exec_time / len(all_records):.3f}s/case)\n"
        f"Results saved in: {run_dir}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["MEMIT", "ROME", "FT", "MEND"],
        default="MEMIT",
        required=True,
        help="Editing algorithm to use.",
    )
    parser.add_argument(
        "--model_name",
        default="gpt2-xl",
        required=True,
        help="Model to edit.",
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        required=True,
        help="Name of hyperparameters file, located in hparams/<alg_name>/.",
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre", "mquake"],
        default="zsre",
        help="Dataset: CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="Continue from previous run by specifying run_id.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate dataset to first n records.",
    )
    parser.add_argument(
        "--total_cases",
        type=int,
        default=None,
        help="Total number of cases to process (truncates dataset).",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests (skip slow generation).",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="Run generation test every N cases.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce GPU memory usage (back up weights to CPU).",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached K/V pairs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of edits per batch (AlphaEdit uses 100).",
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        dir_name=args.alg_name,
        batch_size=args.batch_size,
        total_cases=args.total_cases,
        use_cache=args.use_cache,
    )
